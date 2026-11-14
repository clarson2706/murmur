"""Murmur command-line interface."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from . import __version__

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
import warnings  # noqa: E402

warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")

AUDIO_EXTS = {".mp3", ".qta", ".m4a", ".wav", ".mp4", ".mov", ".flac", ".ogg", ".opus",
              ".webm", ".aac", ".wma", ".mkv", ".amr", ".3gp", ".m4v", ".avi"}
MODEL_ALIASES = {"turbo": "large-v3-turbo", "accurate": "large-v3"}
LARGE_MODELS = {"large-v3", "large-v2", "large-v1", "large"}
SR = 16000

# id(audio) -> (audio, vad_options, timestamps); filled by the prefetch thread, consumed by the pipeline
_VAD_CACHE: dict[int, tuple] = {}


# ---------------------------------------------------------------- output helpers

def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def fmt_ts(seconds: float, srt: bool = False) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    if srt:
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


# ---------------------------------------------------------------- hardware

def register_cuda_libs() -> None:
    """Make pip-installed cuBLAS / cuDNN (the nvidia-* wheels) loadable, so no system CUDA install is needed."""
    import importlib.util
    spec = importlib.util.find_spec("nvidia")
    if not spec or not spec.submodule_search_locations:
        return
    roots = [Path(p) for p in spec.submodule_search_locations]
    if os.name == "nt":
        for bin_dir in (d for r in roots for d in r.glob("*/bin")):
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
    elif sys.platform.startswith("linux"):
        import ctypes
        libs = sorted(f for r in roots for f in r.glob("*/lib/lib*.so*")
                      if f.name.startswith(("libcublas", "libcudnn")))
        for _ in range(2):  # second pass picks up libraries whose dependencies loaded in the first
            for f in libs:
                try:
                    ctypes.CDLL(str(f), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


def gpu_info() -> tuple[str, float] | None:
    """(name, total VRAM in GB) of the first NVIDIA GPU, via nvidia-smi."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        name, mem = out[0].rsplit(",", 1)
        return name.strip(), float(mem) / 1024
    except Exception:
        return None


def pick_batch_size(model: str, fast: bool, vram_gb: float | None) -> int:
    """Largest batch that stays inside VRAM. Overflowing makes Windows page to system RAM (~3x slower)."""
    big = model in LARGE_MODELS
    if vram_gb is None:
        return 8 if big else 16
    if big:
        return 16 if vram_gb >= 11 else 8 if vram_gb >= 7 else 4
    if vram_gb >= 11:
        return 32
    if vram_gb >= 7:
        return 24 if fast else 16
    return 8 if vram_gb >= 5 else 4


# ---------------------------------------------------------------- fast VAD + prefetch

def quiet_cut(audio, target: int, window: int = SR // 2, search: int = 10 * SR) -> int:
    """Sample index near `target` in the quietest half-second, so a split never cuts a word."""
    import numpy as np
    lo, hi = max(0, target - search), min(len(audio), target + search)
    seg = audio[lo:hi]
    n = len(seg) // window
    if n < 2:
        return target
    energy = np.square(seg[: n * window].reshape(n, window)).mean(axis=1)
    return lo + int(energy.argmin()) * window + window // 2


def install_parallel_vad(workers: int) -> None:
    """Silero VAD is a sequential LSTM on one CPU thread (~3 s per 30 min of audio).
    Split long audio at quiet points and run the pieces concurrently instead."""
    import faster_whisper.transcribe as ft
    import faster_whisper.vad as fv
    from concurrent.futures import ThreadPoolExecutor

    orig = getattr(fv, "_murmur_original", fv.get_speech_timestamps)
    fv._murmur_original = orig
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="vad")

    def parallel(audio, vad_options=None, sampling_rate=SR, **kw):
        hit = _VAD_CACHE.pop(id(audio), None)
        if hit and hit[0] is audio and hit[1] == vad_options and not kw:
            return hit[2]
        n = len(audio)
        piece = max(60 * sampling_rate, n // workers)
        if workers < 2 or n < 2 * piece:
            return orig(audio, vad_options, sampling_rate, **kw)
        cuts = [0] + [quiet_cut(audio, k) for k in range(piece, n - piece // 2, piece)] + [n]
        spans = list(zip(cuts, cuts[1:]))
        parts = pool.map(lambda ab: orig(audio[ab[0]:ab[1]], vad_options, sampling_rate, **kw), spans)
        return [{"start": s["start"] + a, "end": s["end"] + a} for (a, _), segs in zip(spans, parts) for s in segs]

    ft.get_speech_timestamps = parallel
    fv.get_speech_timestamps = parallel


def prepare_audio(src: Path, batched: bool):
    """Decode, and in batched mode pre-run VAD. Runs on a background thread for the next file."""
    from faster_whisper import decode_audio
    import faster_whisper.vad as fv
    audio = decode_audio(str(src), sampling_rate=SR)
    if batched:  # the same options BatchedInferencePipeline builds internally, so the cache hits
        opts = fv.VadOptions(max_speech_duration_s=30, min_silence_duration_ms=160)
        _VAD_CACHE[id(audio)] = (audio, opts, fv.get_speech_timestamps(audio, opts))
    return audio


# ---------------------------------------------------------------- inputs / outputs

def collect_inputs(paths: list[str], recursive: bool) -> list[Path]:
    files: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            it = p.rglob("*") if recursive else p.iterdir()
            files += sorted(f for f in it if f.is_file() and f.suffix.lower() in AUDIO_EXTS)
        elif p.is_file():
            files.append(p)
        else:
            log(f"!! not found: {p}")
    seen, out = set(), []
    for f in files:
        key = f.resolve()
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out


def paragraphs(segments: list[dict], gap: float = 2.0, max_len: float = 75.0) -> list[dict]:
    """Group segments into readable paragraphs: break on pauses, or on length at a sentence end."""
    paras: list[dict] = []
    cur: dict | None = None
    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        if cur is not None:
            pause = seg["start"] - cur["end"]
            length = cur["end"] - cur["start"]
            sentence_end = cur["text"].rstrip().endswith((".", "?", "!"))
            if pause >= gap or (length >= max_len and sentence_end) or length >= max_len * 2:
                paras.append(cur)
                cur = None
        if cur is None:
            cur = {"start": seg["start"], "end": seg["end"], "text": text}
        else:
            cur["end"] = seg["end"]
            cur["text"] += " " + text
    if cur:
        paras.append(cur)
    return paras


def cues(segments: list[dict], max_dur: float = 6.0, max_chars: int = 84) -> list[dict]:
    """Subtitle-sized cues built from word timings: at most ~6 s / 2 lines, preferring sentence ends."""
    out: list[dict] = []
    for seg in segments:
        words = seg.get("words")
        if not words:
            out.append({"start": seg["start"], "end": seg["end"], "text": seg["text"].strip()})
            continue
        cur: list[dict] = []
        for w in words:
            if cur:
                text = "".join(x["word"] for x in cur).strip()
                dur = w["end"] - cur[0]["start"]
                if dur > max_dur or len(text) + len(w["word"]) > max_chars or (
                        text.endswith((".", "?", "!")) and w["start"] - cur[0]["start"] > 1.5):
                    out.append({"start": cur[0]["start"], "end": cur[-1]["end"], "text": text})
                    cur = []
            cur.append(w)
        if cur:
            out.append({"start": cur[0]["start"], "end": cur[-1]["end"],
                        "text": "".join(x["word"] for x in cur).strip()})
    return [c for c in out if c["text"]]


def output_stems(files: list[Path], out: str | None) -> dict[Path, Path]:
    """Output path (without extension) per input; keeps the source extension when two inputs share a name."""
    def out_dir(f: Path) -> Path:
        return Path(out) if out else f.parent
    keys = Counter((out_dir(f).resolve(), f.stem.lower()) for f in files)
    stems = {}
    for f in files:
        clash = keys[(out_dir(f).resolve(), f.stem.lower())] > 1
        stems[f] = out_dir(f) / (f"{f.stem}.{f.suffix.lstrip('.').lower()}" if clash else f.stem)
    return stems


def write_outputs(src: Path, stem: Path, segments: list[dict], info: dict, formats: set[str],
                  keep_words: bool = False) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    paras = paragraphs(segments)

    def emit(ext: str, text: str) -> None:
        p = Path(f"{stem}.{ext}")
        p.write_text(text, encoding="utf-8")
        written.append(p)

    if "md" in formats:
        head = (f"# Transcript: {src.name}\n\n"
                f"- Duration: {fmt_ts(info['duration'])}\n"
                f"- Language: {info['language']} ({info['language_probability']:.0%})\n"
                f"- Model: {info['model']}\n"
                f"- Transcribed: {time.strftime('%Y-%m-%d %H:%M')} with Murmur {__version__}\n\n")
        emit("md", head + "\n\n".join(f"**[{fmt_ts(x['start'])}]** {x['text']}" for x in paras) + "\n")
    if "txt" in formats:
        emit("txt", "\n\n".join(x["text"] for x in paras) + "\n")
    if "srt" in formats or "vtt" in formats:
        subs = cues(segments)
        if "srt" in formats:
            emit("srt", "\n".join(f"{i}\n{fmt_ts(c['start'], True)} --> {fmt_ts(c['end'], True)}\n{c['text']}\n"
                                  for i, c in enumerate(subs, 1)))
        if "vtt" in formats:
            emit("vtt", "WEBVTT\n\n" + "\n".join(
                f"{fmt_ts(c['start'], True).replace(',', '.')} --> {fmt_ts(c['end'], True).replace(',', '.')}\n"
                f"{c['text']}\n" for c in subs))
    if "json" in formats:
        segs = segments if keep_words else [{k: v for k, v in s.items() if k != "words"} for s in segments]
        emit("json", json.dumps({"source": src.name, **info, "segments": segs}, indent=1, ensure_ascii=False))
    return written


# ---------------------------------------------------------------- model + transcription

def model_is_cached(name: str) -> bool:
    try:
        from faster_whisper.utils import download_model
        download_model(name, local_files_only=True)
        return True
    except Exception:
        return Path(name).is_dir()


def load_model(name: str, device: str, compute_type: str | None):
    from faster_whisper import WhisperModel
    import ctranslate2

    if device == "auto":
        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    cached = model_is_cached(name)  # if so, load offline: no network call at all after the first run
    if not cached:
        size = "~1.6 GB" if "turbo" in name else "~3 GB" if name in LARGE_MODELS else ""
        log(f"-> first run: downloading model {name} {size} (one time only)")
        from huggingface_hub.utils import enable_progress_bars
        enable_progress_bars()

    def build(dev: str, ct: str | None):
        # int8 weights + fp16 compute: ~10% faster than float16 with no measured accuracy loss, less VRAM
        ct = ct or ("int8_float16" if dev == "cuda" else "int8")
        log(f"-> loading {name} on {dev} ({ct})")
        return WhisperModel(name, device=dev, compute_type=ct, cpu_threads=os.cpu_count() or 4,
                            local_files_only=cached), dev

    t = time.perf_counter()
    try:
        model, device = build(device, compute_type)
    except Exception as e:
        if device != "cuda":
            raise
        log(f"!! GPU unavailable ({str(e).splitlines()[0][:120]}); falling back to CPU (much slower)")
        model, device = build("cpu", None)
    log(f"   loaded in {time.perf_counter() - t:.1f}s")
    return model, device


def transcribe_audio(pipeline, audio, args, model_name: str) -> tuple[list[dict], dict]:
    kwargs = dict(
        language=None if args.language == "auto" else args.language,
        beam_size=args.beam,
        vad_filter=True,
        initial_prompt=args.prompt,
        word_timestamps=args.need_words,
    )
    if args.batch_size > 1:
        kwargs["batch_size"] = args.batch_size
    else:
        kwargs.update(condition_on_previous_text=False, vad_parameters=dict(min_silence_duration_ms=500))
    seg_iter, info = pipeline.transcribe(audio, **kwargs)

    segments: list[dict] = []
    last_report = time.perf_counter()
    for s in seg_iter:
        item = {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
        if args.need_words and s.words:
            item["words"] = [{"start": round(w.start, 2), "end": round(w.end, 2), "word": w.word,
                              "p": round(w.probability, 3)} for w in s.words]
        segments.append(item)
        now = time.perf_counter()
        if now - last_report > 5 and info.duration:
            log(f"   ... {fmt_ts(s.end)} / {fmt_ts(info.duration)} ({s.end / info.duration:.0%})")
            last_report = now
    meta = {"duration": round(info.duration, 2), "language": info.language,
            "language_probability": round(info.language_probability, 3), "model": model_name}
    return segments, meta


# ---------------------------------------------------------------- commands

def check() -> int:
    """Print what Murmur sees on this machine and the settings it would use."""
    register_cuda_libs()
    print(f"Murmur {__version__}  ·  Python {sys.version.split()[0]}  ·  {sys.platform}")
    g = gpu_info()
    print(f"GPU:    {f'{g[0]} ({g[1]:.0f} GB)' if g else 'no NVIDIA GPU found (CPU mode, much slower)'}")
    try:
        import ctranslate2
        n = ctranslate2.get_cuda_device_count()
        print(f"CUDA:   {'ready' if n else 'not available'} (CTranslate2 {ctranslate2.__version__})")
        if n:
            print(f"        compute types: {', '.join(sorted(ctranslate2.get_supported_compute_types('cuda')))}")
    except Exception as e:
        print(f"CUDA:   error: {e}")
    name = MODEL_ALIASES["turbo"]
    print(f"Model:  {name} {'(downloaded)' if model_is_cached(name) else '(downloads ~1.6 GB on first use)'}")
    print(f"Batch:  {pick_batch_size(name, False, g[1] if g else None)} (default), "
          f"{pick_batch_size(name, True, g[1] if g else None)} (--fast)")
    print(f"CPU:    {os.cpu_count()} threads")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="murmur",
        description="Fast, accurate, private transcription. Runs Whisper on your own GPU, "
                    "so audio never leaves your machine.",
        epilog="examples:\n"
               "  murmur lecture.mp3\n"
               "  murmur recordings/ -o transcripts -r\n"
               "  murmur interview.mp4 -f md,srt\n"
               "  murmur podcast.m4a --fast\n"
               "  murmur --check",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("inputs", nargs="*", help="audio/video files and/or folders")
    ap.add_argument("-o", "--out", help="output folder (default: next to each input)")
    ap.add_argument("-m", "--model", default="turbo",
                    help="turbo (default) | large-v3 (~2.7x slower, slightly more robust on very hard audio) "
                         "| any faster-whisper model name or local path")
    ap.add_argument("-l", "--language", default="en", help="language code, or 'auto' to detect (default: en)")
    ap.add_argument("-f", "--formats", default="md,txt,json", help="comma list of md,txt,json,srt,vtt "
                                                                  "(default: md,txt,json)")
    ap.add_argument("--fast", action="store_true", help="greedy decoding and bigger batches: ~10%% faster")
    ap.add_argument("--words", action="store_true", help="add word-level timestamps to the json output")
    ap.add_argument("-r", "--recursive", action="store_true", help="include subfolders")
    ap.add_argument("--force", action="store_true", help="redo files that already have a transcript")
    ap.add_argument("--check", action="store_true", help="show detected hardware and settings, then exit")
    ap.add_argument("--version", action="version", version=f"murmur {__version__}")
    adv = ap.add_argument_group("advanced")
    adv.add_argument("-b", "--batch-size", type=int, help="chunks per GPU batch (default: picked from VRAM); "
                                                          "1 = sequential mode")
    adv.add_argument("--beam", type=int, help="beam size (default 5, or 1 with --fast)")
    adv.add_argument("-p", "--prompt",
                     help="text hint fed to every chunk. Can fix spelling of unusual names, but in testing it made "
                          "the model skip sentences (WER 0.8%% -> 1.3-9.9%%). Off by default; compare before relying on it")
    adv.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    adv.add_argument("--compute-type", help="int8_float16 (default on GPU), float16, int8, float32, ...")
    return ap


def run(args) -> int:
    formats = {f.strip().lower() for f in args.formats.split(",") if f.strip()}
    unknown = formats - {"md", "txt", "json", "srt", "vtt"}
    if unknown or not formats:
        log(f"!! unknown format(s): {', '.join(sorted(unknown)) or '(none)'}; use md,txt,json,srt,vtt")
        return 2
    model_name = MODEL_ALIASES.get(args.model, args.model)
    args.need_words = args.words or bool(formats & {"srt", "vtt"})  # subtitles are cut from word timings

    files = collect_inputs(args.inputs, args.recursive)
    if not files:
        log("no audio or video files found")
        return 1
    stems = output_stems(files, args.out)
    primary = next(f for f in ("md", "txt", "json", "srt", "vtt") if f in formats)
    todo = [f for f in files if args.force or not Path(f"{stems[f]}.{primary}").exists()]
    for f in files:
        if f not in todo:
            log(f"== skip (already done): {f.name}")
    if not todo:
        return 0

    register_cuda_libs()
    g = gpu_info() if args.device != "cpu" else None
    if args.beam is None:
        args.beam = 1 if args.fast else 5
    if args.batch_size is None:
        args.batch_size = pick_batch_size(model_name, args.fast, g[1] if g else None)
    install_parallel_vad(workers=min(8, os.cpu_count() or 4))
    batched = args.batch_size > 1

    from concurrent.futures import ThreadPoolExecutor
    # Decode + VAD the first file while the model loads, then each next file while the GPU works.
    prefetch = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prefetch")
    pending = prefetch.submit(prepare_audio, todo[0], batched)

    from faster_whisper import BatchedInferencePipeline
    model, device = load_model(model_name, args.device, args.compute_type)
    pipeline = BatchedInferencePipeline(model=model) if batched else model
    if device == "cpu" and batched and args.batch_size > 4:
        args.batch_size = 4

    failures = 0
    total_audio = 0.0
    t_run = time.perf_counter()
    for i, f in enumerate(todo, 1):
        log(f"[{i}/{len(todo)}] {f.name}")
        t = time.perf_counter()
        audio, err = None, None
        try:
            audio = pending.result()
        except Exception as e:
            err = e
        if i < len(todo):
            pending = prefetch.submit(prepare_audio, todo[i], batched)
        if audio is None:
            log(f"!! could not read {f.name}: {err}")
            failures += 1
            continue
        try:
            segments, meta = transcribe_audio(pipeline, audio, args, model_name)
        except Exception as e:
            if device == "cuda" and "out of memory" in str(e).lower() and args.batch_size > 4:
                log(f"   GPU out of memory at batch {args.batch_size}; retrying at 4")
                args.batch_size = 4
                segments, meta = transcribe_audio(pipeline, audio, args, model_name)
            else:
                log(f"!! failed: {f.name}: {e}")
                failures += 1
                continue
        del audio
        dt = time.perf_counter() - t
        meta["elapsed_s"] = round(dt, 1)
        written = write_outputs(f, stems[f], segments, meta, formats, keep_words=args.words)
        total_audio += meta["duration"]
        log(f"   done: {fmt_ts(meta['duration'])} audio in {dt:.1f}s "
            f"({meta['duration'] / dt if dt else 0:.0f}x realtime)")
        for w in written:
            print(w, flush=True)  # stdout = the files written, one per line, easy for scripts to parse

    prefetch.shutdown()
    wall = time.perf_counter() - t_run
    if total_audio and wall:
        log(f"== {len(todo) - failures}/{len(todo)} files, {fmt_ts(total_audio)} of audio in {wall:.0f}s "
            f"({total_audio / wall:.0f}x realtime)")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.check:
        return check()
    if not args.inputs:
        ap.print_usage(sys.stderr)
        log("murmur: give one or more audio/video files or folders (or --check)")
        return 2
    try:
        return run(args)
    except KeyboardInterrupt:
        log("\ninterrupted")
        return 130
