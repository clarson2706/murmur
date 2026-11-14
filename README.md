# Murmur

**Fast, accurate, private transcription.** Murmur turns audio and video into clean, timestamped
transcripts with OpenAI's Whisper, running on your own NVIDIA GPU. An hour of audio takes about
35 seconds, and nothing is uploaded anywhere.

```console
$ murmur lecture.m4a
-> loading large-v3-turbo on cuda (int8_float16)
   loaded in 2.9s
[1/1] lecture.m4a
   ... 7:33 / 29:33 (26%)
   ... 21:43 / 29:33 (73%)
   done: 29:33 audio in 17.9s (99x realtime)
lecture.md
lecture.txt
lecture.json
```

## Why Murmur

- **Fast.** About 100x realtime on a mid-range GPU (RTX 3060 Ti), with batched GPU inference,
  int8 weights and a pipelined CPU stage. Details are in [How it's fast](#how-its-fast).
- **Accurate.** Whisper large-v3-turbo with beam search scored 0.8% word error rate in our tests.
  Every default was benchmarked: settings that saved time but lost accuracy were left out.
- **Private.** Fully offline after a one-time model download. Your recordings stay on your machine.
- **Reads almost anything.** mp3, m4a, wav, flac, ogg, opus, aac, wma, Apple Voice Memos (`.qta`),
  and the audio track of mp4, mov, mkv, webm and avi videos. No ffmpeg install needed.
- **Useful output.** Readable paragraphs with timestamps (`.md`), plain text (`.txt`), structured
  segments (`.json`), and subtitles (`.srt`, `.vtt`) cut to 1–6 second cues from word timings.
- **Batch-friendly.** Point it at a folder. Files that already have transcripts are skipped, so
  re-running only does the new ones.
- **No CUDA setup.** The CUDA libraries it needs install as ordinary Python packages.

## Install

You need Python 3.10–3.13 and, for GPU speed, an NVIDIA GPU with a recent driver.
The simplest route is [uv](https://docs.astral.sh/uv/):

```sh
uv tool install git+https://github.com/clarson2706/murmur
murmur --check        # shows your GPU and the settings Murmur will use
```

With pipx: `pipx install git+https://github.com/clarson2706/murmur`.

The first transcription downloads the model (~1.6 GB) once. Without an NVIDIA GPU, Murmur runs on
the CPU instead, which works but is much slower. That includes Macs.

## Use

```sh
murmur talk.mp3                          # writes talk.md, talk.txt and talk.json next to it
murmur recordings/ -o transcripts -r     # a whole folder, including subfolders
murmur interview.mp4 -f md,srt           # pick output formats
murmur podcast.m4a --fast                # ~10% faster, greedy decoding
murmur hard-audio.wav -m large-v3        # full-size model, ~2.7x slower, a bit more robust
murmur meeting.m4a -l auto               # detect the language (default: English)
```

Progress goes to stderr. stdout lists the files written, one per line, so it's easy to script.

| Option | What it does |
|---|---|
| `-o, --out DIR` | Output folder (default: next to each input) |
| `-f, --formats` | Any of `md,txt,json,srt,vtt` (default `md,txt,json`) |
| `-m, --model` | `turbo` (default), `large-v3`, or any faster-whisper model name or path |
| `-l, --language` | Language code, or `auto` (default `en`) |
| `--fast` | Greedy decoding and bigger batches |
| `-r, --recursive` | Include subfolders |
| `--force` | Redo files that already have a transcript |
| `--words` | Word-level timestamps in the `.json` |
| `--check` | Show the hardware and settings detected, then exit |

Advanced options: `-b/--batch-size` (default is picked from your VRAM), `--beam`, `--device`,
`--compute-type`, `-p/--prompt`. Run `murmur --help` for everything.

> **About `--prompt`.** Whisper can take a text hint, and it's often recommended for fixing
> unusual names. In our tests, with batched inference, it made the model **skip whole sentences**
> (WER rose from 0.8% to 1.3–9.9%). Murmur leaves it off by default. Only use it if a transcript
> misspells key names, and compare the result. (`--hotwords` was even worse, at 12–27% WER, so it
> isn't offered.)

### Output

`lecture.md`:

```markdown
# Transcript: lecture.m4a

- Duration: 29:33
- Language: en (100%)
- Model: large-v3-turbo
- Transcribed: 2026-11-14 12:04 with Murmur 0.1.0

**[0:00]** Good morning everyone. Today we are going to talk about the capital asset pricing
model, which relates the expected return on a security to its systematic risk, measured by beta.
The expected return equals the risk-free rate plus beta times the market risk premium. So if the
risk-free rate is 4%, the expected market return is 10%, and the stock has a beta of 1.2, the
expected return is 11.2%...
```

## Performance

Measured on an RTX 3060 Ti (8 GB) and a 12-thread CPU on Windows 11. The 30-minute file is a known text read
by a speech synthesizer, so word error rate (WER) can be measured exactly.

| Mode | Speed | WER |
|---|---|---|
| **default** (turbo, beam 5, int8_float16) | **~100–110x realtime** | **0.8%** |
| `--fast` (beam 1, larger batches) | ~120x | 0.8% |
| `-m large-v3` | ~41x | 0.8% |
| plain faster-whisper, same model, default settings | ~78x | 0.8% |

On a real 75-minute lecture recording it ran at 113x realtime. Synthesized speech is cleaner than a
real room, so expect a higher error rate on real recordings. The relative speeds hold.

### How it's fast

1. **Batched GPU inference.** Speech is cut into chunks of up to 30 seconds at natural pauses, and
   16 chunks run through Whisper together.
2. **int8 weights, fp16 compute.** About 10% faster than plain fp16, with no measured accuracy loss.
3. **Right-sized batches.** The batch size is chosen from your GPU memory. Overfilling VRAM makes
   Windows page to system RAM, which is 3–4x slower, so Murmur doesn't.
4. **Parallel voice detection.** Silero VAD finds the speech, but it's a sequential model on one CPU
   thread. Murmur splits long audio at its quietest moments and runs the pieces in parallel.
5. **Pipelining.** The next file is decoded and voice-detected on a background thread while the GPU
   transcribes the current one. The first file is prepared while the model loads.

At that point the GPU runs near 100%, so it's the limit.

## Privacy

Murmur goes online only to download the model from Hugging Face the first time you use it.
After that it loads the model from disk without contacting any server, and works with no internet
connection. Audio, transcripts and file names never leave your computer.

## Development

```sh
git clone https://github.com/clarson2706/murmur && cd murmur
uv sync
uv run pytest                                   # unit tests, no GPU needed
uv run murmur --check
uv run python tools/wer.py reference.txt transcript.txt   # score a transcript
```

## Credits and license

Murmur is MIT licensed (see [LICENSE](LICENSE)). It's built on excellent open-source work:
[Whisper](https://github.com/openai/whisper) (OpenAI), [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
and [CTranslate2](https://github.com/OpenNMT/CTranslate2) (SYSTRAN / OpenNMT),
[Silero VAD](https://github.com/snakers4/silero-vad), and [PyAV](https://github.com/PyAV-Org/PyAV) / FFmpeg.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
