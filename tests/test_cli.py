"""Fast unit tests: no GPU and no model download needed."""
import json

import numpy as np
import pytest

from murmur import __version__, cli


def test_fmt_ts():
    assert cli.fmt_ts(0) == "0:00"
    assert cli.fmt_ts(75.4) == "1:15"
    assert cli.fmt_ts(3725) == "1:02:05"
    assert cli.fmt_ts(3725.5, srt=True) == "01:02:05,500"


def test_paragraphs_break_on_pause_and_length():
    segs = [{"start": 0, "end": 5, "text": "Hello there."},
            {"start": 5.2, "end": 9, "text": "Same paragraph."},
            {"start": 15, "end": 18, "text": "After a pause."},
            {"start": 18, "end": 18, "text": "   "}]
    paras = cli.paragraphs(segs)
    assert [p["text"] for p in paras] == ["Hello there. Same paragraph.", "After a pause."]
    assert paras[1]["start"] == 15

    long = [{"start": i * 10, "end": i * 10 + 10, "text": f"Sentence {i}."} for i in range(20)]
    assert all(p["end"] - p["start"] <= 160 for p in cli.paragraphs(long))
    assert len(cli.paragraphs(long)) > 1


def test_output_stems_keep_extension_only_on_clash(tmp_path):
    a, b, c = tmp_path / "x.mp3", tmp_path / "x.qta", tmp_path / "y.m4a"
    stems = cli.output_stems([a, b, c], None)
    assert stems[a].name == "x.mp3" and stems[b].name == "x.qta" and stems[c].name == "y"
    out = tmp_path / "out"
    assert cli.output_stems([c], str(out))[c] == out / "y"


def test_collect_inputs(tmp_path):
    (tmp_path / "sub").mkdir()
    for name in ("a.mp3", "b.QTA", "notes.txt", "sub/c.mp4"):
        (tmp_path / name).write_bytes(b"")
    flat = {p.name for p in cli.collect_inputs([str(tmp_path)], recursive=False)}
    deep = {p.name for p in cli.collect_inputs([str(tmp_path)], recursive=True)}
    assert flat == {"a.mp3", "b.QTA"}
    assert deep == {"a.mp3", "b.QTA", "c.mp4"}
    dup = cli.collect_inputs([str(tmp_path / "a.mp3"), str(tmp_path / "a.mp3")], recursive=False)
    assert len(dup) == 1


def test_quiet_cut_lands_in_silence():
    sr = cli.SR
    audio = np.random.default_rng(0).normal(0, 0.3, 60 * sr).astype("float32")
    audio[33 * sr: 34 * sr] = 0.0  # one silent second, 3 s after the target
    cut = cli.quiet_cut(audio, 30 * sr)
    assert 33 * sr <= cut <= 34 * sr


def test_parallel_vad_covers_audio_contiguously(monkeypatch):
    import faster_whisper.vad as fv
    calls = []

    def fake(audio, vad_options=None, sampling_rate=16000, **kw):
        calls.append(len(audio))
        return [{"start": 0, "end": len(audio)}]

    monkeypatch.setattr(fv, "get_speech_timestamps", fake)
    monkeypatch.setattr(fv, "_murmur_original", fake, raising=False)
    cli.install_parallel_vad(workers=4)
    audio = np.zeros(10 * 60 * cli.SR, dtype="float32")
    spans = fv.get_speech_timestamps(audio, None)
    assert len(calls) >= 4  # actually split
    assert spans[0]["start"] == 0 and spans[-1]["end"] == len(audio)
    assert all(a["end"] == b["start"] for a, b in zip(spans, spans[1:]))


def test_pick_batch_size():
    assert cli.pick_batch_size("large-v3-turbo", False, 8) == 16
    assert cli.pick_batch_size("large-v3-turbo", True, 8) == 24
    assert cli.pick_batch_size("large-v3", False, 8) == 8
    assert cli.pick_batch_size("large-v3-turbo", False, 4) == 4
    assert cli.pick_batch_size("large-v3-turbo", False, 24) == 32
    assert cli.pick_batch_size("large-v3-turbo", False, None) == 16


def test_write_outputs_all_formats(tmp_path):
    segs = [{"start": 0.0, "end": 2.5, "text": "Hello world."}, {"start": 6.0, "end": 8.0, "text": "Bye."}]
    info = {"duration": 8.0, "language": "en", "language_probability": 0.99, "model": "large-v3-turbo"}
    src = tmp_path / "talk.mp3"
    written = cli.write_outputs(src, tmp_path / "talk", segs, info, {"md", "txt", "json", "srt", "vtt"})
    assert {p.suffix for p in written} == {".md", ".txt", ".json", ".srt", ".vtt"}
    md = (tmp_path / "talk.md").read_text(encoding="utf-8")
    assert "**[0:00]** Hello world." in md and "**[0:06]** Bye." in md and __version__ in md
    assert (tmp_path / "talk.txt").read_text(encoding="utf-8") == "Hello world.\n\nBye.\n"
    assert "00:00:06,000 --> 00:00:08,000" in (tmp_path / "talk.srt").read_text(encoding="utf-8")
    assert (tmp_path / "talk.vtt").read_text(encoding="utf-8").startswith("WEBVTT")
    data = json.loads((tmp_path / "talk.json").read_text(encoding="utf-8"))
    assert data["source"] == "talk.mp3" and len(data["segments"]) == 2


def test_cues_split_long_segments_on_words():
    words = [{"start": i * 0.5, "end": i * 0.5 + 0.4, "word": f" w{i}" + ("." if i == 9 else "")} for i in range(40)]
    seg = {"start": 0, "end": 20, "text": "x", "words": words}
    cs = cli.cues([seg])
    assert len(cs) > 2
    assert all(c["end"] - c["start"] <= 6.0 for c in cs)
    assert cs[0]["text"].endswith("w9.")  # breaks at the sentence end
    assert cli.cues([{"start": 0, "end": 3, "text": " plain "}]) == [{"start": 0, "end": 3, "text": "plain"}]


def test_json_drops_words_unless_requested(tmp_path):
    segs = [{"start": 0, "end": 1, "text": "Hi.", "words": [{"start": 0, "end": 1, "word": " Hi.", "p": 0.9}]}]
    info = {"duration": 1, "language": "en", "language_probability": 1, "model": "m"}
    cli.write_outputs(tmp_path / "a.mp3", tmp_path / "a", segs, info, {"json", "srt"})
    assert "words" not in json.loads((tmp_path / "a.json").read_text(encoding="utf-8"))["segments"][0]
    cli.write_outputs(tmp_path / "a.mp3", tmp_path / "b", segs, info, {"json"}, keep_words=True)
    assert "words" in json.loads((tmp_path / "b.json").read_text(encoding="utf-8"))["segments"][0]


def test_cli_argument_errors(tmp_path, capsys):
    assert cli.main([]) == 2
    f = tmp_path / "a.mp3"
    f.write_bytes(b"")
    assert cli.main([str(f), "-f", "docx"]) == 2
    assert cli.main([str(tmp_path / "missing.mp3")]) == 1


def test_skips_finished_files(tmp_path, capsys):
    f = tmp_path / "a.mp3"
    f.write_bytes(b"")
    (tmp_path / "a.md").write_text("done", encoding="utf-8")
    assert cli.main([str(f)]) == 0
    assert "skip" in capsys.readouterr().err


def test_version_flag(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--version"])
    assert __version__ in capsys.readouterr().out
