# Changelog

## 0.1.0 (2026-11-14)

First release.

- Transcribes audio and video files, or whole folders, with Whisper large-v3-turbo on an NVIDIA
  GPU, at about 100x realtime.
- Writes Markdown with timestamps, plain text, JSON segments, and SRT/VTT subtitles cut from word timings.
- Picks the batch size from GPU memory, falls back to CPU automatically, and installs CUDA
  libraries as pip packages (no CUDA toolkit needed).
- Runs voice detection in parallel and prepares the next file during transcription.
- Skips files that are already transcribed, and keeps outputs separate when two inputs share a name.
- `murmur --check` shows the detected hardware and the settings it will use.
