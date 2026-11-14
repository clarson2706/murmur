# Third-party notices

Murmur's own code is MIT licensed. It contains no third-party code. The components below are
installed separately as dependencies (by pip or uv), or downloaded at first run, and each keeps its
own license.

| Component | Used for | License |
|---|---|---|
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | Whisper inference, batching, VAD integration | MIT |
| [CTranslate2](https://github.com/OpenNMT/CTranslate2) | GPU/CPU inference engine | MIT |
| [Whisper](https://github.com/openai/whisper) model weights (large-v3, large-v3-turbo), downloaded at first run | Speech recognition model | MIT |
| [Silero VAD](https://github.com/snakers4/silero-vad) (bundled in faster-whisper) | Voice activity detection | MIT |
| [PyAV](https://github.com/PyAV-Org/PyAV) | Audio decoding | BSD-3-Clause |
| FFmpeg libraries (bundled in PyAV's binary wheels) | Audio/video codecs | LGPL v3 or later. The wheels also include codec libraries under other licenses, some GPL (for example x264 and x265). See PyAV's distribution. |
| NVIDIA cuBLAS and cuDNN (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`) | GPU math libraries | NVIDIA software license agreements |
| [Hugging Face Hub](https://github.com/huggingface/huggingface_hub) client | Model download | Apache-2.0 |
| NumPy, tokenizers, onnxruntime and other transitive dependencies | Runtime | Their respective licenses (BSD, Apache-2.0, MIT) |

Because Murmur doesn't redistribute these components, their licenses apply to them as installed.
If you bundle Murmur together with its dependencies (for example, in a standalone installer),
you're responsible for complying with each component's license, including the LGPL/GPL terms of
the FFmpeg build and NVIDIA's redistribution terms.
