"""Word error rate of one or more transcripts against a reference text.

    uv run python tools/wer.py reference.txt transcript.txt [more.txt ...]

Text is lowercased and punctuation is ignored, so the score measures words only.
Note that numbers count as written ("4%" and "four percent" don't match).
"""
import re
import sys
from pathlib import Path


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower().replace("-", " ").replace("%", " percent"))


def wer(ref: list[str], hyp: list[str]) -> float:
    d = list(range(len(hyp) + 1))
    for i in range(1, len(ref) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(hyp) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ref[i - 1] != hyp[j - 1]))
    return d[-1] / max(1, len(ref))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    ref = words(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
    for f in sys.argv[2:]:
        print(f"{Path(f).name}: WER {wer(ref, words(Path(f).read_text(encoding='utf-8-sig'))):.1%}")
