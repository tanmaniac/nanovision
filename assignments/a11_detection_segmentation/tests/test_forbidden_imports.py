"""Anti-cheat: the detection mechanism (boxes, matcher, loss) is built from scratch.

Scans the A11 top-level holed files and their solution copies for ready-made detection ops
and end-to-end DETR/segmentation libraries (torchvision.ops box/giou helpers, detectron2,
ultralytics, mmdet, transformers' detection models). scipy.optimize.linear_sum_assignment is
ALLOWED - it is the matcher's intended tool, the exact Hungarian solver. Comments and string
literals are stripped via tokenize, so a forbidden name in a docstring does not trip it.
"""

import re
import tokenize
from pathlib import Path

_ASSIGN = Path(__file__).resolve().parents[1]

_FILES = [
    _ASSIGN / "boxes.py",
    _ASSIGN / "matcher.py",
    _ASSIGN / "loss.py",
    _ASSIGN / "solution" / "boxes.py",
    _ASSIGN / "solution" / "matcher.py",
    _ASSIGN / "solution" / "loss.py",
]

_FORBIDDEN = [
    r"import\s+torchvision",
    r"from\s+torchvision",
    r"import\s+detectron2",
    r"from\s+detectron2",
    r"import\s+ultralytics",
    r"from\s+ultralytics",
    r"import\s+mmdet",
    r"from\s+mmdet",
    r"import\s+mmcv",
    r"from\s+mmcv",
    r"from\s+transformers",
    r"import\s+transformers",
]

_SKIP = {tokenize.COMMENT, tokenize.STRING, tokenize.NL, tokenize.NEWLINE,
         tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING, tokenize.ENDMARKER}


def _code_only(path: Path) -> str:
    parts = []
    with open(path) as f:
        for tok in tokenize.generate_tokens(f.readline):
            if tok.type in _SKIP:
                continue
            parts.append(tok.string)
    return " ".join(parts)


def test_no_forbidden_imports():
    for f in _FILES:
        code = _code_only(f)
        for pat in _FORBIDDEN:
            assert re.search(pat, code) is None, f"forbidden pattern {pat!r} found in {f}"
