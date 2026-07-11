"""Anti-cheat: the fusion mechanisms use no perception library or external pooling kernel.

Scans the holed fusion.py / transfuser.py, their solutions, and the provided compare.py. Bans
cv2, kornia, mmdet3d, mmcv, spconv, torch_scatter, and open3d (the pillar scatter-MAX must be
plain torch, the projection must be the course geometry), and bans importing the answer key from
the ``solution`` package. Comments and string literals are stripped via tokenize, so a forbidden
name in a docstring does not trip it.
"""

import re
import tokenize
from pathlib import Path

_ASSIGN = Path(__file__).resolve().parents[1]

_FILES = [
    _ASSIGN / "fusion.py",
    _ASSIGN / "transfuser.py",
    _ASSIGN / "compare.py",
    _ASSIGN / "solution" / "fusion.py",
    _ASSIGN / "solution" / "transfuser.py",
]

_FORBIDDEN = [
    r"import\s+cv2",
    r"import\s+kornia",
    r"from\s+kornia",
    r"import\s+mmdet3d",
    r"from\s+mmdet3d",
    r"import\s+mmcv",
    r"from\s+mmcv",
    r"import\s+spconv",
    r"from\s+spconv",
    r"import\s+torch_scatter",
    r"from\s+torch_scatter",
    r"import\s+open3d",
    r"from\s+open3d",
    # The answer key must not be imported directly (tests/viz use the shim path instead).
    r"^\s*import\s+solution\b",
    r"^\s*from\s+solution\b",
    r"from\s+.*\.solution\.",
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


def _raw_lines(path: Path) -> str:
    out = []
    with open(path) as f:
        for line in f:
            if line.lstrip().startswith("#"):
                continue
            out.append(line)
    return "".join(out)


def test_no_forbidden_imports():
    for f in _FILES:
        assert f.exists(), f"expected file not found: {f}"
        code = _code_only(f)
        raw = _raw_lines(f)
        for pat in _FORBIDDEN:
            target = raw if pat.startswith("^") else code
            flags = re.MULTILINE if pat.startswith("^") else 0
            assert re.search(pat, target, flags) is None, f"forbidden pattern {pat!r} in {f}"
