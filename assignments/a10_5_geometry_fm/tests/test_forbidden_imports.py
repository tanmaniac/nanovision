"""Anti-cheat: the pointmap mechanism is built from scratch.

Scans the A10.5 top-level files, the solution copies, and the geometry shim for prebuilt
DUSt3R / MASt3R / CroCo packages, and forbids importing the owned shared geometry file by its
bare name (it must be reached via nanovision.geometry). Comments and string literals are
stripped via tokenize, so a forbidden name in a docstring does not trip it.
"""

import re
import tokenize
from pathlib import Path

_ASSIGN = Path(__file__).resolve().parents[1]
_REPO = _ASSIGN.parents[1]

_FILES = [
    _ASSIGN / "head.py",
    _ASSIGN / "loss.py",
    _ASSIGN / "geometry_fm.py",
    _ASSIGN / "model.py",
    _ASSIGN / "toy_scene.py",
    _ASSIGN / "solution" / "head.py",
    _ASSIGN / "solution" / "loss.py",
    _ASSIGN / "solution" / "geometry_fm.py",
    _REPO / "nanovision" / "geometry.py",
]

_FORBIDDEN = [
    r"import\s+dust3r",
    r"from\s+dust3r",
    r"import\s+mast3r",
    r"from\s+mast3r",
    r"import\s+croco",
    r"from\s+croco",
    # The owned shared geometry file must be reached via nanovision.geometry, never bare.
    r"import\s+geometry_fm",
    r"from\s+geometry_fm",
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
