"""Anti-cheat: the VLA action heads are built from scratch.

Scans the A13 top-level holed files and their solution copies for prebuilt robot / diffusion /
flow-matching libraries and any bare cross-assignment import. Comments and string literals are
stripped via tokenize, so a forbidden name in a docstring does not trip it. This test passes in
both modes (it is a static scan, not a runtime import).
"""

import re
import tokenize
from pathlib import Path

_ASSIGN = Path(__file__).resolve().parents[1]

_FILES = [
    _ASSIGN / "flow.py",
    _ASSIGN / "bc.py",
    _ASSIGN / "ddpm.py",
    _ASSIGN / "solution" / "flow.py",
    _ASSIGN / "solution" / "bc.py",
    _ASSIGN / "solution" / "ddpm.py",
]

_FORBIDDEN = [
    r"import\s+gym\b",
    r"from\s+gym\b",
    r"import\s+gymnasium",
    r"from\s+gymnasium",
    r"import\s+diffusers",
    r"from\s+diffusers",
    r"import\s+robomimic",
    r"from\s+robomimic",
    r"import\s+lerobot",
    r"from\s+lerobot",
    r"import\s+dm_control",
    r"from\s+dm_control",
    r"import\s+torchcfm",
    r"from\s+torchcfm",
    r"import\s+torchdiffeq",
    r"from\s+torchdiffeq",
    # No bare cross-assignment imports: A13 is a leaf and owns its modules locally.
    r"from\s+assignments",
    r"import\s+assignments",
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
