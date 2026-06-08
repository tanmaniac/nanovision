"""Anti-cheat: the flow-matching mechanism is built from scratch.

Scans the A6 top-level files and the solution for prebuilt flow-matching / ODE-solver
libraries. scipy.optimize is allowed (it is the OT-coupling utility). Comments and string
literals are stripped via tokenize, so a forbidden name in a docstring does not trip it.
"""

import re
import tokenize
from pathlib import Path

_ASSIGN = Path(__file__).resolve().parents[1]

_FILES = [
    _ASSIGN / "path.py",
    _ASSIGN / "timesteps.py",
    _ASSIGN / "flow.py",
    _ASSIGN / "coupling.py",
    _ASSIGN / "sampling.py",
    _ASSIGN / "solution" / "path.py",
    _ASSIGN / "solution" / "timesteps.py",
    _ASSIGN / "solution" / "flow.py",
    _ASSIGN / "solution" / "coupling.py",
    _ASSIGN / "solution" / "sampling.py",
]

_FORBIDDEN = [
    r"import\s+torchcfm",
    r"from\s+torchcfm",
    r"import\s+torchdyn",
    r"from\s+torchdyn",
    r"import\s+torchdiffeq",
    r"from\s+torchdiffeq",
    r"import\s+diffusers",
    r"from\s+diffusers",
    r"import\s+k_diffusion",
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
