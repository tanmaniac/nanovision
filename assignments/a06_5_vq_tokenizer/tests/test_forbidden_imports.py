"""Anti-cheat: the codebook and straight-through estimator are built from scratch.

Scans the A6.5 top-level files, the solution, and the nanovision.quantize shim for prebuilt
vector-quantization libraries. Comments and string literals are stripped via tokenize.
"""

import re
import tokenize
from pathlib import Path

_ASSIGN = Path(__file__).resolve().parents[1]
_REPO = _ASSIGN.parents[1]

_FILES = [
    _ASSIGN / "quantize.py",
    _ASSIGN / "vqvae.py",
    _ASSIGN / "prior.py",
    _ASSIGN / "solution" / "quantize.py",
    _ASSIGN / "solution" / "vqvae.py",
    _ASSIGN / "solution" / "prior.py",
    _REPO / "nanovision" / "quantize.py",
]

_FORBIDDEN = [
    r"import\s+vector_quantize_pytorch",
    r"from\s+vector_quantize_pytorch",
    r"import\s+taming",
    r"from\s+taming",
    r"import\s+diffusers",
    r"from\s+diffusers",
    r"VQModel",
    r"VectorQuantizer2",
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
