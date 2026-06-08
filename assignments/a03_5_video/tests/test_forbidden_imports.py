"""Anti-cheat: the mechanism code must not import the taught shortcuts.

Scans the A3.5 top-level files, the solution, and the nanovision.transformer shim for
prebuilt attention/transformer modules, fused SDPA, nn.LayerNorm, timm, transformers,
or any prebuilt video model. Conv3d is allowed: it IS the tubelet mechanism. Comments
and string literals are stripped via tokenize, so naming a forbidden symbol in a
docstring does not trip the test; only real code usage does. Passes with the holes in
place too.
"""

import re
import tokenize
from pathlib import Path

_ASSIGN = Path(__file__).resolve().parents[1]
_REPO = _ASSIGN.parents[1]

_FILES = [
    _ASSIGN / "tubelet.py",       # student top-level
    _ASSIGN / "video_mae.py",
    _ASSIGN / "backbone.py",
    _ASSIGN / "solution" / "tubelet.py",
    _ASSIGN / "solution" / "video_mae.py",
    _ASSIGN / "solution" / "backbone.py",
    _REPO / "nanovision" / "transformer.py",   # shim
]

_FORBIDDEN = [
    r"nn\.MultiheadAttention",
    r"nn\.TransformerEncoder",
    r"nn\.TransformerEncoderLayer",
    r"nn\.Transformer\b",
    r"nn\.LayerNorm",
    r"F\.scaled_dot_product_attention",
    r"functional\.scaled_dot_product_attention",
    r"import\s+timm",
    r"import\s+transformers",
    r"from\s+transformers",
    r"torchvision\.models",
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
