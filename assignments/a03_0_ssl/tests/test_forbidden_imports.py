"""Enforce the anti-cheat contract: the taught mechanisms are not imported.

Scans every .py in the A3 solution for the high-level shortcuts the learner is
supposed to build by hand: prebuilt attention/transformer modules, fused SDPA,
timm/transformers, and nn.LayerNorm (the course builds LayerNorm from scratch).
Comments and string literals are stripped first (via tokenize) so merely naming a
forbidden symbol in a docstring does not trip the test; only real code usage does.

This test must pass on starter too: the holed code still must not import the
forbidden symbols.
"""

import re
import tokenize
from pathlib import Path

_ASSIGN = Path(__file__).resolve().parents[1]

_FILES = [
    _ASSIGN / "backbone.py",  # student top-level
    _ASSIGN / "mae.py",
    _ASSIGN / "dino.py",
] + sorted((_ASSIGN / "solution").glob("*.py"))

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
]

_SKIP = {tokenize.COMMENT, tokenize.STRING, tokenize.NL, tokenize.NEWLINE,
         tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING, tokenize.ENDMARKER}


def _code_only(path: Path) -> str:
    """Source with comments and string literals removed."""
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
