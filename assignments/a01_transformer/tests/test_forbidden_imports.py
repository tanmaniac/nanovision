"""Enforce the anti-cheat contract: the taught mechanism is not imported.

Checks the A1 solution and the shared-library modules it builds for the
high-level shortcuts the learner is supposed to implement by hand. Comments and
string literals are stripped first (via tokenize) so that merely *naming* a
forbidden symbol in a docstring -- e.g. "nn.MultiheadAttention is forbidden
here" -- does not trip the test; only real code usage does.
"""

import re
import tokenize
from pathlib import Path

_ASSIGN = Path(__file__).resolve().parents[1]
_REPO = _ASSIGN.parents[1]

_FILES = [
    _ASSIGN / "solution" / "attention.py",
    _ASSIGN / "solution" / "transformer.py",
    _REPO / "nanovision" / "attention.py",
    _REPO / "nanovision" / "transformer.py",
]

_FORBIDDEN = [
    r"nn\.MultiheadAttention",
    r"nn\.Transformer",  # also catches nn.TransformerEncoder/Decoder
    r"F\.scaled_dot_product_attention",
    r"functional\.scaled_dot_product_attention",
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
