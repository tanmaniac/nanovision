"""Anti-cheat: the attention must be the A1 MultiHeadAttention, not a torch.nn shortcut.

Scans the holed predict.py and its solution (no nanovision shim - predict.py is assignment-local).
The forbidden set is the three high-level attention APIs that would skip building the mechanism:
nn.MultiheadAttention, nn.Transformer* (encoder/decoder/layer), and
F.scaled_dot_product_attention. Comments and string literals are stripped via tokenize, so a
forbidden name in a docstring does not trip the scan. The positive check confirms attention is
imported from nanovision.attention.
"""

import re
import tokenize
from pathlib import Path

_ASSIGN = Path(__file__).resolve().parents[1]

_FILES = [
    _ASSIGN / "predict.py",
    _ASSIGN / "solution" / "predict.py",
]

_FORBIDDEN = [
    r"MultiheadAttention",
    r"Transformer",                 # nn.Transformer / TransformerEncoder / TransformerDecoder / *Layer
    r"scaled_dot_product_attention",
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
        assert f.exists(), f"expected file not found: {f}"
        code = _code_only(f)
        for pat in _FORBIDDEN:
            assert re.search(pat, code) is None, f"forbidden pattern {pat!r} in {f}"


def test_uses_nanovision_attention():
    """The solution must source attention from nanovision.attention, not torch.nn."""
    sol = (_ASSIGN / "solution" / "predict.py").read_text()
    assert "from nanovision.attention import" in sol, "must import attention from nanovision.attention"
    assert "MultiHeadAttention" in sol, "must use the A1 MultiHeadAttention"
