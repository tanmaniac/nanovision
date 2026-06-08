"""Anti-cheat: the VAE and DiT are built from scratch.

Scans the A7 top-level holed files and the solution for prebuilt VAE/DiT/transformer
implementations. nn.Transformer / TransformerEncoder would bypass the adaLN-Zero block, so
they are forbidden too. Comments and string literals are stripped via tokenize, so a
forbidden name in a docstring does not trip it.
"""

import re
import tokenize
from pathlib import Path

_ASSIGN = Path(__file__).resolve().parents[1]

_FILES = [
    _ASSIGN / "vae.py",
    _ASSIGN / "dit.py",
    _ASSIGN / "flow.py",
    _ASSIGN / "solution" / "vae.py",
    _ASSIGN / "solution" / "dit.py",
]

_FORBIDDEN = [
    r"import\s+diffusers",
    r"from\s+diffusers",
    r"import\s+timm",
    r"from\s+timm",
    r"from\s+torchvision",
    r"import\s+torchvision",
    r"nn\.Transformer",
    r"TransformerEncoder",
    r"TransformerDecoder",
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
