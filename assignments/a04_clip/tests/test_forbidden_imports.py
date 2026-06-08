"""Anti-cheat: the mechanism code builds the losses from scratch.

Scans the A4 top-level files and the solution for prebuilt CLIP/SigLIP libraries and for
F.cross_entropy / F.log_softmax / F.nll_loss. clip_loss must build the symmetric
softmax cross-entropy by hand (the in-batch-negative denominator is the lesson), so
cross_entropy is forbidden. F.logsigmoid (a stable elementwise primitive) and
F.normalize (the provided L2-norm) are allowed. Comments and string literals are stripped
via tokenize, so naming a forbidden symbol in a docstring does not trip the test.
"""

import re
import tokenize
from pathlib import Path

_ASSIGN = Path(__file__).resolve().parents[1]

_FILES = [
    _ASSIGN / "losses.py",         # student top-level
    _ASSIGN / "inference.py",
    _ASSIGN / "model.py",
    _ASSIGN / "solution" / "losses.py",
    _ASSIGN / "solution" / "inference.py",
    _ASSIGN / "solution" / "model.py",
]

_FORBIDDEN = [
    r"import\s+open_clip",
    r"import\s+clip\b",
    r"from\s+clip\b",
    r"import\s+timm",
    r"import\s+transformers",
    r"from\s+transformers",
    r"F\.cross_entropy",
    r"functional\.cross_entropy",
    r"F\.nll_loss",
    r"F\.log_softmax",
    r"functional\.log_softmax",
    r"nn\.CrossEntropyLoss",
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
