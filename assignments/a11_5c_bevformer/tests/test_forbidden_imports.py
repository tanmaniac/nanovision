"""Anti-cheat: the attention must come from nanovision.attention, not a prebuilt library module.

Scans the holed bevformer.py, its solution, and the nanovision.bevformer shim. The spatial cross-
attention, deformable attention, and temporal self-attention all reuse MultiHeadAttention from the
transformer assignment, so nn.MultiheadAttention, F.scaled_dot_product_attention, and nn.Transformer
are forbidden. cv2 / kornia projection shortcuts are forbidden too. Comments and string literals are
stripped via tokenize, so a forbidden name in a docstring does not trip the scan.

F.grid_sample and F.affine_grid are ALLOWED: they are the bilinear-sampling substrate (the
view-transform sampling and the ego-motion warp), not the taught attention. The owned bevformer.py
must be reached via nanovision.bevformer, never bare.
"""

import re
import tokenize
from pathlib import Path

_ASSIGN = Path(__file__).resolve().parents[1]
_SHIM = _ASSIGN.parents[1] / "nanovision" / "bevformer.py"

_FILES = [
    _ASSIGN / "bevformer.py",
    _ASSIGN / "solution" / "bevformer.py",
    _SHIM,
]

_FORBIDDEN = [
    r"nn\.MultiheadAttention",
    r"MultiheadAttention",
    r"scaled_dot_product_attention",
    r"nn\.Transformer",
    r"cv2\.projectPoints",
    r"cv2\.solvePnP",
    r"import\s+kornia",
    r"from\s+kornia",
    # The owned shared file must be reached via nanovision.bevformer, not bare.
    r"^\s*import\s+bevformer\b",
    r"^\s*from\s+bevformer\s+import",
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


def _raw_lines(path: Path) -> str:
    out = []
    with open(path) as f:
        for line in f:
            if line.lstrip().startswith("#"):
                continue
            out.append(line)
    return "".join(out)


def test_no_forbidden_imports():
    for f in _FILES:
        assert f.exists(), f"expected file not found: {f}"
        code = _code_only(f)
        raw = _raw_lines(f)
        for pat in _FORBIDDEN:
            target = raw if pat.startswith("^") else code
            flags = re.MULTILINE if pat.startswith("^") else 0
            assert re.search(pat, target, flags) is None, f"forbidden pattern {pat!r} in {f}"
