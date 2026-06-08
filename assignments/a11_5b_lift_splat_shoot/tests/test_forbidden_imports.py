"""Anti-cheat: the splat is the cumsum trick, not a scatter, and no library view-transform.

Scans the holed lift_splat.py, its solution, and the nanovision.lift_splat shim. The splat must
pool with the sort+cumsum trick, so scatter_add / index_add are forbidden IN lift_splat.py (a
test may use scatter_add only in its own oracle). Also forbids cv2 / kornia projection shortcuts
and any external bev_pool / voxel_pooling kernel. Comments and string literals are stripped via
tokenize, so a forbidden name in a docstring does not trip it; grid_sample is NOT forbidden
(viz may use it for an IPM overlay), and the scan is scoped to lift_splat.py so viz is not
penalized. The owned lift_splat.py must be reached via nanovision.lift_splat, never bare.
"""

import re
import tokenize
from pathlib import Path

_ASSIGN = Path(__file__).resolve().parents[1]
_SHIM = _ASSIGN.parents[1] / "nanovision" / "lift_splat.py"

_FILES = [
    _ASSIGN / "lift_splat.py",
    _ASSIGN / "solution" / "lift_splat.py",
    _SHIM,
]

_FORBIDDEN = [
    r"scatter_add",
    r"index_add",
    r"cv2\.projectPoints",
    r"cv2\.solvePnP",
    r"import\s+kornia",
    r"from\s+kornia",
    r"bev_pool",
    r"voxel_pooling",
    # The owned shared file must be reached via nanovision.lift_splat, not bare.
    r"^\s*import\s+lift_splat\b",
    r"^\s*from\s+lift_splat\s+import",
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
