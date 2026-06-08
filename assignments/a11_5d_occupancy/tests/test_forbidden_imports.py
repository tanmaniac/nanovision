"""Anti-cheat: the alpha compositing must be the reused NeRF kernel, not a hand-rolled one.

Scans the holed occupancy.py and its solution (no nanovision shim - occupancy.py is assignment-
local). The forbidden set: cv2 / kornia projection shortcuts, the sparse-conv libraries
MinkowskiEngine and spconv (the dense grid is the exercise), and a hand-rolled cumprod/cumsum of
(1 - alpha) inside occupancy.py (the front-to-back transmittance product MUST come from
nanovision.volume.volume_render). Comments and string literals are stripped via tokenize, so a
forbidden name in a docstring does not trip the scan.

F.grid_sample and F.affine_grid are ALLOWED: they are the trilinear-sampling substrate, not the
compositing. The reuse of volume_render is checked positively as well.
"""

import re
import tokenize
from pathlib import Path

_ASSIGN = Path(__file__).resolve().parents[1]

_FILES = [
    _ASSIGN / "occupancy.py",
    _ASSIGN / "solution" / "occupancy.py",
]

_FORBIDDEN = [
    r"cv2\.projectPoints",
    r"cv2\.solvePnP",
    r"import\s+kornia",
    r"from\s+kornia",
    r"MinkowskiEngine",
    r"spconv",
    # No hand-rolled transmittance: the (1 - alpha) cumulative product is the reused kernel's job.
    r"cumprod",
    r"cumsum",
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


def test_reuses_volume_render():
    """The solution must call the reused NeRF kernel, not re-implement compositing."""
    sol = (_ASSIGN / "solution" / "occupancy.py").read_text()
    assert "volume_render(" in sol, "render_occupancy_rays must call volume_render"
    assert "from nanovision.volume import" in sol, "must import the kernel from nanovision.volume"
