"""Anti-cheat: the mechanism code must not call high-level geometry shortcuts.

We grep the solution and the shared geometry module for the functions that would
defeat the exercise (projecting, solving PnP, or warping a perspective image with
a single library call, or using kornia). numpy / torch / pyquaternion are fine.
"""

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]

FORBIDDEN = [
    "cv2.projectPoints",
    "cv2.solvePnP",
    "cv2.warpPerspective",
    "cv2.findHomography",
    "import kornia",
    "from kornia",
]

FILES = [
    _REPO / "assignments" / "a115a_camera_geometry_bev" / "solution" / "geometry.py",
    _REPO / "nanovision" / "geometry.py",
]


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.name))
def test_no_forbidden_imports(path):
    assert path.exists(), f"expected file not found: {path}"
    text = path.read_text()
    hits = [tok for tok in FORBIDDEN if tok in text]
    assert not hits, f"{path} uses forbidden shortcut(s): {hits}"
