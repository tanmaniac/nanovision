"""Essential-matrix decomposition and pose recovery: the decomposition yields proper
rotations, and cheirality recovers the true relative pose (rotation exactly, translation up
to its sign and the monocular scale) with all points in front of both cameras."""

import numpy as np

from _impl import eight_point, decompose_essential, recover_pose
from config import CX, CY, FX, FY
from _helpers import rotation_angle, translation_angle
import sim


def _normalized(u):
    return (u - [CX, CY]) / [FX, FY]


def test_decompose_gives_proper_rotations():
    sc = sim.two_view_scene(seed=1, noise_px=0.0)
    E = eight_point(_normalized(sc["u1"]), _normalized(sc["u2"]))
    R1, R2, t = decompose_essential(E)
    for R in (R1, R2):
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert abs(np.linalg.det(R) - 1.0) < 1e-9
    assert abs(np.linalg.norm(t) - 1.0) < 1e-9


def test_recover_pose_noise_free():
    sc = sim.two_view_scene(seed=2, noise_px=0.0)
    x1, x2 = _normalized(sc["u1"]), _normalized(sc["u2"])
    E = eight_point(x1, x2)
    R, t = recover_pose(E, x1, x2)
    assert np.degrees(rotation_angle(R, sc["R"])) < 1e-4
    assert np.degrees(translation_angle(t, sc["t"])) < 1e-4


def test_cheirality_all_points_in_front():
    sc = sim.two_view_scene(seed=4, noise_px=0.0)
    x1, x2 = _normalized(sc["u1"]), _normalized(sc["u2"])
    E = eight_point(x1, x2)
    R, t = recover_pose(E, x1, x2)
    # The chosen pose puts the (noise-free) scene points in front of both cameras: the
    # ground-truth 3D points, transformed into each camera, have positive depth.
    X = sc["points3d"]
    assert np.all(X[:, 2] > 0)                       # camera 1 (= world) frame
    Xc2 = X @ R.T + t * np.linalg.norm(sc["t"])      # camera 2, scale restored for the check
    assert np.all(Xc2[:, 2] > 0)
