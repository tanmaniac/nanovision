"""Closed-form point-to-point alignment (Umeyama/Kabsch): recovers a known transform from
matched correspondences to numerical precision, and the determinant correction keeps the
rotation proper (no reflection) on a degenerate planar cloud."""

import numpy as np

from _impl import align_point_to_point
from config import TOL_TIGHT
from _helpers import transform_error
import sim


def test_recovers_known_transform_matched():
    pts, nrm = sim.terrain_cloud(seed=1)
    pair = sim.make_pair(pts, nrm)
    T = np.asarray(align_point_to_point(pair["source"], pair["target"]))
    rot_err, trans_err = transform_error(T, pair["T_align"])
    assert rot_err < 1e-7
    assert trans_err < 1e-7


def test_rotation_is_proper():
    pts, nrm = sim.terrain_cloud(seed=2)
    pair = sim.make_pair(pts, nrm)
    T = np.asarray(align_point_to_point(pair["source"], pair["target"]))
    R = T[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=TOL_TIGHT)
    assert abs(np.linalg.det(R) - 1.0) < TOL_TIGHT


def test_planar_cloud_det_correction():
    # A flat sheet (z = 0) makes H rank-deficient. Mirror the target across x so the closest
    # ORTHOGONAL fit to the data is a reflection (det = -1): an implementation that omits the
    # det(V U^T) correction returns that improper matrix here. The correction must instead
    # return a proper rotation (det = +1) - and on planar data a proper 180-degree rotation
    # reproduces the mirror exactly, so the corrected solve still aligns the clouds.
    pts, _ = sim.planar_cloud(seed=3)
    source = pts
    target = pts.copy()
    target[:, 0] *= -1.0
    T = np.asarray(align_point_to_point(source, target))
    R = T[:3, :3]
    assert abs(np.linalg.det(R) - 1.0) < 1e-9            # proper rotation, not a reflection
    aligned = source @ R.T + T[:3, 3]
    assert np.allclose(aligned, target, atol=1e-9)        # the proper solution still aligns
