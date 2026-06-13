"""The ICP outer loop: from an identity initialization it recovers the known transform in
both modes, point-to-plane converges in far fewer iterations than point-to-point on a
surface-rich cloud, and the max-distance gate rejects unmatched outlier points."""

import numpy as np

from _impl import icp
from config import MAX_ITER, MAX_CORR_DIST
from _helpers import transform_error
import sim


def test_icp_recovers_transform_both_modes():
    pts, nrm = sim.terrain_cloud(seed=1)
    pair = sim.make_pair(pts, nrm)
    for mode in ("point", "plane"):
        T, iters, rms = icp(pair["source"], pair["target"], pair["target_normals"],
                            np.eye(4), MAX_ITER, MAX_CORR_DIST, mode)
        rot_err, trans_err = transform_error(np.asarray(T), pair["T_align"])
        assert rot_err < 1e-4, mode
        assert trans_err < 1e-4, mode
        assert rms < 1e-3, mode


def test_point_to_plane_converges_faster():
    # Iterations to reach a tight rotation error: point-to-plane needs far fewer than
    # point-to-point on a surface with varying normals. Asserted as a wide-margin ordering.
    pts, nrm = sim.terrain_cloud(seed=2)
    pair = sim.make_pair(pts, nrm, rotvec=[0.10, -0.14, 0.10], trans=[0.25, -0.15, 0.10])

    def iters_to_converge(mode, tol_deg=0.01):
        for k in range(1, MAX_ITER):
            T, _, _ = icp(pair["source"], pair["target"], pair["target_normals"],
                          np.eye(4), k, MAX_CORR_DIST, mode)
            if np.degrees(transform_error(np.asarray(T), pair["T_align"])[0]) < tol_deg:
                return k
        return MAX_ITER

    n_point = iters_to_converge("point")
    n_plane = iters_to_converge("plane")
    assert n_plane < n_point
    assert n_plane <= 4 and n_point >= 2 * n_plane


def test_max_distance_gate_rejects_outliers():
    # Add a handful of far-away source points with no real match; the gate must reject them so
    # the recovered transform stays accurate.
    pts, nrm = sim.terrain_cloud(seed=3)
    pair = sim.make_pair(pts, nrm)
    rng = np.random.default_rng(0)
    junk = rng.uniform(20.0, 30.0, size=(40, 3))  # far outside the cloud
    source = np.vstack([pair["source"], junk])
    T, iters, rms = icp(source, pair["target"], pair["target_normals"],
                        np.eye(4), MAX_ITER, MAX_CORR_DIST, "point")
    rot_err, trans_err = transform_error(np.asarray(T), pair["T_align"])
    assert rot_err < 1e-3
    assert trans_err < 1e-3
