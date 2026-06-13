"""Gauss-Newton pose-graph optimization: it drives the total residual to zero and recovers
the ground-truth trajectory (up to the anchored gauge) on a noise-free graph, and a single
loop-closure edge corrects a drifted odometry chain."""

import numpy as np

from _impl import optimize_pose_graph
from config import GN_ITERS
from _helpers import mean_position_error, pose_distance
import sim


def test_recovers_ground_truth_noise_free():
    pg = sim.pose_graph_exact(seed=1)
    ei, ej, meas, info = sim.unpack_edges(pg["edges"])
    opt, cost = optimize_pose_graph(pg["init"], ei, ej, meas, info, GN_ITERS)
    # With exact measurements and pose 0 anchored, the unique solution is the ground truth.
    assert mean_position_error(opt, pg["gt"]) < 1e-6
    assert cost < 1e-12


def test_anchored_pose_stays_fixed():
    pg = sim.pose_graph_exact(seed=2)
    ei, ej, meas, info = sim.unpack_edges(pg["edges"])
    opt, _ = optimize_pose_graph(pg["init"], ei, ej, meas, info, GN_ITERS)
    # Pose 0 fixes the gauge; the optimizer must not move it.
    assert np.allclose(np.asarray(opt[0]), np.asarray(pg["init"][0]), atol=1e-12)


def test_loop_closure_reduces_drift():
    pg = sim.pose_graph_drift(seed=2)
    ei, ej, meas, info = sim.unpack_edges(pg["edges"])
    drift_before = pose_distance(pg["init"][-1], pg["gt"][-1])

    opt, _ = optimize_pose_graph(pg["init"], ei, ej, meas, info, GN_ITERS)
    drift_after = pose_distance(opt[-1], pg["gt"][-1])

    # Optimizing WITHOUT the loop-closure edge leaves the drift untouched, because the
    # odometry edges are already satisfied by the integrated trajectory.
    ei2, ej2, meas2, info2 = sim.unpack_edges(pg["edges"][:-1])
    opt_nolc, _ = optimize_pose_graph(pg["init"], ei2, ej2, meas2, info2, GN_ITERS)
    drift_nolc = pose_distance(opt_nolc[-1], pg["gt"][-1])

    assert drift_after < drift_before / 5.0       # the loop closure corrects the trajectory
    assert abs(drift_nolc - drift_before) < 1e-6  # without it, nothing changes
