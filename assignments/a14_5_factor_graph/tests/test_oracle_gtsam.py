"""Oracle cross-check: the from-scratch Gauss-Newton trajectory agrees with GTSAM's optimizer
on the same pose graph. GTSAM is a labeled comparison only - the graded solver is the C++ one.
Skipped when GTSAM is not installed.

Note: the comparison uses the noise-free graph with isotropic information, so the difference in
tangent-space ordering ([rho; theta] here versus GTSAM's [theta; rho]) does not affect the
optimized trajectory."""

import numpy as np
import pytest

from _impl import optimize_pose_graph
from config import GN_ITERS
from _helpers import trajectory_positions
import sim

gtsam = pytest.importorskip("gtsam")


def test_agrees_with_gtsam_on_pose_graph():
    pg = sim.pose_graph_exact(seed=1)
    ei, ej, meas, info = sim.unpack_edges(pg["edges"])
    ours, _ = optimize_pose_graph(pg["init"], ei, ej, meas, info, GN_ITERS)

    graph = gtsam.NonlinearFactorGraph()
    prior = gtsam.noiseModel.Isotropic.Precision(6, 1e8)  # anchor pose 0
    graph.add(gtsam.PriorFactorPose3(0, gtsam.Pose3(pg["init"][0]), prior))
    for i, j, m, om in pg["edges"]:
        noise = gtsam.noiseModel.Gaussian.Information(np.asarray(om))
        graph.add(gtsam.BetweenFactorPose3(int(i), int(j), gtsam.Pose3(np.asarray(m)), noise))

    values = gtsam.Values()
    for k, T in enumerate(pg["init"]):
        values.insert(k, gtsam.Pose3(np.asarray(T)))
    result = gtsam.LevenbergMarquardtOptimizer(graph, values).optimize()
    theirs = [result.atPose3(k).matrix() for k in range(len(pg["init"]))]

    ours_xy = trajectory_positions(ours)
    theirs_xy = trajectory_positions(theirs)
    assert np.abs(ours_xy - theirs_xy).max() < 1e-3
