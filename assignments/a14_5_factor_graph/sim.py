"""Synthetic pose graphs for the optimizer. Provided.

The ground-truth trajectory is a planar loop. `pose_graph_exact` perturbs each pose and gives
noise-free relative measurements, so Gauss-Newton must recover the ground truth (up to the
anchored gauge). `pose_graph_drift` integrates noisy odometry into a drifted initial
trajectory and adds a single loop-closure edge - the classic drift-then-correct scenario.
"""

import numpy as np

from _helpers import rng, pose_2d, small_perturbation
import config as C


def loop_trajectory(n):
    """n ground-truth poses evenly spaced around a circle, each facing along the path."""
    poses = []
    for k in range(n):
        a = 2.0 * np.pi * k / n
        poses.append(pose_2d(C.RADIUS * np.cos(a), C.RADIUS * np.sin(a), a + np.pi / 2))
    return poses


def _relative(Ti, Tj):
    return np.linalg.inv(Ti) @ Tj


def pose_graph_exact(seed=0, n=None):
    """Ground-truth loop, per-pose perturbed initialization (pose 0 exact), and exact relative
    measurements on the consecutive + loop-closure edges. The unique anchored solution is the
    ground truth."""
    gen = rng(seed)
    n = n if n is not None else C.N_POSES
    gt = loop_trajectory(n)
    info = np.eye(6)

    edges = []
    for i in range(n):
        j = (i + 1) % n  # consecutive edges, including the wrap-around loop closure
        edges.append((i, j, _relative(gt[i], gt[j]), info))

    init = [gt[0]]
    for i in range(1, n):
        init.append(gt[i] @ small_perturbation(gen, C.INIT_ROT_SIGMA, C.INIT_TRANS_SIGMA))
    return {"gt": gt, "init": init, "edges": edges}


def pose_graph_drift(seed=0, n=None):
    """Noisy odometry integrated into a drifting trajectory, plus one loop-closure edge tying
    the last pose back to the first. Odometry edges are (nearly) satisfied by the drifted
    init; the loop-closure residual is what corrects the trajectory."""
    gen = rng(seed)
    n = n if n is not None else C.N_POSES
    gt = loop_trajectory(n)
    odom_info = np.diag([1, 1, 1, 1, 1, 1.0]) / (C.ODOM_SIGMA ** 2)
    loop_info = np.diag([1, 1, 1, 1, 1, 1.0]) / (C.LOOP_SIGMA ** 2)

    edges = []
    init = [gt[0]]
    # Consecutive odometry edges with noisy measurements; integrate them to get the drift.
    for i in range(n - 1):
        meas = _relative(gt[i], gt[i + 1]) @ small_perturbation(gen, C.ODOM_SIGMA, C.ODOM_SIGMA)
        edges.append((i, i + 1, meas, odom_info))
        init.append(init[i] @ meas)

    # One loop-closure edge: last pose back to the first, measured near its true relative pose.
    loop_meas = _relative(gt[n - 1], gt[0]) @ small_perturbation(gen, C.LOOP_SIGMA, C.LOOP_SIGMA)
    edges.append((n - 1, 0, loop_meas, loop_info))

    return {"gt": gt, "init": init, "edges": edges}


def unpack_edges(edges):
    """Split the edge tuples into the parallel arrays the C++ optimizer takes."""
    edge_i = [e[0] for e in edges]
    edge_j = [e[1] for e in edges]
    meas = [e[2] for e in edges]
    info = [e[3] for e in edges]
    return edge_i, edge_j, meas, info
