"""Pose-graph optimization, visualized in Rerun - the drift-then-correct centerpiece.

A robot drives a loop; its odometry drifts, so the raw trajectory does not close. One
loop-closure edge ties the last pose back to the first. Scrubbing the timeline replays the
Gauss-Newton iterations: the drifted trajectory snaps onto the ground-truth loop as the
loop-closure residual is distributed back along the path. A scalar panel plots the total cost
falling, and a static image shows the block-sparsity of the system matrix H (the arrowhead the
loop-closure edge creates).

Run headless (writes out/pose_graph.rrd):   make viz A=a14_5_factor_graph
Open the interactive viewer:                 make viz A=a14_5_factor_graph SHOW=1
"""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import rerun as rr

import _impl as M
import sim
import config as C
from _helpers import trajectory_positions

SHOW = os.environ.get("NANOVISION_VIZ_SHOW") == "1"
_OUT = Path(__file__).parent / "out"

GT_COLOR = [150, 150, 150]
EST_COLOR = [228, 26, 28]
LOOP_COLOR = [55, 126, 184]


def _set_step(k):
    if hasattr(rr, "set_time"):
        rr.set_time("iter", sequence=k)
    else:
        rr.set_time_sequence("iter", k)


def _scalar(path, value):
    if hasattr(rr, "Scalars"):
        rr.log(path, rr.Scalars([value]))
    else:
        rr.log(path, rr.Scalar(value))


def _to3(xy):
    return np.column_stack([xy, np.zeros(len(xy))])


def _sparsity_image(n, edges):
    """A per-pose-block image of the H sparsity: block (i, j) is filled if i == j or an edge
    connects i and j. The wrap-around loop-closure edge is the off-corner that breaks the band."""
    img = np.full((n, n), 255, dtype=np.uint8)
    for i in range(n):
        img[i, i] = 0
    for i, j, _, _ in edges:
        img[i, j] = 0
        img[j, i] = 0
    return img


def main():
    pg = sim.pose_graph_drift(seed=0)
    ei, ej, meas, info = sim.unpack_edges(pg["edges"])
    gt_xy = trajectory_positions(pg["gt"])
    n = len(pg["gt"])

    rr.init("a14_5_factor_graph", spawn=SHOW)
    rr.log("world/ground_truth",
           rr.LineStrips2D([np.vstack([gt_xy, gt_xy[0]])], colors=[GT_COLOR]), static=True)
    rr.log("system/H_sparsity", rr.Image(_sparsity_image(n, pg["edges"])), static=True)

    n_frames = 12
    for k in range(n_frames + 1):
        _set_step(k)
        opt, cost = M.optimize_pose_graph(pg["init"], ei, ej, meas, info, k)
        xy = trajectory_positions(opt)
        rr.log("world/estimate", rr.LineStrips2D([np.vstack([xy, xy[0]])], colors=[EST_COLOR]))
        rr.log("world/poses", rr.Points2D(xy, radii=0.08, colors=[EST_COLOR]))
        # The loop-closure edge: last pose back to the first.
        rr.log("world/loop_closure",
               rr.LineStrips2D([np.vstack([xy[-1], xy[0]])], colors=[LOOP_COLOR]))
        _scalar("metrics/cost", float(cost))

    if not SHOW:
        _OUT.mkdir(parents=True, exist_ok=True)
        rrd = _OUT / "pose_graph.rrd"
        rr.save(str(rrd))
        print(f"wrote {rrd}")
        print("open it with:  rerun", rrd)
        print("or render live with:  make viz A=a14_5_factor_graph SHOW=1")


if __name__ == "__main__":
    main()
