"""ICP convergence, visualized in Rerun.

A source cloud (a bumpy surface moved by a known transform) slides onto a fixed target cloud
as ICP iterates. Point-to-point and point-to-plane run side by side on the same data, so you
can watch point-to-plane reach the target in a handful of iterations while point-to-point
crawls in. A scalar panel plots the RMS correspondence distance for each as it shrinks. Scrub
the timeline to step through the registration.

Run headless (writes out/icp.rrd):   make viz A=a14_4_icp
Open the interactive viewer:          make viz A=a14_4_icp SHOW=1
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

SHOW = os.environ.get("NANOVISION_VIZ_SHOW") == "1"
_OUT = Path(__file__).parent / "out"

TARGET_COLOR = [150, 150, 150]
P2P_COLOR = [228, 26, 28]
P2PLANE_COLOR = [55, 126, 184]


def _set_step(k):
    if hasattr(rr, "set_time"):
        rr.set_time("iter", sequence=k)
    else:
        rr.set_time_sequence("iter", k)


def _scalars(path, value):
    if hasattr(rr, "Scalars"):
        rr.log(path, rr.Scalars([value]))
    else:
        rr.log(path, rr.Scalar(value))


def main():
    pts, nrm = sim.terrain_cloud(seed=0)
    pair = sim.make_pair(pts, nrm, rotvec=[0.10, -0.14, 0.10], trans=[0.25, -0.15, 0.10])
    source, target, normals = pair["source"], pair["target"], pair["target_normals"]

    rr.init("a14_4_icp", spawn=SHOW)
    rr.log("world/target", rr.Points3D(target, radii=0.02, colors=[TARGET_COLOR]), static=True)

    n_frames = 18
    for k in range(n_frames + 1):
        _set_step(k)
        for mode, path, color in (("point", "world/point_to_point", P2P_COLOR),
                                  ("plane", "world/point_to_plane", P2PLANE_COLOR)):
            # Run k iterations from scratch (k = 0 leaves the source at its initial pose).
            T, _, rms = M.icp(source, target, normals, np.eye(4), k, C.MAX_CORR_DIST, mode)
            T = np.asarray(T)
            moved = source @ T[:3, :3].T + T[:3, 3]
            rr.log(path, rr.Points3D(moved, radii=0.02, colors=[color]))
            _scalars(f"metrics/rms_{mode}", float(rms))

    if not SHOW:
        _OUT.mkdir(parents=True, exist_ok=True)
        rrd = _OUT / "icp.rrd"
        rr.save(str(rrd))
        print(f"wrote {rrd}")
        print("open it with:  rerun", rrd)
        print("or render live with:  make viz A=a14_4_icp SHOW=1")


if __name__ == "__main__":
    main()
