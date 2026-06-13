"""Two-view geometry, visualized in Rerun.

A synthetic point cloud is seen by two cameras a short baseline apart, with a third of the
correspondences corrupted into wrong matches. The RANSAC front-end estimates the relative
pose and triangulates the inliers. The 3D view shows the triangulated points and the two
recovered camera frusta. The two 2D views show the correspondences colored by inlier vs
outlier, with a few epipolar lines (l' = F x) drawn in the second image - every inlier sits
on its epipolar line, every outlier does not. A scalar panel tracks the consensus-set size.

Run headless (writes out/multiview.rrd):   make viz A=a14_3_multiview
Open the interactive viewer:                make viz A=a14_3_multiview SHOW=1
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

INLIER_COLOR = [55, 126, 184]
OUTLIER_COLOR = [228, 26, 28]


def _epipolar_segments(F, u1, img_w):
    """For each point u1, the epipolar line l' = F [u1;1] in image 2, as a segment spanning
    the image width (drawn where l'_x x + l'_y y + l'_z = 0)."""
    segs = []
    for u in u1:
        l = F @ np.array([u[0], u[1], 1.0])
        a, b, c = l
        if abs(b) < 1e-9:
            continue
        xs = np.array([0.0, img_w])
        ys = -(a * xs + c) / b
        segs.append(np.stack([xs, ys], axis=1))
    return segs


def main():
    sc = sim.two_view_scene(seed=0, noise_px=0.5, outlier_frac=C.OUTLIER_FRAC)
    K, u1, u2, inlier = sc["K"], sc["u1"], sc["u2"], sc["inlier"]

    T, pts, inliers = M.two_view_relative_pose(
        K, u1, u2, C.RANSAC_THRESH_PX, C.RANSAC_ITERS, 0)
    inliers = [int(i) for i in inliers]
    found_mask = np.zeros(len(u1), dtype=bool)
    found_mask[inliers] = True

    rr.init("a14_3_multiview", spawn=SHOW)

    # 3D: triangulated inlier points and the two camera frusta. Camera 1 is the world
    # origin; camera 2 is at the recovered relative pose (T_2_1 -> camera-to-world is its
    # inverse, the pose at which to draw the frustum).
    rr.log("world/points", rr.Points3D(np.asarray(pts), radii=0.03,
                                        colors=[INLIER_COLOR]), static=True)
    rr.log("world/cam1", rr.Pinhole(image_from_camera=K, width=C.IMG_W, height=C.IMG_H),
           static=True)
    T_c2w = np.linalg.inv(T)
    rr.log("world/cam2", rr.Transform3D(translation=T_c2w[:3, 3], mat3x3=T_c2w[:3, :3]),
           static=True)
    rr.log("world/cam2/image", rr.Pinhole(image_from_camera=K, width=C.IMG_W,
                                          height=C.IMG_H), static=True)

    # 2D: correspondences colored by RANSAC inlier/outlier, and epipolar lines in image 2.
    colors = np.where(found_mask[:, None], INLIER_COLOR, OUTLIER_COLOR)
    rr.log("view1/points", rr.Points2D(u1, colors=colors, radii=2.0), static=True)
    rr.log("view2/points", rr.Points2D(u2, colors=colors, radii=2.0), static=True)

    F, _ = M.ransac_fundamental(u1, u2, C.RANSAC_THRESH_PX, C.RANSAC_ITERS, 0)
    show = inliers[:8]  # a handful of inlier epipolar lines, to keep it readable
    segs = _epipolar_segments(F, u1[show], C.IMG_W)
    if segs:
        rr.log("view2/epipolar", rr.LineStrips2D(segs, colors=[INLIER_COLOR]), static=True)

    print(f"recovered {len(inliers)} inliers of {int(inlier.sum())} true; "
          f"rotation/translation drawn in 3D")
    if not SHOW:
        _OUT.mkdir(parents=True, exist_ok=True)
        rrd = _OUT / "multiview.rrd"
        rr.save(str(rrd))
        print(f"wrote {rrd}")
        print("open it with:  rerun", rrd)
        print("or render live with:  make viz A=a14_3_multiview SHOW=1")


if __name__ == "__main__":
    main()
