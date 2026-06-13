"""Synthetic scenes for the multi-view estimators. Provided.

`two_view_scene` builds a 3D point cloud seen by two cameras a short baseline apart, with
the world frame chosen as camera 1, so the ground-truth relative pose is exactly
T_2_1 = (R, t). `pnp_scene` builds world points seen by one camera at a known pose. Both can
inject Gaussian pixel noise and (two-view only) wrong correspondences as outliers.
"""

import numpy as np

from _helpers import rng, rotvec_to_R, project
import config as C


def _sample_in_view(gen, n, R2, t2):
    """Sample n world points (= camera-1 frame) that are also in front of camera 2 and
    inside both image rectangles."""
    pts = []
    while len(pts) < n:
        u = gen.uniform(0, C.IMG_W)
        v = gen.uniform(0, C.IMG_H)
        d = gen.uniform(C.DEPTH_MIN, C.DEPTH_MAX)
        X = np.array([(u - C.CX) / C.FX * d, (v - C.CY) / C.FY * d, d])
        Xc2 = R2 @ X + t2
        if Xc2[2] <= 0.1:
            continue
        u2 = project(C.K, R2, t2, X[None, :])[0]
        if 0 <= u2[0] < C.IMG_W and 0 <= u2[1] < C.IMG_H:
            pts.append(X)
    return np.array(pts)


def two_view_scene(seed=0, n_points=None, noise_px=0.0, outlier_frac=0.0):
    """Returns a dict with K, the ground-truth R/t (= T_2_1), the 3D points (camera-1
    frame), the pixel correspondences u1/u2, and the boolean inlier mask."""
    gen = rng(seed)
    n = n_points if n_points is not None else C.N_POINTS
    R = rotvec_to_R(C.REL_ROTVEC)
    t = np.asarray(C.REL_TRANS, dtype=float)

    X = _sample_in_view(gen, n, R, t)
    u1 = project(C.K, np.eye(3), np.zeros(3), X)
    u2 = project(C.K, R, t, X)
    if noise_px > 0:
        u1 = u1 + gen.normal(0, noise_px, u1.shape)
        u2 = u2 + gen.normal(0, noise_px, u2.shape)

    inlier = np.ones(n, dtype=bool)
    if outlier_frac > 0:
        n_out = int(round(outlier_frac * n))
        out_idx = gen.choice(n, size=n_out, replace=False)
        # Replace the second-view match with a random pixel (a wrong correspondence).
        u2[out_idx, 0] = gen.uniform(0, C.IMG_W, n_out)
        u2[out_idx, 1] = gen.uniform(0, C.IMG_H, n_out)
        inlier[out_idx] = False

    return {"K": C.K, "R": R, "t": t, "points3d": X, "u1": u1, "u2": u2, "inlier": inlier}


def pnp_scene(seed=0, n_points=40, noise_px=0.0):
    """World points in a box and a single camera at a known pose T_cam_world = (R, t).
    Returns X (world, N,3), u (pixels, N,2), and the ground-truth R, t."""
    gen = rng(seed)
    R = rotvec_to_R(np.array([0.2, 0.4, -0.1]))
    t = np.array([0.3, -0.2, 6.0])  # camera 6 m back, points spread in a 4 m box
    X = gen.uniform(-2.0, 2.0, size=(n_points, 3))
    u = project(C.K, R, t, X)
    if noise_px > 0:
        u = u + gen.normal(0, noise_px, u.shape)
    return {"K": C.K, "R": R, "t": t, "points3d": X, "u": u}
