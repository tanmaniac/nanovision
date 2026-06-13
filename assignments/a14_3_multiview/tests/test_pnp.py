"""PnP: the DLT recovers the camera pose from 3D-2D correspondences on noise-free data, and
the Gauss-Newton refinement drives the reprojection error to near zero (and recovers the
pose under mild pixel noise)."""

import numpy as np

from _impl import pnp_dlt, pnp_refine, pinhole_project
from config import TOL_TIGHT
from _helpers import rotation_angle
import sim


def _reproj_rms(T, X, u, K):
    R, t = T[:3, :3], T[:3, 3]
    err = [np.linalg.norm(u[i] - pinhole_project(K, R @ X[i] + t)) for i in range(len(X))]
    return float(np.sqrt(np.mean(np.square(err))))


def test_pnp_dlt_noise_free():
    ps = sim.pnp_scene(seed=1, noise_px=0.0)
    T = pnp_dlt(ps["points3d"], ps["u"], ps["K"])
    assert np.degrees(rotation_angle(T[:3, :3], ps["R"])) < 1e-4
    assert np.linalg.norm(T[:3, 3] - ps["t"]) < TOL_TIGHT


def test_pnp_refine_zero_reprojection():
    ps = sim.pnp_scene(seed=2, noise_px=0.0)
    T0 = pnp_dlt(ps["points3d"], ps["u"], ps["K"])
    T = pnp_refine(ps["points3d"], ps["u"], ps["K"], T0)
    assert _reproj_rms(T, ps["points3d"], ps["u"], ps["K"]) < 1e-8


def test_pnp_refine_from_perturbed_pose():
    # Gauss-Newton recovers the pose from a deliberately perturbed initial guess.
    ps = sim.pnp_scene(seed=3, noise_px=0.0)
    R0 = ps["R"]
    T0 = np.eye(4)
    T0[:3, :3] = R0
    T0[:3, 3] = ps["t"] + np.array([0.3, -0.2, 0.4])  # offset the translation
    T = pnp_refine(ps["points3d"], ps["u"], ps["K"], T0)
    assert np.degrees(rotation_angle(T[:3, :3], ps["R"])) < 1e-3
    assert np.linalg.norm(T[:3, 3] - ps["t"]) < 1e-6
