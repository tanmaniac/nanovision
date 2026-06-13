"""Linear triangulation reproduces the 3D points on noise-free correspondences, and the
Gauss-Newton refinement reduces reprojection error from a perturbed start."""

import numpy as np

from _impl import triangulate_dlt, triangulate_refine
from config import TOL_TIGHT
import sim


def _projection_matrices(scene):
    K, R, t = scene["K"], scene["R"], scene["t"]
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t[:, None]])
    return P1, P2


def test_dlt_recovers_points_noise_free():
    sc = sim.two_view_scene(seed=1, noise_px=0.0)
    P1, P2 = _projection_matrices(sc)
    X, u1, u2 = sc["points3d"], sc["u1"], sc["u2"]
    for i in range(len(X)):
        Xi = triangulate_dlt(P1, P2, u1[i], u2[i])
        assert np.linalg.norm(Xi - X[i]) < TOL_TIGHT


def test_refine_improves_on_noisy_init():
    sc = sim.two_view_scene(seed=2, noise_px=0.0)
    P1, P2 = _projection_matrices(sc)
    X, u1, u2 = sc["points3d"], sc["u1"], sc["u2"]
    rng = np.random.default_rng(0)
    improved = 0
    for i in range(len(X)):
        X0 = X[i] + rng.normal(0, 0.1, 3)  # perturb the true point
        Xr = triangulate_refine(P1, P2, u1[i], u2[i], X0)
        # Refinement from a perturbed start lands back on the true point.
        assert np.linalg.norm(Xr - X[i]) < np.linalg.norm(X0 - X[i])
        if np.linalg.norm(Xr - X[i]) < 1e-6:
            improved += 1
    assert improved == len(X)
