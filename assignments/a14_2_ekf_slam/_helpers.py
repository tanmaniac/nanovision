"""Test/viz helpers: RNG, angle wrap, covariance ellipses, and NEES. Provided.

These are pure NumPy and do not import the C++ build, so sim.py can use them freely.
"""

import numpy as np


def rng(seed):
    return np.random.default_rng(seed)


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def covariance_ellipse(mean_xy, cov_xy, n_sigma=1.0, n_pts=40):
    """Points tracing the n-sigma covariance ellipse of a 2D Gaussian, for plotting."""
    vals, vecs = np.linalg.eigh(cov_xy)
    vals = np.clip(vals, 0.0, None)
    t = np.linspace(0, 2 * np.pi, n_pts)
    circle = np.stack([np.cos(t), np.sin(t)])
    pts = vecs @ (n_sigma * np.sqrt(vals)[:, None] * circle)
    return pts.T + np.asarray(mean_xy)[None, :]


def robot_nees(mu, P, true_pose):
    """Normalized estimation error squared for the robot pose (3-DOF). Its expectation
    under a consistent filter is 3; persistently larger means overconfident."""
    e = np.asarray(mu[:3]) - np.asarray(true_pose)
    e[2] = wrap_angle(e[2])
    return float(e @ np.linalg.solve(P[:3, :3], e))
