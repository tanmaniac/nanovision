"""Test/viz helpers: RNG, numerical Jacobians, the UKF Python wiring, a unicycle
simulator, and a covariance-ellipse helper. Provided.

The UKF wiring lives here on purpose. The graded C++ holds the unscented-transform
math (sigma points and weights, the weighted mean/covariance recovery, the cross-
covariance). Propagating each sigma point through the model f or h is a trivial loop,
so it is Python glue that calls the C++ models - which also lets the tests drive the
UKF with an arbitrary linear model to compare against the KF.
"""

import numpy as np

from _impl import ukf_sigma_points, ukf_unscented_transform, ukf_cross_covariance


def rng(seed):
    return np.random.default_rng(seed)


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def numerical_jacobian(func, x, eps, angle_out_rows=()):
    """Central-difference Jacobian of func at x. For output rows listed in
    angle_out_rows (e.g. a bearing), the +/- difference is angle-wrapped before
    dividing, so the derivative is correct across the +/-pi seam."""
    x = np.asarray(x, dtype=float)
    f0 = np.asarray(func(x), dtype=float)
    J = np.zeros((f0.size, x.size))
    for j in range(x.size):
        d = np.zeros_like(x)
        d[j] = eps
        fp = np.asarray(func(x + d), dtype=float)
        fm = np.asarray(func(x - d), dtype=float)
        diff = fp - fm
        for r in angle_out_rows:
            diff[r] = wrap_angle(fp[r] - fm[r])
        J[:, j] = diff / (2 * eps)
    return J


def ukf_predict(mu, P, f, Q, alpha, beta, kappa):
    """One UKF predict step. f maps a state to the next state."""
    sigmas, Wm, Wc = ukf_sigma_points(mu, P, alpha, beta, kappa)
    prop = np.stack([np.asarray(f(s)) for s in sigmas])
    return ukf_unscented_transform(prop, Wm, Wc, Q)


def ukf_update(mu, P, z, h, R, alpha, beta, kappa, residual=None):
    """One UKF update step. h maps a state to a predicted measurement. residual(z,
    z_pred) defaults to z - z_pred; pass a wrapping residual for angular measurements."""
    sigmas, Wm, Wc = ukf_sigma_points(mu, P, alpha, beta, kappa)
    zsig = np.stack([np.asarray(h(s)) for s in sigmas])
    z_mean, S = ukf_unscented_transform(zsig, Wm, Wc, R)
    Pxz = ukf_cross_covariance(sigmas, mu, zsig, z_mean, Wc)
    K = Pxz @ np.linalg.inv(S)
    innov = (z - z_mean) if residual is None else residual(z, z_mean)
    mu_upd = np.asarray(mu) + K @ innov
    P_upd = P - K @ S @ K.T
    return mu_upd, P_upd


def simulate_unicycle(v, omega, dt, n_steps, landmark, q_diag, r_diag, seed):
    """Run the ground-truth unicycle and emit a noisy range-bearing reading each step.
    Returns (gt states (n+1, 3), measurements (n, 2))."""
    r = rng(seed)
    x = np.array([0.0, 0.0, 0.0])
    gt = [x.copy()]
    meas = []
    qs = np.sqrt(q_diag)
    rs = np.sqrt(r_diag)
    for _ in range(n_steps):
        x = np.array([
            x[0] + v * dt * np.cos(x[2]),
            x[1] + v * dt * np.sin(x[2]),
            wrap_angle(x[2] + omega * dt),
        ])
        x = x + r.normal(0.0, qs)  # process noise
        x[2] = wrap_angle(x[2])
        gt.append(x.copy())
        dx, dy = landmark[0] - x[0], landmark[1] - x[1]
        z = np.array([np.hypot(dx, dy), wrap_angle(np.arctan2(dy, dx) - x[2])])
        z = z + r.normal(0.0, rs)  # measurement noise
        meas.append(z)
    return np.array(gt), np.array(meas)


def covariance_ellipse(mean_xy, cov_xy, n_sigma=1.0, n_pts=48):
    """Points tracing the n-sigma covariance ellipse of a 2D Gaussian, for plotting."""
    vals, vecs = np.linalg.eigh(cov_xy)
    vals = np.clip(vals, 0.0, None)
    t = np.linspace(0, 2 * np.pi, n_pts)
    circle = np.stack([np.cos(t), np.sin(t)])  # (2, n_pts)
    pts = vecs @ (n_sigma * np.sqrt(vals)[:, None] * circle)
    return (pts.T + np.asarray(mean_xy)[None, :])
