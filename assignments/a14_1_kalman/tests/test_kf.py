"""Linear Kalman filter: predict, the Joseph-form update, and convergence."""

import numpy as np

from _impl import kf_predict, kf_update
from _helpers import rng
from config import TOL_TIGHT, SEED


def _np_predict(mu, P, F, B, u, Q):
    return F @ mu + B @ u, F @ P @ F.T + Q


def _np_update(mu, P, z, H, R):
    y = z - H @ mu
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    mu2 = mu + K @ y
    I = np.eye(P.shape[0])
    IKH = I - K @ H
    P2 = IKH @ P @ IKH.T + K @ R @ K.T
    return mu2, P2


def test_kf_predict_matches_reference():
    F = np.array([[1.0, 0.1], [0.0, 1.0]])
    B = np.array([[0.0], [0.1]])
    u = np.array([2.0])
    Q = np.diag([0.01, 0.04])
    mu = np.array([1.0, -0.5])
    P = np.diag([0.5, 0.3])
    mu2, P2 = kf_predict(mu, P, F, B, u, Q)
    emu, eP = _np_predict(mu, P, F, B, u, Q)
    assert np.allclose(mu2, emu, atol=TOL_TIGHT)
    assert np.allclose(P2, eP, atol=TOL_TIGHT)


def test_kf_update_matches_reference():
    r = rng(SEED)
    H = np.array([[1.0, 0.0]])
    R = np.array([[0.2]])
    mu = np.array([1.0, -0.5])
    A = r.standard_normal((2, 2))
    P = A @ A.T + np.eye(2)
    z = np.array([0.7])
    mu2, P2 = kf_update(mu, P, z, H, R)
    emu, eP = _np_update(mu, P, z, H, R)
    assert np.allclose(mu2, emu, atol=TOL_TIGHT)
    assert np.allclose(P2, eP, atol=TOL_TIGHT)


def test_kf_update_reduces_uncertainty_and_stays_spd():
    r = rng(SEED + 1)
    H = np.array([[1.0, 0.0], [0.0, 1.0]])
    R = np.diag([0.3, 0.3])
    mu = np.array([0.0, 0.0])
    A = r.standard_normal((2, 2))
    P = A @ A.T + np.eye(2)
    z = np.array([0.5, -0.2])
    _, P2 = kf_update(mu, P, z, H, R)
    assert np.trace(P2) < np.trace(P)  # a measurement cannot increase uncertainty
    assert np.allclose(P2, P2.T, atol=TOL_TIGHT)  # symmetric
    np.linalg.cholesky(P2)  # SPD (raises if not)


def test_kf_static_scalar_converges():
    # Constant scalar state observed under noise: the estimate converges to the truth
    # and the variance shrinks toward R/n.
    truth = 2.0
    r = rng(SEED + 2)
    F = np.array([[1.0]])
    B = np.array([[0.0]])
    u = np.array([0.0])
    Q = np.array([[0.0]])
    H = np.array([[1.0]])
    R = np.array([[0.25]])
    mu = np.array([0.0])
    P = np.array([[1.0]])
    n = 400
    for _ in range(n):
        mu, P = kf_predict(mu, P, F, B, u, Q)
        z = np.array([truth + r.normal(0.0, 0.5)])
        mu, P = kf_update(mu, P, z, H, R)
    assert abs(mu[0] - truth) < 0.1
    assert abs(P[0, 0] - 0.25 / n) < 1e-4
