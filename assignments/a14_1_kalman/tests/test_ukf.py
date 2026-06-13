"""UKF unscented transform: weights, exactness on affine maps, and agreement with the
linear KF on a linear-Gaussian system (the UT is exact for affine f/h, so the only gap
is floating point)."""

import numpy as np

from _impl import (kf_predict, kf_update, ukf_sigma_points, ukf_unscented_transform)
from _helpers import rng, ukf_predict, ukf_update
from config import TOL_TIGHT, TOL_AGREE, SEED, UKF_ALPHA, UKF_BETA, UKF_KAPPA


def _spd(r, n):
    A = r.standard_normal((n, n))
    return A @ A.T + n * np.eye(n)


def test_ukf_weights():
    r = rng(SEED)
    n = 3
    mu = r.standard_normal(n)
    P = _spd(r, n)
    sig, Wm, Wc = ukf_sigma_points(mu, P, UKF_ALPHA, UKF_BETA, UKF_KAPPA)
    assert sig.shape == (2 * n + 1, n)
    # Mean weights sum to 1; the covariance weights differ from the mean weights only
    # in the center entry, by exactly (1 - alpha^2 + beta).
    assert abs(np.sum(Wm) - 1.0) < TOL_TIGHT
    assert abs((Wc[0] - Wm[0]) - (1.0 - UKF_ALPHA**2 + UKF_BETA)) < TOL_TIGHT
    assert np.allclose(Wc[1:], Wm[1:], atol=TOL_TIGHT)
    assert sig.shape[0] == Wm.size == Wc.size


def test_ukf_recovers_its_own_gaussian():
    # Sigma points of (mu, P) pushed through the identity recover (mu, P) exactly.
    r = rng(SEED + 1)
    n = 3
    mu = r.standard_normal(n)
    P = _spd(r, n)
    sig, Wm, Wc = ukf_sigma_points(mu, P, UKF_ALPHA, UKF_BETA, UKF_KAPPA)
    mean, cov = ukf_unscented_transform(sig, Wm, Wc, np.zeros((n, n)))
    assert np.allclose(mean, mu, atol=TOL_AGREE)
    assert np.allclose(cov, P, atol=TOL_AGREE)


def test_ukf_exact_on_affine():
    # The unscented transform is exact for an affine map y = A x + b: recovered
    # mean = A mu + b and covariance = A P A^T.
    r = rng(SEED + 2)
    n, m = 3, 2
    mu = r.standard_normal(n)
    P = _spd(r, n)
    A = r.standard_normal((m, n))
    b = r.standard_normal(m)
    sig, Wm, Wc = ukf_sigma_points(mu, P, UKF_ALPHA, UKF_BETA, UKF_KAPPA)
    prop = np.stack([A @ s + b for s in sig])
    mean, cov = ukf_unscented_transform(prop, Wm, Wc, np.zeros((m, m)))
    assert np.allclose(mean, A @ mu + b, atol=TOL_AGREE)
    assert np.allclose(cov, A @ P @ A.T, atol=TOL_AGREE)


def test_ukf_matches_kf_linear_run():
    # A 1D constant-velocity system filtered by both KF and UKF on the same data. The
    # posteriors agree to floating-point: the UT is exact for the affine f and h here.
    r = rng(SEED + 3)
    dt = 0.5
    F = np.array([[1.0, dt], [0.0, 1.0]])
    B = np.zeros((2, 1))
    u = np.zeros(1)
    Q = np.diag([1e-3, 1e-3])
    H = np.array([[1.0, 0.0]])
    R = np.array([[0.2]])

    mu_k = np.array([0.0, 1.0])
    P_k = np.diag([1.0, 1.0])
    mu_u = mu_k.copy()
    P_u = P_k.copy()

    f = lambda x: F @ x
    h = lambda x: H @ x
    for _ in range(40):
        z = np.array([r.normal(0.0, 1.0)])

        mu_k, P_k = kf_predict(mu_k, P_k, F, B, u, Q)
        mu_k, P_k = kf_update(mu_k, P_k, z, H, R)

        mu_u, P_u = ukf_predict(mu_u, P_u, f, Q, UKF_ALPHA, UKF_BETA, UKF_KAPPA)
        mu_u, P_u = ukf_update(mu_u, P_u, z, h, R, UKF_ALPHA, UKF_BETA, UKF_KAPPA)

        assert np.allclose(mu_k, mu_u, atol=TOL_AGREE)
        assert np.allclose(P_k, P_u, atol=TOL_AGREE)
