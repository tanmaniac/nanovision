"""EKF: the nonlinear models, their analytic Jacobians against finite differences,
and a tracking run on the simulated unicycle."""

import numpy as np

from _impl import ekf_f, ekf_F_x, ekf_h, ekf_H, ekf_predict, ekf_update
from _helpers import rng, numerical_jacobian
from config import (TOL_TIGHT, TOL_JAC, FD_STEP, SEED, DT, V, OMEGA, N_STEPS,
                    LANDMARK, Q_DIAG, R_DIAG)


def test_ekf_F_x_matches_numerical():
    r = rng(SEED)
    for _ in range(20):
        x = np.array([r.uniform(-3, 3), r.uniform(-3, 3), r.uniform(-2.5, 2.5)])
        u = np.array([r.uniform(-1, 1), r.uniform(-1, 1)])
        analytic = ekf_F_x(x, u, DT)
        numeric = numerical_jacobian(lambda s: ekf_f(s, u, DT), x, FD_STEP,
                                     angle_out_rows=(2,))
        assert np.allclose(analytic, numeric, atol=TOL_JAC)


def test_ekf_H_matches_numerical():
    r = rng(SEED + 1)
    for _ in range(20):
        x = np.array([r.uniform(-3, 3), r.uniform(-3, 3), r.uniform(-2.5, 2.5)])
        analytic = ekf_H(x, LANDMARK)
        numeric = numerical_jacobian(lambda s: ekf_h(s, LANDMARK), x, FD_STEP,
                                     angle_out_rows=(1,))
        assert np.allclose(analytic, numeric, atol=TOL_JAC)


def test_ekf_h_bearing_is_wrapped():
    # Standing at the origin facing +x with the landmark behind: bearing near +/-pi,
    # and always inside (-pi, pi].
    x = np.array([0.0, 0.0, 0.0])
    r_, phi = ekf_h(x, np.array([-5.0, 0.01]))
    assert -np.pi < phi <= np.pi
    assert abs(abs(phi) - np.pi) < 1e-2


def test_ekf_predict_update_keeps_spd():
    mu = np.array([0.0, 0.0, 0.0])
    P = np.diag([0.1, 0.1, 0.05])
    Q = np.diag(Q_DIAG)
    R = np.diag(R_DIAG)
    u = np.array([V, OMEGA])
    for _ in range(30):
        mu, P = ekf_predict(mu, P, u, DT, Q)
        mu, P = ekf_update(mu, P, ekf_h(mu, LANDMARK), LANDMARK, R)
        assert np.allclose(P, P.T, atol=TOL_TIGHT)
        np.linalg.cholesky(P)


def test_ekf_update_reduces_uncertainty():
    mu = np.array([0.5, -0.3, 0.2])
    P = np.diag([0.4, 0.4, 0.2])
    R = np.diag(R_DIAG)
    z = ekf_h(mu, LANDMARK)
    _, P2 = ekf_update(mu, P, z, LANDMARK, R)
    assert np.trace(P2) < np.trace(P)


def test_ekf_tracks_unicycle():
    # Full filter on the noisy sim. With a single range-bearing landmark the heading is
    # weakly observable, so we check the position estimate stays close to ground truth.
    from _helpers import simulate_unicycle
    gt, meas = simulate_unicycle(V, OMEGA, DT, N_STEPS, LANDMARK, Q_DIAG, R_DIAG,
                                 seed=SEED + 3)
    mu = gt[0].copy()
    P = np.diag([0.2, 0.2, 0.1])
    Q = np.diag(Q_DIAG)
    R = np.diag(R_DIAG)
    u = np.array([V, OMEGA])
    errs = []
    for k in range(N_STEPS):
        mu, P = ekf_predict(mu, P, u, DT, Q)
        mu, P = ekf_update(mu, P, meas[k], LANDMARK, R)
        errs.append(np.linalg.norm(mu[:2] - gt[k + 1][:2]))
    assert np.mean(errs[-20:]) < 0.5  # tracking, not diverging
