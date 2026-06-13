"""The between-factor residual and its analytic Jacobians: the residual is zero when the
estimate matches the measurement, and the analytic edge Jacobians match numerical
differentiation (the test that catches the sign / adjoint-argument bug)."""

import numpy as np

from _impl import between_residual, between_jacobians, se3_exp
from config import TOL_TIGHT


def _rand_T(gen, scale=0.8):
    return np.asarray(se3_exp(gen.normal(0, scale, 6)))


def test_residual_zero_when_estimate_matches():
    gen = np.random.default_rng(0)
    Ti = _rand_T(gen)
    Tj = _rand_T(gen)
    T_meas = np.linalg.inv(Ti) @ Tj  # the exact relative pose
    r = np.asarray(between_residual(Ti, Tj, T_meas))
    assert np.linalg.norm(r) < TOL_TIGHT


def _numerical_jacobian(Ti, Tj, T_meas, which, eps=1e-6):
    r0 = np.asarray(between_residual(Ti, Tj, T_meas))
    J = np.zeros((6, 6))
    for k in range(6):
        d = np.zeros(6)
        d[k] = eps
        if which == "i":
            r1 = np.asarray(between_residual(Ti @ np.asarray(se3_exp(d)), Tj, T_meas))
        else:
            r1 = np.asarray(between_residual(Ti, Tj @ np.asarray(se3_exp(d)), T_meas))
        J[:, k] = (r1 - r0) / eps
    return J


def test_jacobians_match_numerical():
    gen = np.random.default_rng(1)
    for _ in range(10):
        Ti = _rand_T(gen)
        Tj = _rand_T(gen)
        T_meas = np.asarray(se3_exp(gen.normal(0, 0.5, 6)))  # nonzero residual
        Ji, Jj = between_jacobians(Ti, Tj, T_meas)
        Ji, Jj = np.asarray(Ji), np.asarray(Jj)
        assert np.abs(Ji - _numerical_jacobian(Ti, Tj, T_meas, "i")).max() < 1e-5
        assert np.abs(Jj - _numerical_jacobian(Ti, Tj, T_meas, "j")).max() < 1e-5
