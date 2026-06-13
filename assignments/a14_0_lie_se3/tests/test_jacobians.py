"""SO(3) Jacobians: inverse consistency, the J_r = J_l(-w) relation, and the
numerical-vs-analytic right-Jacobian check (the gate on a hand-derived J_r)."""

import numpy as np

from _impl import (
    so3_left_jacobian,
    so3_left_jacobian_inv,
    so3_right_jacobian,
    so3_right_jacobian_inv,
)
from _helpers import rng, rand_rotvec, numerical_right_jacobian
from config import TOL_TIGHT, TOL_JAC, FD_STEP, SEED


def test_left_jacobian_inverse():
    r = rng(SEED)
    for _ in range(30):
        w = rand_rotvec(r)
        assert np.allclose(so3_left_jacobian(w) @ so3_left_jacobian_inv(w), np.eye(3), atol=TOL_TIGHT)


def test_right_jacobian_inverse():
    r = rng(SEED + 1)
    for _ in range(30):
        w = rand_rotvec(r)
        assert np.allclose(so3_right_jacobian(w) @ so3_right_jacobian_inv(w), np.eye(3), atol=TOL_TIGHT)


def test_right_equals_left_of_negative():
    # J_r(w) = J_l(-w) = J_l(w)^T.
    r = rng(SEED + 2)
    for _ in range(30):
        w = rand_rotvec(r)
        assert np.allclose(so3_right_jacobian(w), so3_left_jacobian(-w), atol=TOL_TIGHT)
        assert np.allclose(so3_right_jacobian(w), so3_left_jacobian(w).T, atol=TOL_TIGHT)


def test_right_jacobian_numerical():
    # The analytic J_r must match central differences of its defining property.
    r = rng(SEED + 3)
    for _ in range(20):
        w = rand_rotvec(r)
        num = numerical_right_jacobian(w, FD_STEP)
        ana = so3_right_jacobian(w)
        assert np.allclose(ana, num, atol=TOL_JAC), f"max err {np.abs(ana - num).max():.2e}"
