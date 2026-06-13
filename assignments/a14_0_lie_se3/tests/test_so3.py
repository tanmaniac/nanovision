"""SO(3): hat/vee, exp/log round-trips, and a known rotation."""

import numpy as np
import pytest

from _impl import hat3, vee3, so3_exp, so3_log
from _helpers import rng, rand_rotvec, rand_so3
from config import TOL_TIGHT, SEED


def test_hat_vee_inverse():
    r = rng(SEED)
    for _ in range(20):
        w = r.standard_normal(3)
        assert np.allclose(vee3(hat3(w)), w, atol=TOL_TIGHT)
    # hat is skew-symmetric
    W = hat3(r.standard_normal(3))
    assert np.allclose(W, -W.T, atol=TOL_TIGHT)


def test_hat_cross_product():
    r = rng(SEED + 1)
    w = r.standard_normal(3)
    v = r.standard_normal(3)
    assert np.allclose(hat3(w) @ v, np.cross(w, v), atol=TOL_TIGHT)


def test_exp_is_rotation():
    r = rng(SEED + 2)
    for _ in range(20):
        R = rand_so3(r)
        assert np.allclose(R.T @ R, np.eye(3), atol=TOL_TIGHT)
        assert abs(np.linalg.det(R) - 1.0) < TOL_TIGHT


def test_exp_log_roundtrip():
    r = rng(SEED + 3)
    for _ in range(50):
        w = rand_rotvec(r)
        assert np.allclose(so3_log(so3_exp(w)), w, atol=TOL_TIGHT)


def test_exp_small_angle():
    # The small-angle branch must agree with a tiny finite rotation.
    w = np.array([1e-10, -2e-10, 5e-11])
    R = so3_exp(w)
    assert np.allclose(R, np.eye(3) + hat3(w), atol=1e-15)


def test_known_rotation_z_90():
    R = so3_exp(np.array([0.0, 0.0, np.pi / 2]))
    expected = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(R, expected, atol=TOL_TIGHT)
