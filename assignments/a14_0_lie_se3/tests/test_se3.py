"""SE(3): hat/vee, exp/log round-trips, box-plus/box-minus consistency."""

import numpy as np

from _impl import hat6, vee6, se3_exp, se3_log, se3_boxplus, se3_boxminus
from _helpers import rng, rand_twist, rand_se3
from config import TOL_TIGHT, SEED


def test_hat6_vee6_inverse():
    r = rng(SEED)
    for _ in range(20):
        xi = r.standard_normal(6)
        assert np.allclose(vee6(hat6(xi)), xi, atol=TOL_TIGHT)


def test_hat6_structure():
    # xi = [rho; theta]; the bottom row is zero, top-left is skew, top-right is rho.
    xi = np.arange(1.0, 7.0)
    X = hat6(xi)
    assert np.allclose(X[3, :], 0.0, atol=TOL_TIGHT)
    assert np.allclose(X[:3, :3], -X[:3, :3].T, atol=TOL_TIGHT)
    assert np.allclose(X[:3, 3], xi[:3], atol=TOL_TIGHT)


def test_exp_is_se3():
    r = rng(SEED + 1)
    for _ in range(20):
        T = rand_se3(r)
        assert np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=TOL_TIGHT)
        assert np.allclose(T[3, :], [0, 0, 0, 1], atol=TOL_TIGHT)


def test_exp_log_roundtrip():
    r = rng(SEED + 2)
    for _ in range(50):
        xi = rand_twist(r)
        assert np.allclose(se3_log(se3_exp(xi)), xi, atol=TOL_TIGHT)


def test_boxplus_boxminus_inverse():
    r = rng(SEED + 3)
    for _ in range(50):
        T1 = rand_se3(r)
        xi = rand_twist(r, trans=0.5)
        T2 = se3_boxplus(T1, xi)
        assert np.allclose(se3_boxminus(T2, T1), xi, atol=TOL_TIGHT)
        # boxplus(T1, boxminus(T2, T1)) == T2
        assert np.allclose(se3_boxplus(T1, se3_boxminus(T2, T1)), T2, atol=TOL_TIGHT)
