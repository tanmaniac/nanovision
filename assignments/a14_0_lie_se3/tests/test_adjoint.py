"""SE(3) adjoint: the conjugation identity and its direction (body -> spatial)."""

import numpy as np

from _impl import se3_exp, se3_adjoint
from _helpers import rng, rand_twist, rand_se3
from config import TOL_TIGHT, SEED


def test_adjoint_conjugation_identity():
    # T exp(xi) T^-1 == exp(Ad_T xi).
    r = rng(SEED)
    for _ in range(30):
        T = rand_se3(r)
        xi = rand_twist(r, trans=0.3)
        lhs = T @ se3_exp(xi) @ np.linalg.inv(T)
        rhs = se3_exp(se3_adjoint(T) @ xi)
        assert np.allclose(lhs, rhs, atol=TOL_TIGHT)


def test_adjoint_direction_body_to_spatial():
    # T exp(xi) == exp(Ad_T xi) T: Ad_T turns a right/body twist into a left/spatial one.
    r = rng(SEED + 1)
    for _ in range(30):
        T = rand_se3(r)
        xi = rand_twist(r, trans=0.3)
        lhs = T @ se3_exp(xi)
        rhs = se3_exp(se3_adjoint(T) @ xi) @ T
        assert np.allclose(lhs, rhs, atol=TOL_TIGHT)


def test_adjoint_identity_element():
    assert np.allclose(se3_adjoint(np.eye(4)), np.eye(6), atol=TOL_TIGHT)
