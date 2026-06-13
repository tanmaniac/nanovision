"""Information (canonical) form: the moment<->information round-trip, equivalence of
the additive information update with the KF update, and additive multi-sensor fusion."""

import numpy as np

from _impl import (kf_update, moments_to_information, information_to_moments,
                   information_update)
from _helpers import rng
from config import TOL_TIGHT, TOL_AGREE, SEED


def _spd(r, n):
    A = r.standard_normal((n, n))
    return A @ A.T + n * np.eye(n)


def test_information_roundtrip():
    r = rng(SEED)
    n = 4
    mu = r.standard_normal(n)
    P = _spd(r, n)
    eta, Omega = moments_to_information(mu, P)
    assert np.allclose(Omega, np.linalg.inv(P), atol=TOL_AGREE)
    mu2, P2 = information_to_moments(eta, Omega)
    assert np.allclose(mu2, mu, atol=TOL_AGREE)
    assert np.allclose(P2, P, atol=TOL_AGREE)


def test_information_update_equals_kf_update():
    r = rng(SEED + 1)
    n = 3
    mu = r.standard_normal(n)
    P = _spd(r, n)
    H = r.standard_normal((2, n))
    R = _spd(r, 2)
    z = r.standard_normal(2)

    mu_kf, P_kf = kf_update(mu, P, z, H, R)

    eta, Omega = moments_to_information(mu, P)
    eta2, Omega2 = information_update(eta, Omega, z, H, R)
    mu_inf, P_inf = information_to_moments(eta2, Omega2)

    assert np.allclose(mu_inf, mu_kf, atol=TOL_AGREE)
    assert np.allclose(P_inf, P_kf, atol=TOL_AGREE)


def test_information_fuses_two_sensors_additively():
    # Two measurements fused by adding both information contributions equal applying the
    # two KF updates in sequence - the additivity that makes the information form natural
    # for multi-sensor fusion.
    r = rng(SEED + 2)
    n = 3
    mu = r.standard_normal(n)
    P = _spd(r, n)
    H1, H2 = r.standard_normal((2, n)), r.standard_normal((1, n))
    R1, R2 = _spd(r, 2), _spd(r, 1)
    z1, z2 = r.standard_normal(2), r.standard_normal(1)

    mu_kf, P_kf = kf_update(mu, P, z1, H1, R1)
    mu_kf, P_kf = kf_update(mu_kf, P_kf, z2, H2, R2)

    eta, Omega = moments_to_information(mu, P)
    eta, Omega = information_update(eta, Omega, z1, H1, R1)
    eta, Omega = information_update(eta, Omega, z2, H2, R2)
    mu_inf, P_inf = information_to_moments(eta, Omega)

    assert np.allclose(mu_inf, mu_kf, atol=TOL_AGREE)
    assert np.allclose(P_inf, P_kf, atol=TOL_AGREE)
