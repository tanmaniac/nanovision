"""The implementation-independent covariance statements of an EKF-SLAM update: the joint
covariance determinant does not increase, and the information matrix does not decrease in
the Loewner (PSD) order. These hold for any correct update; the per-landmark marginal
determinant is NOT asserted (a Schur complement of the joint, it is not monotone)."""

import numpy as np

from _impl import range_bearing, slam_add_landmark, slam_update
from config import TOL_TIGHT


def _two_landmark_state():
    robot = np.array([0.0, 0.0, 0.2])
    P = np.diag([0.05, 0.05, 0.02])
    R = np.diag([0.12, 0.02])
    mu, P = slam_add_landmark(robot, P, np.asarray(range_bearing(robot, [4.0, 1.0])), R)
    mu, P = slam_add_landmark(mu, P, np.asarray(range_bearing(robot, [1.0, 3.0])), R)
    return mu, P, R


def test_update_does_not_increase_joint_determinant():
    mu, P, R = _two_landmark_state()
    # A measurement of landmark 0 (use the predicted reading so only the covariance moves).
    z = np.asarray(range_bearing(mu[:3], mu[3:5]))
    _, P2 = slam_update(mu, P, z, 0, R)
    assert np.linalg.det(P2) <= np.linalg.det(P) * (1.0 + 1e-9)


def test_update_does_not_decrease_information_loewner():
    mu, P, R = _two_landmark_state()
    z = np.asarray(range_bearing(mu[:3], mu[3:5]))
    _, P2 = slam_update(mu, P, z, 0, R)
    Omega1 = np.linalg.inv(P)
    Omega2 = np.linalg.inv(P2)
    # Omega2 - Omega1 must be positive semidefinite (a measurement adds information).
    min_eig = np.linalg.eigvalsh(Omega2 - Omega1).min()
    assert min_eig > -1e-7


def test_update_keeps_joint_symmetric_psd():
    mu, P, R = _two_landmark_state()
    z = np.asarray(range_bearing(mu[:3], mu[3:5])) + np.array([0.1, 0.02])
    _, P2 = slam_update(mu, P, z, 0, R)
    assert np.allclose(P2, P2.T, atol=TOL_TIGHT)
    np.linalg.cholesky(P2 + 1e-12 * np.eye(P2.shape[0]))
