"""State augmentation: a new landmark is placed at the right world position and the joint
covariance grows by two consistent rows/columns."""

import numpy as np

from _impl import range_bearing, slam_add_landmark
from config import TOL_TIGHT


def test_landmark_initialized_at_truth():
    robot = np.array([1.0, 2.0, 0.5])
    lm_true = np.array([4.0, 3.0])
    z = np.asarray(range_bearing(robot, lm_true))  # noise-free observation
    P = np.diag([0.01, 0.01, 0.005])
    R = np.diag([0.1, 0.01])

    mu2, P2 = slam_add_landmark(robot, P, z, R)
    assert mu2.shape == (5,)
    assert np.allclose(mu2[3:5], lm_true, atol=1e-9)  # inverse model recovers the landmark
    assert P2.shape == (5, 5)


def test_augmented_covariance_symmetric_psd():
    robot = np.array([0.0, 0.0, 0.0])
    z = np.array([3.0, 0.4])
    P = np.diag([0.02, 0.02, 0.01])
    R = np.diag([0.1, 0.02])
    _, P2 = slam_add_landmark(robot, P, z, R)
    assert np.allclose(P2, P2.T, atol=TOL_TIGHT)
    np.linalg.cholesky(P2 + 1e-12 * np.eye(P2.shape[0]))  # PSD
    # The existing robot block is carried over unchanged.
    assert np.allclose(P2[:3, :3], P, atol=TOL_TIGHT)
