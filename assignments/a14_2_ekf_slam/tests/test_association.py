"""Data association: the Mahalanobis gate matches a real observation to its landmark and
rejects a spurious one, and an empty map associates nothing."""

import numpy as np

from _impl import range_bearing, slam_add_landmark, slam_associate
from config import GATE


def _two_landmark_state():
    robot = np.array([0.0, 0.0, 0.0])
    P = np.diag([0.04, 0.04, 0.02])
    R = np.diag([0.12, 0.02])
    mu, P = slam_add_landmark(robot, P, np.asarray(range_bearing(robot, [4.0, 0.5])), R)
    mu, P = slam_add_landmark(mu, P, np.asarray(range_bearing(robot, [-1.0, 3.5])), R)
    return mu, P, R


def test_empty_map_associates_nothing():
    robot = np.array([0.0, 0.0, 0.0])
    P = np.diag([0.04, 0.04, 0.02])
    R = np.diag([0.12, 0.02])
    assert slam_associate(robot, P, np.array([3.0, 0.1]), R, GATE) == -1


def test_matching_observation_associates_to_its_landmark():
    # A realistically noisy reading of each landmark (not the exact prediction, which would
    # make d^2 ~ 0 trivially) still associates to the right index and passes the gate.
    mu, P, R = _two_landmark_state()
    z0 = np.asarray(range_bearing(mu[:3], mu[3:5])) + np.array([0.08, np.deg2rad(1.5)])
    z1 = np.asarray(range_bearing(mu[:3], mu[5:7])) + np.array([-0.06, np.deg2rad(-1.2)])
    assert slam_associate(mu, P, z0, R, GATE) == 0
    assert slam_associate(mu, P, z1, R, GATE) == 1


def test_spurious_observation_is_rejected():
    mu, P, R = _two_landmark_state()
    # A reading far from both landmarks in range and bearing.
    z = np.array([2.0, np.deg2rad(170.0)])
    assert slam_associate(mu, P, z, R, GATE) == -1
