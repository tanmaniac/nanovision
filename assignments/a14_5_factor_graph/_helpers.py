"""Pure-NumPy test/viz helpers: RNG, planar SE(3) poses, small perturbations, error metrics.
Provided. These do not import the C++ build, so sim.py can use them freely.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def rng(seed):
    return np.random.default_rng(seed)


def pose_2d(x, y, yaw):
    """A planar pose as a 4x4 SE(3): yaw about +z, translation in the z = 0 plane."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
    T[:2, 3] = [x, y]
    return T


def small_perturbation(gen, rot_sigma, trans_sigma):
    """A small random 4x4 transform: a Gaussian rotvec and translation of the given scales."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(gen.normal(0, rot_sigma, 3)).as_matrix()
    T[:3, 3] = gen.normal(0, trans_sigma, 3)
    return T


def position(T):
    """The xy position of a pose (the planar trajectory point)."""
    return np.asarray(T)[:2, 3]


def trajectory_positions(poses):
    return np.array([position(T) for T in poses])


def pose_distance(Ta, Tb):
    """Euclidean distance between the translations of two poses."""
    return float(np.linalg.norm(np.asarray(Ta)[:3, 3] - np.asarray(Tb)[:3, 3]))


def mean_position_error(poses, gt):
    """Mean translation error of a trajectory against ground truth (gauge already fixed by the
    anchored first pose)."""
    return float(np.mean([pose_distance(p, g) for p, g in zip(poses, gt)]))
