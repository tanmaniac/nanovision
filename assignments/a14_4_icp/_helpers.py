"""Pure-NumPy test/viz helpers: RNG, rotations, transforms, pose error metrics. Provided.

These do not import the C++ build, so sim.py can use them freely.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def rng(seed):
    return np.random.default_rng(seed)


def rotvec_to_R(rotvec):
    return Rotation.from_rotvec(np.asarray(rotvec)).as_matrix()


def make_transform(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def apply_transform(T, X):
    """Apply a 4x4 transform to points X (N,3); returns (N,3)."""
    return X @ T[:3, :3].T + T[:3, 3]


def transform_normals(T, N):
    """Rotate unit normals by a transform's rotation (translation does not affect them)."""
    return N @ T[:3, :3].T


def rotation_angle(R_a, R_b):
    """Geodesic angle (radians) between two rotations."""
    R = R_a @ R_b.T
    c = (np.trace(R) - 1.0) / 2.0
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def transform_error(T_est, T_true):
    """Rotation angle (radians) and translation distance between two 4x4 transforms."""
    return rotation_angle(T_est[:3, :3], T_true[:3, :3]), float(
        np.linalg.norm(T_est[:3, 3] - T_true[:3, 3]))
