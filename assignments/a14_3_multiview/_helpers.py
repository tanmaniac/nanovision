"""Pure-NumPy test/viz helpers: RNG, rotations, projection, pose error metrics. Provided.

These do not import the C++ build, so sim.py can use them freely.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def rng(seed):
    return np.random.default_rng(seed)


def rotvec_to_R(rotvec):
    """3-vector axis-angle to a 3x3 rotation matrix."""
    return Rotation.from_rotvec(np.asarray(rotvec)).as_matrix()


def project(K, R, t, X):
    """Project world points X (N,3) into a camera at pose (R, t) = T_cam_world. Returns
    pixels (N,2). Points with non-positive depth come back with meaningless pixels."""
    Xc = X @ R.T + t  # (N,3) camera-frame points
    z = Xc[:, 2:3]
    uv = Xc[:, :2] / z
    return uv * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])


def rotation_angle(R_a, R_b):
    """Geodesic angle (radians) between two rotations."""
    R = R_a @ R_b.T
    c = (np.trace(R) - 1.0) / 2.0
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def translation_angle(t_a, t_b):
    """Angle (radians) between two translation directions; the monocular scale is free, so
    only the direction is comparable. Sign-folded to [0, pi/2]."""
    a = np.asarray(t_a) / np.linalg.norm(t_a)
    b = np.asarray(t_b) / np.linalg.norm(t_b)
    c = abs(float(a @ b))
    return float(np.arccos(np.clip(c, -1.0, 1.0)))
