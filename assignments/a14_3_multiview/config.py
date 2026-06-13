"""Intrinsics, the synthetic two-view scene, noise, and RANSAC settings for a14_3. Provided."""

import numpy as np

TOL_TIGHT = 1e-6   # noise-free geometric recovery
TOL_LOOSE = 1e-2   # noisy / up-to-scale recovery
SEED = 0

# Pinhole intrinsics (OpenCV axes: +x right, +y down, +z forward). 640x480 image.
FX = FY = 500.0
CX, CY = 320.0, 240.0
IMG_W, IMG_H = 640, 480
K = np.array([[FX, 0.0, CX],
              [0.0, FY, CY],
              [0.0, 0.0, 1.0]])

# The synthetic scene: a cloud of 3D points viewed by two cameras a short baseline apart.
N_POINTS = 120
DEPTH_MIN, DEPTH_MAX = 4.0, 12.0     # point depth range in front of camera 1
# True relative pose T_2_1 = (R, t): a small rotation plus a sideways/forward baseline.
REL_ROTVEC = np.array([0.03, -0.12, 0.02])   # ~7 deg, mostly a yaw
REL_TRANS = np.array([0.8, -0.05, 0.25])      # baseline (its scale is the monocular gauge)

# Noise and outliers for the robust-estimation tests.
NOISE_PX = 0.5          # Gaussian pixel noise std on each projected point
OUTLIER_FRAC = 0.35     # fraction of correspondences that are wrong matches

# RANSAC: a Sampson-distance inlier threshold in pixels, and the iteration budget.
RANSAC_THRESH_PX = 1.5
RANSAC_ITERS = 500
