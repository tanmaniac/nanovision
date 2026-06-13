"""Tolerances, the simulated world, noise, and the association gate for a14_2. Provided."""

import numpy as np
from scipy.stats import chi2

TOL_TIGHT = 1e-9
SEED = 0

# Simulation: a robot driving a loop among point landmarks, observing range-bearing.
DT = 0.1
V = 1.2  # forward speed
SENSOR_RANGE = 6.0  # landmarks beyond this are not observed

# Landmarks scattered around a roughly circular loop the robot drives.
LANDMARKS = np.array([
    [4.0, 0.0], [5.0, 4.0], [2.0, 6.0], [-2.0, 5.5], [-5.0, 2.0],
    [-5.0, -2.0], [-2.0, -5.0], [2.0, -5.0], [5.0, -3.0], [0.0, 3.0],
])

# Robot process noise on [px, py, theta] and measurement noise on [range, bearing].
Q_DIAG = np.array([0.03, 0.03, 0.015]) ** 2
R_DIAG = np.array([0.12, np.deg2rad(2.5)]) ** 2

# Initial robot-pose covariance (the start pose is known but not perfectly).
INIT_P_DIAG = np.array([0.05, 0.05, 0.03])

# Mahalanobis gate: chi-square 95th percentile for the 2-DOF range-bearing innovation.
GATE = float(chi2.ppf(0.95, df=2))  # ~= 5.991
