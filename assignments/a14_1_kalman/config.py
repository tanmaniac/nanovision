"""Tolerances, UKF parameters, and simulation settings for a14_1. Provided."""

import numpy as np

# Algebraic-identity tolerance (information round-trip, KF-vs-UKF agreement).
TOL_TIGHT = 1e-9
# KF/UKF posterior agreement on a linear-Gaussian run (floating-point, not modeling).
TOL_AGREE = 1e-6
# Numerical-vs-analytic Jacobian tolerance (central differences, step FD_STEP).
TOL_JAC = 1e-6
FD_STEP = 1e-6
SEED = 0

# UKF scaling parameters. A benign, well-conditioned set: alpha = 0.5 keeps
# n + lambda = alpha^2 * n > 0 (so the Cholesky never sees a tiny/indefinite matrix),
# beta = 2 is Gaussian-optimal, kappa = 0. The textbook alpha = 1e-3 makes the sigma
# spread ~1e-3 and the center weights ~1e6, which is exact in theory but loses digits
# to cancellation; we use 0.5 so the deterministic agreement test has clean margin.
# (kappa = 3 - n is another common choice but can drive n + lambda <= 0.)
UKF_ALPHA = 0.5
UKF_BETA = 2.0
UKF_KAPPA = 0.0

# EKF demo simulation: a unicycle driving a gentle arc, observed by range-bearing
# to a fixed landmark.
DT = 0.1
N_STEPS = 120
V = 1.0  # forward speed
OMEGA = 0.25  # turn rate (rad/s)
LANDMARK = np.array([6.0, 2.0])
# Process noise (on [px, py, theta]) and measurement noise (on [range, bearing]).
Q_DIAG = np.array([0.02, 0.02, 0.01]) ** 2
R_DIAG = np.array([0.15, np.deg2rad(3.0)]) ** 2
