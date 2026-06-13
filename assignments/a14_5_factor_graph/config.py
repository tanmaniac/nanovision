"""Tolerances, the synthetic pose graph, and noise settings for a14_5. Provided."""

import numpy as np

TOL_TIGHT = 1e-9
TOL = 1e-6
SEED = 0

# The ground-truth trajectory: a planar loop of N_POSES poses on a circle of radius RADIUS.
N_POSES = 24
RADIUS = 5.0

# Initialization perturbation for the exact-recovery problem (per pose, pose 0 left exact).
INIT_ROT_SIGMA = 0.15
INIT_TRANS_SIGMA = 0.4

# Odometry and loop-closure measurement noise for the drift-then-correct problem.
ODOM_SIGMA = 0.02
LOOP_SIGMA = 0.05

GN_ITERS = 20
