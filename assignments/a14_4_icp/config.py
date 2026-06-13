"""Tolerances, the synthetic clouds, the known test transform, and ICP settings. Provided."""

import numpy as np

TOL_TIGHT = 1e-9   # closed-form alignment on exact correspondences
TOL = 1e-6
SEED = 0

N_POINTS = 800

# The known transform that separates the source cloud from the target. ICP (and the matched
# closed-form solve) must recover its inverse, mapping source back onto target. Kept modest so
# nearest-neighbor correspondences from the identity initialization are mostly correct.
PERTURB_ROTVEC = np.array([0.06, -0.10, 0.08])   # ~9 deg
PERTURB_TRANS = np.array([0.15, -0.10, 0.08])

# ICP outer-loop settings.
MAX_ITER = 60
MAX_CORR_DIST = 1.0   # reject correspondences farther than this (units of the cloud)
