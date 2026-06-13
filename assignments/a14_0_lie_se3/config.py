"""Tolerances and viz parameters for a14_0. Provided."""

# Round-trip / algebraic-identity tolerance (exp.log, hat.vee, adjoint identity).
TOL_TIGHT = 1e-9
# Numerical-vs-analytic Jacobian tolerance (central differences, step 1e-6).
TOL_JAC = 1e-6
# Finite-difference step for the numerical Jacobian check.
FD_STEP = 1e-6
# Seed for the random twists/transforms the tests sample.
SEED = 0

# viz: number of interpolation steps between the two poses.
VIZ_STEPS = 60
