"""Test helpers: random group elements and a numerical right-Jacobian. Provided."""

import numpy as np

from _impl import so3_exp, so3_log, se3_exp


def rng(seed):
    return np.random.default_rng(seed)


def rand_rotvec(r, lo=0.1, hi=2.0):
    """A random SO(3) tangent vector with angle in [lo, hi] (avoids the th~pi edge)."""
    axis = r.standard_normal(3)
    axis /= np.linalg.norm(axis)
    return axis * r.uniform(lo, hi)


def rand_so3(r):
    return so3_exp(rand_rotvec(r))


def rand_twist(r, trans=1.0):
    """A random se(3) twist [rho; theta]."""
    rho = r.uniform(-trans, trans, 3)
    theta = rand_rotvec(r)
    return np.concatenate([rho, theta])


def rand_se3(r):
    return se3_exp(rand_twist(r))


def numerical_right_jacobian(w, eps):
    """J_r(w) by central differences from its defining property
    exp(w + dw) ~ exp(w) exp(J_r(w) dw), i.e. J_r(w) dw = log(exp(w)^-1 exp(w + dw))."""
    R_inv = so3_exp(w).T  # rotation inverse is the transpose
    J = np.zeros((3, 3))
    for i in range(3):
        d = np.zeros(3)
        d[i] = eps
        plus = so3_log(R_inv @ so3_exp(w + d))
        minus = so3_log(R_inv @ so3_exp(w - d))
        J[:, i] = (plus - minus) / (2 * eps)
    return J
