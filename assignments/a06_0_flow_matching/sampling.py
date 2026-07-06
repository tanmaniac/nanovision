"""Euler ODE sampling and the rectified-flow straightness metric.

Sampling integrates dx/dt = v_theta(x, t) forward from t=0 (noise) to t=1 (data) with the
forward Euler method on a uniform t-grid. With straight trajectories (from OT coupling or
reflow), a handful of Euler steps reaches data quality that a diffusion sampler needs
hundreds of steps for.
"""

import torch
from torch import Tensor


def euler_sample(model, x0: Tensor, n_steps: int, *, return_traj: bool = False):
    """Integrate dx/dt = v(x, t) from t=0 to t=1 with n_steps forward-Euler steps.

    Return the final x1_hat, or the full trajectory stacked as (n_steps+1, B, D) if
    return_traj. See the sampling and straightness section of the README.
    """
    raise NotImplementedError("implement Euler ODE sampling")


def straightness(model, x0: Tensor, n_steps: int) -> Tensor:
    """The rectified-flow straightness metric (Liu et al. 2022): how far the instantaneous
    velocity departs from the net chord, averaged over the Euler trajectory and the batch.

    Zero exactly when the velocity is constant along the trajectory, i.e. a straight line at
    constant speed. See the sampling and straightness section of the README.
    """
    raise NotImplementedError("implement the straightness metric")
