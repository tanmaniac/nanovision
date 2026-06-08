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

    Step: x <- x + v(x, t) * dt with dt = 1/n_steps, t running 0, dt, 2dt, ... At step k use
    t = k*dt for the whole batch. Return the final x1_hat, or the full trajectory stacked as
    (n_steps+1, B, D) if return_traj.
    """
    raise NotImplementedError("implement Euler ODE sampling")


def straightness(model, x0: Tensor, n_steps: int) -> Tensor:
    """The rectified-flow straightness metric (Liu et al. 2022).

    Run euler_sample with return_traj. The chord is x1_hat - x0 (the last minus the first
    trajectory point). Average ||(chord) - v(x_t, t)||^2 over the trajectory points (the
    same t grid as the Euler steps) and the batch. It is 0 iff the velocity is constant
    along the trajectory, i.e. the path is a straight line at constant speed.
    """
    raise NotImplementedError("implement the straightness metric")
