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

    Returns the final x1_hat, or the full trajectory (n_steps+1, B, D) if return_traj.
    """
    dt = 1.0 / n_steps
    x = x0
    traj = [x]
    for k in range(n_steps):
        t = torch.full((x.shape[0],), k * dt, device=x.device, dtype=x.dtype)
        x = x + model(x, t) * dt
        traj.append(x)
    if return_traj:
        return torch.stack(traj, dim=0)
    return x


def straightness(model, x0: Tensor, n_steps: int) -> Tensor:
    """The rectified-flow straightness metric (Liu et al. 2022).

    Along the Euler trajectory, measure how far the instantaneous velocity v(x_t, t) departs
    from the net displacement (chord) x1_hat - x0, averaged over trajectory points and the
    batch: E_t,x[ ||(x1_hat - x0) - v(x_t, t)||^2 ]. It is 0 iff the velocity is constant
    along the trajectory, i.e. the path is a straight line at constant speed.
    """
    traj = euler_sample(model, x0, n_steps, return_traj=True)   # (n_steps+1, B, D)
    chord = traj[-1] - traj[0]                                   # (B, D)
    dt = 1.0 / n_steps
    total = x0.new_zeros(())
    for k in range(n_steps):
        t = torch.full((x0.shape[0],), k * dt, device=x0.device, dtype=x0.dtype)
        v = model(traj[k], t)
        total = total + ((chord - v) ** 2).flatten(1).sum(dim=1).mean()
    return total / n_steps
