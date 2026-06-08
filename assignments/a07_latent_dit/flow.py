"""The conditional flow-matching objective and Euler sampler, in latent space. Provided.

This is the flow-matching assignment's linear-interpolant objective with class
conditioning added; the denoiser now sees a VAE latent instead of a pixel image and is
passed a class label y. The convention is identical to that assignment and must not be
flipped:

    t = 0 is noise x0 ~ N(0, I); t = 1 is data x1 (here the VAE latent).
    path:   x_t = (1 - t) x0 + t x1
    target: u = x1 - x0   (constant in t)
    model predicts v_theta(x_t, t, y); sampling integrates dx/dt = v from t=0 to t=1.

cfm_loss takes x0 as an argument (it does not sample noise internally) so a test can fix
x0 and get a deterministic regression target. The viz/train loop samples a fresh
x0 = torch.randn_like(x1) each step.
"""

import torch
from torch import Tensor


def cfm_loss(model, x0: Tensor, x1: Tensor, y: Tensor, t: Tensor) -> Tensor:
    """Conditional flow-matching loss in latent space.

    Build x_t = (1 - t) x0 + t x1, target u = x1 - x0, predict v = model(x_t, t, y), and
    return the per-image-sum, batch-mean squared error (sum over the C,H,W latent dims,
    mean over the batch). t (B,) is supplied by the caller; the loss is deterministic given
    its inputs.
    """
    tv = t.view(-1, *([1] * (x1.dim() - 1)))
    x_t = (1.0 - tv) * x0 + tv * x1
    target = x1 - x0
    pred = model(x_t, t, y)
    return ((pred - target) ** 2).flatten(1).sum(dim=1).mean()


def euler_sample(model, x0: Tensor, y: Tensor, n_steps: int) -> Tensor:
    """Integrate dx/dt = v(x, t, y) from t=0 to t=1 with n_steps forward-Euler steps.

    Step: x <- x + v(x, t, y) * dt with dt = 1/n_steps, t running 0, dt, 2dt, ... At step k
    the whole batch uses t = k*dt. Returns the final latent x1_hat.
    """
    dt = 1.0 / n_steps
    x = x0
    for k in range(n_steps):
        t = torch.full((x.shape[0],), k * dt, device=x.device, dtype=x.dtype)
        x = x + model(x, t, y) * dt
    return x
