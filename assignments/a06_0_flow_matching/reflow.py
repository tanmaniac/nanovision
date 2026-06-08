"""Rectified-flow reflow pair generation. Provided.

After training an initial flow, generate (x0, x1_hat) pairs by integrating the learned ODE
from noise to its t=1 endpoint, then retrain the CFM objective on these pairs to get the
2-rectified flow, whose trajectories are straighter by construction (Liu et al. 2022). This
is just euler_sample plus pairing; the lesson is what retraining on these pairs does, shown
in viz. Use enough Euler steps that x1_hat approximates the true ODE endpoint.
"""

import torch
from torch import Tensor

from sampling import euler_sample


def reflow_pairs(model, n: int, dim: int, n_steps: int = 100,
                 generator: torch.Generator | None = None,
                 device: str = "cpu") -> tuple[Tensor, Tensor]:
    """Draw x0 ~ N(0, I), integrate the learned flow to x1_hat, return the (x0, x1_hat) pairs."""
    x0 = torch.randn(n, dim, generator=generator, device=device)
    with torch.no_grad():
        x1_hat = euler_sample(model, x0, n_steps)
    return x0, x1_hat
