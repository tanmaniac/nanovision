"""Timestep sampling for training: uniform or logit-normal (SD3).

Uniform sampling spends capacity on the near-noise and near-data extremes, which carry
little perceptual information. Logit-normal sampling, t = sigmoid(loc + scale*z) with
z ~ N(0,1) (Esser et al. 2024), concentrates training mass near t=0.5. For the linear path
the CFM loss is unweighted, so the timestep distribution IS the loss weighting.
"""

import torch
from torch import Tensor


def sample_timesteps(n: int, dist: str = "uniform", loc: float = 0.0, scale: float = 1.0,
                     generator: torch.Generator | None = None,
                     device: str = "cpu") -> Tensor:
    """Return (n,) timesteps in (0, 1)."""
    if dist == "uniform":
        return torch.rand(n, generator=generator, device=device)
    if dist == "logit_normal":
        z = torch.randn(n, generator=generator, device=device)
        return torch.sigmoid(loc + scale * z)
    raise ValueError(f"unknown timestep distribution {dist!r}")
