"""Timestep sampling for training: uniform or logit-normal (SD3).

Uniform sampling spends capacity on the near-noise and near-data extremes, which carry
little perceptual information. Logit-normal sampling (Esser et al. 2024) concentrates training
mass near t=0.5. For the linear path the CFM loss is unweighted, so the timestep distribution
is the loss weighting. See the logit-normal timestep sampling section of the README.
"""

import torch
from torch import Tensor


def sample_timesteps(n: int, dist: str = "uniform", loc: float = 0.0, scale: float = 1.0,
                     generator: torch.Generator | None = None,
                     device: str = "cpu") -> Tensor:
    """Return (n,) timesteps in (0, 1).

    Support dist="uniform" and dist="logit_normal" (see the logit-normal timestep sampling
    section of the README). Raise ValueError on an unknown dist.
    """
    raise NotImplementedError("implement uniform and logit-normal timestep sampling")
