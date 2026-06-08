"""Hyperparameters for the 2D flow-matching toy.

The core of A6 is conditional flow matching on a 2D distribution where the velocity field
and the trajectories are fully visible. The model is a small MLP over (x, t); the data is a
2D mixture (eight Gaussians by default, two-moons as a secondary target). The image-scale
demo in viz reuses A5's U-Net and is not part of the graded tests.
"""

from dataclasses import dataclass


@dataclass
class FlowConfig:
    data_dim: int = 2
    mlp_width: int = 128
    mlp_depth: int = 3
    time_dim: int = 64          # sinusoidal time-embedding width

    toy: str = "8gauss"         # "8gauss" or "two_moons"
    batch: int = 256

    # logit-normal timestep sampling (SD3); loc/scale of the underlying normal.
    t_loc: float = 0.0
    t_scale: float = 1.0

    n_steps: int = 100          # Euler steps for sampling and reflow-pair generation
