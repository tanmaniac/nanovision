"""The linear (rectified-flow) conditional path and its constant velocity.

Convention: t=0 is noise x0 ~ N(0, I), t=1 is data x1. The path is the straight line
x_t = (1-t) x0 + t x1, so the conditional velocity dx_t/dt = x1 - x0 is constant in t.
"""

import torch
from torch import Tensor


def linear_path(x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
    """x_t = (1 - t) * x0 + t * x1. t is (B,); broadcast it over the feature dims."""
    raise NotImplementedError("implement the linear conditional path")


def linear_velocity(x0: Tensor, x1: Tensor) -> Tensor:
    """The constant conditional velocity of the linear path: x1 - x0."""
    raise NotImplementedError("implement the linear conditional velocity")
