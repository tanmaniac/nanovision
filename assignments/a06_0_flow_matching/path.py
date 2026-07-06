"""The linear (rectified-flow) conditional path and its constant velocity.

Convention: t=0 is noise x0 ~ N(0, I), t=1 is data x1. The path is the straight line between
noise and data, so the conditional velocity is constant in t. See the linear path section of
the README.
"""

import torch
from torch import Tensor


def linear_path(x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
    """The linear path point x_t at time t. t is (B,); broadcast it over the feature dims.

    See the linear path section of the README.
    """
    raise NotImplementedError("implement the linear conditional path")


def linear_velocity(x0: Tensor, x1: Tensor) -> Tensor:
    """The constant conditional velocity of the linear path. See the linear path section of the README."""
    raise NotImplementedError("implement the linear conditional velocity")
