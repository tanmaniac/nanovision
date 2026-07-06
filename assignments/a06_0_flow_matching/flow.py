"""The conditional flow matching objective and the score-velocity relation.

CFM regresses the network velocity onto the conditional target along the linear path
(Lipman et al. 2022). This conditional objective has the same gradient as the intractable
marginal objective, so the network learns the marginal field, the conditional average of the
velocity. Where paths from different x1 cross the same x_t, that average curves the marginal
field, which OT coupling reduces.
"""

import torch
from torch import Tensor

from path import linear_path, linear_velocity


def cfm_loss(model, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
    """Conditional flow matching loss: MSE between the predicted velocity and the target.

    t (B,) is supplied by the caller (sampled via sample_timesteps), so the loss is
    deterministic given its inputs. Reduce by summing over feature dims and averaging over the
    batch. Do not weight by t; the timestep distribution does the weighting.

    See the conditional flow matching objective section of the README.
    """
    raise NotImplementedError("implement the conditional flow matching loss")


def score_from_velocity(v: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
    """The exact score-velocity relation for the linear path.

    Defined only for t < 1: the score is singular at t=1, where the conditional path collapses
    to a point mass at x1. See the diffusion-flow equivalence section of the README.
    """
    raise NotImplementedError("implement the score from the velocity")
