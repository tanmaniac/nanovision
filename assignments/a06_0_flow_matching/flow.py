"""The conditional flow matching objective and the score-velocity relation.

CFM regresses the network velocity onto the conditional target u = x1 - x0 along the linear
path (Lipman et al. 2022). This conditional objective has the same gradient as the
intractable marginal objective, so the network learns the marginal field
v(x, t) = E[x1 - x0 | x_t = x], the conditional average of the velocity. Where paths from
different x1 cross the same x_t, that average curves the marginal field, which is what OT
coupling reduces.
"""

import torch
from torch import Tensor

from path import linear_path, linear_velocity


def cfm_loss(model, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
    """Conditional flow matching loss: MSE between the predicted velocity and x1 - x0.

    Build x_t on the linear path, target u = x1 - x0, predict v = model(x_t, t), and return
    the mean squared error (sum over feature dims, mean over the batch). t (B,) is supplied
    by the caller (sampled via sample_timesteps), so the loss is deterministic given its
    inputs. Do not weight by t; the timestep distribution does the weighting.
    """
    raise NotImplementedError("implement the conditional flow matching loss")


def score_from_velocity(v: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
    """The exact score-velocity relation for the linear path: score = (t*v - x_t)/(1 - t).

    Derivation (this convention): x_t | x1 ~ N(t*x1, (1-t)^2 I), so the conditional score is
    -(x_t - t*x1)/(1-t)^2; with u = (x1 - x_t)/(1-t) substituted, this reduces to
    (t*v - x_t)/(1-t). The score is singular at t=1 (the conditional becomes a point mass at
    x1), so this is defined only for t < 1.
    """
    raise NotImplementedError("implement the score from the velocity")
