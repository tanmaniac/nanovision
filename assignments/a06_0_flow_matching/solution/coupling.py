"""Minibatch optimal-transport coupling (Tong et al. 2023, OT-CFM).

With independent coupling (random x0 paired with random x1), paths from different data
points cross, so the marginal velocity field is curved and needs many integration steps.
Pairing each x0 with its optimal x1 inside the minibatch, under the squared-L2 cost, gives
the discrete optimal-transport plan (a permutation, since the marginals are uniform), which
straightens the trajectories without any reflow.
"""

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor


def ot_coupling(x0: Tensor, x1: Tensor) -> tuple[Tensor, Tensor]:
    """Reorder x1 so row i of x0 pairs with its optimal x1 under squared-L2 cost.

    Builds C[i,j] = ||x0[i] - x1[j]||^2, solves the assignment with the Hungarian algorithm
    (the exact minimizer), and returns (x1_reordered, perm) where x1_reordered = x1[perm].
    linear_sum_assignment returns row indices already sorted, so the permutation to apply to
    x1 is the column indices.
    """
    cost = torch.cdist(x0, x1) ** 2                  # (B, B) squared-L2
    row, col = linear_sum_assignment(cost.detach().cpu().numpy())
    perm = torch.as_tensor(col, device=x1.device, dtype=torch.long)
    return x1[perm], perm
