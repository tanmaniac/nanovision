"""The behavior-cloning baseline and action chunking.

Behavior cloning (BC) treats expert demonstrations as supervised data: predict the expert's action
from the conditioning. Single-step BC (H=1) is the compounding-error baseline; at rollout, small
errors push the point mass into states absent from the demos and the error compounds.

Action chunking (ACT, Zhao et al. 2023) predicts an H-step chunk and executes it open-loop before
re-querying, cutting the decision frequency by H and forcing internally consistent trajectories.

BCPolicy regresses the conditional MEAN action chunk. When p(a|c) is unimodal (goal visible) the
mean is correct and BC matches the flow head; when p(a|c) is multimodal (goal hidden) the mean is a
point between modes that the expert never takes, so BC fails where the flow head samples a mode.

Shapes: c is (B, cond_in); the chunk is (B, H, 2).
"""

import torch
from torch import Tensor, nn


class BCPolicy(nn.Module):
    """An MLP c -> (H, 2) regressing an H-step action chunk directly. Provided."""

    def __init__(self, cfg, cond_in: int):
        super().__init__()
        self.H = cfg.chunk
        self.act_dim = cfg.act_dim
        w = cfg.hidden
        self.net = nn.Sequential(
            nn.Linear(cond_in, w), nn.SiLU(),
            nn.Linear(w, w), nn.SiLU(),
            nn.Linear(w, self.H * self.act_dim),
        )

    def forward(self, c: Tensor) -> Tensor:
        return self.net(c).reshape(c.shape[0], self.H, self.act_dim)


def bc_loss(policy: BCPolicy, a_chunk: Tensor, c: Tensor) -> Tensor:
    """The behavior-cloning loss: MSE between the predicted and demonstrated chunk."""
    return ((policy(c) - a_chunk) ** 2).mean()


def chunk_actions(actions: Tensor, H: int) -> Tensor:
    """Build overlapping H-step chunks: (B, T, 2) -> (B, T-H+1, H, 2)."""
    B, T, A = actions.shape
    idx = torch.arange(T - H + 1, device=actions.device)[:, None] + torch.arange(H, device=actions.device)[None, :]
    return actions[:, idx]                      # (B, T-H+1, H, 2)


def de_chunk(chunks: Tensor) -> Tensor:
    """Inverse of chunk_actions: full first chunk, then the last action of each subsequent chunk."""
    first = chunks[:, 0]                         # (B, H, 2)
    if chunks.shape[1] == 1:
        return first
    rest = chunks[:, 1:, -1]                     # (B, M-1, 2)
    return torch.cat([first, rest], dim=1)      # (B, H + M - 1, 2) = (B, T, 2)


def receding_horizon_indices(T: int, H: int) -> list[int]:
    """Open-loop chunk start indices: 0, H, 2H, ... with the last clamped to T-H."""
    starts = []
    k = 0
    while k * H < T:
        starts.append(min(k * H, T - H))
        k += 1
    # Drop a duplicate if the clamp made the last two starts equal.
    out = []
    for s in starts:
        if not out or out[-1] != s:
            out.append(s)
    return out
