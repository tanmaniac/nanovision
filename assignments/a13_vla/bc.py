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
    """An MLP c -> (H, 2) regressing an H-step action chunk directly. Provided.

    H=1 is the single-step baseline. This is plain regression: no noise input, no sampling.
    """

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
    """The behavior-cloning loss. HOLE.

    Plain regression: predict the chunk with policy(c) and return the mean squared error against
    the demonstrated a_chunk (mean over all elements). a_chunk is (B, H, 2), c is (B, cond_in).
    """
    raise NotImplementedError("implement the behavior-cloning MSE loss")


def chunk_actions(actions: Tensor, H: int) -> Tensor:
    """Build overlapping H-step chunks from a per-step action sequence. HOLE.

    actions is (B, T, 2). Return overlapping windows of length H: chunk i is actions[:, i:i+H],
    stacked to (B, T-H+1, H, 2). These overlapping windows are the chunk-training targets; the
    overlap means the windows are redundant, so de_chunk defines the inverse used to test the
    round-trip. Requires H <= T.
    """
    raise NotImplementedError("implement overlapping action chunking")


def de_chunk(chunks: Tensor) -> Tensor:
    """Reconstruct the per-step sequence from overlapping chunks. HOLE.

    chunks is (B, M, H, 2) where M = T-H+1 from chunk_actions. The inverse rule: take the full
    first chunk (its H steps), then append the LAST action of each subsequent chunk. This recovers
    the original (B, T, 2) sequence exactly, because chunk i+1's last action is actions[i+H], the
    one new step that window introduced. Return (B, M + H - 1, 2) = (B, T, 2).
    """
    raise NotImplementedError("implement the de-chunk inverse of chunk_actions")


def receding_horizon_indices(T: int, H: int) -> list[int]:
    """The chunk start indices for open-loop receding-horizon execution. HOLE.

    Execute an H-step chunk, then re-query: starts are 0, H, 2H, ... up to T. The last chunk is
    clamped so it does not run past T (start = min(kH, T-H)). Return the list of start indices,
    with duplicates from the clamp removed, covering all T steps. Requires H <= T.
    """
    raise NotImplementedError("implement the receding-horizon chunk start indices")
