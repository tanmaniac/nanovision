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

    Plain regression: the mean squared error between the policy's predicted chunk and the
    demonstrated a_chunk. a_chunk is (B, H, 2), c is (B, cond_in).

    See the behavior cloning and compounding error section of the README.
    """
    raise NotImplementedError("implement the behavior-cloning MSE loss")


def chunk_actions(actions: Tensor, H: int) -> Tensor:
    """Build overlapping H-step chunks from a per-step action sequence. HOLE.

    actions is (B, T, 2); return the overlapping length-H windows stacked to (B, T-H+1, H, 2). These
    windows are the chunk-training targets; because they overlap they are redundant, so de_chunk is
    the inverse used to test the round-trip. Requires H <= T.

    See the action chunking section of the README.
    """
    raise NotImplementedError("implement overlapping action chunking")


def de_chunk(chunks: Tensor) -> Tensor:
    """Reconstruct the per-step sequence from overlapping chunks. HOLE.

    chunks is (B, M, H, 2) where M = T-H+1 from chunk_actions; this is the exact inverse of
    chunk_actions, returning (B, M + H - 1, 2) = (B, T, 2).

    See the action chunking section of the README.
    """
    raise NotImplementedError("implement the de-chunk inverse of chunk_actions")


def receding_horizon_indices(T: int, H: int) -> list[int]:
    """The chunk start indices for open-loop receding-horizon execution. HOLE.

    Execute an H-step chunk, then re-query. Return the list of chunk start indices covering all T
    steps, with the final chunk clamped so it does not run past T and any duplicate from that clamp
    removed. Requires H <= T.

    See the action chunking section of the README.
    """
    raise NotImplementedError("implement the receding-horizon chunk start indices")
