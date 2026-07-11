"""TransFuser's attention fusion: self-attention over the concatenated token sets.

This is the holed file the learner fills in. The reference lives in ``solution/transfuser.py``
with identical docstrings; only the hole body below differs. This is an assignment-LOCAL module,
imported by bare name (``from transfuser import TransFuserBlock``); the attention block it builds
on comes through ``nanovision.transformer``.

TransFuser (arXiv:2104.09224) tokenizes the image and the LiDAR BEV pseudo-image at several
resolution stages, CONCATENATES the two token sets into one sequence, runs one self-attention
block over the union so every image token can attend to every LiDAR token and back, then splits
the sequence and adds each half onto its own stream. This is self-attention over the concatenated
tokens, not query/key-value cross-attention. The course isolates this fusion block and drops the
driving pipeline (the waypoint GRU); the contrast to BEVFusion is learned all-to-all token mixing
instead of a fixed channel concat.
"""

import torch
import torch.nn as nn
from torch import Tensor

from nanovision.transformer import TransformerBlock


class TransFuserBlock(nn.Module):
    """One attention-fusion block over concatenated camera and LiDAR tokens.

    A single pre-norm self-attention block runs over the union of the two token sets. Positional
    encoding is off (``pos="none"``): the tokens carry no 1D order, so attention is permutation
    equivariant and the block mixes camera and LiDAR tokens symmetrically. The block's output is
    split back and added onto each input stream (the fusion skip that lets each conv branch
    continue with its own features updated by cross-modal attention).

    Args:
        dim: per-token model dimension (both streams share it).
        n_heads: attention heads.
    """

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.block = TransformerBlock(
            dim, n_heads, causal=False, cross_attn=False, pos="none"
        )

    def forward(self, cam_tokens: Tensor, lidar_tokens: Tensor) -> tuple[Tensor, Tensor]:
        """Fuse two token sets with one self-attention block.

        Concatenate along the sequence axis, run the block over the union, split back at the
        camera/LiDAR boundary, and residual-add each half onto its input stream.

        Args:
            cam_tokens: (B, Sc, dim) camera tokens.
            lidar_tokens: (B, Sl, dim) LiDAR tokens.

        Returns:
            (cam_out, lidar_out) with the input shapes preserved.
        """
        raise NotImplementedError("TransFuserBlock.forward")
