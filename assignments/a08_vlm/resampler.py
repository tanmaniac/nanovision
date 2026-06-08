"""A Perceiver-style cross-attention resampler, the alternative connector to the MLP.

A fixed set of Q learned query vectors cross-attends over the projected patch features and
returns Q output tokens, regardless of how many patches N came in. This is the connector
family of Flamingo's Perceiver Resampler and BLIP-2's Q-Former: it compresses a variable
number of patches to a fixed token budget, so the context cost does not grow with image
resolution.

This implementation is a deliberate single-layer miniature. A real Perceiver Resampler or
Q-Former stacks several blocks, each with query self-attention plus a feed-forward network,
and the queries also attend to each other. Here there is one cross-attention with one
residual and one norm, so the gradcheck target is unambiguous. A8 still PREPENDS these Q
tokens to the text the same way it prepends the MLP's patch tokens; it does not use
Flamingo's gated cross-attention injection into the LM layers.
"""

import torch
import torch.nn as nn
from torch import Tensor

from nanovision.attention import MultiHeadAttention


class PerceiverResampler(nn.Module):
    """Q learned queries cross-attend over patch features: (B, N, dim_v) -> (B, Q, dim_l).

    A small input projection maps the ViT features to dim_l first, then the queries (an
    nn.Parameter expanded to the batch) attend to them with one MultiHeadAttention layer.
    The output length is Q for any N.
    """

    def __init__(self, dim_v: int, dim_l: int, n_queries: int, n_heads: int):
        super().__init__()
        self.queries = nn.Parameter(torch.zeros(1, n_queries, dim_l))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.in_proj = nn.Linear(dim_v, dim_l)
        self.attn = MultiHeadAttention(dim_l, n_heads)
        self.norm = nn.LayerNorm(dim_l)

    def forward(self, feats: Tensor) -> Tensor:
        """feats (B, N, dim_v) -> (B, Q, dim_l).

        Project feats to dim_l, expand the queries to the batch, then
        out = norm(queries + attn(queries, kv=feats_proj)): one cross-attention with one
        residual and one norm.
        """
        raise NotImplementedError("implement the single cross-attention resampler forward")
