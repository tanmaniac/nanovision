"""A1 - attention from scratch. Fill the holes, then run the tests.

The reference implementation lives in this assignment's `solution/attention.py` (read it if you
get stuck). Do not import it here; implement the bodies yourself.

Shape convention: tensors are (B, H, S, Dh) inside attention - B batch, H heads,
S sequence length, Dh per-head dim. Attention weights are (B, H, Sq, Sk). At the
module boundary MultiHeadAttention takes and returns (B, S, dim).
"""

import math
from typing import Optional

import torch
from torch import Tensor, nn


def scaled_dot_product_attention(
    q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
) -> tuple[Tensor, Tensor]:
    """Attention(q, k, v) = softmax(q k^T / sqrt(Dh) + mask) v.

    Args:
        q: (B, H, Sq, Dh) queries.
        k: (B, H, Sk, Dh) keys.
        v: (B, H, Sk, Dh) values.
        mask: optional additive mask broadcastable to (B, H, Sq, Sk); 0 keeps a
            position, -inf forbids it.

    Returns:
        out: (B, H, Sq, Dh), the attention-weighted sum of values.
        attn: (B, H, Sq, Sk), the softmax weights (each row sums to 1).

    Implement:
        1. scores = q @ k^T / sqrt(Dh)              -> (B, H, Sq, Sk)
        2. if mask is not None: scores = scores + mask
        3. numerically stable softmax over the last dim (subtract the row max
           before exp), giving attn
        4. out = attn @ v
    Do NOT use F.scaled_dot_product_attention.
    """
    raise NotImplementedError(
        "A1 Task 1: implement scaled dot-product attention (stable softmax)"
    )


class MultiHeadAttention(nn.Module):
    """Multi-head attention with self-, cross-, and grouped-query support.

    Project x (and kv if given) to Q, K, V; split into n_heads heads of size
    dim // n_heads; run scaled_dot_product_attention per head; concat heads;
    project back to dim. kv=None is self-attention; kv given is cross-attention.
    n_kv_heads < n_heads is GQA/MQA: KV is projected to fewer heads, each shared
    across a group of query heads (repeat_interleave to expand before attention).
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        causal: bool = False,
        n_kv_heads: Optional[int] = None,
    ):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        assert n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.head_dim = dim // n_heads
        self.causal = causal

        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def forward(
        self, x: Tensor, kv: Optional[Tensor] = None, mask: Optional[Tensor] = None
    ) -> Tensor:
        """x: (B, Sq, dim). kv: (B, Sk, dim) or None. Returns (B, Sq, dim).

        Implement:
            1. src = x if kv is None else kv
            2. project x->q (n_heads), src->k,v (n_kv_heads); reshape each to
               (B, H, S, Dh) via .view(...).transpose(1, 2)
            3. if n_kv_heads != n_heads: repeat_interleave k, v on the head dim by
               n_heads // n_kv_heads (GQA/MQA)
            4. if mask is None and self.causal: build an additive (1,1,Sq,Sk) mask
               with -inf above the diagonal
            5. call scaled_dot_product_attention; merge heads back to
               (B, Sq, n_heads*head_dim); apply out_proj
        """
        raise NotImplementedError(
            "A1 Task 2: implement MultiHeadAttention.forward (self/cross/GQA)"
        )
