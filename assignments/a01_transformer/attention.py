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
    """Scaled dot-product attention over a single set of heads.

    Args:
        q: (B, H, Sq, Dh) queries.
        k: (B, H, Sk, Dh) keys.
        v: (B, H, Sk, Dh) values.
        mask: optional additive mask broadcastable to (B, H, Sq, Sk); 0 keeps a
            position, -inf forbids it.

    Returns:
        out: (B, H, Sq, Dh), the attention-weighted sum of values.
        attn: (B, H, Sq, Sk), the softmax weights (each row sums to 1).

    Use a numerically stable softmax. Do NOT use F.scaled_dot_product_attention.
    See the scaled dot-product attention section of the README.
    """
    raise NotImplementedError(
        "A1 Task 1: implement scaled dot-product attention (stable softmax)"
    )


class MultiHeadAttention(nn.Module):
    """Multi-head attention with self-, cross-, and grouped-query support.

    kv=None is self-attention; kv given is cross-attention. n_kv_heads < n_heads
    is GQA/MQA: KV uses fewer heads, each shared across a group of query heads.
    See the multi-head attention section of the README.
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

        When mask is None and self.causal, build the causal mask internally.
        See the multi-head attention section of the README.
        """
        raise NotImplementedError(
            "A1 Task 2: implement MultiHeadAttention.forward (self/cross/GQA)"
        )
