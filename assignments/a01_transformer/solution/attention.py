"""Scaled dot-product attention and multi-head attention, built from scratch.

This is the core of A1 and is reused by every later assignment (ViT, CLIP, DiT,
BEVFormer, VLM, VLA). The softmax over QK^T is implemented by hand with the
standard max-subtraction trick for numerical stability. The high-level shortcuts
(`F.scaled_dot_product_attention`, `nn.MultiheadAttention`) are forbidden here -
the point is to build the mechanism.

Shape convention across this module: tensors are (B, H, S, Dh) inside attention,
where B is batch, H is heads, S is sequence length, Dh is per-head dimension.
Attention weights are (B, H, Sq, Sk). At the module boundary MultiHeadAttention
takes and returns (B, S, dim).
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
        mask: optional additive mask broadcastable to (B, H, Sq, Sk); use 0 to
            keep a position and -inf to forbid it.

    Returns:
        out: (B, H, Sq, Dh), the attention-weighted sum of values.
        attn: (B, H, Sq, Sk), the softmax weights (rows sum to 1).

    The softmax is computed by subtracting the per-row max before exponentiating,
    so large logits do not overflow.
    """
    dh = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dh)
    if mask is not None:
        scores = scores + mask
    scores = scores - scores.amax(dim=-1, keepdim=True)
    weights = torch.exp(scores)
    attn = weights / weights.sum(dim=-1, keepdim=True)
    out = torch.matmul(attn, v)
    return out, attn


class MultiHeadAttention(nn.Module):
    """Multi-head attention supporting self-, cross-, and grouped-query attention.

    The input is projected to Q, K, V, split into `n_heads` heads of size
    `dim // n_heads`, run through scaled dot-product attention per head, then the
    heads are concatenated and projected back to `dim`.

    Self- vs cross-attention is just where K and V come from: with `kv=None` they
    come from `x` (self-attention); with `kv` given they come from `kv`
    (cross-attention, e.g. a decoder attending to encoder memory).

    Grouped/multi-query attention: when `n_kv_heads < n_heads`, K and V are
    projected to fewer heads and each KV head is shared across a group of query
    heads. `n_kv_heads == 1` is multi-query attention; `n_kv_heads == n_heads`
    (the default) is ordinary multi-head attention. This shrinks the KV-cache at
    inference while keeping most of the quality.

    Args:
        dim: model dimension.
        n_heads: number of query heads; must divide `dim`.
        causal: if True, build and apply a causal mask in self-attention.
        n_kv_heads: number of key/value heads; defaults to `n_heads`. Must divide
            `n_heads`.

    forward(x, kv=None, mask=None):
        x: (B, Sq, dim). kv: (B, Sk, dim) or None. mask: additive mask
        broadcastable to (B, n_heads, Sq, Sk) or None. Returns (B, Sq, dim).
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
        B, Sq, _ = x.shape
        src = x if kv is None else kv
        Sk = src.shape[1]

        # Project and split into heads: (B, H, S, Dh).
        q = self.q_proj(x).view(B, Sq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(src).view(B, Sk, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(src).view(B, Sk, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # GQA/MQA: repeat each KV head to cover its group of query heads.
        if self.n_kv_heads != self.n_heads:
            repeat = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        if mask is None and self.causal:
            mask = self._causal_mask(Sq, Sk, x.device, x.dtype)

        out, _ = scaled_dot_product_attention(q, k, v, mask)
        out = out.transpose(1, 2).reshape(B, Sq, self.n_heads * self.head_dim)
        return self.out_proj(out)

    @staticmethod
    def _causal_mask(sq: int, sk: int, device, dtype) -> Tensor:
        """Additive (1, 1, sq, sk) mask with -inf above the diagonal."""
        full = torch.full((sq, sk), float("-inf"), device=device, dtype=dtype)
        return torch.triu(full, diagonal=1).view(1, 1, sq, sk)
