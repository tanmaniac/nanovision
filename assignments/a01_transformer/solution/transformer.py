"""Transformer blocks, encoder/decoder stacks, and positional schemes.

The LLaMA-style configuration is the core/default: pre-norm with RMSNorm, rotary
position embedding (RoPE), and a SwiGLU feed-forward. The 2017 alternatives -
LayerNorm, sinusoidal or learned absolute positional encodings, and a GELU MLP -
are selectable for the historical contrast but are not the default.

Shape convention: hidden states are (B, S, dim); attention weights are
(B, H, Sq, Sk). RoPE operates on per-head tensors of shape (B, H, S, Dh).
"""

import math
from typing import Optional

import torch
from torch import Tensor, nn

from nanovision.attention import MultiHeadAttention, scaled_dot_product_attention
from nanovision.primitives import MLP, LayerNorm, RMSNorm, SwiGLU


def build_causal_mask(seq_len: int) -> Tensor:
    """Additive (seq_len, seq_len) mask with -inf above the diagonal.

    Adding this to the attention logits before softmax forbids each position from
    attending to later positions, which is what makes a decoder autoregressive.
    Entry (i, j) is 0 for j <= i and -inf for j > i.
    """
    full = torch.full((seq_len, seq_len), float("-inf"))
    return torch.triu(full, diagonal=1)


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sin/cos absolute positional encoding (Vaswani et al., 2017).

        PE(pos, 2i)   = sin(pos / 10000^(2i/dim))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/dim))

    Historical contrast to RoPE. forward(x) adds the table to token embeddings;
    x is (B, S, dim), output is (B, S, dim).
    """

    def __init__(self, dim: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[: x.shape[1]].unsqueeze(0)


class LearnedPositionalEncoding(nn.Module):
    """Learned absolute positional embedding (GPT-2 style), historical contrast.

    A lookup table of `max_len` position vectors added to the token embeddings.
    forward(x): x is (B, S, dim), output is (B, S, dim).
    """

    def __init__(self, dim: int, max_len: int = 4096):
        super().__init__()
        self.pos = nn.Embedding(max_len, dim)

    def forward(self, x: Tensor) -> Tensor:
        idx = torch.arange(x.shape[1], device=x.device)
        return x + self.pos(idx).unsqueeze(0)


def _rope_freqs(seq_len: int, head_dim: int, base: float, device, dtype) -> tuple[Tensor, Tensor]:
    """Per-position cos/sin tables of shape (seq_len, head_dim) for RoPE."""
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float) / half))
    pos = torch.arange(seq_len, device=device, dtype=torch.float)
    angles = torch.outer(pos, inv_freq)  # (seq_len, half)
    # Repeat each frequency across its adjacent pair so both channels of a pair rotate by the
    # same angle, matching the paper's (eq. 34) adjacent-pair _rotate_half below.
    angles = angles.repeat_interleave(2, dim=-1)  # (seq_len, head_dim)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def _rotate_half(x: Tensor) -> Tensor:
    """Rotate adjacent channel pairs (Su et al. 2021, eq. 34): [x1, x2, x3, x4, ...] -> [-x2, x1, -x4, x3, ...]."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(q: Tensor, k: Tensor, base: float = 10000.0) -> tuple[Tensor, Tensor]:
    """Apply rotary position embedding to q and k (Su et al., 2021).

    Each pair of channels is rotated by an angle proportional to the position, so
    the dot product of a rotated query and key depends only on their relative
    offset. This is the core positional scheme of the modern stack.

        q' = q * cos + rotate_half(q) * sin

    Args:
        q: (B, H, S, Dh) queries. k: (B, H, S, Dh) keys. Dh must be even.
        base: RoPE base frequency (theta), 10000 by default.

    Returns:
        (q_rot, k_rot), each (B, H, S, Dh).
    """
    seq_len, head_dim = q.shape[-2], q.shape[-1]
    cos, sin = _rope_freqs(seq_len, head_dim, base, q.device, q.dtype)
    cos = cos.view(1, 1, seq_len, head_dim)
    sin = sin.view(1, 1, seq_len, head_dim)
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot


class _RoPEAttention(nn.Module):
    """Multi-head self/cross attention with RoPE applied to q and k.

    Used inside TransformerBlock when pos="rope". Mirrors MultiHeadAttention but
    rotates the per-head q/k before the dot product. Supports GQA via n_kv_heads.
    """

    def __init__(self, dim: int, n_heads: int, causal: bool = False, n_kv_heads: Optional[int] = None):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        assert n_heads % self.n_kv_heads == 0
        self.head_dim = dim // n_heads
        assert self.head_dim % 2 == 0, "RoPE needs an even head_dim"
        self.causal = causal
        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def forward(self, x: Tensor, kv: Optional[Tensor] = None, mask: Optional[Tensor] = None) -> Tensor:
        B, Sq, _ = x.shape
        src = x if kv is None else kv
        Sk = src.shape[1]
        q = self.q_proj(x).view(B, Sq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(src).view(B, Sk, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(src).view(B, Sk, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k)
        if self.n_kv_heads != self.n_heads:
            repeat = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        if mask is None and self.causal:
            full = torch.full((Sq, Sk), float("-inf"), device=x.device, dtype=x.dtype)
            mask = torch.triu(full, diagonal=1).view(1, 1, Sq, Sk)
        out, _ = scaled_dot_product_attention(q, k, v, mask)
        out = out.transpose(1, 2).reshape(B, Sq, self.n_heads * self.head_dim)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block; LLaMA-style defaults are the core.

        h = x + Attn(Norm(x))
        y = h + FFN(Norm(h))

    With cross_attn=True a third sub-layer attends to encoder memory `kv`:

        h = x + SelfAttn(Norm(x))
        h = h + CrossAttn(Norm(h), kv)
        y = h + FFN(Norm(h))

    Args:
        dim: model dimension.
        n_heads: number of attention heads.
        mlp_ratio: feed-forward expansion. For SwiGLU the inner width is set to
            ~ (2/3) * mlp_ratio * dim (the 8/3 rule when mlp_ratio=4) so the
            parameter count matches a GELU MLP; for the GELU MLP it is
            mlp_ratio * dim.
        causal: causal self-attention (decoder).
        cross_attn: add a cross-attention sub-layer.
        norm: "rms" (default, LLaMA-style) or "layer" (historical).
        ffn: "swiglu" (default) or "mlp" (GELU, historical).
        pos: "rope" (default, applied inside attention), or "none" when absolute
            positional encoding is added at the embedding layer instead.
        n_kv_heads: KV heads for GQA/MQA; defaults to n_heads.

    forward(x, kv=None, mask=None): x is (B, S, dim); returns (B, S, dim).
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        mlp_ratio: float = 4,
        causal: bool = False,
        cross_attn: bool = False,
        norm: str = "rms",
        ffn: str = "swiglu",
        pos: str = "rope",
        n_kv_heads: Optional[int] = None,
    ):
        super().__init__()
        self.cross_attn = cross_attn

        def make_norm() -> nn.Module:
            return RMSNorm(dim) if norm == "rms" else LayerNorm(dim)

        def make_attn(causal_: bool) -> nn.Module:
            if pos == "rope":
                return _RoPEAttention(dim, n_heads, causal=causal_, n_kv_heads=n_kv_heads)
            return MultiHeadAttention(dim, n_heads, causal=causal_, n_kv_heads=n_kv_heads)

        self.norm1 = make_norm()
        self.attn = make_attn(causal)
        if cross_attn:
            self.norm_cross = make_norm()
            # Cross-attention is never causal and does not use RoPE on the
            # cross-positions; keep it plain multi-head.
            self.cross = MultiHeadAttention(dim, n_heads, causal=False, n_kv_heads=n_kv_heads)
        self.norm2 = make_norm()

        if ffn == "swiglu":
            hidden = int(2 * mlp_ratio * dim / 3)
            hidden = (hidden + 7) // 8 * 8  # round to a multiple of 8
            self.ffn = SwiGLU(dim, hidden)
        else:
            self.ffn = MLP(dim, int(mlp_ratio * dim))

    def forward(self, x: Tensor, kv: Optional[Tensor] = None, mask: Optional[Tensor] = None) -> Tensor:
        x = x + self.attn(self.norm1(x), mask=mask)
        if self.cross_attn:
            assert kv is not None, "cross_attn block needs kv (encoder memory)"
            x = x + self.cross(self.norm_cross(x), kv=kv)
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    """Stack of N non-causal pre-norm blocks (bidirectional self-attention)."""

    def __init__(self, dim: int, n_heads: int, depth: int, mlp_ratio: float = 4,
                 norm: str = "rms", ffn: str = "swiglu", pos: str = "rope",
                 n_kv_heads: Optional[int] = None):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads, mlp_ratio, causal=False, cross_attn=False,
                             norm=norm, ffn=ffn, pos=pos, n_kv_heads=n_kv_heads)
            for _ in range(depth)
        ])

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        for block in self.blocks:
            x = block(x, mask=mask)
        return x


class TransformerDecoder(nn.Module):
    """Stack of N causal pre-norm blocks, optionally with cross-attention to memory.

    With cross_attn=True each block attends to `memory` (encoder output) after its
    causal self-attention, giving the encoder-decoder configuration.
    """

    def __init__(self, dim: int, n_heads: int, depth: int, mlp_ratio: float = 4,
                 cross_attn: bool = False, norm: str = "rms", ffn: str = "swiglu",
                 pos: str = "rope", n_kv_heads: Optional[int] = None):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads, mlp_ratio, causal=True, cross_attn=cross_attn,
                             norm=norm, ffn=ffn, pos=pos, n_kv_heads=n_kv_heads)
            for _ in range(depth)
        ])

    def forward(self, x: Tensor, memory: Optional[Tensor] = None,
                mask: Optional[Tensor] = None) -> Tensor:
        for block in self.blocks:
            x = block(x, kv=memory, mask=mask)
        return x
