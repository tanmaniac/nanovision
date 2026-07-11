"""A1 - transformer blocks and positional schemes.

The reference implementation lives in this assignment's solution/transformer.py
(read it if you get stuck). Do not import it here; implement the bodies yourself.

The LLaMA-style configuration is the core/default: pre-norm RMSNorm, RoPE, and a
SwiGLU feed-forward. LayerNorm, sinusoidal/learned absolute encodings, and the
GELU MLP are selectable for the historical contrast.

Attention (A1) and the primitives (A0/A1) come from `nanovision.*`, which routes
to your top-level code or to solution/ depending on NANOVISION_IMPL.
"""

import math
from typing import Optional

import torch
from torch import Tensor, nn

from nanovision.attention import MultiHeadAttention, scaled_dot_product_attention
from nanovision.primitives import MLP, LayerNorm, RMSNorm, SwiGLU


def build_causal_mask(seq_len: int) -> Tensor:
    """Additive (seq_len, seq_len) mask that makes attention causal.

    Adding this to the attention logits before softmax forbids each position from
    attending to later positions. Entry (i, j) is 0 for j <= i and -inf for j > i.
    See the causal mask section of the README.
    """
    mask = torch.triu(torch.ones([seq_len, seq_len]) * float("-inf"), diagonal=1)
    return mask


def apply_rope(q: Tensor, k: Tensor, base: float = 10000.0) -> tuple[Tensor, Tensor]:
    """Rotary position embedding (Su et al., 2021), the core positional scheme.

    Rotate each pair of channels by an angle proportional to the position so the
    q.k dot product depends only on the relative offset between q and k.

    Args:
        q: (B, H, S, Dh). k: (B, H, S, Dh). Dh must be even. base: theta (10000).

    Returns:
        (q_rot, k_rot), each (B, H, S, Dh).

    The module-level _rotate_half helper is provided for you.
    See the rotary position embedding section of the README.
    """
    _, _, S, d = q.shape
    half = d // 2
    thetas = base ** (-2 * torch.linspace(0, half - 1, steps=half) / half)
    thetas = thetas.unsqueeze(1).repeat([1, 2]).flatten()
    m = torch.linspace(0, S - 1, steps=S)
    cos = torch.cos(m.outer(thetas))
    sin = torch.sin(m.outer(thetas))
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin

    return q_rot, k_rot


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sin/cos absolute positional encoding (Vaswani et al., 2017).

        PE(pos, 2i)   = sin(pos / 10000^(2i/dim))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/dim))

    Historical contrast to RoPE. forward(x) adds the table to (B, S, dim) -> (B, S, dim).
    """

    def __init__(self, dim: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float) * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[: x.shape[1]].unsqueeze(0)


class LearnedPositionalEncoding(nn.Module):
    """Learned absolute positional embedding (GPT-2 style), historical contrast.

    forward(x): x is (B, S, dim), output is (B, S, dim).
    """

    def __init__(self, dim: int, max_len: int = 4096):
        super().__init__()
        self.pos = nn.Embedding(max_len, dim)

    def forward(self, x: Tensor) -> Tensor:
        idx = torch.arange(x.shape[1], device=x.device)
        return x + self.pos(idx).unsqueeze(0)


def _rotate_half(x: Tensor) -> Tensor:
    # Su et al. (2021) eq. 34 convention: rotate adjacent channel pairs.
    # [x1, x2, x3, x4, ...] -> [-x2, x1, -x4, x3, ...]
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class _RoPEAttention(nn.Module):
    """Multi-head attention with RoPE applied to q and k (used when pos="rope").

    Provided in full so TransformerBlock can use RoPE once you implement
    apply_rope above. Supports GQA via n_kv_heads.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        causal: bool = False,
        n_kv_heads: Optional[int] = None,
    ):
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

    def forward(
        self, x: Tensor, kv: Optional[Tensor] = None, mask: Optional[Tensor] = None
    ) -> Tensor:
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

    Each sub-layer normalizes its input before the operation and adds a residual.
    With cross_attn=True a cross-attention sub-layer (attending to kv) sits
    between the self-attention and FFN sub-layers.

    Args: norm in {"rms","layer"}; ffn in {"swiglu","mlp"}; pos in {"rope","none"};
    n_kv_heads for GQA/MQA.

    forward(x, kv=None, mask=None): x is (B, S, dim); returns (B, S, dim).
    See the pre-norm block section of the README.
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
                return _RoPEAttention(
                    dim, n_heads, causal=causal_, n_kv_heads=n_kv_heads
                )
            return MultiHeadAttention(
                dim, n_heads, causal=causal_, n_kv_heads=n_kv_heads
            )

        self.norm1 = make_norm()
        self.attn = make_attn(causal)
        if cross_attn:
            self.norm_cross = make_norm()
            self.cross = MultiHeadAttention(
                dim, n_heads, causal=False, n_kv_heads=n_kv_heads
            )
        self.norm2 = make_norm()

        if ffn == "swiglu":
            hidden = int(2 * mlp_ratio * dim / 3)
            hidden = (hidden + 7) // 8 * 8
            self.ffn = SwiGLU(dim, hidden)
        else:
            self.ffn = MLP(dim, int(mlp_ratio * dim))

    def forward(
        self, x: Tensor, kv: Optional[Tensor] = None, mask: Optional[Tensor] = None
    ) -> Tensor:
        """Pre-norm residual sub-layers: self-attention, optional cross-attention,
        then the feed-forward, each wrapped in a residual.

        See the pre-norm block section of the README.
        """
        block1 = x + self.attn(self.norm1(x))
        block2 = block1 + self.ffn(self.norm2(block1))
        return block2


class TransformerEncoder(nn.Module):
    """Stack of N non-causal pre-norm blocks (bidirectional self-attention)."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        depth: int,
        mlp_ratio: float = 4,
        norm: str = "rms",
        ffn: str = "swiglu",
        pos: str = "rope",
        n_kv_heads: Optional[int] = None,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim,
                    n_heads,
                    mlp_ratio,
                    causal=False,
                    cross_attn=False,
                    norm=norm,
                    ffn=ffn,
                    pos=pos,
                    n_kv_heads=n_kv_heads,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        for block in self.blocks:
            x = block(x, mask=mask)
        return x


class TransformerDecoder(nn.Module):
    """Stack of N causal pre-norm blocks, optionally cross-attending to memory."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        depth: int,
        mlp_ratio: float = 4,
        cross_attn: bool = False,
        norm: str = "rms",
        ffn: str = "swiglu",
        pos: str = "rope",
        n_kv_heads: Optional[int] = None,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim,
                    n_heads,
                    mlp_ratio,
                    causal=True,
                    cross_attn=cross_attn,
                    norm=norm,
                    ffn=ffn,
                    pos=pos,
                    n_kv_heads=n_kv_heads,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self, x: Tensor, memory: Optional[Tensor] = None, mask: Optional[Tensor] = None
    ) -> Tensor:
        for block in self.blocks:
            x = block(x, kv=memory, mask=mask)
        return x
