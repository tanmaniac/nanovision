"""Tasks 1-7 shapes. Run first."""

import torch

from attention import MultiHeadAttention, scaled_dot_product_attention
from primitives import RMSNorm, SwiGLU
from transformer import (
    TransformerBlock,
    TransformerDecoder,
    TransformerEncoder,
    apply_rope,
    build_causal_mask,
)


def test_sdpa_shapes():
    B, H, Sq, Sk, Dh = 2, 3, 5, 7, 8
    q = torch.randn(B, H, Sq, Dh)
    k = torch.randn(B, H, Sk, Dh)
    v = torch.randn(B, H, Sk, Dh)
    out, attn = scaled_dot_product_attention(q, k, v)
    assert out.shape == (B, H, Sq, Dh)
    assert attn.shape == (B, H, Sq, Sk)
    # softmax rows sum to 1
    assert torch.allclose(attn.sum(-1), torch.ones(B, H, Sq), atol=1e-6)


def test_mha_self_cross_gqa():
    x = torch.randn(2, 6, 16)
    kv = torch.randn(2, 9, 16)
    assert MultiHeadAttention(16, 4)(x).shape == (2, 6, 16)            # self
    assert MultiHeadAttention(16, 4)(x, kv=kv).shape == (2, 6, 16)     # cross
    assert MultiHeadAttention(16, 4, n_kv_heads=2)(x).shape == (2, 6, 16)  # GQA
    assert MultiHeadAttention(16, 4, n_kv_heads=1)(x).shape == (2, 6, 16)  # MQA


def test_build_causal_mask():
    m = build_causal_mask(4)
    assert m.shape == (4, 4)
    assert torch.isinf(m[0, 1]) and m[0, 1] < 0
    assert m[1, 0] == 0.0
    assert torch.equal(torch.triu(torch.ones(4, 4), diagonal=1).bool(), torch.isinf(m))


def test_apply_rope_shapes_and_norm():
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    qr, kr = apply_rope(q, k)
    assert qr.shape == q.shape and kr.shape == k.shape
    # rotation preserves per-vector norm
    assert torch.allclose(qr.norm(dim=-1), q.norm(dim=-1), atol=1e-5)


def test_rmsnorm_swiglu_shapes():
    x = torch.randn(3, 5, 16)
    assert RMSNorm(16)(x).shape == (3, 5, 16)
    assert SwiGLU(16, 32)(x).shape == (3, 5, 16)


def test_block_and_stacks():
    x = torch.randn(2, 6, 16)
    kv = torch.randn(2, 9, 16)
    assert TransformerBlock(16, 4)(x).shape == (2, 6, 16)
    assert TransformerBlock(16, 4, norm="layer", ffn="mlp", pos="none")(x).shape == (2, 6, 16)
    assert TransformerBlock(16, 4, cross_attn=True, causal=True)(x, kv=kv).shape == (2, 6, 16)
    assert TransformerEncoder(16, 4, 2)(x).shape == (2, 6, 16)
    assert TransformerDecoder(16, 4, 2)(x).shape == (2, 6, 16)
    assert TransformerDecoder(16, 4, 2, cross_attn=True)(x, memory=kv).shape == (2, 6, 16)
