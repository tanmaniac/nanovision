"""Tasks 1 + 3: a causal mask zeros the attention weights above the diagonal.

A decoder position must not attend to later positions. Adding the build_causal_mask
mask to the logits before softmax should drive every upper-triangle weight to ~0.
"""

import torch

from nanovision.attention import scaled_dot_product_attention
from nanovision.transformer import build_causal_mask


def test_causal_upper_triangle_is_zero():
    B, H, S, Dh = 2, 2, 6, 8
    q = torch.randn(B, H, S, Dh)
    k = torch.randn(B, H, S, Dh)
    v = torch.randn(B, H, S, Dh)
    mask = build_causal_mask(S)
    _, attn = scaled_dot_product_attention(q, k, v, mask)
    upper = torch.triu(torch.ones(S, S), diagonal=1).bool()
    assert torch.allclose(attn[..., upper], torch.zeros_like(attn[..., upper]), atol=1e-6)
    # rows still normalize to 1 over the allowed (lower-triangular) positions
    assert torch.allclose(attn.sum(-1), torch.ones(B, H, S), atol=1e-6)


def test_mha_causal_matches_explicit_mask():
    torch.manual_seed(0)
    from nanovision.attention import MultiHeadAttention

    x = torch.randn(1, 5, 8)
    causal_mha = MultiHeadAttention(8, 2, causal=True)
    # Same module, mask passed explicitly, must match the internal causal path.
    mask = build_causal_mask(5).view(1, 1, 5, 5)
    out_internal = causal_mha(x)
    out_explicit = causal_mha(x, mask=mask)
    assert torch.allclose(out_internal, out_explicit, atol=1e-6)
