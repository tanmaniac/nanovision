"""Tasks 1, 2, 5 gradients (float64 gradcheck). Run after shapes."""

import torch
from torch import nn

from nanovision.gradcheck import check_gradients
from nanovision.attention import MultiHeadAttention, scaled_dot_product_attention
from nanovision.primitives import RMSNorm, SwiGLU


class _SDPAModule(nn.Module):
    """Wrap the SDPA function so check_gradients can run it (returns out only)."""

    def forward(self, q, k, v):
        out, _ = scaled_dot_product_attention(q, k, v)
        return out


def test_sdpa_gradcheck():
    B, H, Sq, Sk, Dh = 1, 2, 3, 4, 4
    q = torch.randn(B, H, Sq, Dh, dtype=torch.double)
    k = torch.randn(B, H, Sk, Dh, dtype=torch.double)
    v = torch.randn(B, H, Sk, Dh, dtype=torch.double)
    assert check_gradients(_SDPAModule(), (q, k, v))


def test_mha_gradcheck():
    m = MultiHeadAttention(8, 2)
    x = torch.randn(2, 3, 8, dtype=torch.double)
    assert check_gradients(m, (x,))


def test_mha_gqa_gradcheck():
    m = MultiHeadAttention(8, 4, n_kv_heads=2)
    x = torch.randn(2, 3, 8, dtype=torch.double)
    assert check_gradients(m, (x,))


def test_rmsnorm_gradcheck():
    m = RMSNorm(6)
    x = torch.randn(4, 6, dtype=torch.double)
    assert check_gradients(m, (x,))


def test_swiglu_gradcheck():
    m = SwiGLU(6, 12)
    x = torch.randn(4, 6, dtype=torch.double)
    assert check_gradients(m, (x,))
