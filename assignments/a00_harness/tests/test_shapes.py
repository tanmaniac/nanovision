"""Task 1-3 shapes. Run first."""

import torch

from nanovision.primitives import MLP, LayerNorm, gelu


def test_gelu_shape_and_zero():
    x = torch.randn(4, 8)
    y = gelu(x)
    assert y.shape == x.shape
    # GELU(0) == 0
    assert torch.allclose(gelu(torch.zeros(3)), torch.zeros(3), atol=1e-6)


def test_layernorm_shape_and_stats():
    ln = LayerNorm(16)
    x = torch.randn(5, 16) * 3 + 2
    y = ln(x)
    assert y.shape == x.shape
    # default weight=1, bias=0 -> output has ~zero mean, ~unit std over last dim
    assert torch.allclose(y.mean(-1), torch.zeros(5), atol=1e-5)
    assert torch.allclose(y.std(-1, unbiased=False), torch.ones(5), atol=1e-4)


def test_mlp_shape():
    mlp = MLP(16, 32)
    x = torch.randn(5, 7, 16)
    assert mlp(x).shape == (5, 7, 16)
