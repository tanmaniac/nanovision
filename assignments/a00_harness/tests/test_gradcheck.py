"""Task 1-3 gradients (float64 gradcheck). Run after shapes."""

import torch

from nanovision.gradcheck import assert_shapes, check_gradients
from primitives import MLP, LayerNorm


def test_layernorm_gradcheck():
    m = LayerNorm(6)
    x = torch.randn(4, 6, dtype=torch.double)
    assert check_gradients(m, (x,))


def test_mlp_gradcheck():
    m = MLP(6, 12)
    x = torch.randn(3, 6, dtype=torch.double)
    assert check_gradients(m, (x,))


def test_assert_shapes_helper():
    m = MLP(8, 16)
    assert_shapes(m, [
        {"args": (torch.randn(2, 8),), "expected": (2, 8)},
        {"args": (torch.randn(2, 5, 8),), "expected": (2, 5, 8)},
    ])
