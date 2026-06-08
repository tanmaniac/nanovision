"""Positional encoding: output shape, exact per-band values, and float64 gradcheck."""

import torch

from encoding import PositionalEncoding


def test_shape_with_include_input():
    L = 6
    enc = PositionalEncoding(L, include_input=True)
    x = torch.randn(4, 5, 3)
    out = enc(x)
    D = 3
    assert out.shape == (4, 5, D + D * 2 * L)


def test_shape_without_include_input():
    L = 4
    enc = PositionalEncoding(L, include_input=False)
    x = torch.randn(7, 3)
    out = enc(x)
    D = 3
    assert out.shape == (7, D * 2 * L)


def test_per_band_values_exact():
    # For a 1D input, the include_input channel is x, then per band k the pair
    # (sin(2^k pi x), cos(2^k pi x)). Check each band matches the closed form.
    L = 5
    enc = PositionalEncoding(L, include_input=True)
    x = torch.linspace(-1.0, 1.0, 11).reshape(-1, 1)   # (11, 1)
    out = enc(x)                                       # (11, 1 + 1*2*L)
    assert torch.allclose(out[:, 0:1], x)
    for k in range(L):
        sin_ch = out[:, 1 + 2 * k]
        cos_ch = out[:, 1 + 2 * k + 1]
        assert torch.allclose(sin_ch, torch.sin(2.0 ** k * torch.pi * x[:, 0]), atol=1e-6)
        assert torch.allclose(cos_ch, torch.cos(2.0 ** k * torch.pi * x[:, 0]), atol=1e-6)


def test_gradcheck():
    enc = PositionalEncoding(3, include_input=True).double()
    x = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a: enc(a), (x,))
