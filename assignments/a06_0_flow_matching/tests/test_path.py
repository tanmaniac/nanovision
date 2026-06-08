"""The linear path endpoints and the constant velocity."""

import torch

from path import linear_path, linear_velocity


def test_endpoints_and_midpoint():
    g = torch.Generator().manual_seed(0)
    x0 = torch.randn(5, 2, generator=g)
    x1 = torch.randn(5, 2, generator=g)
    z = torch.zeros(5)
    one = torch.ones(5)
    half = torch.full((5,), 0.5)
    assert torch.allclose(linear_path(x0, x1, z), x0)
    assert torch.allclose(linear_path(x0, x1, one), x1)
    assert torch.allclose(linear_path(x0, x1, half), 0.5 * (x0 + x1))


def test_velocity_is_displacement():
    g = torch.Generator().manual_seed(1)
    x0 = torch.randn(5, 2, generator=g)
    x1 = torch.randn(5, 2, generator=g)
    assert torch.allclose(linear_velocity(x0, x1), x1 - x0)
