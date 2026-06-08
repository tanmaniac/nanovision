"""The straightness metric is ~0 for a constant field and positive for a curved one."""

import torch

from sampling import straightness


class ConstVelocity:
    def __init__(self, v):
        self.v = v

    def __call__(self, x, t):
        return self.v


class Rotation:
    """A curved field: velocity perpendicular to x, so trajectories bend."""

    def __call__(self, x, t):
        return torch.stack([-x[..., 1], x[..., 0]], dim=-1)


def test_constant_field_is_straight():
    g = torch.Generator().manual_seed(0)
    x0 = torch.randn(64, 2, generator=g)
    v = torch.randn(64, 2, generator=g)
    s = straightness(ConstVelocity(v), x0, 50)
    assert s.item() < 1e-10


def test_curved_field_is_not_straight():
    g = torch.Generator().manual_seed(1)
    x0 = torch.randn(64, 2, generator=g)
    s = straightness(Rotation(), x0, 50)
    assert s.item() > 0.1
