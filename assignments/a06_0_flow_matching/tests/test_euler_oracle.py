"""Euler sampling is exact for the constant-velocity oracle.

The linear path is a straight line, so its conditional velocity x1 - x0 is constant. A
velocity field that returns that constant integrates from x0 to x1 EXACTLY with forward
Euler at any number of steps, including 1 (no discretization error for a constant field).
"""

import torch

from sampling import euler_sample


class ConstVelocity:
    """Returns the same per-sample constant velocity regardless of x and t."""

    def __init__(self, v):
        self.v = v

    def __call__(self, x, t):
        return self.v


def test_oracle_reconstructs_x1_exactly():
    g = torch.Generator().manual_seed(0)
    x0 = torch.randn(16, 2, generator=g)
    x1 = torch.randn(16, 2, generator=g)
    oracle = ConstVelocity(x1 - x0)
    for n_steps in (1, 4, 50):
        out = euler_sample(oracle, x0, n_steps)
        assert torch.allclose(out, x1, atol=1e-6), n_steps
