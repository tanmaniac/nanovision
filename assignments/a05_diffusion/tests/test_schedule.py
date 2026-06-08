"""The two noising schedules: endpoints, monotonicity, and a cosine reference."""

import math

import torch

from schedule import cosine_alpha_bar, linear_alpha_bar


def test_endpoints_and_monotonicity():
    T = 50
    for betas, abar in (cosine_alpha_bar(T), linear_alpha_bar(T, 1e-4, 2e-2)):
        assert abar.shape == (T,) and betas.shape == (T,)
        assert abar[0] > 0.99                      # least noised, close to clean
        assert torch.all(abar[1:] <= abar[:-1] + 1e-6)   # non-increasing
        assert torch.all(betas > 0) and torch.all(betas <= 1.0)

    # alpha_bar -> 0 by the end is the cosine schedule's job at small T; the linear
    # schedule's constants are calibrated for T=1000 and do NOT reach 0 at T=50.
    _, abar_cos = cosine_alpha_bar(T)
    assert abar_cos[-1] < 0.1


def test_cosine_reference():
    T = 10
    s = 0.008
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos((steps / T + s) / (1.0 + s) * math.pi / 2.0) ** 2
    ref = (f / f[0])[1:].float()                   # abar at t=1..T
    _, abar = cosine_alpha_bar(T, s)
    assert torch.allclose(abar, ref, atol=1e-5)
