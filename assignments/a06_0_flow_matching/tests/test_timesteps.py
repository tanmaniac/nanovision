"""Uniform and logit-normal timestep sampling."""

import torch

from timesteps import sample_timesteps


def test_uniform_range():
    t = sample_timesteps(20000, "uniform", generator=torch.Generator().manual_seed(0))
    assert (t > 0).all() and (t < 1).all()
    assert abs(t.mean().item() - 0.5) < 0.02


def test_logit_normal_concentrates_mid():
    g = torch.Generator().manual_seed(0)
    n = 40000
    t = sample_timesteps(n, "logit_normal", loc=0.0, scale=1.0, generator=g)
    assert (t > 0).all() and (t < 1).all()
    # median ~ sigmoid(loc) = 0.5 (sigmoid commutes with quantiles).
    assert abs(t.median().item() - 0.5) < 0.02
    # logit-normal(0,1) puts ~0.73 of its mass in [0.25, 0.75]; uniform puts exactly 0.5.
    mid = ((t > 0.25) & (t < 0.75)).float().mean().item()
    assert mid > 0.65


def test_unknown_dist_raises():
    try:
        sample_timesteps(4, "nope")
        assert False, "expected ValueError"
    except ValueError:
        pass
