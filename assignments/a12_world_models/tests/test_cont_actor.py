"""The continuous Tanh-Normal actor is reparameterized, bounded, and respects the log-std floor.

ContActor.sample must (a) return an action in (-1, 1) (it is a tanh output), (b) carry a nonzero
gradient from the action back to the actor parameters - the reparameterization that makes dynamics
backprop possible, and (c) clamp log_std to the floor log(0.1) so the policy keeps exploration noise
and cannot collapse to a delta. These are gradient-exists / bound checks, not training-convergence
asserts. CPU, no dm_control.
"""

import math

import torch

from actor_critic import LOGSTD_MAX, LOGSTD_MIN, ContActor
from config import WorldModelConfig


def _cfg():
    cfg = WorldModelConfig()
    cfg.h_dim = 32
    cfg.n_cat = 4
    cfg.n_cls = 4
    return cfg


def test_action_is_in_open_interval():
    cfg = _cfg()
    torch.manual_seed(0)
    actor = ContActor(cfg)
    h = torch.randn(16, cfg.h_dim)
    z = torch.randn(16, cfg.n_cat * cfg.n_cls)
    a, ent = actor.sample(h, z)
    assert a.shape == (16, cfg.action_dim)
    assert torch.all(a > -1.0) and torch.all(a < 1.0), "tanh output must lie in (-1, 1)"
    assert torch.isfinite(ent).all()


def test_sample_is_reparameterized_grad_to_params():
    # The sampled action must be differentiable w.r.t. the actor parameters (reparameterization).
    cfg = _cfg()
    torch.manual_seed(0)
    actor = ContActor(cfg)
    h = torch.randn(8, cfg.h_dim)
    z = torch.randn(8, cfg.n_cat * cfg.n_cls)
    a, _ = actor.sample(h, z)              # NOT greedy: goes through mean + std * eps
    a.sum().backward()
    grads = [p.grad for p in actor.parameters() if p.grad is not None]
    assert grads, "no gradient reached the actor parameters"
    total = sum(g.abs().sum() for g in grads)
    assert total > 0, "action gradient w.r.t. actor params is zero (not reparameterized)"


def test_log_std_respects_floor():
    # Drive the raw log_std head far negative; the clamp must hold it at the floor log(0.1).
    cfg = _cfg()
    torch.manual_seed(0)
    actor = ContActor(cfg)
    with torch.no_grad():
        # Push the final linear's bias for the log_std channel very negative.
        actor.net[-1].weight.zero_()
        actor.net[-1].bias[:] = torch.tensor([0.0, -100.0])
    h = torch.randn(4, cfg.h_dim)
    z = torch.randn(4, cfg.n_cat * cfg.n_cls)
    _, log_std = actor.dist(h, z)
    assert torch.allclose(log_std, torch.full_like(log_std, LOGSTD_MIN), atol=1e-5), log_std
    assert math.isclose(LOGSTD_MIN, math.log(0.1), rel_tol=1e-9)
    assert LOGSTD_MIN < LOGSTD_MAX
