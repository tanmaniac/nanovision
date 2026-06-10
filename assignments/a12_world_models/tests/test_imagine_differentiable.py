"""The imagined return is differentiable w.r.t. the actor THROUGH the dynamics.

This is the test that guards the headline mechanism: dynamics backprop. imagine_dynamics rolls the
prior forward with the reparameterized actor in the loop and decodes the reward / value WITHOUT
no_grad, so the imagined lambda-return must carry a nonzero gradient back to the actor parameters
through the world model. If the rollout were wrapped in no_grad, or the reward were decoded from the
wrong state, or the action were not reparameterized, this gradient would be zero or absent.

This is a gradient-EXISTS check, not a convergence assert. CPU, no dm_control.
"""

import torch

from actor_critic import Critic, ContActor, imagine_dynamics
from config import WorldModelConfig
from world_model import WorldModel


def _cfg():
    cfg = WorldModelConfig()
    cfg.obs_size = 16
    cfg.embed_dim = 32
    cfg.h_dim = 32
    cfg.n_cat = 4
    cfg.n_cls = 4
    cfg.n_bins = 31
    cfg.horizon = 3
    return cfg


def test_imagined_return_has_grad_to_actor_through_dynamics():
    cfg = _cfg()
    torch.manual_seed(0)
    model = WorldModel(cfg)
    actor = ContActor(cfg)
    critic = Critic(cfg)

    B = 8
    h = torch.randn(B, cfg.h_dim)
    z = torch.randn(B, cfg.n_cat * cfg.n_cls)

    returns, ents, H_h, H_z = imagine_dynamics(model, actor, critic, h, z, cfg)
    assert returns.shape == (B, cfg.horizon)
    assert ents.shape == (B, cfg.horizon)
    assert H_h.shape == (B, cfg.horizon + 1, cfg.h_dim)
    assert H_z.shape == (B, cfg.horizon + 1, cfg.n_cat * cfg.n_cls)
    assert returns.requires_grad, "the imagined return must stay attached to the graph (no no_grad)"

    returns.mean().backward()
    grads = [p.grad for p in actor.parameters() if p.grad is not None]
    assert grads, "no gradient reached the actor: the dynamics-backprop path is broken"
    total = sum(g.abs().sum() for g in grads)
    assert total > 0, "the imagined return gradient w.r.t. the actor is zero through the dynamics"


def test_one_step_return_depends_on_action():
    # With horizon 1 the only learnable thing between the actor and the return is the action -> the
    # gradient must still reach the actor parameters through forward_h and the reward head.
    cfg = _cfg()
    cfg.horizon = 1
    torch.manual_seed(1)
    model = WorldModel(cfg)
    actor = ContActor(cfg)
    critic = Critic(cfg)
    h = torch.randn(4, cfg.h_dim)
    z = torch.randn(4, cfg.n_cat * cfg.n_cls)

    returns, _, _, _ = imagine_dynamics(model, actor, critic, h, z, cfg)
    returns.sum().backward()
    total = sum(p.grad.abs().sum() for p in actor.parameters() if p.grad is not None)
    assert total > 0, "a one-step imagined return must carry gradient to the actor through dynamics"
