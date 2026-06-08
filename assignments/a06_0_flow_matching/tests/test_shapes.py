"""Shape contracts for the path, timesteps, loss, model, sampler, and coupling."""

import torch

from config import FlowConfig
from coupling import ot_coupling
from flow import cfm_loss
from model import VelocityMLP
from path import linear_path, linear_velocity
from sampling import euler_sample
from timesteps import sample_timesteps


def test_shapes():
    cfg = FlowConfig()
    B, D = 8, cfg.data_dim
    x0 = torch.randn(B, D)
    x1 = torch.randn(B, D)
    t = sample_timesteps(B)
    assert t.shape == (B,) and (t > 0).all() and (t < 1).all()
    assert linear_path(x0, x1, t).shape == (B, D)
    assert linear_velocity(x0, x1).shape == (B, D)

    model = VelocityMLP(cfg).eval()
    assert model(x0, t).shape == (B, D)
    assert cfm_loss(model, x0, x1, t).shape == ()
    assert euler_sample(model, x0, 10).shape == (B, D)
    assert euler_sample(model, x0, 10, return_traj=True).shape == (11, B, D)

    x1r, perm = ot_coupling(x0, x1)
    assert x1r.shape == (B, D) and perm.shape == (B,)
