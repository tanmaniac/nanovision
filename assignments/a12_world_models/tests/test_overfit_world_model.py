"""Overfit one short synthetic sequence: reconstruction drops, KL settles near the free-bits floor.

Bounded CPU training on a tiny config (a few short synthetic-frame sequences), so the test runs in
seconds and needs no dm_control. The world model must drive its reconstruction MSE down and hold the
two summed KL terms in a loose band around the 1-nat free-bits floor (not collapsed to 0, not
diverging). The thresholds are loose bands, not tuned values.
"""

import numpy as np
import torch

from config import WorldModelConfig
from world_model import WorldModel


def _tiny_cfg():
    cfg = WorldModelConfig()
    # Shrink so CPU overfit is fast; the holes do not depend on these sizes.
    cfg.obs_size = 16
    cfg.embed_dim = 64
    cfg.h_dim = 64
    cfg.n_cat = 8
    cfg.n_cls = 8
    cfg.n_bins = 63
    return cfg


def _batch(cfg, B=2, T=6):
    # A simple, learnable target sequence: a bright row that moves over time, so frames differ.
    obs = np.full((B, T, cfg.obs_ch, cfg.obs_size, cfg.obs_size), 0.1, np.float32)
    for b in range(B):
        for t in range(T):
            obs[b, t, :, (t + b) % cfg.obs_size, :] = 0.9
    rng = np.random.default_rng(0)
    actions = rng.uniform(-1, 1, (B, T, cfg.action_dim)).astype(np.float32)
    rewards = rng.random((B, T)).astype(np.float32)
    conts = np.ones((B, T), np.float32)
    return {
        "obs": torch.as_tensor(obs),
        "actions": torch.as_tensor(actions),
        "rewards": torch.as_tensor(rewards),
        "conts": torch.as_tensor(conts),
    }


def test_overfit_one_sequence():
    cfg = _tiny_cfg()
    torch.manual_seed(0)
    model = WorldModel(cfg)
    batch = _batch(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)

    hist = {"recon": [], "kl_dyn": [], "kl_rep": []}
    for _ in range(400):
        total, parts = model.loss(batch)
        opt.zero_grad()
        total.backward()
        opt.step()
        hist["recon"].append(float(parts["recon"]))
        hist["kl_dyn"].append(float(parts["kl_dyn"]))
        hist["kl_rep"].append(float(parts["kl_rep"]))

    recon = hist["recon"][-1]
    assert recon < 0.05, f"reconstruction MSE {recon:.4f} should overfit below 0.05"

    kl_dyn = sum(hist["kl_dyn"][-20:]) / 20
    kl_rep = sum(hist["kl_rep"][-20:]) / 20
    assert 0.8 <= kl_dyn <= 6.0, f"kl_dyn settled at {kl_dyn:.3f}, outside the 0.8-6.0 band"
    assert 0.8 <= kl_rep <= 6.0, f"kl_rep settled at {kl_rep:.3f}, outside the 0.8-6.0 band"
