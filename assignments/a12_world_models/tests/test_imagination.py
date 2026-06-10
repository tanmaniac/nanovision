"""Imagination is prior-only (no decoder, no env) and is action-conditional.

Two robust checks on a tiny world model trained briefly on synthetic frames (no dm_control). First,
the prior rollout produces finite states and never touches the decoder (asserted with a spy on the
decoder). Second, the prior is action-conditional: stepping from the same start under two different
continuous forces gives different next states, so the GRU genuinely reads the action.

The quantitative accuracy of imagination and the real-env policy transfer are NOT asserted here;
they are measured in viz.py (the transfer number is a measured statistic, not a pytest threshold).
"""

import numpy as np
import torch

from config import WorldModelConfig
from world_model import WorldModel


def _tiny_cfg():
    cfg = WorldModelConfig()
    cfg.obs_size = 16
    cfg.embed_dim = 64
    cfg.h_dim = 64
    cfg.n_cat = 8
    cfg.n_cls = 8
    cfg.n_bins = 63
    return cfg


def _batch(cfg, B=2, T=6):
    obs = np.full((B, T, cfg.obs_ch, cfg.obs_size, cfg.obs_size), 0.1, np.float32)
    for b in range(B):
        for t in range(T):
            obs[b, t, :, (t + b) % cfg.obs_size, :] = 0.9
    rng = np.random.default_rng(0)
    return {
        "obs": torch.as_tensor(obs),
        "actions": torch.as_tensor(rng.uniform(-1, 1, (B, T, cfg.action_dim)).astype(np.float32)),
        "rewards": torch.as_tensor(rng.random((B, T)).astype(np.float32)),
        "conts": torch.as_tensor(np.ones((B, T), np.float32)),
    }


def _trained_model(cfg, steps):
    torch.manual_seed(0)
    model = WorldModel(cfg)
    batch = _batch(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    for _ in range(steps):
        total, _ = model.loss(batch)
        opt.zero_grad()
        total.backward()
        opt.step()
    model.eval()
    return model


def _start_state(model, cfg):
    obs0 = _batch(cfg)["obs"][:, 0]
    with torch.no_grad():
        return model.encode_start(obs0)


def test_prior_rollout_is_finite_and_decoder_free():
    cfg = _tiny_cfg()
    model = _trained_model(cfg, steps=5)

    called = {"n": 0}
    orig = model.decoder.forward

    def spy(*a, **k):
        called["n"] += 1
        return orig(*a, **k)

    model.decoder.forward = spy
    h, z = _start_state(model, cfg)
    with torch.no_grad():
        for _ in range(cfg.horizon):
            a = torch.zeros(h.shape[0], cfg.action_dim)
            h = model.rssm.forward_h(h, z, a)
            _, z, _ = model.rssm.prior(h, greedy=True)
    model.decoder.forward = orig

    assert torch.isfinite(h).all() and torch.isfinite(z).all()
    assert called["n"] == 0, "the prior rollout must not call the decoder"


def test_imagination_is_action_conditional():
    cfg = _tiny_cfg()
    model = _trained_model(cfg, steps=200)
    h, z = _start_state(model, cfg)

    with torch.no_grad():
        def step(force):
            a = torch.full((h.shape[0], cfg.action_dim), force, dtype=torch.float32)
            h2 = model.rssm.forward_h(h, z, a)
            _, _, probs = model.rssm.prior(h2, greedy=True)
            return h2, probs

        h_a, p_a = step(-1.0)
        h_b, p_b = step(1.0)

    # Different continuous forces move the deterministic state differently and yield different prior
    # logits, so the GRU is reading the action.
    assert not torch.allclose(h_a, h_b, atol=1e-4), "forward_h must depend on the action"
    assert not torch.allclose(p_a, p_b, atol=1e-4), "the prior must depend on the action through h"
