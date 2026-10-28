"""Shared tiny config and synthetic batch for the a12 tests. Not a test module."""

import torch

from config import small_config


def tiny_cfg(**over):
    """small_config with 16x16 frames: same structure as the full agent, fast on CPU."""
    base = dict(obs_size=16, seed=0)
    base.update(over)
    return small_config(**base)


def synthetic_batch(cfg, B=2, T=8, seed=0):
    """A learnable sequence: a bright row moving down the frame, random actions and rewards."""
    g = torch.Generator().manual_seed(seed)
    s = cfg.obs_size
    image = torch.full((B, T, 3, s, s), 0.1)
    for b in range(B):
        for t in range(T):
            image[b, t, :, (2 * t + 3 * b) % s, :] = 0.9
    data = {
        "image": image,
        "action": torch.rand(B, T, cfg.action_dim, generator=g) * 2 - 1,
        "reward": torch.rand(B, T, 1, generator=g),
        "is_first": torch.zeros(B, T, dtype=torch.bool),
        "is_last": torch.zeros(B, T, 1),
        "is_terminal": torch.zeros(B, T, 1),
    }
    data["is_first"][:, 0] = True
    data["action"][:, 0] = 0.0
    return data
