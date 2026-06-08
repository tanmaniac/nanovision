"""Overfit the DiT on a fixed, deterministic flow-matching regression target.

Decoupled from VAE convergence (the honest choice): the target latents are fixed random
tensors. x0, t, and y are all fixed and seeded, so cfm_loss(model, x0, x1, y, t) is a
single regression and the relative-drop threshold is well-posed. Do NOT resample x0 each
step - that injects irreducible variance and breaks the relative threshold.
"""

import torch

from config import DiTConfig
from dit import DiT
from flow import cfm_loss


def test_dit_overfit():
    torch.manual_seed(0)
    cfg = DiTConfig()
    g = torch.Generator().manual_seed(0)
    B, C, hw = 8, cfg.latent_dim, cfg.image_size // cfg.f

    x1 = torch.randn(B, C, hw, hw, generator=g)          # fixed target latents
    x0 = torch.randn(B, C, hw, hw, generator=g)          # fixed noise
    t = 0.05 + 0.9 * torch.rand(B, generator=g)          # fixed per-row time
    y = torch.randint(0, cfg.num_classes, (B,), generator=g)

    model = DiT(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    first = None
    for _ in range(2000):
        opt.zero_grad()
        loss = cfm_loss(model, x0, x1, y, t)
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    final = loss.item()
    assert final < 0.05, f"final {final} (start {first})"
    assert final < 0.01 * first, f"final {final} not < 1% of first {first}"
