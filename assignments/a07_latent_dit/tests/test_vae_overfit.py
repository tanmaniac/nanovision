"""Overfit the VAE on 8 images: the recon MSE drops below a measured threshold.

This exercises the full encode -> reparameterize -> decode -> vae_loss loop. recon is the
per-image-sum, batch-mean squared error over 256 pixels, so the absolute number depends on
that sum reduction; the threshold below was set from the measured floor.
"""

import torch

from config import DiTConfig
from vae import KLVAE, vae_loss

from nanovision.data import toy


def test_vae_overfit():
    torch.manual_seed(0)
    cfg = DiTConfig()
    x, _ = toy.diffusion_image_batch(8, num_classes=cfg.num_classes,
                                     size=cfg.image_size, channels=cfg.channels, seed=0)
    vae = KLVAE(cfg)
    opt = torch.optim.Adam(vae.parameters(), lr=2e-3)
    first_recon = None
    for _ in range(800):
        x_hat, mu, logvar = vae(x)
        total, recon, kl = vae_loss(x, x_hat, mu, logvar, cfg.beta)
        opt.zero_grad()
        total.backward()
        opt.step()
        if first_recon is None:
            first_recon = recon.item()
    final_recon = recon.item()
    assert final_recon < 5.0, f"final recon {final_recon} (start {first_recon})"
