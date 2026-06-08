"""Overfit one batch: the VQ-VAE reconstructs it and uses several codes (no collapse)."""

import torch

from config import VQConfig
from vqvae import VQVAE, vq_vae_loss

from nanovision.data import toy
from nanovision.quantize import codebook_perplexity


def test_recon_overfit():
    torch.manual_seed(0)
    cfg = VQConfig()
    x, _ = toy.diffusion_image_batch(8, num_classes=3, size=cfg.img_size,
                                     channels=cfg.channels, seed=0)
    vae = VQVAE(cfg)
    opt = torch.optim.Adam(vae.parameters(), lr=3e-3)
    for _ in range(1500):
        opt.zero_grad()
        x_hat, idx, vq = vae(x)
        vq_vae_loss(x, x_hat, vq).backward()
        opt.step()

    with torch.no_grad():
        x_hat, idx, _ = vae(x)
        recon = (x_hat - x).pow(2).mean().item()
        ppl = codebook_perplexity(idx, cfg.num_codes).item()
    assert recon < 0.05, f"recon MSE {recon}"
    assert ppl > 3.0, f"codebook perplexity {ppl} (collapsed)"
