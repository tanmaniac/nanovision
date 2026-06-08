"""Shape contracts for the VAE encoder/decoder, reparameterize, and the DiT."""

import torch

from config import DiTConfig
from dit import DiT
from vae import Decoder, Encoder, KLVAE, reparameterize


def test_shapes():
    cfg = DiTConfig()
    B = 8
    C, hw = cfg.latent_dim, cfg.image_size // cfg.f
    x = torch.randn(B, cfg.channels, cfg.image_size, cfg.image_size)

    enc = Encoder(cfg.channels, cfg.latent_dim)
    mu, logvar = enc(x)
    assert mu.shape == (B, C, hw, hw)
    assert logvar.shape == (B, C, hw, hw)

    z = reparameterize(mu, logvar)
    assert z.shape == (B, C, hw, hw)

    dec = Decoder(cfg.channels, cfg.latent_dim)
    x_hat = dec(z)
    assert x_hat.shape == (B, cfg.channels, cfg.image_size, cfg.image_size)

    vae = KLVAE(cfg)
    x_hat2, mu2, logvar2 = vae(x)
    assert x_hat2.shape == (B, cfg.channels, cfg.image_size, cfg.image_size)
    assert mu2.shape == (B, C, hw, hw) and logvar2.shape == (B, C, hw, hw)

    dit = DiT(cfg).eval()
    t = torch.rand(B)
    y = torch.randint(0, cfg.num_classes, (B,))
    v = dit(z, t, y)
    assert v.shape == (B, C, hw, hw)
