"""Shape contracts for the encoder, quantizer, decoder, and prior."""

import torch

from config import VQConfig
from prior import TokenPrior
from vqvae import VQVAE

from nanovision.quantize import VectorQuantizer


def test_vqvae_shapes():
    cfg = VQConfig()
    vae = VQVAE(cfg).eval()
    x = torch.randn(2, cfg.channels, cfg.img_size, cfg.img_size)
    x_hat, indices, vq_loss = vae(x)
    assert x_hat.shape == x.shape
    assert indices.shape == (2, cfg.grid, cfg.grid)
    assert indices.min() >= 0 and indices.max() < cfg.num_codes
    assert vq_loss.shape == ()


def test_quantizer_shapes():
    cfg = VQConfig()
    q = VectorQuantizer(cfg.num_codes, cfg.code_dim, cfg.beta)
    z_e = torch.randn(2, cfg.code_dim, cfg.grid, cfg.grid)
    z_q, idx, vq = q(z_e)
    assert z_q.shape == z_e.shape
    assert idx.shape == (2, cfg.grid, cfg.grid)
    assert vq.shape == ()


def test_prior_shapes():
    cfg = VQConfig()
    prior = TokenPrior(cfg).eval()
    L = cfg.grid * cfg.grid
    tokens = torch.randint(0, cfg.num_codes + 1, (2, L))
    logits = prior(tokens)
    assert logits.shape == (2, L, cfg.num_codes)
