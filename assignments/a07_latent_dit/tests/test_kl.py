"""kl_divergence matches the closed form for fixed (mu, logvar)."""

import math

import torch

from vae import kl_divergence


def test_kl_zero_at_standard_normal():
    # mu = 0, logvar = 0 -> N(0, I), KL = 0.
    mu = torch.zeros(4, 4, 4, 4)
    logvar = torch.zeros(4, 4, 4, 4)
    assert torch.allclose(kl_divergence(mu, logvar), torch.zeros(()), atol=1e-7)


def test_kl_known_value():
    # Constant mu, logvar over all latent dims; per-element KL is the same scalar, so the
    # per-image sum is D * that scalar and the batch mean leaves it unchanged.
    B, C, H, W = 5, 4, 4, 4
    D = C * H * W
    mu_val, logvar_val = 0.5, math.log(2.0)   # sigma^2 = 2
    mu = torch.full((B, C, H, W), mu_val)
    logvar = torch.full((B, C, H, W), logvar_val)
    per_elem = 0.5 * (math.exp(logvar_val) + mu_val**2 - 1.0 - logvar_val)
    expected = D * per_elem
    out = kl_divergence(mu, logvar)
    assert torch.allclose(out, torch.tensor(expected), atol=1e-5), (out.item(), expected)
