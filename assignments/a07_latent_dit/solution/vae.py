"""A small KL-regularized VAE that compresses 16x16 images into a 4x4x4 latent.

The encoder maps an image to a per-latent-pixel Gaussian (mu, logvar); reparameterize
draws a sample z; the decoder reconstructs the image. A light KL penalty (beta small)
pulls the latent toward a unit Gaussian without erasing its spatial structure, so the
diffusion model later sees a smooth, roughly-standardized latent. This is the
continuous-latent route used by latent diffusion (LDM, SD3, FLUX); the contrast is the
discrete VQ codebook used for autoregressive token models.

Encoder/Decoder/KLVAE are provided. The student fills reparameterize, kl_divergence, and
vae_loss.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _groups(c: int) -> int:
    return 8 if c % 8 == 0 else 1


def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
    """Sample z = mu + sigma * eps with sigma = exp(0.5*logvar), eps ~ N(0, I).

    The reparameterization trick: routing the randomness through eps (drawn independently
    of the parameters) keeps the sample differentiable in mu and logvar.
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(mu)
    return mu + std * eps


def kl_divergence(mu: Tensor, logvar: Tensor) -> Tensor:
    """KL( N(mu, sigma^2) || N(0, I) ), summed over latent dims and averaged over the batch.

    Closed form per element: 0.5 * (exp(logvar) + mu^2 - 1 - logvar). Sum over the (C, H, W)
    latent dims, then mean over the batch.
    """
    kl = 0.5 * (torch.exp(logvar) + mu**2 - 1.0 - logvar)
    return kl.flatten(1).sum(dim=1).mean()


def vae_loss(x: Tensor, x_hat: Tensor, mu: Tensor, logvar: Tensor, beta: float):
    """VAE loss = recon + beta * KL.

    recon is the squared error summed over pixels per image then averaged over the batch
    (the same per-image-sum, batch-mean reduction the flow-matching loss uses). Returns the
    three scalars (total, recon, kl) so the test and viz can see the split.
    """
    recon = ((x_hat - x) ** 2).flatten(1).sum(dim=1).mean()
    kl = kl_divergence(mu, logvar)
    total = recon + beta * kl
    return total, recon, kl


class Encoder(nn.Module):
    """Two stride-2 conv blocks (Conv + GroupNorm + SiLU), then a 1x1 conv to 2C channels.

    1 -> 32 -> 64 channels with spatial 16 -> 8 -> 4, then a 1x1 conv to 2C = 8 channels.
    forward returns (mu, logvar), each (B, C, 4, 4), by chunking the 8 channels in half.
    """

    def __init__(self, channels: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 32, 3, stride=2, padding=1),   # 16 -> 8
            nn.GroupNorm(_groups(32), 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),         # 8 -> 4
            nn.GroupNorm(_groups(64), 64),
            nn.SiLU(),
            nn.Conv2d(64, 2 * latent_dim, 1),                  # -> 2C channels
        )

    def forward(self, x: Tensor):
        h = self.net(x)
        mu, logvar = h.chunk(2, dim=1)
        return mu, logvar


class Decoder(nn.Module):
    """1x1 conv C -> 64, then two nearest-upsample + conv blocks back to a 16x16 image.

    Spatial 4 -> 8 -> 16, channels 64 -> 32 -> 1, final Tanh (images live in [-1, 1]).
    """

    def __init__(self, channels: int, latent_dim: int):
        super().__init__()
        self.in_conv = nn.Conv2d(latent_dim, 64, 1)
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),       # 4 -> 8
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GroupNorm(_groups(32), 32),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),       # 8 -> 16
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GroupNorm(_groups(32), 32),
            nn.SiLU(),
            nn.Conv2d(32, channels, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.up(self.in_conv(z))


class KLVAE(nn.Module):
    """Encoder + Decoder with the reparameterized sampling in forward."""

    def __init__(self, cfg):
        super().__init__()
        self.encoder = Encoder(cfg.channels, cfg.latent_dim)
        self.decoder = Decoder(cfg.channels, cfg.latent_dim)

    def encode(self, x: Tensor):
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, x: Tensor):
        mu, logvar = self.encode(x)
        z = reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar
