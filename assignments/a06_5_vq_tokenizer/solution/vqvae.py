"""The VQ-VAE: a convolutional encoder, the shared vector quantizer, and a decoder.

The encoder downsamples a 16x16 image to a 4x4 grid of code vectors; the quantizer snaps
each to a codebook entry; the decoder reconstructs the image. Trained end to end with the
reconstruction loss plus the quantizer's codebook and commitment losses; the straight-through
estimator carries the reconstruction gradient through the quantization.
"""

import torch
from torch import Tensor, nn

from nanovision.primitives import ConvNeXtBlock
from nanovision.quantize import VectorQuantizer


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        c = cfg.hidden
        self.net = nn.Sequential(
            nn.Conv2d(cfg.channels, c, 4, stride=2, padding=1),   # 16 -> 8
            nn.SiLU(),
            nn.Conv2d(c, c, 4, stride=2, padding=1),              # 8 -> 4
            ConvNeXtBlock(c),
            nn.Conv2d(c, cfg.code_dim, 1),                        # (B, D, 4, 4)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        c = cfg.hidden
        self.net = nn.Sequential(
            nn.Conv2d(cfg.code_dim, c, 1),
            ConvNeXtBlock(c),
            nn.ConvTranspose2d(c, c, 4, stride=2, padding=1),     # 4 -> 8
            nn.SiLU(),
            nn.ConvTranspose2d(c, cfg.channels, 4, stride=2, padding=1),  # 8 -> 16
            nn.Tanh(),                                            # images are in [-1, 1]
        )

    def forward(self, z_q: Tensor) -> Tensor:
        return self.net(z_q)


class VQVAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = Encoder(cfg)
        self.quantizer = VectorQuantizer(cfg.num_codes, cfg.code_dim, cfg.beta)
        self.decoder = Decoder(cfg)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        z_e = self.encoder(x)
        z_q, indices, vq_loss = self.quantizer(z_e)
        x_hat = self.decoder(z_q)
        return x_hat, indices, vq_loss

    def decode_indices(self, indices: Tensor) -> Tensor:
        """Decode a token grid (B, H, W) of code indices to an image (for prior samples)."""
        z_q = self.quantizer.codebook(indices).permute(0, 3, 1, 2).contiguous()
        return self.decoder(z_q)


def vq_vae_loss(x: Tensor, x_hat: Tensor, vq_loss: Tensor) -> Tensor:
    """The total VQ-VAE objective: reconstruction MSE plus the quantizer's vq loss."""
    recon = (x_hat - x).pow(2).mean()
    return recon + vq_loss
