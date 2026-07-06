"""Vector quantization with the straight-through estimator (van den Oord et al. 2017).

The codebook is K learned vectors of dimension D. Each encoder vector is snapped to its
nearest code; the argmin has zero gradient almost everywhere, so the straight-through
estimator copies the decoder's gradient straight back to the encoder. The codebook is
trained by the codebook loss (not through the estimator, which detaches the code), and the
encoder is pulled toward its chosen code by the commitment loss.
"""

import torch
from torch import Tensor, nn


class VectorQuantizer(nn.Module):
    def __init__(self, num_codes: int, code_dim: int, beta: float = 0.25):
        super().__init__()
        self.num_codes = num_codes
        self.beta = beta
        self.codebook = nn.Embedding(num_codes, code_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)

    def forward(self, z_e: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Quantize z_e (B, D, H, W) to its nearest codebook vectors.

        Returns (z_q_ste (B, D, H, W), indices (B, H, W), vq_loss). z_q_ste carries the
        straight-through gradient: the decoder sees the hard codebook lookup, but the gradient
        reaches the encoder as if quantization were the identity. vq_loss combines the codebook
        term (stop-gradient on the encoder) and the commitment term (stop-gradient on the code,
        weighted by beta).

        See the straight-through estimator and the losses sections of the README.
        """
        raise NotImplementedError("implement vector quantization with the straight-through estimator")


def codebook_perplexity(indices: Tensor, num_codes: int) -> Tensor:
    """exp(-sum_k p_k log p_k) over the code-usage distribution. 1 = total collapse,
    num_codes = uniform usage. The collapse diagnostic."""
    counts = torch.bincount(indices.reshape(-1), minlength=num_codes).float()
    p = counts / counts.sum()
    p = p[p > 0]
    return torch.exp(-(p * p.log()).sum())
