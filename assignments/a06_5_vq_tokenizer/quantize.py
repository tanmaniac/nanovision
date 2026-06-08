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
        """Quantize z_e (B, D, H, W) to the nearest codebook vectors.

        Steps:
        1. Move D to the last axis and flatten to (B*H*W, D) (use .contiguous() after the
           permute so the reshape is valid).
        2. Squared distance to each of the K codes
           ||z_e||^2 - 2 z_e . e + ||e||^2; indices = argmin over codes.
        3. z_q = codebook[indices], reshaped back to (B, D, H, W) (permute back, contiguous).
        4. codebook loss = ||sg[z_e] - z_q||^2 (mean); commitment = ||z_e - sg[z_q]||^2 (mean);
           vq_loss = codebook + beta * commitment. (sg = .detach())
        5. straight-through: z_q_ste = z_e + (z_q - z_e).detach().
        Return (z_q_ste (B, D, H, W), indices (B, H, W), vq_loss).
        """
        raise NotImplementedError("implement vector quantization with the straight-through estimator")


def codebook_perplexity(indices: Tensor, num_codes: int) -> Tensor:
    """exp(-sum_k p_k log p_k) over the code-usage distribution. 1 = total collapse,
    num_codes = uniform usage. The collapse diagnostic."""
    counts = torch.bincount(indices.reshape(-1), minlength=num_codes).float()
    p = counts / counts.sum()
    p = p[p > 0]
    return torch.exp(-(p * p.log()).sum())
