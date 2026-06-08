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
        """z_e is (B, D, H, W). Returns (z_q_ste (B, D, H, W), indices (B, H, W), vq_loss)."""
        B, D, H, W = z_e.shape
        flat = z_e.permute(0, 2, 3, 1).contiguous().view(-1, D)        # (B*H*W, D)

        E = self.codebook.weight                                        # (K, D)
        dist = (flat.pow(2).sum(1, keepdim=True)                        # ||z_e||^2
                - 2 * flat @ E.t()                                      # -2 z_e . e
                + E.pow(2).sum(1))                                      # ||e||^2
        idx = dist.argmin(dim=1)                                        # (B*H*W,)
        z_q = self.codebook(idx).view(B, H, W, D).permute(0, 3, 1, 2).contiguous()

        codebook = (z_e.detach() - z_q).pow(2).mean()                   # trains the codes
        commitment = (z_e - z_q.detach()).pow(2).mean()                 # trains the encoder
        vq_loss = codebook + self.beta * commitment

        z_q_ste = z_e + (z_q - z_e).detach()                           # straight-through
        return z_q_ste, idx.view(B, H, W), vq_loss


def codebook_perplexity(indices: Tensor, num_codes: int) -> Tensor:
    """exp(-sum_k p_k log p_k) over the code-usage distribution. 1 = total collapse,
    num_codes = uniform usage. The collapse diagnostic."""
    counts = torch.bincount(indices.reshape(-1), minlength=num_codes).float()
    p = counts / counts.sum()
    p = p[p > 0]
    return torch.exp(-(p * p.log()).sum())
