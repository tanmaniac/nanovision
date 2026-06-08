"""Vector quantization: nearest-neighbor lookup and the codebook/commitment losses."""

import torch

from config import VQConfig

from nanovision.quantize import VectorQuantizer


def test_nearest_neighbor_and_losses():
    cfg = VQConfig()
    torch.manual_seed(0)
    q = VectorQuantizer(cfg.num_codes, cfg.code_dim, cfg.beta)
    z_e = torch.randn(2, cfg.code_dim, 3, 3)
    z_q_ste, idx, vq = q(z_e)

    # Independent brute-force nearest code.
    flat = z_e.permute(0, 2, 3, 1).reshape(-1, cfg.code_dim)
    ref_idx = torch.cdist(flat, q.codebook.weight).argmin(dim=1).reshape(2, 3, 3)
    assert (idx == ref_idx).all()

    # The forward value is the hard codebook lookup (the detach does not change it).
    z_q_hard = q.codebook(idx).permute(0, 3, 1, 2)
    assert torch.allclose(z_q_ste, z_q_hard, atol=1e-6)

    # vq_loss = ||sg[z_e] - z_q||^2 + beta ||z_e - sg[z_q]||^2.
    ref_codebook = (z_e.detach() - z_q_hard).pow(2).mean()
    ref_commit = (z_e - z_q_hard.detach()).pow(2).mean()
    assert torch.allclose(vq, ref_codebook + cfg.beta * ref_commit, atol=1e-6)
