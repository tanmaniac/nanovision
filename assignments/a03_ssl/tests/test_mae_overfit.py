"""MAE overfits one batch (masked-patch MSE falls below a tolerance). Runs late.

End-to-end signal: with random_masking, append_mask_tokens, and mae_loss all
correct, the full encode -> decode -> masked-MSE pipeline memorizes a single fixed
batch of synthetic images. The mask pattern is held fixed across steps (the loop
re-seeds before each forward) so this is a clean memorization signal: the masked-
patch MSE drops well below 0.05 (see solution_notes in ASSIGNMENT).
"""

import torch

from config import SSLConfig
from mae import MAE
from nanovision.determinism import set_seed


def test_mae_overfits_one_batch():
    set_seed(0)
    cfg = SSLConfig()
    img = torch.randn(cfg.overfit_batch, 3, 32, 32)

    model = MAE(img_size=cfg.img_size, patch=cfg.patch, in_chans=cfg.in_chans,
                enc_dim=cfg.enc_dim, enc_depth=cfg.enc_depth, enc_heads=cfg.enc_heads,
                dec_dim=cfg.dec_dim, dec_depth=cfg.dec_depth, dec_heads=cfg.dec_heads,
                mask_ratio=cfg.mask_ratio)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.mae_lr)

    losses = []
    for _ in range(cfg.mae_steps):
        torch.manual_seed(0)  # hold the mask pattern fixed across steps
        loss, _, _ = model(img)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < 0.05, f"masked-patch MSE should drop; final {losses[-1]:.4f}"
    assert losses[-1] < 0.1 * losses[0]
