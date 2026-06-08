"""Video MAE overfits one batch (masked-tubelet MSE falls below a tolerance).

End-to-end signal: with TubeletEmbedding, tube_masking, and video_mae_loss all
correct, the encode -> decode -> masked-MSE pipeline memorizes a single fixed batch of
synthetic clips. The tube mask is held fixed across steps (the loop re-seeds before
each forward) so this is a clean memorization signal: the masked-tubelet MSE drops
well below 0.05. The toy clips have independently moving blobs, so the masked tubes
are not recoverable by copying one visible column; this checks the mechanism plumbing,
not that the representation is good (see the README).
"""

import torch

from config import VideoSSLConfig
from nanovision.data.toy import video_batch
from nanovision.determinism import set_seed
from video_mae import VideoMAE


def test_video_mae_overfits_one_batch():
    set_seed(0)
    cfg = VideoSSLConfig()
    vid = video_batch(batch=cfg.overfit_batch, n_frames=cfg.n_frames, size=cfg.img_size,
                      channels=cfg.in_chans, n_blobs=cfg.n_blobs, seed=0)

    model = VideoMAE(
        img_size=cfg.img_size, patch=cfg.patch, tubelet_t=cfg.tubelet_t,
        n_frames=cfg.n_frames, in_chans=cfg.in_chans, enc_dim=cfg.enc_dim,
        enc_depth=cfg.enc_depth, enc_heads=cfg.enc_heads, dec_dim=cfg.dec_dim,
        dec_depth=cfg.dec_depth, dec_heads=cfg.dec_heads, mlp_ratio=cfg.mlp_ratio,
        mask_ratio=cfg.mask_ratio,
    )
    opt = torch.optim.Adam(model.parameters(), lr=cfg.mae_lr)

    losses = []
    for _ in range(cfg.mae_steps):
        torch.manual_seed(1234)  # hold the tube mask fixed across steps
        loss, _, _ = model(vid)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < 0.05, f"masked-tubelet MSE should drop; final {losses[-1]:.4f}"
    assert losses[-1] < 0.1 * losses[0]
