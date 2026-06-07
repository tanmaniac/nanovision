"""Tasks 1-6 shapes. Run first.

Covers MAE random masking and the full MAE forward, and the DINO student/teacher
head outputs and the center buffer.
"""

import torch

from backbone import DINOModel
from config import SSLConfig
from dino import build_student_teacher
from mae import MAE, random_masking


def test_random_masking_shapes():
    B, N, D = 2, 64, 32
    x = torch.randn(B, N, D)
    x_kept, mask, ids_restore = random_masking(x, mask_ratio=0.75)
    n_keep = round(0.25 * N)             # 16
    assert x_kept.shape == (B, n_keep, D)
    assert mask.shape == (B, N)
    assert ids_restore.shape == (B, N)
    # 48 masked patches per row (1s), 16 kept (0s).
    assert torch.allclose(mask.sum(dim=1), torch.full((B,), float(N - n_keep)))


def test_mae_forward_shapes():
    cfg = SSLConfig()
    model = MAE(img_size=cfg.img_size, patch=cfg.patch, in_chans=cfg.in_chans,
                enc_dim=cfg.enc_dim, enc_depth=cfg.enc_depth, enc_heads=cfg.enc_heads,
                dec_dim=cfg.dec_dim, dec_depth=cfg.dec_depth, dec_heads=cfg.dec_heads,
                mask_ratio=cfg.mask_ratio)
    img = torch.randn(2, 3, 32, 32)
    loss, pred, mask = model(img)
    patch_dim = cfg.patch * cfg.patch * cfg.in_chans     # 48
    assert pred.shape == (2, 64, patch_dim)
    assert mask.shape == (2, 64)
    assert loss.ndim == 0


def test_dino_head_shapes():
    cfg = SSLConfig()
    student, teacher = build_student_teacher(cfg)
    img = torch.randn(3, 3, 32, 32)
    s_out = student(img)
    t_out = teacher(img)
    assert s_out.shape == (3, cfg.out_dim)
    assert t_out.shape == (3, cfg.out_dim)


def test_center_buffer_shape():
    cfg = SSLConfig()
    center = torch.zeros(1, cfg.out_dim)
    assert center.shape == (1, cfg.out_dim)
