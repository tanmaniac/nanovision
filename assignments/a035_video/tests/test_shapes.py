"""Shape tests: tubelet embedding, tube masking, and the video MAE forward."""

import torch

from backbone import VideoViTEncoder
from config import VideoSSLConfig
from nanovision.data.toy import video_batch
from nanovision.transformer import TubeletEmbedding
from video_mae import VideoMAE, tube_masking

cfg = VideoSSLConfig()


def _grid():
    t_prime = cfg.n_frames // cfg.tubelet_t
    s = (cfg.img_size // cfg.patch) ** 2
    return t_prime, s, t_prime * s


def _build_mae():
    return VideoMAE(
        img_size=cfg.img_size, patch=cfg.patch, tubelet_t=cfg.tubelet_t,
        n_frames=cfg.n_frames, in_chans=cfg.in_chans, enc_dim=cfg.enc_dim,
        enc_depth=cfg.enc_depth, enc_heads=cfg.enc_heads, dec_dim=cfg.dec_dim,
        dec_depth=cfg.dec_depth, dec_heads=cfg.dec_heads, mlp_ratio=cfg.mlp_ratio,
        mask_ratio=cfg.mask_ratio,
    )


def test_tubelet_embed_shape():
    t_prime, s, n = _grid()
    vid = torch.randn(2, cfg.in_chans, cfg.n_frames, cfg.img_size, cfg.img_size)
    te = TubeletEmbedding(cfg.in_chans, cfg.enc_dim, cfg.patch, cfg.tubelet_t)
    out = te(vid)
    assert out.shape == (2, n, cfg.enc_dim)


def test_tube_masking_shape():
    t_prime, s, n = _grid()
    x = torch.randn(2, n, cfg.enc_dim)
    x_kept, mask, ids_restore = tube_masking(x, t_prime, cfg.mask_ratio)
    n_keep = t_prime * round((1 - cfg.mask_ratio) * s)
    assert x_kept.shape == (2, n_keep, cfg.enc_dim)
    assert mask.shape == (2, n)
    assert ids_restore.shape == (2, n)
    assert int(mask[0].sum().item()) == n - n_keep


def test_video_mae_forward_shape():
    t_prime, s, n = _grid()
    vid = video_batch(batch=2, n_frames=cfg.n_frames, size=cfg.img_size,
                      channels=cfg.in_chans, n_blobs=cfg.n_blobs, seed=0)
    mae = _build_mae()
    loss, pred, mask = mae(vid)
    tubelet_dim = cfg.tubelet_t * cfg.patch * cfg.patch * cfg.in_chans
    assert pred.shape == (2, n, tubelet_dim)
    assert mask.shape == (2, n)
    assert loss.ndim == 0
