"""Shape contracts for the projector, the resampler, and the full VLM forward."""

import torch

from config import VLMConfig
from projector import MLPProjector
from resampler import PerceiverResampler
from vlm import VLM

from nanovision.data import toy


def test_projector_shape():
    cfg = VLMConfig()
    proj = MLPProjector(cfg.vit_dim, cfg.dim_l)
    feats = torch.randn(8, cfg.n_patches, cfg.vit_dim)
    assert proj(feats).shape == (8, cfg.n_patches, cfg.dim_l)


def test_resampler_shape():
    cfg = VLMConfig()
    res = PerceiverResampler(cfg.vit_dim, cfg.dim_l, cfg.n_queries, cfg.lm_heads)
    feats = torch.randn(8, cfg.n_patches, cfg.vit_dim)
    assert res(feats).shape == (8, cfg.n_queries, cfg.dim_l)


def test_vlm_forward_shape():
    cfg = VLMConfig()
    model = VLM(cfg).eval()
    img, tok = toy.image_text_batch(
        batch=8, size=cfg.img_size, channels=cfg.in_chans, n_classes=cfg.n_classes,
        n_attrs=cfg.n_attrs, vocab_size=cfg.vocab_size, max_len=cfg.max_len, seed=0,
    )
    logits, n_visual = model(img, tok)
    assert n_visual == cfg.n_patches == 16
    assert logits.shape == (8, cfg.n_patches + cfg.max_len, cfg.vocab_size)
    assert logits.shape[1] == 24
