"""Shape and L2-norm tests for the dual encoder, the losses, and zero-shot."""

import torch
import torch.nn.functional as F

from config import CLIPConfig
from inference import zero_shot_classify
from losses import clip_loss, siglip_loss
from model import CLIPModel
from nanovision.data.toy import image_text_batch

cfg = CLIPConfig()


def _batch(n=None):
    n = n or cfg.overfit_batch
    return image_text_batch(batch=n, size=cfg.img_size, channels=cfg.in_chans,
                            n_classes=cfg.n_classes, n_attrs=cfg.n_attrs,
                            vocab_size=cfg.vocab_size, max_len=cfg.max_len, seed=0)


def test_encode_shapes_and_norm():
    imgs, toks = _batch()
    m = CLIPModel(cfg)
    fi, ft = m(imgs, toks)
    assert fi.shape == (cfg.overfit_batch, cfg.embed_dim)
    assert ft.shape == (cfg.overfit_batch, cfg.embed_dim)
    assert torch.allclose(fi.norm(dim=-1), torch.ones(cfg.overfit_batch), atol=1e-5)
    assert torch.allclose(ft.norm(dim=-1), torch.ones(cfg.overfit_batch), atol=1e-5)


def test_losses_scalar():
    imgs, toks = _batch()
    m = CLIPModel(cfg)
    fi, ft = m(imgs, toks)
    lc = clip_loss(fi, ft, m.logit_scale_clamped())
    ls = siglip_loss(fi, ft, m.logit_scale_clamped(), m.bias)
    assert lc.ndim == 0 and ls.ndim == 0
    assert lc.item() > 0 and ls.item() > 0


def test_zero_shot_shapes():
    imgs, toks = _batch()
    m = CLIPModel(cfg)
    fi = m.encode_image(imgs)
    k = cfg.n_classes
    ft = F.normalize(torch.randn(k, cfg.embed_dim), dim=-1)
    logits, preds = zero_shot_classify(fi, ft)
    assert logits.shape == (cfg.overfit_batch, k)
    assert preds.shape == (cfg.overfit_batch,)
