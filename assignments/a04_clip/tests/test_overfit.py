"""Overfit one batch: the dual encoder aligns N image-caption pairs.

With siglip_loss (and with clip_loss), training the tiny dual encoder on one fixed batch
drives the similarity-matrix diagonal to be the maximum of its row and column for every
pair: matched image and caption are nearest neighbors in the shared space.
"""

import torch

from config import CLIPConfig
from model import CLIPModel
from nanovision.data.toy import image_text_batch
from nanovision.determinism import set_seed
from losses import clip_loss, siglip_loss


def _aligned(model, imgs, toks):
    with torch.no_grad():
        fi, ft = model(imgs, toks)
        s = fi @ ft.T
        n = s.shape[0]
        rows = (s.argmax(dim=1) == torch.arange(n)).all()
        cols = (s.argmax(dim=0) == torch.arange(n)).all()
    return bool(rows and cols)


def _train(loss_name):
    set_seed(0)
    cfg = CLIPConfig()
    imgs, toks = image_text_batch(batch=cfg.overfit_batch, size=cfg.img_size,
                                  channels=cfg.in_chans, n_classes=cfg.n_classes,
                                  n_attrs=cfg.n_attrs, vocab_size=cfg.vocab_size,
                                  max_len=cfg.max_len, seed=0)
    model = CLIPModel(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    for _ in range(cfg.steps):
        fi, ft = model(imgs, toks)
        if loss_name == "clip":
            loss = clip_loss(fi, ft, model.logit_scale_clamped())
        else:
            loss = siglip_loss(fi, ft, model.logit_scale_clamped(), model.bias)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return _aligned(model, imgs, toks)


def test_siglip_overfits_to_alignment():
    assert _train("siglip")


def test_clip_overfits_to_alignment():
    assert _train("clip")
