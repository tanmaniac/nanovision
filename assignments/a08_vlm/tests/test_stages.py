"""The two-stage freeze curriculum toggles requires_grad correctly.

Stage 1: projector/connector, token embedding, and output head train; the ViT and decoder
are frozen. Stage 2: the decoder also unfreezes; the ViT stays frozen. A stage-1 optimizer
step changes projector params but leaves decoder params unchanged.
"""

import copy

import torch

from config import VLMConfig
from vlm import VLM, vlm_loss

from nanovision.data import toy


def _all_frozen(module):
    return all(not p.requires_grad for p in module.parameters())


def _all_trainable(module):
    return all(p.requires_grad for p in module.parameters())


def test_stage1_freezes_vit_and_decoder():
    model = VLM(VLMConfig())
    model.set_stage(1)
    assert _all_frozen(model.vit)
    assert _all_frozen(model.decoder)
    assert _all_trainable(model.connector)
    assert _all_trainable(model.embed)
    assert _all_trainable(model.head)


def test_stage2_unfreezes_decoder_keeps_vit_frozen():
    model = VLM(VLMConfig())
    model.set_stage(2)
    assert _all_frozen(model.vit)
    assert _all_trainable(model.decoder)
    assert _all_trainable(model.connector)


def test_stage1_step_moves_projector_not_decoder():
    torch.manual_seed(0)
    cfg = VLMConfig()
    model = VLM(cfg)
    img, tok = toy.image_text_batch(
        batch=4, size=cfg.img_size, channels=cfg.in_chans, n_classes=cfg.n_classes,
        n_attrs=cfg.n_attrs, vocab_size=cfg.vocab_size, max_len=cfg.max_len, seed=0,
    )
    model.set_stage(1)
    proj_before = copy.deepcopy(model.connector.fc1.weight.detach())
    dec_before = copy.deepcopy(next(model.decoder.parameters()).detach())

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    opt.zero_grad()
    logits, n_visual = model(img, tok)
    vlm_loss(logits, tok, n_visual).backward()
    opt.step()

    assert not torch.allclose(proj_before, model.connector.fc1.weight)
    assert torch.allclose(dec_before, next(model.decoder.parameters()))
