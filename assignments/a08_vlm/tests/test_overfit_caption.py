"""Overfit one image_text_batch and check generate reproduces the captions exactly.

Stage 2 trains the projector, embedding, head, and decoder on 4 fixed (image, caption) pairs.
The caption cross-entropy falls to near zero within ~300 steps, and greedy generate
reproduces each caption's class + attr + EOS tokens exactly. Measured final loss ~7e-4
(see the README's validation section).
"""

import torch

from config import VLMConfig
from vlm import VLM, vlm_loss

from nanovision.data import toy


def test_overfit_caption():
    torch.manual_seed(0)
    cfg = VLMConfig()
    model = VLM(cfg)
    img, tok = toy.image_text_batch(
        batch=4, size=cfg.img_size, channels=cfg.in_chans, n_classes=cfg.n_classes,
        n_attrs=cfg.n_attrs, vocab_size=cfg.vocab_size, max_len=cfg.max_len, seed=0,
    )
    model.set_stage(2)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-3)
    for _ in range(300):
        opt.zero_grad()
        logits, n_visual = model(img, tok)
        loss = vlm_loss(logits, tok, n_visual)
        loss.backward()
        opt.step()
    final = loss.item()
    assert final < 0.05, f"final caption loss {final}"

    # Greedy generate must reproduce the [class, attr, EOS] prefix of each caption exactly.
    gen = model.generate(img, max_new_tokens=3)
    assert torch.equal(gen, tok[:, :3]), f"generated {gen.tolist()} != {tok[:, :3].tolist()}"
