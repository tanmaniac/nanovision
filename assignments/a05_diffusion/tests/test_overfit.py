"""Overfit one batch: the v-prediction loss drops sharply on a fixed 4-image batch.

This checks the training loop end to end (schedule, q_sample, v-target, the U-Net's time
injection, the loss). It does not assert sampling reconstructs the specific images:
sampling integrates the learned score from fresh noise and is only constrained on the
training (x_t, t) points, so a 4-image batch does not pin down a global trajectory.
"""

import torch

from config import DiffusionConfig
from diffusion import diffusion_loss
from schedule import cosine_alpha_bar
from unet import TimeEmbeddedUNet

from nanovision.data import toy


def test_overfit_one_batch():
    torch.manual_seed(0)
    cfg = DiffusionConfig(base_width=32)
    T = 50
    _, abar = cosine_alpha_bar(T)
    x0, labels = toy.diffusion_image_batch(n=4, num_classes=cfg.num_classes,
                                           size=cfg.img_size, channels=cfg.channels, seed=0)
    model = TimeEmbeddedUNet(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    gen = torch.Generator().manual_seed(0)

    first = None
    for step in range(400):
        opt.zero_grad()
        loss = diffusion_loss(model, x0, abar, kind="v", num_classes=cfg.num_classes,
                              cfg_drop_prob=0.1, labels=labels, generator=gen)
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    final = loss.item()
    assert final < 0.1, f"final loss {final} not below 0.1 (start {first})"
    assert final < 0.5 * first
