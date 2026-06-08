"""Shape contracts for the schedule, q_sample, the U-Net, and the two samplers."""

import torch

from config import DiffusionConfig
from diffusion import q_sample
from sampling import ddim_sample, ddpm_sample
from schedule import cosine_alpha_bar
from unet import TimeEmbeddedUNet


def test_shapes():
    cfg = DiffusionConfig(base_width=16)
    T = 10
    _, abar = cosine_alpha_bar(T)
    assert abar.shape == (T,)

    B = 2
    x0 = torch.randn(B, cfg.channels, cfg.img_size, cfg.img_size)
    t = torch.randint(0, T, (B,))
    eps = torch.randn_like(x0)
    assert q_sample(x0, t, eps, abar).shape == x0.shape

    model = TimeEmbeddedUNet(cfg).eval()
    labels = torch.tensor([0, 1])
    with torch.no_grad():
        assert model(x0, t, labels).shape == x0.shape
        assert model(x0, t, None).shape == x0.shape

        shape = (B, cfg.channels, cfg.img_size, cfg.img_size)
        assert ddpm_sample(model, shape, abar, kind="v").shape == shape
        ts = list(range(T - 1, -1, -2))
        assert ddim_sample(model, shape, abar, ts, kind="v").shape == shape
