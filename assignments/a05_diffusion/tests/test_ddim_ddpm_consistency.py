"""DDIM with eta=1 on the full grid matches the beta_tilde DDPM sampler.

DDIM eta=1 produces variance beta_tilde_t, so on the full consecutive timestep list and
with the same noise it follows the same trajectory as the DDPM ancestral sampler using
variance="beta_tilde" (NOT "beta", whose noise scale differs).
"""

import torch

from config import DiffusionConfig
from sampling import ddim_sample, ddpm_sample
from schedule import cosine_alpha_bar
from unet import TimeEmbeddedUNet


def test_ddim_eta1_matches_beta_tilde_ddpm():
    torch.manual_seed(0)
    cfg = DiffusionConfig(base_width=16)
    model = TimeEmbeddedUNet(cfg).eval()
    T = 8
    _, abar = cosine_alpha_bar(T)
    shape = (2, cfg.channels, cfg.img_size, cfg.img_size)
    ts = list(range(T - 1, -1, -1))

    with torch.no_grad():
        x_ddpm = ddpm_sample(model, shape, abar, kind="v", variance="beta_tilde",
                             clip_x0=False, generator=torch.Generator().manual_seed(42))
        x_ddim = ddim_sample(model, shape, abar, ts, kind="v", eta=1.0, clip_x0=False,
                             generator=torch.Generator().manual_seed(42))
    assert torch.allclose(x_ddpm, x_ddim, atol=1e-4, rtol=1e-3)
