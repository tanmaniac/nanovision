"""The spectral-bias invariant: the Fourier encoding must let the MLP fit the scene's
high-frequency surface texture markedly better than a raw-coordinate MLP can.

This gates the lesson the a09 ablation teaches (Tancik et al. 2020 / NeRF). It is a *fitting*
test, not a novel-view test: both models train on the same rays and we compare reconstruction
error. The encoded model resolves the stripes; the raw-coordinate model is spectrally biased
and blurs them, so its error stays far higher. If the scene ever reverts to a smooth albedo
(no high-frequency content), both models fit equally and this test fails - which is the point.

It uses a small capture (a few low-res views, a few hundred steps) so it stays CPU-only, but
it is heavier than the other unit tests; the spectral gap is what is asserted, with margin.
"""

import torch

from config import NeRFConfig
from model import NeRFMLP

from nanovision.data import toy
from nanovision.volume import (
    deltas_from_z,
    sample_along_rays,
    stratified_sample_rays,
    volume_render,
)

# A small textured capture: enough rays that fitting (not memorization) is the bottleneck,
# small enough to stay on CPU.
_H, _N_VIEWS, _STEPS, _NRAYS = 20, 4, 500, 512


def _fit_reconstruction_mse(pos_L, ro, rd, tg, near, far, cfg):
    """Train a NeRF MLP (with or without encoding) on the rays and return its final MSE over
    the full ray set."""
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    model = NeRFMLP(pos_L=pos_L, dir_L=cfg.dir_L, hidden=cfg.hidden, n_layers=cfg.n_layers,
                    include_input=cfg.include_input, scene_bound=cfg.scene_bound)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    zl = torch.linspace(0.0, 1.0, cfg.n_samples)
    for _ in range(_STEPS):
        idx = torch.randperm(ro.shape[0], generator=g)[:_NRAYS]
        o, d, t = ro[idx], rd[idx], tg[idx]
        z = (near + (far - near) * zl).expand(o.shape[0], cfg.n_samples).contiguous()
        pts = sample_along_rays(o, d, z)
        sigma, rgb = model(pts, d)
        color, _ = volume_render(sigma, rgb, deltas_from_z(z), white_background=cfg.white_background)
        loss = ((color - t) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        z = (near + (far - near) * zl).expand(ro.shape[0], cfg.n_samples).contiguous()
        pts = sample_along_rays(ro, rd, z)
        sigma, rgb = model(pts, rd)
        color, _ = volume_render(sigma, rgb, deltas_from_z(z), white_background=cfg.white_background)
        return ((color - tg) ** 2).mean().item()


def test_encoding_resolves_high_frequency_texture():
    cfg = NeRFConfig()
    images, poses, K, near, far = toy.nerf_synthetic_scene(
        n_views=_N_VIEWS, H=_H, W=_H,
        radius=cfg.radius, sphere_sigma=cfg.sphere_sigma, cam_dist=cfg.cam_dist,
        texture_freq=cfg.texture_freq, focal_mult=cfg.focal_mult,
    )
    ro, rd, tg = [], [], []
    for v in range(_N_VIEWS):
        o, d, _ = stratified_sample_rays(_H, _H, K, poses[v], near, far, cfg.n_samples, perturb=False)
        ro.append(o); rd.append(d); tg.append(images[v].reshape(-1, 3))
    ro, rd, tg = torch.cat(ro), torch.cat(rd), torch.cat(tg)

    enc_mse = _fit_reconstruction_mse(cfg.pos_L, ro, rd, tg, near, far, cfg)
    raw_mse = _fit_reconstruction_mse(0, ro, rd, tg, near, far, cfg)

    # The encoded model must fit the high-frequency texture clearly better than the
    # raw-coordinate model. Measured ratio ~0.2; assert a comfortable margin.
    assert enc_mse < 0.5 * raw_mse, f"encoding did not resolve the texture: enc {enc_mse:.4f} vs raw {raw_mse:.4f}"
    # And the raw-coordinate model genuinely fails to fit it (not both trivially fitting a
    # smooth scene), which is the spectral bias the ablation is about.
    assert raw_mse > 5e-3, f"raw-coordinate MSE {raw_mse:.4f} too low; is the scene still textured?"
