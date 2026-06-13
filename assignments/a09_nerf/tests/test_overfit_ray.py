"""Overfit a small batch of rays from the toy scene end to end.

This exercises encoding -> NeRFMLP -> volume_render against ground-truth pixels that were
rendered by the INDEPENDENT closed-form ray-sphere chord (not by volume_render). A passing
run therefore shows the discretized quadrature converges to the analytic Beer-Lambert
integral, not that it matches its own renderer.
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


def test_overfit_rays():
    torch.manual_seed(0)
    cfg = NeRFConfig()
    images, poses, K, near, far = toy.nerf_synthetic_scene(
        n_views=cfg.n_views, H=cfg.H, W=cfg.W,
        radius=cfg.radius, sphere_sigma=cfg.sphere_sigma, cam_dist=cfg.cam_dist,
        texture_freq=cfg.texture_freq, focal_mult=cfg.focal_mult,
    )
    # Build all rays from the training views (drop the held-out last view).
    g = torch.Generator().manual_seed(0)
    rays_o, rays_d, target = [], [], []
    for v in range(cfg.n_views - 1):
        o, d, z = stratified_sample_rays(cfg.H, cfg.W, K, poses[v], near, far,
                                         cfg.n_samples, perturb=False)
        rays_o.append(o)
        rays_d.append(d)
        target.append(images[v].reshape(-1, 3))
    rays_o = torch.cat(rays_o, 0)
    rays_d = torch.cat(rays_d, 0)
    target = torch.cat(target, 0)

    # Sample a fixed batch of 256 rays to overfit.
    idx = torch.randperm(rays_o.shape[0], generator=g)[:256]
    ro, rd, tgt = rays_o[idx], rays_d[idx], target[idx]

    model = NeRFMLP(
        pos_L=cfg.pos_L, dir_L=cfg.dir_L, hidden=cfg.hidden,
        n_layers=cfg.n_layers, include_input=cfg.include_input,
        scene_bound=cfg.scene_bound,
    )
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    z_lo = torch.linspace(0.0, 1.0, cfg.n_samples)
    z_vals = (near + (far - near) * z_lo).expand(ro.shape[0], cfg.n_samples).contiguous()
    deltas = deltas_from_z(z_vals)

    first = None
    for _ in range(800):
        pts = sample_along_rays(ro, rd, z_vals)
        sigma, rgb = model(pts, rd)
        color, _ = volume_render(sigma, rgb, deltas, white_background=cfg.white_background)
        loss = ((color - tgt) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    final = loss.item()
    # Measured floor on this batch is well under 1e-3; assert a comfortable bound.
    assert final < 5e-3, f"final MSE {final} (start {first})"
    assert final < 0.05 * first
