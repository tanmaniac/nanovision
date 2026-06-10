"""Fit the toy scene with 3D Gaussian splatting and compare its render speed to the NeRF.

Run from the repo root: `python -m assignments.a10_gaussian_splatting.viz` (or
`make viz A=a10_gaussian_splatting`). Writes figures to out/. Provided, not graded.

Fits a cloud of Gaussians to the posed colored-sphere images (the same scene the NeRF
assignment fits) by gradient descent through the differentiable splat rasterizer, on the
GPU when one is present. Shows the target next to the render for a training view and a
held-out view, the held-out PSNR over training, and a wall-clock inference-time comparison
against the ray-marched NeRF on the same scene. The speedup is measured and printed, not
assumed: it depends on the NeRF width and sample count and the Gaussian count.
"""

import os
import sys
import time
from pathlib import Path

from nanovision.viz import SHOW, finish, plt  # noqa: E402  (sets the matplotlib backend)
import torch  # noqa: E402

_here = Path(__file__).parent
_impl = _here / "solution" if os.environ.get("NANOVISION_IMPL") == "solution" else _here
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_impl))

from config import SplatConfig  # noqa: E402
from gaussian import GaussianModel  # noqa: E402
from project import project_gaussians  # noqa: E402
from render import splat_render  # noqa: E402

from nanovision.data import toy  # noqa: E402
from nanovision.determinism import default_device  # noqa: E402
from nanovision.geometry import invert_transform  # noqa: E402

_OUT = _here / "out"
_OUT.mkdir(exist_ok=True)


def _psnr(pred, target):
    mse = ((pred - target) ** 2).mean().clamp(min=1e-10)
    return (-10.0 * torch.log10(mse)).item()


def _render_view(model, K, c2w, H, W, dilation):
    w2c = invert_transform(c2w)
    means2d, cov2d, depths = project_gaussians(model, K, w2c, dilation=dilation)
    return splat_render(means2d, cov2d, model.colors, model.opacities, depths, H, W, bg=1.0)


def _fit(cfg, images, poses, K, dev):
    model = GaussianModel.random_init(cfg.n_gaussians, spread=cfg.init_spread,
                                      init_scale=cfg.init_scale, seed=0, device=dev)
    opt = torch.optim.Adam([
        {"params": [model.means], "lr": cfg.lr_means},
        {"params": [model.log_scales], "lr": cfg.lr_scales},
        {"params": [model.quats], "lr": cfg.lr_quats},
        {"params": [model.opacity_logits], "lr": cfg.lr_opacity},
        {"params": [model.color_logits], "lr": cfg.lr_color},
    ])
    n_train = images.shape[0] - 1                       # last view held out
    heldout_psnr = []
    for step in range(cfg.n_steps):
        v = step % n_train
        target = images[v]
        pred = _render_view(model, K, poses[v], cfg.H, cfg.W, cfg.dilation)
        loss = (pred - target).abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 25 == 0:
            with torch.no_grad():
                ho = _render_view(model, K, poses[-1], cfg.H, cfg.W, cfg.dilation)
                heldout_psnr.append((step, _psnr(ho, images[-1])))
    return model, heldout_psnr


def _nerf_inference_time(cfg, K, c2w, dev, reps=20):
    """Wall-clock to render one view with the ray-marched NeRF (untrained; timing only)."""
    sys.path.insert(0, str(Path(_here).parents[1] / "assignments" / "a09_nerf"))
    impl9 = "solution" if os.environ.get("NANOVISION_IMPL") == "solution" else ""
    if impl9:
        sys.path.insert(0, str(Path(_here).parents[1] / "assignments" / "a09_nerf" / "solution"))
    from model import NeRFMLP  # noqa: E402
    from nanovision.volume import (deltas_from_z, sample_along_rays,  # noqa: E402
                                   stratified_sample_rays, volume_render)

    n_samples = 32
    model = NeRFMLP(pos_L=6, dir_L=4, hidden=128, n_layers=4,
                    include_input=True, scene_bound=4.0).to(dev)
    near, far = 2.5, 5.5
    o, d, _ = stratified_sample_rays(cfg.H, cfg.W, K, c2w, near, far, n_samples, perturb=False)
    zl = torch.linspace(0.0, 1.0, n_samples, device=dev)
    z_vals = (near + (far - near) * zl).expand(o.shape[0], n_samples).contiguous()
    deltas = deltas_from_z(z_vals)

    def once():
        with torch.no_grad():
            pts = sample_along_rays(o, d, z_vals)
            sigma, rgb = model(pts, d)
            volume_render(sigma, rgb, deltas, white_background=True)

    once()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        once()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) / reps


def _splat_inference_time(model, cfg, K, c2w, dev, reps=20):
    def once():
        with torch.no_grad():
            _render_view(model, K, c2w, cfg.H, cfg.W, cfg.dilation)

    once()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        once()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) / reps


def main():
    dev = default_device()
    cfg = SplatConfig()
    torch.manual_seed(0)

    images, poses, K, _, _ = toy.nerf_synthetic_scene(n_views=cfg.n_views, H=cfg.H, W=cfg.W,
                                                      device=str(dev))
    poses = poses.to(dev)
    K = K.to(dev)

    model, heldout_psnr = _fit(cfg, images, poses, K, dev)

    # Target vs render for a training view and the held-out view.
    with torch.no_grad():
        train_pred = _render_view(model, K, poses[0], cfg.H, cfg.W, cfg.dilation)
        held_pred = _render_view(model, K, poses[-1], cfg.H, cfg.W, cfg.dilation)
    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    for ax, im, title in [
        (axes[0, 0], images[0], "train view: target"),
        (axes[0, 1], train_pred, "train view: render"),
        (axes[1, 0], images[-1], "held-out: target"),
        (axes[1, 1], held_pred, "held-out: render"),
    ]:
        ax.imshow(im.detach().cpu().numpy()); ax.set_title(title); ax.axis("off")
    plt.tight_layout(); finish(_OUT / "splat_fit.png")

    steps, psnrs = zip(*heldout_psnr)
    plt.figure(figsize=(4.5, 3.2))
    plt.plot(steps, psnrs)
    plt.xlabel("step"); plt.ylabel("held-out PSNR (dB)"); plt.tight_layout()
    finish(_OUT / "psnr_curve.png")

    # Inference-time comparison on the same view.
    t_splat = _splat_inference_time(model, cfg, K, poses[-1], dev)
    t_nerf = _nerf_inference_time(cfg, K, poses[-1], dev)
    ratio = t_nerf / t_splat
    plt.figure(figsize=(4.5, 3.2))
    plt.bar(["NeRF (ray march)", "splat"], [t_nerf * 1e3, t_splat * 1e3], color=["C0", "C1"])
    plt.ylabel("ms / view"); plt.title(f"splat is {ratio:.1f}x faster here")
    plt.tight_layout(); finish(_OUT / "speed.png")

    print(f"device {dev}; held-out PSNR final {psnrs[-1]:.1f} dB; "
          f"splat {t_splat*1e3:.2f} ms vs NeRF {t_nerf*1e3:.2f} ms per {cfg.H}x{cfg.W} view "
          f"({ratio:.1f}x). figures in {_OUT}")


if __name__ == "__main__":
    main()
    if SHOW:
        plt.show()
