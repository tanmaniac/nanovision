"""Render the toy NeRF scene and the spectral-bias ablation. Provided, not graded.

Run from the repo root: `python -m assignments.a09_nerf.viz` (or
`make viz A=a09_nerf`). Writes figures to out/.

Three panels:
- the closed-form ground truth of the held-out view next to the trained model's render;
- the spectral-bias ablation: a model trained with the Fourier encoding vs one trained on
  raw coordinates only (no encoding). The no-encoding render is visibly blurry at the sphere
  boundary, since a raw-coordinate MLP fits low frequencies first;
- the held-out PSNR after a short training run.
"""

import os
import sys
from pathlib import Path

from nanovision.viz import SHOW, finish, plt  # noqa: E402  (sets the matplotlib backend)
import torch  # noqa: E402

_here = Path(__file__).parent
_impl = _here / "solution" if os.environ.get("NANOVISION_IMPL") == "solution" else _here
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_impl))

from config import NeRFConfig  # noqa: E402
from model import NeRFMLP  # noqa: E402

from nanovision.data import toy  # noqa: E402
from nanovision.determinism import default_device  # noqa: E402
from nanovision.volume import (  # noqa: E402
    deltas_from_z,
    sample_along_rays,
    stratified_sample_rays,
    volume_render,
)

_OUT = _here / "out"
_OUT.mkdir(exist_ok=True)


def _render_view(model, K, c2w, near, far, cfg):
    o, d, _ = stratified_sample_rays(cfg.H, cfg.W, K, c2w, near, far, cfg.n_samples, perturb=False)
    zl = torch.linspace(0.0, 1.0, cfg.n_samples, device=K.device)
    z_vals = (near + (far - near) * zl).expand(o.shape[0], cfg.n_samples).contiguous()
    deltas = deltas_from_z(z_vals)
    with torch.no_grad():
        pts = sample_along_rays(o, d, z_vals)
        sigma, rgb = model(pts, d)
        color, _ = volume_render(sigma, rgb, deltas, white_background=cfg.white_background)
    return color.reshape(cfg.H, cfg.W, 3).clamp(0, 1)


def _train(cfg, images, poses, K, near, far, pos_L, steps, seed=0):
    torch.manual_seed(seed)
    dev = K.device
    g = torch.Generator(device=dev).manual_seed(seed)
    ro_l, rd_l, tg_l = [], [], []
    for v in range(cfg.n_views - 1):
        o, d, _ = stratified_sample_rays(cfg.H, cfg.W, K, poses[v], near, far, cfg.n_samples, perturb=False)
        ro_l.append(o); rd_l.append(d); tg_l.append(images[v].reshape(-1, 3))
    ro = torch.cat(ro_l); rd = torch.cat(rd_l); tg = torch.cat(tg_l)
    model = NeRFMLP(pos_L=pos_L, dir_L=cfg.dir_L, hidden=cfg.hidden, n_layers=cfg.n_layers,
                    include_input=cfg.include_input, scene_bound=cfg.scene_bound).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    zl = torch.linspace(0.0, 1.0, cfg.n_samples, device=dev)
    for _ in range(steps):
        idx = torch.randperm(ro.shape[0], generator=g, device=dev)[:512]
        o, d, t = ro[idx], rd[idx], tg[idx]
        z_vals = (near + (far - near) * zl).expand(o.shape[0], cfg.n_samples).contiguous()
        deltas = deltas_from_z(z_vals)
        pts = sample_along_rays(o, d, z_vals)
        sigma, rgb = model(pts, d)
        color, _ = volume_render(sigma, rgb, deltas, white_background=cfg.white_background)
        loss = ((color - t) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def main():
    cfg = NeRFConfig()
    dev = default_device()
    images, poses, K, near, far = toy.nerf_synthetic_scene(
        n_views=cfg.n_views, H=cfg.H, W=cfg.W,
        radius=cfg.radius, sphere_sigma=cfg.sphere_sigma, cam_dist=cfg.cam_dist,
    )
    images, poses, K = images.to(dev), poses.to(dev), K.to(dev)
    gt = images[-1].clamp(0, 1)

    with_enc = _train(cfg, images, poses, K, near, far, pos_L=cfg.pos_L, steps=1500)
    no_enc = _train(cfg, images, poses, K, near, far, pos_L=0, steps=1500)

    r_with = _render_view(with_enc, K, poses[-1], near, far, cfg)
    r_no = _render_view(no_enc, K, poses[-1], near, far, cfg)

    def _psnr(a, b):
        mse = ((a - b) ** 2).mean()
        return (-10.0 * torch.log10(mse)).item()

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    axes[0].imshow(gt.cpu().numpy()); axes[0].set_title("ground truth (held-out)")
    axes[1].imshow(r_with.cpu().numpy()); axes[1].set_title(f"with encoding\nPSNR {_psnr(r_with, gt):.1f} dB")
    axes[2].imshow(r_no.cpu().numpy()); axes[2].set_title(f"no encoding\nPSNR {_psnr(r_no, gt):.1f} dB")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    finish(_OUT / "heldout_and_ablation.png")
    print(f"wrote {_OUT/'heldout_and_ablation.png'}; "
          f"PSNR with-enc {_psnr(r_with, gt):.2f} dB, no-enc {_psnr(r_no, gt):.2f} dB")


if __name__ == "__main__":
    main()
    if SHOW:
        plt.show()
