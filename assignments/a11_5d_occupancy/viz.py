"""Train a small occupancy field by rendering supervision and visualize the result.

Run from the repo root: `python -m assignments.a11_5d_occupancy.viz` (or
`make viz A=a11_5d_occupancy`). Writes figures to out/.

Two figures:
1. A few Z layers of the predicted occupancy grid next to the ground-truth occupancy, as
   heatmaps. Depth-only rendering supervision pulls up the occupancy where rays terminate.
2. Rendered depth vs analytic GT depth for one camera, as a scatter (the rendering integral
   inverting to the hard ray-box geometry).

GPU-aware: the field and the training batch move to default_device() (CUDA if present); tensors
handed to matplotlib are moved back to CPU. With NUSCENES_DATAROOT set a real loader would go
here; the synthetic occupancy_toy_scene runs everywhere.
"""

import os
import sys
from pathlib import Path

from nanovision.viz import SHOW, finish, plt  # noqa: E402  (sets the matplotlib backend)
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

_here = Path(__file__).parent
_impl = _here / "solution" if os.environ.get("NANOVISION_IMPL") == "solution" else _here
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_impl))

from config import OccConfig  # noqa: E402
from occupancy import render_occupancy_rays  # noqa: E402

from nanovision.data import toy  # noqa: E402
from nanovision.determinism import default_device  # noqa: E402

_OUT = _here / "out"
_OUT.mkdir(exist_ok=True)


def _z_vals(n_rays, z_near, z_far, n_samples, device):
    mids = torch.linspace(0.0, 1.0, n_samples, device=device)
    z = z_near + (z_far - z_near) * mids
    return z[None].expand(n_rays, n_samples).contiguous()


def main():
    torch.manual_seed(0)
    dev = default_device()
    cfg = OccConfig()

    if os.environ.get("NUSCENES_DATAROOT"):
        print("NUSCENES_DATAROOT set, but this viz uses the synthetic toy; ignoring.")
    scene = toy.occupancy_toy_scene(grid=cfg.grid, bounds=cfg.grid_bounds, n_classes=cfg.n_classes,
                                    n_boxes=cfg.n_boxes, n_cams=cfg.n_cams, img=cfg.img, seed=0)
    rays_o = scene["rays_o"].to(dev)
    rays_d = scene["rays_d"].to(dev)
    gt_depth = scene["gt_depth"].to(dev)
    gt_sem = scene["gt_sem"].to(dev)
    occ_gt = scene["occ_gt"]                              # [Z, Y, X], keep on CPU for plotting
    hit = gt_depth < cfg.z_far

    z_vals = _z_vals(rays_o.shape[0], cfg.z_near, cfg.z_far, cfg.n_samples, dev)

    # Learnable occupancy + semantic field, optimized by depth + semantic rendering loss.
    occ_logit = torch.zeros(*cfg.grid, device=dev, requires_grad=True)
    sem_logit = torch.zeros(cfg.n_classes, *cfg.grid, device=dev, requires_grad=True)
    opt = torch.optim.Adam([occ_logit, sem_logit], lr=0.1)
    for step in range(600):
        occ = torch.sigmoid(occ_logit)
        depth, sem_out, _ = render_occupancy_rays(
            occ, sem_logit, rays_o, rays_d, z_vals, cfg.grid_bounds, cfg.z_far)
        loss = F.smooth_l1_loss(depth, gt_depth)
        if hit.any():
            loss = loss + F.cross_entropy(sem_out[hit], gt_sem[hit])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 150 == 0:
            print(f"step {step:4d}  loss {loss.item():.4f}")

    with torch.no_grad():
        occ = torch.sigmoid(occ_logit).cpu()             # [Z, Y, X]
        depth, _, _ = render_occupancy_rays(
            occ.to(dev), sem_logit, rays_o, rays_d, z_vals, cfg.grid_bounds, cfg.z_far)
        depth = depth.cpu()
    print(f"final depth error {(depth - gt_depth.cpu()).abs().mean().item():.4f} m")

    # Figure 1: a few Z layers, predicted occupancy over GT occupancy.
    Z = cfg.Z
    layers = sorted({0, Z // 2, Z - 1})
    fig, axes = plt.subplots(2, len(layers), figsize=(3 * len(layers), 6))
    for col, zi in enumerate(layers):
        axes[0, col].imshow(occ[zi].numpy(), origin="lower", cmap="magma", vmin=0, vmax=1)
        axes[0, col].set_title(f"pred occ  Z={zi}")
        axes[1, col].imshow(occ_gt[zi].numpy(), origin="lower", cmap="magma", vmin=0, vmax=1)
        axes[1, col].set_title(f"GT occ  Z={zi}")
        for r in range(2):
            axes[r, col].set_xlabel("X"); axes[r, col].set_ylabel("Y")
    fig.suptitle("Predicted vs ground-truth occupancy (depth-supervised)")
    fig.tight_layout()
    finish(_OUT / "occupancy_slices.png")

    # Figure 2: rendered depth vs analytic GT depth for the hit rays.
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    gd = gt_depth.cpu()
    ax.scatter(gd[hit.cpu()].numpy(), depth[hit.cpu()].numpy(), s=10, alpha=0.6)
    lim = [cfg.z_near, gd[hit.cpu()].max().item() + 0.5]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel("analytic GT depth (m)"); ax.set_ylabel("rendered depth (m)")
    ax.set_title("Rendered depth vs analytic ray-box depth")
    fig.tight_layout()
    finish(_OUT / "rendered_depth.png")
    print(f"wrote {_OUT / 'occupancy_slices.png'} and {_OUT / 'rendered_depth.png'}")


if __name__ == "__main__":
    main()
    if SHOW:
        plt.show()
