"""Train BEVFormer on one toy multi-cam scene and visualize the geometry and the result.

Run from the repo root: `python -m assignments.a11_5c_bevformer.viz` (or
`make viz A=a11_5c_bevformer`). Writes figures to out/.

Two figures:
1. For a few BEV cells, the projected reference points overlaid on each camera image. This shows
   the geometry-as-attention-prior: a BEV cell reaches back into image space and samples the
   features at exactly these pixels (the query-pull view transform), the opposite of the depth-push
   in Lift-Splat-Shoot.
2. The predicted BEV occupancy next to the ground-truth occupancy after a short overfit.

GPU-aware: the model and the training batch move to default_device() (CUDA if present), while the
tensors handed to matplotlib are moved back to CPU. With NUSCENES_DATAROOT set this would load a
real scene; the toy fallback runs everywhere.
"""

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

_here = Path(__file__).parent
_impl = _here / "solution" if os.environ.get("NANOVISION_IMPL") == "solution" else _here
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_impl))

from config import BEVFormerConfig  # noqa: E402

from nanovision.data import toy  # noqa: E402
from nanovision.determinism import default_device  # noqa: E402
from nanovision.geometry import CameraRig  # noqa: E402
from nanovision.bevformer import (  # noqa: E402
    BEVFormerSeg,
    bev_reference_points,
    project_reference_points,
)

_OUT = _here / "out"
_OUT.mkdir(exist_ok=True)


def _rig(scene, cfg, device):
    K, E = scene["K"].to(device), scene["E"].to(device)
    Ks = {f"cam{i}": K for i in range(cfg.n_cams)}
    Es = {f"cam{i}": E[i] for i in range(cfg.n_cams)}
    sizes = {f"cam{i}": (cfg.img, cfg.img) for i in range(cfg.n_cams)}
    return CameraRig(Ks, Es, sizes)


def main():
    torch.manual_seed(0)
    dev = default_device()
    cfg = BEVFormerConfig()

    if os.environ.get("NUSCENES_DATAROOT"):
        print("NUSCENES_DATAROOT set, but this viz uses the synthetic toy; ignoring.")
    scene = toy.bev_multicam_scene(n_cams=cfg.n_cams, n_vehicles=4, img=cfg.img,
                                   stride=cfg.stride, focal=cfg.f, seed=0)
    rig = _rig(scene, cfg, dev)
    imgs = scene["images"][0].to(dev)                    # (n_cam, 3, 32, 32)
    gt = scene["bev_gt"][0].to(dev)                      # (nx, ny)

    # Figure 1: projected reference points for a few BEV cells, overlaid per camera.
    grid = cfg.bev_grid()
    ref = bev_reference_points(grid, cfg.n_heights, cfg.z_min, cfg.z_max).to(dev)
    uv, valid = project_reference_points(ref, rig, (cfg.img, cfg.img))
    # Pick a few occupied cells to trace.
    cells = (gt.cpu() > 0.5).nonzero()[:3].tolist()
    fig, axes = plt.subplots(1, cfg.n_cams, figsize=(3 * cfg.n_cams, 3))
    colors = ["red", "lime", "cyan"]
    for c in range(cfg.n_cams):
        ax = axes[c]
        ax.imshow(imgs[c].cpu().permute(1, 2, 0).numpy())
        for ci, (i, j) in enumerate(cells):
            for h in range(cfg.n_heights):
                if not bool(valid[c, i, j, h]):
                    continue
                gx, gy = uv[c, i, j, h].tolist()
                u = (gx + 1.0) / 2.0 * cfg.img - 0.5
                v = (gy + 1.0) / 2.0 * cfg.img - 0.5
                ax.scatter([u], [v], s=18, c=colors[ci % len(colors)], edgecolors="black")
        ax.set_title(f"cam{c}")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("BEV cell reference points projected into each camera (query-pull)")
    fig.tight_layout()
    fig.savefig(_OUT / "reference_points.png", dpi=120)
    plt.close(fig)

    # Train a short overfit, then show predicted vs GT occupancy.
    model = BEVFormerSeg(cfg).to(dev)
    backbone = torch.nn.Sequential(
        torch.nn.Conv2d(3, cfg.dim, 3, stride=2, padding=1), torch.nn.ReLU(),
        torch.nn.Conv2d(cfg.dim, cfg.dim, 3, stride=2, padding=1), torch.nn.ReLU(),
    ).to(dev)
    opt = torch.optim.Adam(list(model.parameters()) + list(backbone.parameters()), lr=1e-2)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for step in range(1500):
        logit = model(backbone(imgs), rig)[0]
        loss = loss_fn(logit, gt)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 300 == 0:
            print(f"step {step:4d}  BCE {loss.item():.4f}")
    print(f"final BCE {loss.item():.4f}")

    with torch.no_grad():
        prob = torch.sigmoid(model(backbone(imgs), rig)[0]).cpu()
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    axes[0].imshow(prob.numpy(), origin="lower", cmap="magma", vmin=0, vmax=1)
    axes[0].set_title("predicted occupancy")
    axes[1].imshow(gt.cpu().numpy(), origin="lower", cmap="magma", vmin=0, vmax=1)
    axes[1].set_title("ground truth")
    for ax in axes:
        ax.set_xlabel("ego y (lateral)"); ax.set_ylabel("ego x (forward)")
    fig.tight_layout()
    fig.savefig(_OUT / "bev_occupancy.png", dpi=120)
    plt.close(fig)
    print(f"wrote {_OUT / 'reference_points.png'} and {_OUT / 'bev_occupancy.png'}")


if __name__ == "__main__":
    main()
