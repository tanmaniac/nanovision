"""Train Lift-Splat-Shoot on one toy BEV scene and visualize the result. Provided, not graded.

Run from the repo root: `python -m assignments.a11_5b_lift_splat_shoot.viz` (or
`make viz A=a11_5b_lift_splat_shoot`). Writes figures to out/.

Two figures:
1. The per-pixel depth distribution as a bar chart over the D bins, for a few feature cells
   that a vehicle projects to. After overfit the softmax concentrates near the vehicle's true
   depth (because the lifted feature only lands in the right BEV pillar at that depth).
2. The predicted BEV occupancy next to the ground-truth occupancy.

GPU-aware: the model and the training batch move to default_device() (CUDA if present), while
the tensors handed to matplotlib are moved back to CPU. With NUSCENES_DATAROOT set this would
load a real scene; the toy fallback runs everywhere.
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

from config import LSSConfig  # noqa: E402

from nanovision.data import toy  # noqa: E402
from nanovision.determinism import default_device  # noqa: E402
from nanovision.lift_splat import LiftSplatShoot  # noqa: E402

_OUT = _here / "out"
_OUT.mkdir(exist_ok=True)


def _load_scene(cfg):
    # nuScenes hook: a real loader would go here when NUSCENES_DATAROOT is set. The toy is the
    # default so the demo runs without the dataset.
    if os.environ.get("NUSCENES_DATAROOT"):
        print("NUSCENES_DATAROOT set, but this toy demo uses the synthetic scene; "
              "see A11.5a for the nuScenes loader.")
    return toy.bev_toy_scene(
        n_vehicles=3, img=cfg.img, stride=cfg.stride,
        d_min=cfg.d_min, d_max=cfg.d_max, d_step=cfg.d_step,
        focal=cfg.f, cam_height=cfg.cam_height, seed=0,
    )


def main():
    torch.manual_seed(0)
    dev = default_device()
    cfg = LSSConfig()
    scene = _load_scene(cfg)
    image = scene["image"].to(dev)
    K, E = scene["K"].to(dev), scene["E"].to(dev)
    gt = scene["bev_gt"][None, None].to(dev)

    model = LiftSplatShoot(cfg).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for _ in range(1500):
        loss = loss_fn(model(image, K, E), gt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"trained: final BCE {loss.item():.5f} on device {dev}")

    # Figure 2 data: predicted vs GT BEV occupancy.
    with torch.no_grad():
        prob = torch.sigmoid(model(image, K, E))[0, 0].cpu()
    gt_cpu = scene["bev_gt"].cpu()

    # Figure 1 data: the depth distribution at the feature cells each vehicle projects to.
    with torch.no_grad():
        feat = model.backbone(image)
        depth_logits, _ = model.depth_lift(feat)         # (1, D, Hf, Wf)
        alpha = torch.softmax(depth_logits, dim=1)[0].cpu()  # (D, Hf, Wf)
    bins = cfg.bins()
    mask = scene["depth_mask"]
    cells = torch.nonzero(mask.cpu(), as_tuple=False)    # (n, 2) feature cells with a vehicle
    labels = scene["depth_bin_labels"].cpu()

    fig, axes = plt.subplots(1, max(1, cells.shape[0]), figsize=(3.2 * max(1, cells.shape[0]), 3))
    if cells.shape[0] <= 1:
        axes = [axes]
    for ax, (i, j) in zip(axes, cells.tolist()):
        ax.bar(bins.numpy(), alpha[:, i, j].numpy(), width=0.7)
        true_d = bins[labels[i, j]].item()
        ax.axvline(true_d, color="r", linestyle="--", label=f"GT bin {true_d:.0f} m")
        ax.set_title(f"cell ({i},{j})")
        ax.set_xlabel("depth (m)")
        ax.set_ylabel("softmax prob")
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(_OUT / "depth_distribution.png", dpi=120)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.4))
    # BEV grid is (nx forward, ny lateral); show forward up the page.
    axes[0].imshow(gt_cpu.numpy(), origin="lower", aspect="auto", cmap="magma")
    axes[0].set_title("ground-truth BEV occupancy")
    axes[1].imshow(prob.numpy(), origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1)
    axes[1].set_title("predicted BEV occupancy")
    for ax in axes:
        ax.set_xlabel("y cell (lateral)")
        ax.set_ylabel("x cell (forward)")
    plt.tight_layout()
    plt.savefig(_OUT / "bev_occupancy.png", dpi=120)
    plt.close()
    print(f"wrote {_OUT/'depth_distribution.png'} and {_OUT/'bev_occupancy.png'}")


if __name__ == "__main__":
    main()
