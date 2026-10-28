"""Train Lift-Splat-Shoot on one toy BEV scene, with and without depth supervision. Provided, not graded.

Run from the repo root: `python -m assignments.a11_5b_lift_splat_shoot.viz` (or
`make viz A=a11_5b_lift_splat_shoot`). Writes figures to out/.

Two models start from the same initialization and train for the same 1500 Adam steps on the same
scene: (i) the BEV occupancy loss alone (vanilla LSS, implicit depth), and (ii) the BEV loss plus
BEVDepth's depth-supervision loss (``bevdepth_depth_loss`` on the scene's depth-bin labels at the
vehicle feature cells, the same labels the tests use).

Two figures:
1. depth_distribution.png: the predicted depth distribution (softmax over the D bins) at each
   labeled vehicle feature cell, one row per model, with the label bin marked. The script prints
   the probability each model puts on the label bin.
2. bev_occupancy.png: ground-truth BEV occupancy next to the predicted occupancy of both models.

GPU-aware: the models and the training batch move to default_device() (CUDA if present), while
the tensors handed to matplotlib are moved back to CPU. With NUSCENES_DATAROOT set this would
load a real scene; the toy fallback runs everywhere.
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

from config import LSSConfig  # noqa: E402

from nanovision.data import toy  # noqa: E402
from nanovision.determinism import default_device  # noqa: E402
from nanovision.lift_splat import LiftSplatShoot, bevdepth_depth_loss  # noqa: E402

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


def _train(model, scene, gt, depth_weight, steps=1500):
    """Adam at lr 1e-2 on BCE(BEV) + depth_weight * BEVDepth depth CE. Returns final losses."""
    image, K, E = scene["image"], scene["K"], scene["E"]
    labels = scene["depth_bin_labels"][None]          # (1, Hf, Wf)
    mask = scene["depth_mask"][None]                  # (1, Hf, Wf)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    bce = torch.nn.BCEWithLogitsLoss()
    for _ in range(steps):
        loss_bev = bce(model(image, K, E), gt)
        depth_logits, _ = model.depth_lift(model.backbone(image))
        loss_depth = bevdepth_depth_loss(depth_logits, labels, mask)
        loss = loss_bev + depth_weight * loss_depth
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss_bev.item(), loss_depth.item()


def _depth_probs(model, image):
    with torch.no_grad():
        depth_logits, _ = model.depth_lift(model.backbone(image))
    return torch.softmax(depth_logits, dim=1)[0].cpu()   # (D, Hf, Wf)


def _iou(model, scene, gt):
    with torch.no_grad():
        pred = model(scene["image"], scene["K"], scene["E"]) > 0.0
    g = gt > 0.5
    return ((pred & g).sum() / (pred | g).sum().clamp(min=1)).item()


def main():
    torch.manual_seed(0)
    dev = default_device()
    cfg = LSSConfig()
    scene = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in _load_scene(cfg).items()}
    gt = scene["bev_gt"][None, None]

    # Same initialization for both models.
    init = LiftSplatShoot(cfg).to(dev)
    models = {}
    for name, w in (("BEV loss only", 0.0), ("BEV loss + depth supervision", 1.0)):
        model = LiftSplatShoot(cfg).to(dev)
        model.load_state_dict(init.state_dict())
        lb, ld = _train(model, scene, gt, depth_weight=w)
        models[name] = model
        print(f"{name}: final BCE {lb:.2e}, depth CE {ld:.2e}, BEV IoU {_iou(model, scene, gt):.2f}"
              f" (device {dev})")

    bins = cfg.bins()
    mask = scene["depth_mask"].cpu()
    labels = scene["depth_bin_labels"].cpu()
    cells = torch.nonzero(mask, as_tuple=False).tolist()   # feature cells with a vehicle
    n = max(1, len(cells))

    # Figure 1: depth distributions at the labeled cells, one row per model.
    fig, axes = plt.subplots(len(models), n, figsize=(3.2 * n, 2.8 * len(models)),
                             squeeze=False, sharey=True)
    for r, (name, model) in enumerate(models.items()):
        alpha = _depth_probs(model, scene["image"])
        for c, (i, j) in enumerate(cells):
            ax = axes[r][c]
            k = int(labels[i, j])
            ax.bar(bins.numpy(), alpha[:, i, j].numpy(), width=0.7)
            ax.bar([bins[k].item()], [alpha[k, i, j].item()], width=0.7, color="tab:red")
            ax.axvline(bins[k].item(), color="r", linestyle="--",
                       label=f"label bin {bins[k].item():.0f} m")
            ax.set_ylim(0, 1)
            ax.set_title(f"{name}\ncell ({i},{j})", fontsize=9)
            ax.set_xlabel("depth (m)")
            if c == 0:
                ax.set_ylabel("softmax prob")
            ax.legend(fontsize=7)
            print(f"  {name}: cell ({i},{j}) label {bins[k].item():.0f} m -> "
                  f"prob {alpha[k, i, j].item():.3f}, argmax {bins[alpha[:, i, j].argmax()].item():.0f} m")
    plt.tight_layout()
    finish(_OUT / "depth_distribution.png")

    # Figure 2: GT vs predicted BEV occupancy for both models.
    fig, axes = plt.subplots(1, 1 + len(models), figsize=(3.6 * (1 + len(models)), 3.4))
    # BEV grid is (nx forward, ny lateral); show forward up the page.
    axes[0].imshow(scene["bev_gt"].cpu().numpy(), origin="lower", aspect="auto", cmap="magma")
    axes[0].set_title("ground-truth BEV occupancy", fontsize=9)
    for ax, (name, model) in zip(axes[1:], models.items()):
        with torch.no_grad():
            prob = torch.sigmoid(model(scene["image"], scene["K"], scene["E"]))[0, 0].cpu()
        ax.imshow(prob.numpy(), origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1)
        ax.set_title(f"predicted, {name}", fontsize=9)
    for ax in axes:
        ax.set_xlabel("y cell (lateral)")
        ax.set_ylabel("x cell (forward)")
    plt.tight_layout()
    finish(_OUT / "bev_occupancy.png")
    print(f"wrote {_OUT/'depth_distribution.png'} and {_OUT/'bev_occupancy.png'}")


if __name__ == "__main__":
    main()
    if SHOW:
        plt.show()
