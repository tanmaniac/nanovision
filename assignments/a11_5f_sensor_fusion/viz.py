"""Compare camera-only, LiDAR-only, and fused BEV segmentation on the toy. Provided, not graded.

Run from the repo root: `python -m assignments.a11_5f_sensor_fusion.viz` (or
`make viz A=a11_5f_sensor_fusion`). Writes two figures to out/:
1. One held-out scene shown four ways: ground-truth vehicle occupancy, and the camera-only,
   LiDAR-only, and fused predictions. The camera smears each vehicle along its lateral column
   (depth ambiguity); the LiDAR fires on clutter too (no class signal); the fused map keeps only
   the vehicle cells.
2. A bar chart of the held-out mean IoU for the three heads.

GPU-aware via default_device(). With NUSCENES_DATAROOT set the loader exists, but this demo uses
the synthetic scene so it runs everywhere.
"""

import os
import sys
from pathlib import Path

from nanovision.viz import SHOW, finish, plt  # noqa: E402  (sets the matplotlib backend)
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

_here = Path(__file__).parent
_impl = _here / "solution" if os.environ.get("NANOVISION_IMPL") == "solution" else _here
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_impl))

from config import FusionConfig  # noqa: E402
from compare import compare_modalities  # noqa: E402
from fusion import BEVFuser, LidarPillarEncoder  # noqa: E402

from nanovision.data import toy  # noqa: E402
from nanovision.determinism import default_device  # noqa: E402

_OUT = _here / "out"
_OUT.mkdir(exist_ok=True)


def _train_three_heads(cfg, grid, train, dev):
    """Train the three heads on the train scenes; return their forward closures."""
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(30.0, device=dev))

    torch.manual_seed(0)
    cam_net = nn.Sequential(
        nn.Conv2d(cfg.n_classes, cfg.fuse_hidden, 3, padding=1), nn.ReLU(),
        nn.Conv2d(cfg.fuse_hidden, cfg.fuse_hidden, 3, padding=1), nn.ReLU(),
        nn.Conv2d(cfg.fuse_hidden, 1, 1),
    ).to(dev)
    torch.manual_seed(0)
    lid_enc = LidarPillarEncoder(grid, cfg.lidar_hidden, cfg.lidar_channels).to(dev)
    lid_head = nn.Conv2d(cfg.lidar_channels, 1, 1).to(dev)
    torch.manual_seed(0)
    fus_enc = LidarPillarEncoder(grid, cfg.lidar_hidden, cfg.lidar_channels).to(dev)
    fuser = BEVFuser(cfg.n_classes, cfg.lidar_channels, cfg.fuse_hidden, cfg.fuse_channels).to(dev)
    fus_head = nn.Conv2d(cfg.fuse_channels, 1, 1).to(dev)

    cam_fwd = lambda sc: cam_net(sc["cam_bev"][None])
    lid_fwd = lambda sc: lid_head(lid_enc(sc["lidar"])[None])
    fus_fwd = lambda sc: fus_head(fuser(sc["cam_bev"], fus_enc(sc["lidar"]))[None])

    heads = {
        "camera": (cam_fwd, list(cam_net.parameters())),
        "lidar": (lid_fwd, list(lid_enc.parameters()) + list(lid_head.parameters())),
        "fused": (fus_fwd, list(fus_enc.parameters()) + list(fuser.parameters()) + list(fus_head.parameters())),
    }
    for fwd, params in heads.values():
        opt = torch.optim.Adam(params, lr=1e-2)
        for _ in range(400):
            opt.zero_grad()
            loss = sum(loss_fn(fwd(sc), sc["bev_gt"][None, None]) for sc in train) / len(train)
            loss.backward()
            opt.step()
    return {k: v[0] for k, v in heads.items()}


def main():
    dev = default_device()
    cfg = FusionConfig()
    grid = cfg.bev_grid()

    train = [toy.bev_fusion_scene(seed=s, device=dev) for s in range(12)]
    heads = _train_three_heads(cfg, grid, train, dev)

    # Figure 1: one held-out scene, four panels.
    scene = toy.bev_fusion_scene(seed=200, device=dev)
    gt = scene["bev_gt"].cpu().numpy()
    with torch.no_grad():
        preds = {k: torch.sigmoid(fwd(scene))[0, 0].cpu().numpy() for k, fwd in heads.items()}

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
    axes[0].imshow(gt, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1)
    axes[0].set_title("ground truth")
    for ax, name in zip(axes[1:], ["camera", "lidar", "fused"]):
        ax.imshow(preds[name], origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1)
        ax.set_title(f"{name}-only" if name != "fused" else "fused")
    for ax in axes:
        ax.set_xlabel("y cell (lateral)")
        ax.set_ylabel("x cell (forward)")
    plt.tight_layout()
    finish(_OUT / "fusion_bev.png")

    # Figure 2: held-out mean IoU for the three heads (a fresh, seeded comparison).
    res = compare_modalities(cfg, seed=0, device=str(dev))
    names = ["camera", "lidar", "fused"]
    vals = [res[n] for n in names]
    print(f"held-out mean IoU  camera {vals[0]:.3f}  lidar {vals[1]:.3f}  fused {vals[2]:.3f}")

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.bar(names, vals, color=["#4c78a8", "#e45756", "#54a24b"])
    ax.set_ylabel("held-out mean IoU")
    ax.set_ylim(0, 1)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
    plt.tight_layout()
    finish(_OUT / "fusion_iou.png")
    print(f"wrote {_OUT/'fusion_bev.png'} and {_OUT/'fusion_iou.png'}")


if __name__ == "__main__":
    main()
    if SHOW:
        plt.show()
