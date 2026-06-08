"""Overfit GeometryFM on the sphere stereo set and visualize the result. Provided.

Run from the repo root:
  NANOVISION_IMPL=solution python -m assignments.a10_5_geometry_fm.viz
(the holes must be filled, so run in solution mode or after implementing).

Writes to out/:
  pointmaps.png        - predicted vs GT pointmaps as 3D scatter, colored by confidence.
  reprojection.png     - the cross-view reprojection-consistency error map.
  cross_ablation.png   - the loss/error floor with vs without cross-attention.

Trains on the GPU when present (default_device); tensors feeding matplotlib are moved to CPU.
The cross-attention ablation trains a second model with the cross memory zeroed and compares
the floors, so the plot MEASURES whether cross-attention helps rather than asserting it.
"""

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from dataclasses import replace  # noqa: E402

_here = Path(__file__).parent
_impl = _here / "solution" if os.environ.get("NANOVISION_IMPL") == "solution" else _here
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_impl))

from config import GeometryFMConfig  # noqa: E402
from model import GeometryFM  # noqa: E402
from loss import pointmap_loss, normalize_scale  # noqa: E402
from toy_scene import stereo_pointmap_gt  # noqa: E402

from nanovision.determinism import default_device  # noqa: E402
from nanovision.geometry import reproject_pointmap  # noqa: E402

_OUT = _here / "out"
_OUT.mkdir(exist_ok=True)

_PAIRS = [(0, 3), (1, 4), (2, 5), (0, 4), (1, 5), (2, 6), (3, 6), (3, 7)]


def _build_batch(cfg, dev):
    batch = [stereo_pointmap_gt(replace(cfg, view1=a, view2=b)) for a, b in _PAIRS]
    i1 = torch.stack([d["img1"] for d in batch]).to(dev)
    i2 = torch.stack([d["img2"] for d in batch]).to(dev)
    g1 = torch.stack([d["gt_pts1"] for d in batch]).to(dev)
    g2 = torch.stack([d["gt_pts2"] for d in batch]).to(dev)
    v1 = torch.stack([d["valid1"] for d in batch]).to(dev)
    v2 = torch.stack([d["valid2"] for d in batch]).to(dev)
    meta = batch
    return (i1, i2, g1, g2, v1, v2), meta


def _train(cfg, batch, use_cross, dev, steps, seed=0):
    torch.manual_seed(seed)
    model = GeometryFM(cfg).to(dev)
    i1, i2, g1, g2, v1, v2 = batch
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    for _ in range(steps):
        opt.zero_grad()
        p1, c1, p2, c2 = model(i1, i2, use_cross=use_cross)
        loss = pointmap_loss(p1, p2, g1, g2, c1, c2, v1, v2, cfg.alpha)
        loss.backward()
        opt.step()
    return model, float(loss)


def _pointmap_error(model, batch, use_cross):
    i1, i2, g1, g2, v1, v2 = batch
    with torch.no_grad():
        p1, c1, p2, c2 = model(i1, i2, use_cross=use_cross)
        pred = torch.stack([p1, p2], dim=1)
        gt = torch.stack([g1, g2], dim=1)
        val = torch.stack([v1, v2], dim=1)
        z = normalize_scale(pred, val)
        zb = normalize_scale(gt, val)
        ell = (pred / z.view(-1, 1, 1, 1, 1) - gt / zb.view(-1, 1, 1, 1, 1)).norm(dim=-1)
        m = val.float()
        return float((ell * m).sum() / m.sum())


def main():
    dev = default_device()
    cfg = GeometryFMConfig()
    batch, meta = _build_batch(cfg, dev)
    steps = cfg.n_steps

    model, final_loss = _train(cfg, batch, use_cross=True, dev=dev, steps=steps)
    err_cross = _pointmap_error(model, batch, use_cross=True)

    model_nc, final_loss_nc = _train(cfg, batch, use_cross=False, dev=dev, steps=steps)
    err_nocross = _pointmap_error(model_nc, batch, use_cross=False)

    print(f"cross:   final_loss={final_loss:.4f} pointmap_err={err_cross:.4f}")
    print(f"nocross: final_loss={final_loss_nc:.4f} pointmap_err={err_nocross:.4f}")

    i1, i2, g1, g2, v1, v2 = batch
    with torch.no_grad():
        p1, c1, p2, c2 = model(i1, i2, use_cross=True)

    # Pointmap scatter for the first pair: predicted vs GT, colored by confidence.
    b = 0
    fig = plt.figure(figsize=(10, 5))
    for col, (pts, conf, gt, valid, name) in enumerate([
        (p1[b], c1[b], g1[b], v1[b], "view 1"),
        (p2[b], c2[b], g2[b], v2[b], "view 2"),
    ]):
        ax = fig.add_subplot(1, 2, col + 1, projection="3d")
        mask = valid.cpu().reshape(-1)
        P = pts.detach().cpu().reshape(-1, 3)[mask]
        G = gt.detach().cpu().reshape(-1, 3)[mask]
        C = conf.detach().cpu().reshape(-1)[mask]
        ax.scatter(G[:, 0], G[:, 1], G[:, 2], c="gray", marker="o", s=40, label="GT", alpha=0.5)
        sc = ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=C, cmap="viridis", marker="^", s=40,
                        label="pred")
        ax.set_title(f"{name} (cam1 frame)")
        ax.legend()
        fig.colorbar(sc, ax=ax, shrink=0.6, label="confidence")
    fig.tight_layout()
    fig.savefig(_OUT / "pointmaps.png", dpi=110)
    plt.close(fig)

    # Reprojection-consistency error map: pred view-2 points reprojected into image 2 vs the
    # patch centers image 2 actually observes.
    grid = cfg.img_size // cfg.patch
    cen = (torch.arange(grid).float() + 0.5) * cfg.patch - 0.5
    vs, us = torch.meshgrid(cen, cen, indexing="ij")
    centers = torch.stack([us, vs], dim=-1)  # (grid, grid, 2)
    T12 = meta[b]["T_1to2"].to(dev)
    K = meta[b]["K"].to(dev)
    with torch.no_grad():
        px = reproject_pointmap(p2[b:b + 1], T12, K)[0].cpu()
    err_map = (px - centers).norm(dim=-1)
    err_map = err_map * v2[b].cpu().float()
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(err_map.numpy(), cmap="magma")
    ax.set_title("reprojection error (px), view 2")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(_OUT / "reprojection.png", dpi=110)
    plt.close(fig)

    # Cross-attention ablation bar chart.
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["cross-attn", "no cross-attn"], [err_cross, err_nocross],
           color=["tab:green", "tab:red"])
    ax.set_ylabel("normalized pointmap error")
    ax.set_title("cross-attention lowers the error floor")
    fig.tight_layout()
    fig.savefig(_OUT / "cross_ablation.png", dpi=110)
    plt.close(fig)

    print(f"wrote figures to {_OUT}")


if __name__ == "__main__":
    main()
