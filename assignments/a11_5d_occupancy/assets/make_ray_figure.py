"""Figure for the notes: our hit ray through the toy occupancy scene.

Run from this directory: python make_ray_figure.py. Trains the occupancy and semantic grids by
rendering supervision exactly as the rendering test does (zero-logit init, Adam lr 0.1, 500
steps, reference solution), then plots one camera ray that hits a box: a top view of the scene
with the ray, and the trained occupancy and compositing weights along that ray against the
analytic first-hit depth.
"""

import os
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
_assign = _here.parent
sys.path.insert(0, str(_assign.parents[1]))          # repo root, for the nanovision package
os.environ["NANOVISION_IMPL"] = "solution"            # nanovision.volume loads the NeRF solution
sys.path.insert(0, str(_assign))                      # config.py
sys.path.insert(0, str(_assign / "solution"))         # reference occupancy.py

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from config import OccConfig  # noqa: E402
from occupancy import render_occupancy_rays  # noqa: E402

from nanovision.data import toy  # noqa: E402
from nanovision.volume import sample_along_rays  # noqa: E402


def main():
    torch.manual_seed(0)
    cfg = OccConfig()
    sc = toy.occupancy_toy_scene(grid=cfg.grid, bounds=cfg.grid_bounds, n_classes=cfg.n_classes,
                                 n_boxes=cfg.n_boxes, n_cams=cfg.n_cams, img=cfg.img, seed=0)
    ro, rd, gd, gs = sc["rays_o"], sc["rays_d"], sc["gt_depth"], sc["gt_sem"]
    R = ro.shape[0]
    z = (cfg.z_near + (cfg.z_far - cfg.z_near) * torch.linspace(0, 1, cfg.n_samples))
    z = z[None].expand(R, -1).contiguous()
    hit = gd < cfg.z_far

    ol = torch.zeros(*cfg.grid, requires_grad=True)
    sl = torch.zeros(cfg.n_classes, *cfg.grid, requires_grad=True)
    opt = torch.optim.Adam([ol, sl], lr=0.1)
    for _ in range(500):
        d, so, _ = render_occupancy_rays(torch.sigmoid(ol), sl, ro, rd, z, cfg.grid_bounds, cfg.z_far)
        loss = F.smooth_l1_loss(d, gd) + F.cross_entropy(so[hit], gs[hit])
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        occ = torch.sigmoid(ol)
        d, _, w = render_occupancy_rays(occ, sl, ro, rd, z, cfg.grid_bounds, cfg.z_far)
        (x0, x1), (y0, y1), (z0, z1) = cfg.grid_bounds
        pts = sample_along_rays(ro, rd, z)                                      # [R, N, 3]
        g = torch.stack([2 * (pts[..., 0] - x0) / (x1 - x0) - 1, 2 * (pts[..., 1] - y0) / (y1 - y0) - 1,
                         2 * (pts[..., 2] - z0) / (z1 - z0) - 1], -1)[None, :, :, None, :]
        o_all = F.grid_sample(occ[None, None], g, align_corners=False).reshape(R, -1)
        gt_all = F.grid_sample(sc["occ_gt"][None, None], g, mode="nearest", align_corners=False).reshape(R, -1)
        # Our hit ray: a ray that passes squarely through a box (longest chord inside it) among
        # those whose rendered depth error is no worse than the scene-wide mean error.
        ok = hit & ((d - gd).abs() <= (d - gd).abs().mean())
        r = int(torch.where(ok, gt_all.sum(-1), torch.full_like(gd, -1.0)).argmax())
        o_i, gt_i = o_all[r], gt_all[r]
    zr, wr = z[r], w[r]
    print(f"ray {r}: GT depth {gd[r]:.3f} m, rendered {d[r]:.3f} m, accumulated opacity {wr.sum():.3f}, "
          f"samples with o > 0.01: {(o_i > 0.01).sum().item()}, peak o {o_i.max():.3f}")

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.6]})
    foot = sc["sem_gt"].max(0).values.float()
    a0.imshow(foot.numpy(), origin="lower", extent=(x0, x1, y0, y1), cmap="Greys", vmin=0, vmax=3)
    cams = sc["E"][:, :3, 3]
    a0.scatter(cams[:, 0], cams[:, 1], c="tab:blue", marker="^", s=60, label="cameras")
    end = ro[r] + rd[r] * (gd[r] + 1.0)
    a0.plot([ro[r, 0], end[0]], [ro[r, 1], end[1]], c="tab:red", lw=1.5, label="our hit ray")
    hp = ro[r] + rd[r] * gd[r]
    a0.scatter([hp[0]], [hp[1]], c="tab:red", s=30, zorder=5)
    a0.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ls="--", ec="k", lw=0.8))
    a0.set_xlim(-7, 7); a0.set_ylim(-7, 7); a0.set_aspect("equal")
    a0.set_xlabel("x (m)"); a0.set_ylabel("y (m)")
    a0.set_title("Top view: box footprints, grid (dashed), cameras")
    a0.legend(loc="lower left", fontsize=8)

    a1.step(zr, gt_i, where="mid", c="k", lw=1.2, label="ground-truth occupancy along the ray")
    a1.plot(zr, o_i, c="tab:orange", lw=1.5, label="trained occupancy $o_i$")
    a1.bar(zr, wr, width=(zr[1] - zr[0]) * 0.8, color="tab:blue", alpha=0.6, label="compositing weight $w_i$")
    a1.axvline(gd[r], c="k", ls="--", lw=1, label=f"analytic depth {gd[r]:.2f} m")
    a1.axvline(d[r], c="tab:red", ls=":", lw=1.5, label=f"rendered depth {d[r]:.2f} m")
    a1.set_xlim(gd[r] - 2.5, gd[r] + 3.5); a1.set_ylim(0, 1.05)
    a1.set_xlabel("distance along the ray (m)")
    a1.set_title("Along the ray after training on depth and class only")
    a1.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(_here / "hit_ray.png", dpi=110)


if __name__ == "__main__":
    main()
