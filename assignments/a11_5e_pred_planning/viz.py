"""Train the multimodal trajectory head on the ambiguous BEV scene and visualize the modes.

Run from the repo root: `python -m assignments.a11_5e_pred_planning.viz` (CPU default; pass
`--device cuda` to use a GPU). Writes figures to out/.

Two figures:
1. The shared-context ambiguous scene under the working (annealed) recipe: the BEV feature grid,
   the agent, the ground-truth futures (the fan of intentions), and the K predicted modes colored
   by their softmax score. The modes should cover the distinct intentions.
2. A three-panel comparison on the same ambiguous batch: K=1 (mode averaging - one path through
   the middle of every intention), K=6 hard WTA (dead modes - only some intentions covered), and
   K=6 soft->hard annealed (every intention covered). This is the dead-mode -> coverage fix shown
   visually.

GPU-aware: the head and batch move to the chosen device; tensors handed to matplotlib are moved
back to CPU. The tests never import this file.
"""

import argparse
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

from _train import train_head  # noqa: E402
from config import PredConfig  # noqa: E402
from predict import min_ade  # noqa: E402

from nanovision.data import toy  # noqa: E402

_OUT = _here / "out"
_OUT.mkdir(exist_ok=True)


def _to_ego(local, yaw, start):
    """Map an agent-centric trajectory (T, 2) back to ego meters for plotting."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    R = torch.tensor([[c, -s], [s, c]])
    return (local @ R.T) + start


def _plot_scene(ax, scene, trajs, scores, title):
    """Draw the BEV blob, GT futures, and predicted modes (ego meters) on one axis."""
    bev = scene["bev_feat"]
    bev_x, bev_y, res = scene["bev_x"], scene["bev_y"], scene["res"]
    # BEV feature magnitude as a faint background (presence channel).
    mag = bev.norm(dim=0).cpu().numpy()
    extent = [bev_y[0], bev_y[1], bev_x[0], bev_x[1]]  # imshow x=ny(=y), y=nx(=x)
    ax.imshow(mag, origin="lower", extent=extent, cmap="Greys", alpha=0.5)

    yaws, starts = scene["agent_yaw"], scene["agents_xy"]
    fut_ego = scene["futures_ego"].cpu()
    # GT futures (one per agent), plotted in ego (y horizontal, x vertical).
    for i in range(fut_ego.shape[0]):
        ax.plot(fut_ego[i, :, 1], fut_ego[i, :, 0], color="tab:green", lw=2, alpha=0.7,
                label="GT future" if i == 0 else None)
    ax.scatter([starts[0, 1].item()], [starts[0, 0].item()], c="k", s=40, zorder=5, label="agent")

    # Predicted modes for agent 0, colored by softmax score.
    w = torch.softmax(scores[0], dim=0).cpu()
    cmap = plt.get_cmap("autumn")
    for k in range(trajs.shape[1]):
        ego = _to_ego(trajs[0, k].cpu(), yaws[0].cpu(), starts[0].cpu())
        ax.plot(ego[:, 1], ego[:, 0], color=cmap(1.0 - w[k].item()), lw=1.5,
                label="modes" if k == 0 else None)
    ax.set_xlabel("y / left (m)")
    ax.set_ylabel("x / forward (m)")
    ax.set_title(title)
    ax.set_aspect("equal")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=400)
    args = ap.parse_args()
    dev = torch.device(args.device)

    cfg = PredConfig()
    scene = toy.pred_toy_scene(channels=cfg.in_ch, horizon=cfg.horizon,
                               shared_context=True, seed=0, device=str(dev))
    gt = scene["futures_local"]

    # Figure 1: the working (annealed) head on the ambiguous scene.
    head = train_head(scene, n_modes=cfg.n_modes, recipe="annealed", steps=args.steps, seed=0,
                      tau0=cfg.tau0, anneal_frac=cfg.anneal_frac, device=str(dev))
    with torch.no_grad():
        trajs, scores = head(scene["bev_feat"], scene["centers"])
        ade = min_ade(trajs, gt).item()
    fig, ax = plt.subplots(figsize=(5, 5))
    _plot_scene(ax, scene, trajs, scores, f"annealed K={cfg.n_modes}  min_ade={ade:.3f} m")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(_OUT / "modes_annealed.png", dpi=120)
    plt.close(fig)

    # Figure 2: K=1 vs hard K=6 vs annealed K=6 on the same batch.
    # Seed 2 is a seed where hard WTA collapses to 2 winning modes (see test_wta_beats_single_mode).
    seed = 2
    configs = [
        ("K=1 (averaging)", 1, "hard"),
        ("K=6 hard (dead modes)", cfg.n_modes, "hard"),
        ("K=6 annealed (coverage)", cfg.n_modes, "annealed"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (title, K, recipe) in zip(axes, configs):
        h = train_head(scene, n_modes=K, recipe=recipe, steps=args.steps, seed=seed,
                       tau0=cfg.tau0, anneal_frac=cfg.anneal_frac, device=str(dev))
        with torch.no_grad():
            tr, sc = h(scene["bev_feat"], scene["centers"])
            a = min_ade(tr, gt).item()
        _plot_scene(ax, scene, tr, sc, f"{title}\nmin_ade={a:.3f} m")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Mode averaging, the dead-mode collapse, and the annealing fix")
    fig.tight_layout()
    fig.savefig(_OUT / "compare_recipes.png", dpi=120)
    plt.close(fig)
    print(f"wrote {_OUT / 'modes_annealed.png'} and {_OUT / 'compare_recipes.png'}")


if __name__ == "__main__":
    main()
