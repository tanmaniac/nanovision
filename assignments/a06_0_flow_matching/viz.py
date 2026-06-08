"""Render flow matching on the 2D toy: the velocity field, trajectories under independent
vs OT coupling vs reflow, the straightness metric, few-step samples, and timestep
histograms. Provided.

Run from the repo root: `python -m assignments.a06_0_flow_matching.viz` (or
`make viz A=a06_0_flow_matching`). Writes figures to out/.

The trajectories under OT coupling and after one reflow step are visibly straighter than
under independent coupling, and the straightness metric drops accordingly. Few-step Euler
samples are recognizable with as few as 2-4 steps once the flow is straight.
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

from config import FlowConfig  # noqa: E402
from coupling import ot_coupling  # noqa: E402
from flow import cfm_loss  # noqa: E402
from model import VelocityMLP  # noqa: E402
from reflow import reflow_pairs  # noqa: E402
from sampling import euler_sample, straightness  # noqa: E402
from timesteps import sample_timesteps  # noqa: E402

from nanovision.data import toy  # noqa: E402

_OUT = _here / "out"
_OUT.mkdir(exist_ok=True)


def _data(cfg, n, g):
    return toy.eight_gaussians(n, generator=g) if cfg.toy == "8gauss" else toy.two_moons(n, generator=g)


def _train(cfg, steps, coupling, g, pairs=None):
    model = VelocityMLP(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(steps):
        if pairs is not None:                       # reflow: fixed (x0, x1_hat) pairs
            idx = torch.randint(0, pairs[0].shape[0], (cfg.batch,), generator=g)
            x0, x1 = pairs[0][idx], pairs[1][idx]
        else:
            x1 = _data(cfg, cfg.batch, g)
            x0 = torch.randn(cfg.batch, cfg.data_dim, generator=g)
            if coupling == "ot":
                x1, _ = ot_coupling(x0, x1)
        t = sample_timesteps(cfg.batch, "logit_normal", cfg.t_loc, cfg.t_scale, generator=g)
        opt.zero_grad()
        cfm_loss(model, x0, x1, t).backward()
        opt.step()
    return model


def _plot_trajectories(ax, model, cfg, g, title):
    x0 = torch.randn(200, cfg.data_dim, generator=g)
    with torch.no_grad():
        traj = euler_sample(model, x0, 50, return_traj=True).numpy()
    for i in range(0, 200, 3):
        ax.plot(traj[:, i, 0], traj[:, i, 1], lw=0.4, alpha=0.5, color="C0")
    data = _data(cfg, 400, g).numpy()
    ax.scatter(data[:, 0], data[:, 1], s=2, color="C1", alpha=0.5)
    ax.set_title(title); ax.set_aspect("equal"); ax.axis("off")


class _UNetCfg:
    img_size = 16
    channels = 1
    num_classes = 3
    base_width = 32


def _image_demo(steps=2500):
    """Flow matching at image scale, reusing A5's U-Net with the objective swapped to CFM.

    Same backbone as A5 diffusion; only the path, target, and sampler change. Samples with
    Euler from noise. Runs in solution mode (the shim loads A5's filled U-Net).
    """
    from nanovision.unet import TimeEmbeddedUNet

    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    cfg = _UNetCfg()
    model = TimeEmbeddedUNet(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    x1_all, labels_all = toy.diffusion_image_batch(64, num_classes=cfg.num_classes,
                                                   size=cfg.img_size, channels=cfg.channels, seed=0)
    for _ in range(steps):
        idx = torch.randint(0, 64, (16,), generator=g)
        x1, lab = x1_all[idx], labels_all[idx]
        x0 = torch.randn_like(x1)
        t = sample_timesteps(16, "logit_normal", generator=g)
        x_t = (1 - t.view(-1, 1, 1, 1)) * x0 + t.view(-1, 1, 1, 1) * x1
        pred = model(x_t, t, lab)
        opt.zero_grad(); ((pred - (x1 - x0)) ** 2).mean().backward(); opt.step()

    model.eval()
    cls = torch.arange(cfg.num_classes)
    fig, axes = plt.subplots(1, cfg.num_classes, figsize=(2 * cfg.num_classes, 2.2))
    with torch.no_grad():
        x0 = torch.randn(cfg.num_classes, cfg.channels, cfg.img_size, cfg.img_size,
                         generator=torch.Generator().manual_seed(1))
        out = euler_sample(lambda x, t: model(x, t, cls), x0, 20)
    for j in range(cfg.num_classes):
        axes[j].imshow(out[j, 0].numpy(), cmap="gray", vmin=-1, vmax=1)
        axes[j].set_title(f"CFM class {j}"); axes[j].axis("off")
    plt.tight_layout(); plt.savefig(_OUT / "image_cfm.png", dpi=120); plt.close()


def main():
    cfg = FlowConfig()
    g = torch.Generator().manual_seed(0)

    indep = _train(cfg, 4000, "independent", g)
    ot = _train(cfg, 4000, "ot", g)
    pairs = reflow_pairs(indep, 10000, cfg.data_dim, cfg.n_steps, generator=g)
    rect = _train(cfg, 4000, "independent", g, pairs=pairs)

    # Trajectories: independent vs OT vs 2-rectified.
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    _plot_trajectories(axes[0], indep, cfg, torch.Generator().manual_seed(1), "independent coupling")
    _plot_trajectories(axes[1], ot, cfg, torch.Generator().manual_seed(1), "OT coupling")
    _plot_trajectories(axes[2], rect, cfg, torch.Generator().manual_seed(1), "2-rectified (reflow)")
    plt.tight_layout(); plt.savefig(_OUT / "trajectories.png", dpi=120); plt.close()

    # Straightness metric for the three.
    x0 = torch.randn(512, cfg.data_dim, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        s = [straightness(m, x0, 50).item() for m in (indep, ot, rect)]
    plt.figure(figsize=(4.5, 3.2))
    plt.bar(["independent", "OT", "2-rectified"], s, color=["C0", "C2", "C3"])
    plt.ylabel("straightness (lower = straighter)"); plt.tight_layout()
    plt.savefig(_OUT / "straightness.png", dpi=120); plt.close()

    # Few-step samples from the OT model.
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    data = _data(cfg, 400, torch.Generator().manual_seed(3)).numpy()
    for ax, n in zip(axes, (1, 2, 4, 10, 100)):
        x0 = torch.randn(400, cfg.data_dim, generator=torch.Generator().manual_seed(4))
        with torch.no_grad():
            sm = euler_sample(ot, x0, n).numpy()
        ax.scatter(data[:, 0], data[:, 1], s=3, color="C1", alpha=0.3)
        ax.scatter(sm[:, 0], sm[:, 1], s=3, color="C0", alpha=0.6)
        ax.set_title(f"{n}-step Euler"); ax.set_aspect("equal"); ax.axis("off")
    plt.tight_layout(); plt.savefig(_OUT / "few_step.png", dpi=120); plt.close()

    # Timestep distributions.
    plt.figure(figsize=(4.5, 3.2))
    u = sample_timesteps(20000, "uniform", generator=g).numpy()
    ln = sample_timesteps(20000, "logit_normal", 0.0, 1.0, generator=g).numpy()
    plt.hist(u, bins=40, alpha=0.5, label="uniform", density=True)
    plt.hist(ln, bins=40, alpha=0.5, label="logit-normal", density=True)
    plt.xlabel("t"); plt.legend(); plt.tight_layout()
    plt.savefig(_OUT / "timesteps.png", dpi=120); plt.close()

    _image_demo()
    print(f"wrote figures to {_OUT}; straightness indep/OT/rect = {[round(x, 3) for x in s]}")


if __name__ == "__main__":
    main()
