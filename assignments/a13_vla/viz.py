"""Train the action heads on the point-mass reacher and visualize the VLA lessons. Provided.

Run from the repo root: `python -m assignments.a13_vla.viz` (uses the GPU when present). Writes
five panels to out/:

1. Rollout trajectories for single-step BC, chunked BC, and the flow head over the four goals.
2. Chunk-size ablation: rollout success and trajectory variance for H in {1, 4, 16}, with per-seed
   spread, against the straight-line expert ceiling. The headline ablation lives here (measured),
   not in a unit test.
3. Flow vs DDPM at matched training, plus a flow inference-step sweep (1, 2, 5, 10 Euler steps)
   against DDPM at its full step count.
4. The flow-matching ODE path: z_t from t=0 (noise) to t=1 (action) for one conditioning.
5. Multimodality in the goal-dropped mode: the BC regressor's single averaged action vs a scatter
   of flow-head samples clustering on the distinct goal directions. This is where the flow head's
   distribution modeling is load-bearing on this toy.

GPU-aware via nanovision.determinism.default_device; tensors handed to matplotlib are moved to CPU.
The tests never import this file.
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

_here = Path(__file__).parent
_impl = _here / "solution" if os.environ.get("NANOVISION_IMPL") == "solution" else _here
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_impl))

import env as ENV  # noqa: E402
from _train import (  # noqa: E402
    build_batch, rollout_success, train_bc, train_ddpm, train_flow,
)
from config import VLAConfig  # noqa: E402
from ddpm import ddpm_sample, make_schedule  # noqa: E402
from flow import flow_sample  # noqa: E402

from nanovision.determinism import default_device  # noqa: E402

_OUT = _here / "out"
_OUT.mkdir(exist_ok=True)


def _rollout_traj(head, kind, cfg, gi, p0, dev, *, max_steps=60, goal_conditioned=True):
    """One open-loop rollout; returns the (L+1, 2) state path."""
    abar = make_schedule(cfg.ddpm_T).to(dev) if kind == "ddpm" else None
    p = np.asarray(p0, dtype=np.float32).copy()
    goal = ENV.GOALS[gi]
    path = [p.copy()]
    steps = 0
    while steps < max_steps:
        c = ENV.make_condition(p, gi, goal_conditioned=goal_conditioned, repr=cfg.repr)
        ct = torch.from_numpy(c[None]).float().to(dev)
        with torch.no_grad():
            if kind == "flow":
                chunk = flow_sample(head, ct, cfg.chunk, cfg.n_flow_steps)[0].cpu().numpy()
            elif kind == "bc":
                chunk = head(ct)[0].cpu().numpy()
            elif kind == "ddpm":
                chunk = ddpm_sample(head, ct, cfg.chunk, abar)[0].cpu().numpy()
        for h in range(cfg.chunk):
            p = ENV.step_env(p, chunk[h], v_max=cfg.v_max)
            path.append(p.copy())
            steps += 1
            if np.linalg.norm(p - goal) < cfg.eps or steps >= max_steps:
                break
        if np.linalg.norm(p - goal) < cfg.eps:
            break
    return np.stack(path, axis=0)


def panel_rollouts(cfg, dev):
    demos = ENV.collect_demos(300, seed=0, repr=cfg.repr, goal_conditioned=True)
    a1, c1 = build_batch(demos, 1, device=dev)
    aH, cH = build_batch(demos, cfg.chunk, device=dev)

    cfg1 = VLAConfig(chunk=1)
    bc1 = train_bc(a1, c1, cfg1, steps=2000, device=dev)
    bcH = train_bc(aH, cH, cfg, steps=2000, device=dev)
    flowH = train_flow(aH, cH, cfg, steps=5000, device=dev)

    rng = np.random.default_rng(7)
    starts = []
    for _ in range(20):
        gi = int(rng.integers(0, cfg.n_goals))
        starts.append((gi, ENV.sample_start(gi, rng)))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (head, kind, cc, title) in zip(
        axes,
        [(bc1, "bc", cfg1, "single-step BC (H=1)"),
         (bcH, "bc", cfg, f"chunked BC (H={cfg.chunk})"),
         (flowH, "flow", cfg, f"flow head (H={cfg.chunk})")],
    ):
        for gi, p0 in starts:
            path = _rollout_traj(head, kind, cc, gi, p0, dev)
            ax.plot(path[:, 0], path[:, 1], lw=0.8, alpha=0.6, color="C0")
        ax.scatter(ENV.GOALS[:, 0], ENV.GOALS[:, 1], c="tab:green", s=80, marker="*", zorder=5)
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
    fig.suptitle("Open-loop rollouts: compounding drift under single-step BC vs chunked/flow")
    fig.tight_layout()
    fig.savefig(_OUT / "rollouts.png", dpi=120)
    plt.close(fig)


def panel_chunk_ablation(cfg, dev, seeds=(0, 1, 2)):
    # Ablate the deterministic BC policy across chunk sizes. BC isolates the compounding-error
    # lesson: a single-step policy (H=1) re-decides every step and drifts off the demo manifold,
    # while a longer chunk commits to an internally consistent segment. The flow head carries its
    # own sampling noise, which would confound this monotone trend, so it stays in the multimodal
    # panel where its distribution modeling is what matters.
    demos = ENV.collect_demos(300, seed=0, repr=cfg.repr, goal_conditioned=True)
    Hs = [1, 4, 16]
    means, stds = [], []
    for H in Hs:
        cfgH = VLAConfig(chunk=H)
        rates = []
        for s in seeds:
            a, c = build_batch(demos, H, device=dev)
            head = train_bc(a, c, cfgH, steps=4000, seed=s, device=dev)
            rates.append(rollout_success(head, "bc", cfgH, n=64, seed=100 + s, device=dev))
        means.append(float(np.mean(rates)))
        stds.append(float(np.std(rates)))
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar([str(h) for h in Hs], means, yerr=stds, capsize=5, color="C0")
    ax.axhline(1.0, ls="--", color="tab:green", label="straight-line expert ceiling")
    ax.set_xlabel("chunk size H")
    ax.set_ylabel("rollout success rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Chunk-size ablation (behavior cloning, per-seed spread)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(_OUT / "chunk_ablation.png", dpi=120)
    plt.close(fig)
    print("chunk ablation (BC) H={1,4,16} success:", [round(m, 3) for m in means],
          "std:", [round(s, 3) for s in stds])
    return means, stds


def panel_flow_vs_ddpm(cfg, dev):
    demos = ENV.collect_demos(300, seed=0, repr=cfg.repr, goal_conditioned=True)
    a, c = build_batch(demos, cfg.chunk, device=dev)
    flowH = train_flow(a, c, cfg, steps=5000, device=dev)
    ddpmH = train_ddpm(a, c, cfg, steps=5000, device=dev)

    flow_rate = rollout_success(flowH, "flow", cfg, n=64, seed=200, device=dev)
    ddpm_rate = rollout_success(ddpmH, "ddpm", cfg, n=64, seed=200, device=dev)

    # Flow inference-step sweep: reconstruction error on the training chunks vs number of Euler
    # steps. The TRAINED sampler is only approximately step-count-invariant, so it degrades at 1-2
    # steps; only the analytic constant-field oracle is exactly invariant (see test_flow_sample_ode).
    flow_steps = [1, 2, 5, 10]
    flow_mae = []
    for ns in flow_steps:
        with torch.no_grad():
            samp = flow_sample(flowH, c, cfg.chunk, ns)
        flow_mae.append((samp - a).abs().mean().item())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(["flow", "DDPM"], [flow_rate, ddpm_rate], color=["C0", "C3"])
    axes[0].axhline(1.0, ls="--", color="tab:green")
    axes[0].set_ylabel("rollout success rate")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title(f"flow vs DDPM (matched training)")
    axes[1].plot(flow_steps, flow_mae, "o-", color="C0")
    axes[1].set_xlabel("flow Euler steps")
    axes[1].set_ylabel("chunk reconstruction MAE")
    axes[1].set_title("flow few-step sweep (trained field, only approx. step-invariant)")
    fig.tight_layout()
    fig.savefig(_OUT / "flow_vs_ddpm.png", dpi=120)
    plt.close(fig)
    print(f"flow success={flow_rate:.3f} ddpm success={ddpm_rate:.3f} "
          f"flow MAE @ steps {flow_steps} = {[round(m, 3) for m in flow_mae]}")


def panel_flow_path(cfg, dev):
    demos = ENV.collect_demos(300, seed=0, repr=cfg.repr, goal_conditioned=True)
    a, c = build_batch(demos, cfg.chunk, device=dev)
    flowH = train_flow(a, c, cfg, steps=5000, device=dev)

    # Integrate one conditioning and record the z_t trajectory, plotting the first action dim.
    ci = c[:1]
    n_steps = cfg.n_flow_steps
    z = torch.randn(1, cfg.chunk, cfg.act_dim, device=dev)
    dt = 1.0 / n_steps
    traj = [z.clone()]
    with torch.no_grad():
        for k in range(n_steps):
            t = torch.full((1, 1, 1), k * dt, device=dev)
            z = z + dt * flowH(z, t, ci)
            traj.append(z.clone())
    traj = torch.cat(traj, dim=0).cpu().numpy()           # (n_steps+1, chunk, 2)
    ts = np.linspace(0, 1, n_steps + 1)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for h in range(cfg.chunk):
        ax.plot(ts, traj[:, h, 0], "-o", ms=3, label=f"step {h} vx")
    ax.set_xlabel("integration time t (0 = noise, 1 = action)")
    ax.set_ylabel("z_t (vx component)")
    ax.set_title("flow-matching ODE path: noise transported to the action chunk")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(_OUT / "flow_path.png", dpi=120)
    plt.close(fig)


def panel_multimodal(cfg, dev):
    # Goal-dropped mode: the goal is hidden from c, so from a fixed state the expert action points
    # to one of the four goals. A regressor averages them; the flow head samples a mode.
    demos = ENV.collect_demos(400, seed=0, repr=cfg.repr, goal_conditioned=False)
    a, c = build_batch(demos, cfg.chunk, device=dev)
    bcH = train_bc(a, c, cfg, steps=3000, device=dev)
    flowH = train_flow(a, c, cfg, steps=4000, device=dev)

    # Fixed query state at the center, goal hidden. The first action step is what we visualize.
    p = np.array([0.5, 0.5], dtype=np.float32)
    cq = ENV.make_condition(p, 0, goal_conditioned=False, repr=cfg.repr)
    cq = torch.from_numpy(cq[None]).float().to(dev)
    with torch.no_grad():
        bc_a = bcH(cq)[0, 0].cpu().numpy()
        samples = []
        for _ in range(200):
            s = flow_sample(flowH, cq, cfg.chunk, cfg.n_flow_steps)[0, 0].cpu().numpy()
            samples.append(s)
    samples = np.stack(samples, axis=0)

    fig, ax = plt.subplots(figsize=(5, 5))
    # Expert directions from the center toward each goal (first step), for reference.
    for gi in range(cfg.n_goals):
        ea = ENV.expert_action(p, ENV.GOALS[gi], v_max=cfg.v_max)
        ax.arrow(0, 0, ea[0], ea[1], color="tab:green", alpha=0.5, width=0.001,
                 head_width=0.006, length_includes_head=True,
                 label="expert modes" if gi == 0 else None)
    ax.scatter(samples[:, 0], samples[:, 1], s=8, alpha=0.4, color="C0", label="flow samples")
    ax.scatter([bc_a[0]], [bc_a[1]], c="C3", s=120, marker="X", zorder=6,
               label="BC regressor (averaged)")
    ax.axhline(0, color="k", lw=0.3)
    ax.axvline(0, color="k", lw=0.3)
    ax.set_xlabel("vx")
    ax.set_ylabel("vy")
    ax.set_title("Goal hidden: regressor averages the modes, flow samples them")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(_OUT / "multimodal.png", dpi=120)
    plt.close(fig)
    # The averaged BC action should sit near the origin (modes cancel); samples spread to corners.
    print(f"BC averaged |action|={np.linalg.norm(bc_a):.4f}  "
          f"flow sample spread std={samples.std(0).mean():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    dev = torch.device(args.device) if args.device else default_device()
    torch.manual_seed(0)
    print(f"device: {dev}")
    cfg = VLAConfig()

    panel_rollouts(cfg, dev)
    panel_chunk_ablation(cfg, dev)
    panel_flow_vs_ddpm(cfg, dev)
    panel_flow_path(cfg, dev)
    panel_multimodal(cfg, dev)
    print(f"wrote figures to {_OUT}")


if __name__ == "__main__":
    main()
