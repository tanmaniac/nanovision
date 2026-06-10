"""Train the pixel reacher policies and the point-mass side-demo, and write the panels. Provided.

Run from the repo root with headless rendering:

    MUJOCO_GL=egl python -m assignments.a13_vla.viz

Writes panels to out/:

1. reacher_rollouts.png - frames from a reacher episode under the flow policy and the BC policy,
   both reading only the 64x64 camera image. Shows the finger reaching the target from pixels.
2. chunk_ablation.png - reach success vs chunk size H for behavior cloning from pixels, with the
   random-torque floor. The MEASURED ablation lives here, not in a unit test.
3. multimodal.png - the retained point-mass side-demo: with the goal hidden, the BC regressor's
   single predicted action collapses toward the origin (averaging the four goal directions) while
   the flow head's samples spread to the four directions. This is the generative-vs-regression
   lesson; the reacher cannot show it because the image fixes the target.
4. flow_path.png - the flow-matching ODE path: z_t integrated from t=0 (noise) to t=1 (action) for
   one conditioning.

GPU-aware via nanovision.determinism.default_device. dm_control is used here (viz is not graded);
the mechanism tests never import this file. Set MUJOCO_GL=egl for the reacher panels.
"""

import argparse
import os
import sys
from pathlib import Path

from nanovision.viz import SHOW, finish, plt  # noqa: E402  (sets the matplotlib backend)
import numpy as np  # noqa: E402
import torch  # noqa: E402

_here = Path(__file__).parent
_impl = _here / "solution" if os.environ.get("NANOVISION_IMPL") == "solution" else _here
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_impl))

import env as ENV  # noqa: E402
from _train import (  # noqa: E402
    build_batch_pointmass, build_pixel_batch, pixel_action_fn,
    pixel_rollout_success, train_bc, train_flow, train_pixel_bc, train_pixel_flow,
)
from config import VLAConfig  # noqa: E402
from flow import flow_sample  # noqa: E402

from nanovision.determinism import default_device  # noqa: E402

_OUT = _here / "out"
_OUT.mkdir(exist_ok=True)


def _collect_demos(cfg, n_success=200):
    demos = ENV.collect_demos(n_success=n_success, seed0=0, max_steps=cfg.max_steps)
    print(f"collected {demos['obs'].shape[0]} filtered demos, T={demos['T']}, "
          f"expert reach_frac={demos['reach_frac']:.3f} over {demos['n_tried']} episodes")
    return demos


def panel_reacher_rollouts(cfg, dev, demos):
    """Render frames of one reacher episode under the flow policy and the BC policy."""
    obs, a = build_pixel_batch(demos, cfg.chunk, device=dev)
    enc_f, flow = train_pixel_flow(obs, a, cfg, steps=3000, device=dev, seed=0)
    enc_b, bc = train_pixel_bc(obs, a, cfg, steps=3000, device=dev, seed=0)

    def run_and_grab(enc, head, kind, seed):
        fn = pixel_action_fn(enc, head, kind, cfg, device=dev)
        env = ENV.make_reacher(seed=seed)
        env.reset()
        frames = []
        steps = 0
        hit = False
        while steps < cfg.max_steps and not hit:
            frames.append(env.physics.render(96, 96, camera_id=0))
            o = ENV.render_obs(env)[None]
            chunk = fn(o)[0]
            for h in range(cfg.chunk):
                ts = env.step(np.clip(chunk[h], -1, 1))
                steps += 1
                if ENV.reached(ts):
                    hit = True
                    break
                if steps >= cfg.max_steps:
                    break
        frames.append(env.physics.render(96, 96, camera_id=0))
        return frames, hit

    rows = [("flow", enc_f, flow, "flow"), ("bc", enc_b, bc, "BC")]
    fig, axes = plt.subplots(2, 6, figsize=(15, 5.2))
    for r, (kind, enc, head, label) in enumerate(rows):
        frames, hit = run_and_grab(enc, head, kind, seed=2001)
        picks = np.linspace(0, len(frames) - 1, 6).astype(int)
        for col, fi in enumerate(picks):
            axes[r, col].imshow(frames[fi])
            axes[r, col].axis("off")
            axes[r, col].set_title(f"{label} t={fi}", fontsize=8)
        axes[r, 0].set_ylabel(label)
    fig.suptitle("Reacher controlled from 64x64 pixels: flow vs BC action head reaching the target")
    fig.tight_layout()
    finish(_OUT / "reacher_rollouts.png", dpi=110)


def panel_chunk_ablation(cfg, dev, demos, Hs=(1, 4, 8), seeds=(0, 1)):
    """Reach success vs chunk size for BC from pixels, against the random-torque floor."""
    means, stds = [], []
    for H in Hs:
        cfgH = VLAConfig(chunk=H)
        obs, a = build_pixel_batch(demos, H, device=dev)
        rates = []
        for s in seeds:
            enc, bc = train_pixel_bc(obs, a, cfgH, steps=3000, device=dev, seed=s)
            rates.append(pixel_rollout_success(enc, bc, "bc", cfgH, n=48, seed0=2000, device=dev))
        means.append(float(np.mean(rates)))
        stds.append(float(np.std(rates)))
    rand = ENV.random_reach_success(n=96, seed0=2000, max_steps=cfg.max_steps)

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.bar([str(h) for h in Hs], means, yerr=stds, capsize=5, color="C0", label="BC from pixels")
    ax.axhline(rand, ls="--", color="tab:red", label=f"random-torque floor ({rand:.2f})")
    ax.set_xlabel("chunk size H")
    ax.set_ylabel("reach success rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Chunk-size ablation: BC from pixels vs random (per-seed spread)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    finish(_OUT / "chunk_ablation.png")
    print("chunk ablation (BC from pixels) H=", list(Hs), "success:", [round(m, 3) for m in means],
          "std:", [round(s, 3) for s in stds], "random floor:", round(rand, 3))
    return means, stds, rand


def panel_multimodal(cfg, dev):
    """Retained point-mass side-demo: regressor averages multimodal actions, flow samples a mode."""
    demos = ENV.collect_demos_pointmass(400, seed=0, repr=cfg.repr, goal_conditioned=False)
    a, c = build_batch_pointmass(demos, cfg.chunk, device=dev)
    bcH = train_bc(a, c, cfg, steps=3000, device=dev)
    flowH = train_flow(a, c, cfg, steps=4000, device=dev)

    p = np.array([0.5, 0.5], dtype=np.float32)
    cq = ENV.make_condition(p, 0, goal_conditioned=False, repr=cfg.repr)
    cq = torch.from_numpy(cq[None]).float().to(dev)
    with torch.no_grad():
        bc_a = bcH(cq)[0, 0].cpu().numpy()
        samples = np.stack(
            [flow_sample(flowH, cq, cfg.chunk, cfg.n_flow_steps)[0, 0].cpu().numpy() for _ in range(200)]
        )

    fig, ax = plt.subplots(figsize=(5, 5))
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
    ax.set_title("Point-mass side-demo: goal hidden, regressor averages, flow samples")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    fig.tight_layout()
    finish(_OUT / "multimodal.png")
    print(f"point-mass multimodal: BC averaged |action|={np.linalg.norm(bc_a):.4f}  "
          f"flow sample spread std={samples.std(0).mean():.4f}")


def panel_flow_path(cfg, dev, demos):
    """The flow-matching ODE path: z_t from t=0 (noise) to t=1 (action) for one image conditioning."""
    obs, a = build_pixel_batch(demos, cfg.chunk, device=dev)
    enc, flowH = train_pixel_flow(obs, a, cfg, steps=3000, device=dev, seed=0)
    with torch.no_grad():
        ci = enc(obs[:1])
    n_steps = cfg.n_flow_steps
    z = torch.randn(1, cfg.chunk, cfg.act_dim, device=dev)
    dt = 1.0 / n_steps
    traj = [z.clone()]
    with torch.no_grad():
        for k in range(n_steps):
            t = torch.full((1, 1, 1), k * dt, device=dev)
            z = z + dt * flowH(z, t, ci)
            traj.append(z.clone())
    traj = torch.cat(traj, dim=0).cpu().numpy()
    ts = np.linspace(0, 1, n_steps + 1)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for h in range(cfg.chunk):
        ax.plot(ts, traj[:, h, 0], "-o", ms=3, label=f"chunk step {h}, torque 0")
    ax.set_xlabel("integration time t (0 = noise, 1 = action)")
    ax.set_ylabel("z_t (joint-0 torque)")
    ax.set_title("Flow-matching ODE path: noise transported to the torque chunk")
    ax.legend(fontsize=7)
    fig.tight_layout()
    finish(_OUT / "flow_path.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--n_demos", type=int, default=200)
    args = ap.parse_args()
    dev = torch.device(args.device) if args.device else default_device()
    torch.manual_seed(0)
    print(f"device: {dev}")
    cfg = VLAConfig()

    demos = _collect_demos(cfg, n_success=args.n_demos)
    panel_reacher_rollouts(cfg, dev, demos)
    panel_chunk_ablation(cfg, dev, demos)
    panel_flow_path(cfg, dev, demos)
    panel_multimodal(cfg, dev)
    print(f"wrote figures to {_OUT}")


if __name__ == "__main__":
    main()
    if SHOW:
        plt.show()
