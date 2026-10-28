"""Figures and animations from a trained DreamerV3 cartpole agent. Provided.

Run from the repo root after training (see _train.py). The default run, cartpole-swingup for 200k
environment steps, takes about 3.2 h on one RTX 4080 laptop GPU at full clock with torch.compile
(on by default for CUDA), plus about 25 min the first time it compiles:

    python assignments/a12_world_models/_train.py --seed 0 \\
        --save assignments/a12_world_models/out/dreamer.pt | tee assignments/a12_world_models/out/train_s0.log
    make viz-mine A=a12_world_models          # your code; reads out/dreamer.pt and out/train*.log

`make viz` (NANOVISION_IMPL=solution) reads out/dreamer_solution.pt and out/train_solution*.log
instead, so a reference run and your own run can sit side by side; train the reference with
NANOVISION_IMPL=solution and those file names. --ckpt and --logs override the defaults.

Everything after training runs on the CPU (the dm_control renderer uses MUJOCO_GL=egl). Outputs in
out/:

1. learning_curve.png: eval return vs environment steps for every log found, against the
   uniform-random and zero-force baselines (50 seeded episodes each, the same env and reward
   accounting as the evaluation, without rendering; cached in
   out/baselines_<task>.json).
2. imagination_vs_real.png: a greedy-policy episode and a uniform-random episode. The world model
   observes the first 5 real frames, then predicts the next 60 open loop, fed only the recorded
   actions. Real frames over imagined frames, and real vs predicted reward.
3. imagination_divergence.png: per-pixel squared error between imagined and real frames as the
   open-loop horizon grows to 100 steps, averaged over 40 start points per episode, for the greedy
   and the random episode. Two references: the decoder's error on the same frames when it sees
   them (posterior reconstruction), and predicting that the last observed frame never changes.
   The bottom row is the same comparison for the predicted reward.
4. policy_vs_imagination.gif: the whole greedy episode in three panels: the real frames, the world
   model's imagination fed the recorded actions, and its imagination with the actor choosing the
   actions from the imagined states (the setting the actor is trained in).
   random_vs_imagination.gif: the uniform-random episode, real next to imagined.
"""

import argparse
import dataclasses
import glob
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

from nanovision.viz import SHOW, finish, plt  # noqa: E402  (sets the matplotlib backend)
import numpy as np  # noqa: E402
import torch  # noqa: E402

_here = Path(__file__).parent
_impl = _here / "solution" if os.environ.get("NANOVISION_IMPL") == "solution" else _here
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_impl))

import _train  # noqa: E402
from env import ACTION_REPEAT, TIME_LIMIT, CartpoleEnv, make_env  # noqa: E402
from nets import twohot_decode  # noqa: E402

_OUT = _here / "out"
_SUFFIX = "_solution" if os.environ.get("NANOVISION_IMPL") == "solution" else ""
CONTEXT = 5           # real frames the world model observes before imagining
SHOW_STEPS = 60       # open-loop steps in the still figure
DIV_HORIZON = 100     # open-loop steps in the divergence figure


# ============================================================================ helpers


def load_agent(path):
    """Agent from a _train.py checkpoint, on the CPU in float32 and without torch.compile."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dataclasses.replace(ckpt["config"], device="cpu", compute_dtype="float32", compile=False)
    agent = _train.Agent(cfg)
    agent.load_state_dict(ckpt["agent"])
    agent.eval()
    return agent, cfg


def parse_log(path):
    """(env steps, eval returns) from the 'step N eval_return X' lines of a _train.py log."""
    steps, rets = [], []
    for line in Path(path).read_text().splitlines():
        m = re.search(r"step (\d+) eval_return ([-\d.]+)", line)
        if m:
            steps.append(int(m.group(1)))
            rets.append(float(m.group(2)))
    return np.array(steps), np.array(rets)


def physics_return(seed, action_fn, task):
    """Return of one episode without rendering: the reward CartpoleEnv.step computes (force clipped
    to [-1, 1], action repeat 2, 500 agent steps), on the same seeded dm_control task. Rendering
    costs about 50 ms a frame on the CPU and does not affect the physics."""
    env = make_env(seed, task)
    env.reset()
    total = 0.0
    for _ in range(TIME_LIMIT // ACTION_REPEAT):
        a = np.clip(np.asarray(action_fn(), np.float64).reshape(1), -1.0, 1.0)
        for _ in range(ACTION_REPEAT):
            total += env.step(a).reward or 0.0
    return total


def measure_baselines(n_episodes, task):
    """Mean and std of uniform-random and zero-force returns, one episode per env seed 0..n-1."""
    rng = np.random.default_rng(0)
    out = {}
    for name, fn in [("random", lambda: rng.uniform(-1.0, 1.0, 1)), ("zero_force", lambda: np.zeros(1))]:
        rets = np.array([physics_return(seed, fn, task) for seed in range(n_episodes)])
        out[name] = {"mean": float(rets.mean()), "std": float(rets.std()), "n": int(len(rets)),
                     "returns": [round(float(r), 1) for r in rets], "task": task}
    return out


@torch.no_grad()
def run_episode(agent, policy, seed):
    """One episode. policy is 'greedy' (the agent's mean action) or 'random' (uniform in [-1, 1]).

    The agent filters every frame either way, so the posterior states are available for both.
    Returns frames (T, 3, 64, 64) uint8, actions (T, A) where actions[t] produced frame t (zero at
    t = 0), rewards (T,) where rewards[t] was earned by arriving at frame t, and the posterior
    states h (T, D), z (T, S, K).
    """
    env = CartpoleEnv(seed, task=agent.cfg.task)
    rng = np.random.default_rng(seed)
    state = agent.initial_state(1)
    obs, done = env.reset(), False
    act = torch.zeros(1, agent.cfg.action_dim)
    frames, actions, rewards, hs, zs = [], [], [], [], []
    reward, prev = 0.0, act
    while True:
        frames.append(obs["image"])
        actions.append(prev[0].numpy().copy())
        rewards.append(reward)
        tobs = {"image": torch.as_tensor(obs["image"])[None],
                "is_first": torch.tensor([obs["is_first"]])}
        act, state = agent.act(tobs, state, greedy=True)
        hs.append(state[0][0].clone())
        zs.append(state[1][0].clone())
        if policy == "random":
            act = torch.as_tensor(rng.uniform(-1.0, 1.0, (1, agent.cfg.action_dim)), dtype=torch.float32)
            state = (state[0], state[1], act)
        if done:
            break
        obs, reward, done = env.step(act[0].numpy())
        prev = act.clamp(-1, 1)
    return {"frames": np.stack(frames), "actions": np.stack(actions), "rewards": np.array(rewards, np.float32),
            "h": torch.stack(hs), "z": torch.stack(zs)}


@torch.no_grad()
def open_loop(agent, ep, starts, horizon):
    """Imagine `horizon` frames from each start s, launched from the posterior state at frame s - 1.

    The prior is fed the recorded actions: step k predicts frame s + k from actions[s + k].
    Returns imagined frames (N, horizon, 3, 64, 64) float in [0, 1] and rewards (N, horizon).
    """
    wm = agent.wm
    idx = torch.as_tensor(starts) - 1
    h, z = ep["h"][idx], ep["z"][idx]
    acts = torch.as_tensor(ep["actions"])
    frames, rewards = [], []
    for k in range(horizon):
        a = acts[torch.as_tensor(starts) + k]
        h, z = wm.rssm.img_step(h, z, a)
        frames.append(wm.decoder(h, z))
        rewards.append(twohot_decode(wm.reward(wm.rssm.get_feat(h, z)), wm.bins)[:, 0])
    return torch.stack(frames, 1), torch.stack(rewards, 1)


@torch.no_grad()
def posterior_recon(agent, ep):
    """Decoder output from the posterior state of every frame (the model sees each frame)."""
    wm = agent.wm
    recon = torch.cat([wm.decoder(ep["h"][i:i + 100], ep["z"][i:i + 100]) for i in range(0, len(ep["h"]), 100)])
    feat = wm.rssm.get_feat(ep["h"], ep["z"])
    return recon, twohot_decode(wm.reward(feat), wm.bins)[:, 0]


def _real(ep):
    return torch.as_tensor(ep["frames"]).float() / 255.0


def _hwc(img):
    return img.permute(1, 2, 0).clamp(0, 1).numpy()


# ============================================================================ figures


def fig_learning_curve(logs, baselines, task):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    curves = [parse_log(p) for p in logs]
    curves = [(p, s, r) for p, (s, r) in zip(logs, curves) if len(s)]
    for i, (p, s, r) in enumerate(curves):
        ax.plot(s, r, "-o", ms=3.5, lw=1.4, color=f"C{i}", alpha=0.85, label=Path(p).stem)
    if len(curves) > 1:
        n = min(len(s) for _, s, _ in curves)
        mean = np.mean([r[:n] for _, _, r in curves], 0)
        ax.plot(curves[0][1][:n], mean, "-", lw=2.6, color="black", label=f"mean of {len(curves)} runs")
    ax.axhline(1000, color="0.6", lw=1, ls=":")
    ax.text(0, 1008, "ceiling 1000", fontsize=8, color="0.4", va="bottom")
    for key, color, label in [("zero_force", "tab:red", "zero force"), ("random", "tab:purple", "uniform random")]:
        b = baselines[key]
        ax.axhline(b["mean"], color=color, ls="--", lw=1.2,
                   label=f"{label}: {b['mean']:.0f} +- {b['std']:.0f} ({b['n']} episodes)")
    ax.set_xlabel("environment steps (action repeat 2 counted)")
    ax.set_ylabel("eval return (greedy, 2 episodes)")
    ax.set_ylim(0, 1080)
    ax.set_title(f"DreamerV3 on cartpole-{task} from 64x64 pixels")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return finish(_OUT / "learning_curve.png")


def fig_imagination_vs_real(agent, episodes):
    """For each episode: real frames over open-loop imagined frames, then real vs predicted reward."""
    start = CONTEXT
    ks = [1] + list(range(5, SHOW_STEPS + 1, 5))                       # open-loop steps shown
    ncol = len(ks) + 1
    fig = plt.figure(figsize=(1.05 * ncol, 4.6 * len(episodes)))
    outer = fig.add_gridspec(len(episodes), 1, hspace=0.28)
    for e, (name, ep) in enumerate(episodes.items()):
        imag, imag_rew = open_loop(agent, ep, [start], SHOW_STEPS)
        real = _real(ep)
        gs = outer[e].subgridspec(3, ncol, height_ratios=[1, 1, 1.4], hspace=0.3, wspace=0.06)
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(_hwc(real[start - 1]))
        ax.set_title(f"t={start - 1}\nlast observed", fontsize=7, color="tab:blue")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel("real", fontsize=9)
        for j, k in enumerate(ks):
            t = start + k - 1
            ax = fig.add_subplot(gs[0, j + 1])
            ax.imshow(_hwc(real[t]))
            ax.set_title(f"t={t}", fontsize=7)
            ax.axis("off")
            ax = fig.add_subplot(gs[1, j + 1])
            ax.imshow(_hwc(imag[0, k - 1]))
            ax.set_title(f"imagined +{k}", fontsize=7, color="tab:orange")
            ax.axis("off")
        ax = fig.add_subplot(gs[2, :])
        t_all = np.arange(0, start + SHOW_STEPS)
        ax.plot(t_all, ep["rewards"][t_all], "-", color="black", lw=1.6, label="real reward")
        ax.plot(np.arange(start, start + SHOW_STEPS), imag_rew[0].numpy(), "-", color="tab:orange", lw=1.6,
                label="predicted reward, open loop")
        ax.axvspan(-0.5, start - 0.5, color="tab:blue", alpha=0.12, label=f"{CONTEXT} observed frames")
        ax.set_ylim(-0.1, 2.1)
        ax.set_xlim(-0.5, start + SHOW_STEPS)
        ax.set_ylabel("reward / step", fontsize=9)
        ax.set_title(f"{name} episode (return {ep['rewards'].sum():.0f}): the world model sees frames 0-{start - 1}, "
                     f"then gets only the recorded actions", fontsize=9, loc="left")
        ax.legend(fontsize=7.5, loc="upper left", ncol=3)
        ax.grid(alpha=0.3)
    ax.set_xlabel("agent step t (one agent step = 2 physics steps)")
    fig.subplots_adjust(top=0.96, bottom=0.05, left=0.06, right=0.98)
    return finish(_OUT / "imagination_vs_real.png")


def divergence_curves(agent, ep, horizon):
    """Per-step pixel MSE of open-loop frames, posterior reconstructions, and a frozen last frame."""
    real = _real(ep)
    T = len(real)
    starts = list(range(CONTEXT, T - horizon, 10))
    imag, imag_rew = open_loop(agent, ep, starts, horizon)
    recon, post_rew = posterior_recon(agent, ep)
    idx = torch.as_tensor(starts)[:, None] + torch.arange(horizon)[None]          # (N, horizon)
    target = real[idx]
    mse = lambda a, b: ((a - b) ** 2).mean((-3, -2, -1))  # noqa: E731
    rew = torch.as_tensor(ep["rewards"])
    return {
        "imag": mse(imag, target).numpy(),
        "recon": mse(recon[idx], target).numpy(),
        "frozen": mse(real[torch.as_tensor(starts) - 1][:, None], target).numpy(),
        "rew_imag": (imag_rew - rew[idx]).abs().numpy(),
        "rew_post": (post_rew[idx] - rew[idx]).abs().numpy(),
        "n_starts": len(starts),
    }


def fig_divergence(curves):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True)
    k = np.arange(1, DIV_HORIZON + 1)
    for col, (name, c) in enumerate(curves.items()):
        ax = axes[0, col]
        for key, color, label in [("imag", "tab:orange", "open-loop imagination"),
                                  ("frozen", "0.5", "last observed frame held fixed"),
                                  ("recon", "tab:blue", "posterior reconstruction (sees the frame)")]:
            m = c[key].mean(0)
            lo, hi = np.percentile(c[key], [25, 75], axis=0)
            ax.plot(k, m, color=color, lw=1.8, label=label)
            ax.fill_between(k, lo, hi, color=color, alpha=0.15, lw=0)
        ax.set_title(f"{name} policy ({c['n_starts']} start points)", fontsize=10)
        ax.set_ylabel("pixel MSE (frames in [0, 1])")
        ax.grid(alpha=0.3)
        ax = axes[1, col]
        for key, color, label in [("rew_imag", "tab:orange", "open-loop imagination"),
                                  ("rew_post", "tab:blue", "posterior (sees the frame)")]:
            ax.plot(k, c[key].mean(0), color=color, lw=1.8, label=label)
        ax.set_xlabel("open-loop steps since the last observed frame")
        ax.set_ylabel("|predicted - real reward|")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    fig.suptitle("Where imagination drifts from reality (mean, band = 25th-75th percentile over start points)",
                 fontsize=10)
    fig.tight_layout()
    return finish(_OUT / "imagination_divergence.png")


@torch.no_grad()
def actor_loop(agent, ep, start, horizon):
    """Imagine `horizon` frames from the posterior state at frame start - 1, with the actor choosing
    each action (its mean) from the imagined state, as in the imagination the actor is trained on."""
    wm = agent.wm
    h, z = ep["h"][start - 1:start], ep["z"][start - 1:start]
    frames, rewards = [], []
    for _ in range(horizon):
        a = agent.actor(wm.rssm.get_feat(h, z)).mode
        h, z = wm.rssm.img_step(h, z, a)
        frames.append(wm.decoder(h, z))
        rewards.append(twohot_decode(wm.reward(wm.rssm.get_feat(h, z)), wm.bins)[:, 0])
    return torch.stack(frames, 1), torch.stack(rewards, 1)


def make_gif(agent, ep, path, title, with_actor=False, scale=3, every=2):
    """A whole episode as an animated GIF: the real frames next to the world model's imagination of
    the same episode (open loop after CONTEXT observed frames, fed the recorded actions). With
    with_actor, a third panel lets the actor drive the imagination instead."""
    from PIL import Image, ImageDraw

    T = len(ep["frames"])
    panels = [("recorded actions", *open_loop(agent, ep, [CONTEXT], T - CONTEXT))]
    if with_actor:
        panels.append(("actor in the loop", *actor_loop(agent, ep, CONTEXT, T - CONTEXT)))
    real = ep["frames"]
    S = real.shape[-1] * scale
    pad, head = 6, 44
    n = 1 + len(panels)
    frames = []
    for t in range(0, T, every):
        canvas = Image.new("RGB", (n * S + (n + 1) * pad, S + head + pad), (255, 255, 255))
        d = ImageDraw.Draw(canvas)
        d.text((pad, 3), f"{title}   t={t}   (5 observed frames, then imagination only)", fill=(0, 0, 0))
        r = Image.fromarray(np.transpose(real[t], (1, 2, 0))).resize((S, S), Image.NEAREST)
        canvas.paste(r, (pad, head))
        d.text((pad, 17), "real", fill=(0, 0, 0))
        d.text((pad, 29), f"reward {ep['rewards'][t]:.2f}", fill=(0, 0, 0))
        for i, (label, imag, rew) in enumerate(panels):
            x = pad + (i + 1) * (S + pad)
            if t >= CONTEXT:
                im = (imag[0, t - CONTEXT].permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
                canvas.paste(Image.fromarray(im).resize((S, S), Image.NEAREST), (x, head))
                d.text((x, 17), f"imagined, {label}", fill=(200, 90, 0))
                d.text((x, 29), f"+{t - CONTEXT + 1} steps  reward {float(rew[0, t - CONTEXT]):.2f}", fill=(200, 90, 0))
            else:
                canvas.paste(r, (x, head))
                d.text((x, 17), "observed", fill=(30, 100, 200))
        frames.append(canvas.quantize(colors=96, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=80, loop=0, optimize=True)
    return path


# ============================================================================ main


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ckpt", default=str(_OUT / f"dreamer{_SUFFIX}.pt"))
    p.add_argument("--logs", nargs="*", default=None,
                   help=f"training logs (default: out/train{_SUFFIX}*.log)")
    p.add_argument("--seed", type=int, default=100, help="env seed of the visualized episodes")
    p.add_argument("--baseline-episodes", type=int, default=50)
    a = p.parse_args()
    _OUT.mkdir(exist_ok=True)
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "4"))))

    if not Path(a.ckpt).exists():
        sys.exit(f"no checkpoint at {a.ckpt}; train first (see the docstring of viz.py)")
    agent, cfg = load_agent(a.ckpt)

    base_path = _OUT / f"baselines_{cfg.task}.json"
    if base_path.exists() and json.loads(base_path.read_text())["random"]["n"] == a.baseline_episodes:
        baselines = json.loads(base_path.read_text())
    else:
        baselines = measure_baselines(a.baseline_episodes, cfg.task)
        base_path.write_text(json.dumps(baselines, indent=1))
    print({k: (round(v["mean"], 1), round(v["std"], 1)) for k, v in baselines.items()})

    logs = a.logs if a.logs is not None else sorted(glob.glob(str(_OUT / f"train{_SUFFIX}*.log")))
    if logs:
        print("wrote", fig_learning_curve(logs, baselines, cfg.task))

    torch.manual_seed(0)
    greedy = run_episode(agent, "greedy", a.seed)
    rand = run_episode(agent, "random", a.seed)
    print(f"greedy episode return {greedy['rewards'].sum():.1f}, random episode return {rand['rewards'].sum():.1f} "
          f"(env seed {a.seed}, CPU float32)")
    print("wrote", fig_imagination_vs_real(agent, {"greedy policy": greedy, "uniform-random": rand}))
    curves = {"greedy": divergence_curves(agent, greedy, DIV_HORIZON),
              "uniform-random": divergence_curves(agent, rand, DIV_HORIZON)}
    for name, c in curves.items():
        for k in (1, 15, 45, 100):
            print(f"{name} k={k}: imag {c['imag'][:, k - 1].mean():.5f} frozen {c['frozen'][:, k - 1].mean():.5f} "
                  f"recon {c['recon'][:, k - 1].mean():.5f} |rew err| imag {c['rew_imag'][:, k - 1].mean():.3f} "
                  f"post {c['rew_post'][:, k - 1].mean():.3f}")
    print("wrote", fig_divergence(curves))
    T = len(greedy["frames"])
    _, rew_open = open_loop(agent, greedy, [CONTEXT], T - CONTEXT)
    _, rew_actor = actor_loop(agent, greedy, CONTEXT, T - CONTEXT)
    print(f"greedy episode, predicted return of the imagined steps: recorded actions {rew_open.sum():.1f}, "
          f"actor in the loop {rew_actor.sum():.1f}, real {greedy['rewards'][CONTEXT:].sum():.1f}")
    for ep, name, title, with_actor in [(greedy, "policy_vs_imagination.gif", "greedy policy", True),
                                        (rand, "random_vs_imagination.gif", "random actions", False)]:
        path = make_gif(agent, ep, _OUT / name, title, with_actor)
        print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
    if SHOW:
        plt.show()
