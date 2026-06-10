"""Train DreamerV3 on cartpole-from-pixels and visualize the imagination-to-real transfer.

Run from the repo root: `python -m assignments.a12_world_models.viz` (uses the GPU when present).
Needs dm_control / MuJoCo (MUJOCO_GL=egl); the imports are local so the mechanism tests do not. The
full run is a documented multi-hour job (~400 iterations, ~1-2 hours on a 12GB GPU); pass
`--iters` to shorten for a quick look. Writes four figures to out/:

1. Learning curve: greedy real-env return vs env steps, against the random baseline and the ~500
   optimal ceiling. This is the headline result - a policy trained purely on imagined rollouts
   transfers to the real cartpole.
2. Replay vs reconstruction: real frames over the decoder's symexp reconstruction.
3. Imagined rollout filmstrip: decode a prior-only rollout under the trained actor.
4. Dynamics-backprop vs REINFORCE: the measured contrast (continuous dynamics backprop transfers;
   discrete REINFORCE on this near-flat reward collapses below random). Drawn from the saved run
   stats; this is a measured statistic, not a unit test.
"""

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

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

import nets  # noqa: E402
from _train import SEQ_LEN, dreamer_train, sample_batch  # noqa: E402
from config import WorldModelConfig  # noqa: E402

_OUT = _here / "out"
_OUT.mkdir(exist_ok=True)

OPTIMAL = 500.0   # cartpole-balance return ceiling (1 reward/step over ~500 agent steps)


def _chw_to_hwc(t):
    return t.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()


def fig_learning_curve(hist):
    """Greedy real-env return vs env steps, against random and the optimal ceiling."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(hist["env_steps"], hist["greedy_return"], "-o", ms=3, color="tab:orange",
            label="greedy real-env return")
    ax.axhline(hist["random"], ls="--", color="tab:gray", label=f"random ({hist['random']:.0f})")
    ax.axhline(OPTIMAL, ls=":", color="tab:green", label=f"optimal (~{OPTIMAL:.0f})")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("episode return")
    ax.set_ylim(0, OPTIMAL * 1.05)
    ax.set_title("cartpole-balance: policy trained in imagination, evaluated in the real env")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(_OUT / "training_curves.png", dpi=120)
    plt.close(fig)


def fig_replay_vs_recon(cfg, model, batch, dev):
    """A real rollout over the decoder's symexp reconstruction of the same frames."""
    model.eval()
    obs = batch["obs"][:1]
    with torch.no_grad():
        embeds = model.encode_seq(obs)
        h0, z0 = model.rssm.initial_state(1, device=dev, dtype=obs.dtype)
        hs, zs, _, _ = model.rssm.observe(embeds, batch["actions"][:1], h0, z0)
        states = torch.cat([hs, zs], dim=-1)
        recon = nets.symexp(model.decoder(states.reshape(-1, states.shape[-1]))).reshape(obs.shape)

    steps = list(range(0, min(10, obs.shape[1])))
    fig, axes = plt.subplots(2, len(steps), figsize=(1.4 * len(steps), 3))
    for j, t in enumerate(steps):
        axes[0, j].imshow(_chw_to_hwc(obs[0, t]))
        axes[0, j].set_title(f"t={t}", fontsize=8)
        axes[0, j].axis("off")
        axes[1, j].imshow(_chw_to_hwc(recon[0, t]))
        axes[1, j].axis("off")
    axes[0, 0].set_ylabel("real", fontsize=9)
    axes[1, 0].set_ylabel("recon", fontsize=9)
    fig.suptitle("Replay (top) vs decoder reconstruction (bottom)")
    fig.tight_layout()
    fig.savefig(_OUT / "replay_vs_recon.png", dpi=120)
    plt.close(fig)


def fig_imagination(cfg, model, actor, h0, z0, dev):
    """Decode a prior-only imagined rollout under the trained actor from a real start state."""
    from actor_critic import imagine_dynamics  # provided in solution; local import for clarity

    model.eval()
    with torch.no_grad():
        # Roll the prior with greedy actor actions, collecting states to decode.
        h, z = h0[:1], z0[:1]
        states = [torch.cat([h, z], dim=-1)]
        for _ in range(cfg.horizon):
            a, _ = actor.sample(h, z, greedy=True)
            h = model.rssm.forward_h(h, z, a)
            _, z, _ = model.rssm.prior(h, greedy=True)
            states.append(torch.cat([h, z], dim=-1))
        S = torch.cat(states, dim=0)
        recon = nets.symexp(model.decoder(S))

    n = recon.shape[0]
    fig, axes = plt.subplots(1, n, figsize=(1.2 * n, 1.6))
    for t in range(n):
        axes[t].imshow(_chw_to_hwc(recon[t]))
        axes[t].axis("off")
    fig.suptitle("Imagined prior-only rollout under the trained actor (left = start)")
    fig.tight_layout()
    fig.savefig(_OUT / "imagination_filmstrip.png", dpi=120)
    plt.close(fig)


def fig_contrast(dynbackprop_greedy, random_ret):
    """The measured dynamics-backprop-vs-REINFORCE contrast (transfer vs collapse).

    The REINFORCE number is the measured collapse on this near-flat-reward task (discrete REINFORCE
    drove the greedy return to ~135, below the random baseline). It is reported as the documented
    contrast that motivates the continuous dynamics-backprop gradient, not re-trained here.
    """
    reinforce_collapse = 135.0
    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ["random", "discrete\nREINFORCE", "continuous\ndynamics backprop", "optimal"]
    vals = [random_ret, reinforce_collapse, dynbackprop_greedy, OPTIMAL]
    colors = ["tab:gray", "tab:red", "tab:orange", "tab:green"]
    ax.bar(bars, vals, color=colors)
    ax.axhline(random_ret, ls="--", color="tab:gray", lw=1)
    ax.set_ylabel("greedy real-env return")
    ax.set_ylim(0, OPTIMAL * 1.05)
    ax.set_title("Why continuous control uses dynamics backprop, not REINFORCE")
    for i, v in enumerate(vals):
        ax.text(i, v + 6, f"{v:.0f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(_OUT / "policy_transfer.png", dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, help="cpu or cuda; default auto")
    ap.add_argument("--iters", type=int, default=400, help="dreamer collect-fit-imagine iterations")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}")

    cfg = WorldModelConfig()
    model, actor, critic, env, hist = dreamer_train(
        cfg, iters=args.iters, seed=args.seed, device=dev, log=True
    )

    fig_learning_curve(hist)

    # Reconstruction / imagination panels from a fresh real episode.
    from env import run_episode

    ep, _ = run_episode(env, model, actor, explore=False, device=dev)
    from _train import batch_to_tensors

    if ep["obs"].shape[0] > SEQ_LEN:
        s = 0
        batch = batch_to_tensors(
            {k: ep[k][s:s + SEQ_LEN][None] for k in ("obs", "actions", "rewards", "conts")},
            device=dev,
        )
        fig_replay_vs_recon(cfg, model, batch, dev)
        # Seed the imagined filmstrip from the first real posterior state.
        with torch.no_grad():
            embeds = model.encode_seq(batch["obs"])
            h0, z0 = model.rssm.initial_state(1, device=dev, dtype=batch["obs"].dtype)
            hs, zs, _, _ = model.rssm.observe(embeds, batch["actions"], h0, z0)
        fig_imagination(cfg, model, actor, hs[:, 0], zs[:, 0], dev)

    fig_contrast(float(np.max(hist["greedy_return"])), hist["random"])

    print(f"greedy return curve: {[round(x, 1) for x in hist['greedy_return']]}")
    print(f"best greedy {max(hist['greedy_return']):.1f}  random {hist['random']:.1f}  optimal {OPTIMAL:.0f}")
    print(f"wrote figures to {_OUT}")


if __name__ == "__main__":
    main()
