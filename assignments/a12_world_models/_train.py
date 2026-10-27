"""The DreamerV3 training loop for cartpole-from-pixels. Provided, not a hole.

Ported from r2dreamer (commit becf338): `dreamer.py` (the agent's update and acting), `optim/`
(LaProp and adaptive gradient clipping), `buffer.py` (replay with the online queue) and
`trainer.py` (collection, update schedule, evaluation). Plain PyTorch and NumPy.

Run from the repo root (GPU strongly recommended; the full run takes hours):

    python assignments/a12_world_models/_train.py --seed 0

Schedule, all in environment steps (agent steps x action repeat 2, summed over the 4 envs):
- Each loop iteration steps all 4 envs once (8 environment steps) and then runs
  batch_size * batch_length / train_ratio * action_repeat = 16 * 64 / 512 * 2 = 4 environment
  steps per update, so 2 updates per iteration once training has started.
- Training starts after batch_length + 2 = 66 iterations (528 environment steps).
- Evaluation runs at step 0 and every 10k steps: 2 eval envs, one episode each, greedy (mean)
  actions, recurrent state carried through the episode. The reported return is their mean.

Replay layout (the reference's, see `Replay`): row t of env e holds the frame o_t, the reward r_t
earned by arriving at o_t (0 on a reset frame), the flags of o_t, the action a_t chosen after
seeing o_t (0 when o_t is the last frame), and the posterior state (h_t, z_t) the acting policy
computed at o_t. A training window is 65 consecutive rows of one env: row 0 only supplies the
initial state (h_0, z_0), and rows 1..64 are the sequence, each paired with the previous row's
action. Windows run across episode boundaries; is_first resets the RSSM inside observe().
"""

import argparse
import collections
import copy
import os
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

_here = Path(__file__).resolve().parent
if os.environ.get("NANOVISION_IMPL") == "solution" and str(_here / "solution") not in sys.path:
    sys.path.insert(0, str(_here / "solution"))
if str(_here) not in sys.path:
    sys.path.insert(1, str(_here))

from actor_critic import Actor, Critic, ReturnEMA, imagination_losses, replay_value_loss  # noqa: E402
from config import Config, small_config  # noqa: E402
from env import CartpoleEnv, step_envs  # noqa: E402  (dm_control itself is imported when an env is built)
from world_model import WorldModel  # noqa: E402

MAX_CONSECUTIVE_NONFINITE = 1000


# ============================================================================ optimizer


class LaProp(Optimizer):
    """LaProp (Ziyin et al. 2020): Adam with the momentum taken AFTER dividing by the RMS.

    Ported from r2dreamer's optim/laprop.py (MIT, Z-T-WANG/LaProp-Optimizer), keeping only the
    path the agent uses (no amsgrad, no centering, no weight decay). Bias correction tracks the
    learning-rate-weighted EMA, so it stays correct while the warmup changes the learning rate.
    """

    def __init__(self, params, lr=4e-4, betas=(0.9, 0.999), eps=1e-15):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_lr_1"] = 0.0
                    state["exp_avg_lr_2"] = 0.0
                    state["exp_avg_sq"] = torch.zeros_like(p)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                state["exp_avg_lr_1"] = state["exp_avg_lr_1"] * beta1 + (1 - beta1) * lr
                state["exp_avg_lr_2"] = state["exp_avg_lr_2"] * beta2 + (1 - beta2)
                step_size = 1 / (state["exp_avg_lr_1"] / lr if lr != 0.0 else 1.0)
                denom = exp_avg_sq.div(state["exp_avg_lr_2"]).sqrt_().add_(group["eps"])
                exp_avg.mul_(beta1).add_(grad / denom, alpha=(1 - beta1) * lr)
                p.add_(exp_avg, alpha=-step_size)


@torch.no_grad()
def clip_grad_agc_(params, clip: float, pmin: float):
    """Adaptive gradient clipping, per tensor: scale g so that ||g|| <= clip * max(||p||, pmin)."""
    params = [p for p in params if p.grad is not None]
    if not params:
        return
    grads = [p.grad for p in params]
    pnorm = torch._foreach_norm(params, ord=2)
    gnorm = torch._foreach_norm(grads, ord=2)
    upper = torch._foreach_mul(torch._foreach_maximum(pnorm, pmin), clip)
    scale = torch._foreach_reciprocal(torch._foreach_maximum(torch._foreach_div(gnorm, upper), 1.0))
    torch._foreach_mul_(grads, scale)


# ============================================================================ agent


class Agent(nn.Module):
    """World model, actor, critic, slow critic and one optimizer over all trained parameters."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.wm = WorldModel(cfg)
        feat = self.wm.rssm.feat_dim
        self.actor = Actor(cfg, feat)
        self.critic = Critic(cfg, feat)
        self.slow_critic = copy.deepcopy(self.critic)
        for p in self.slow_critic.parameters():
            p.requires_grad_(False)
        self.return_ema = ReturnEMA(cfg.ret_ema_alpha, cfg.ret_quantiles)
        self._slow_updates = 0
        self.params = [p for n, p in self.named_parameters() if not n.startswith("slow_critic.")]
        self.opt = LaProp(self.params, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps)
        warmup = cfg.warmup
        self.sched = LambdaLR(self.opt, lambda step: min(1.0, (step + 1) / warmup) if warmup else 1.0)
        self.nf_streak = 0
        self.nf_skipped = 0
        self._grad_fn = self._loss_and_backward
        if cfg.compile:
            # The reference compiles the same function with CUDA graphs.
            self._grad_fn = torch.compile(self._loss_and_backward, mode="reduce-overhead")

    @property
    def device(self):
        return next(self.parameters()).device

    def autocast(self):
        if self.cfg.compute_dtype == "float32":
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=getattr(torch, self.cfg.compute_dtype))

    # ---------------------------------------------------------------- acting

    def initial_state(self, n: int):
        h, z = self.wm.rssm.initial(n, self.device)
        return h, z, torch.zeros(n, self.cfg.action_dim, device=self.device)

    @torch.no_grad()
    def act(self, obs: dict, state, greedy: bool):
        """One policy step. obs: image (n, 3, H, W) uint8 and is_first (n,) bool tensors.

        Returns (action (n, A), new state (h, z, action)). greedy takes the Gaussian's mean.
        """
        with self.autocast():
            h, z, prev_action = state
            embed = self.wm.encoder(obs["image"].float() / 255.0)
            h, z, _ = self.wm.rssm.obs_step(h, z, prev_action, embed, obs["is_first"])
            dist = self.actor(self.wm.rssm.get_feat(h, z))
            action = dist.mode if greedy else dist.rsample()
        return action, (h, z, action)

    # ---------------------------------------------------------------- learning

    def _loss_and_backward(self, data, h0, z0):
        cfg = self.cfg
        losses, h, z, feat = self.wm.loss(data, h0, z0)
        B, T = h.shape[:2]
        ac_losses, imag_ret, metrics = imagination_losses(
            self.wm, self.actor, self.critic, self.slow_critic, self.return_ema,
            h.reshape(B * T, -1), z.reshape(B * T, *z.shape[2:]), cfg)
        losses.update(ac_losses)
        # The reference rebuilds feat here rather than reusing it; doing the same keeps the gradient
        # accumulation order, and so the float rounding, identical to it.
        feat = self.wm.rssm.get_feat(h, z)
        losses["repval"] = replay_value_loss(self.critic, self.slow_critic, feat, data, imag_ret, cfg)
        total = sum(v * cfg.loss_scales[k] for k, v in losses.items())
        total.backward()
        return losses, metrics, total

    @torch.no_grad()
    def _update_slow_critic(self):
        if self._slow_updates % self.cfg.slow_target_update == 0:
            mix = self.cfg.slow_target_fraction
            for v, s in zip(self.critic.parameters(), self.slow_critic.parameters()):
                s.data.copy_(mix * v.data + (1 - mix) * s.data)
        self._slow_updates += 1

    def update(self, data: dict, h0, z0) -> dict:
        """One gradient step on a replayed batch. Returns detached scalar metrics."""
        self._update_slow_critic()          # before the loss, as in the reference
        if self.cfg.compile:
            torch.compiler.cudagraph_mark_step_begin()
        with self.autocast():
            losses, metrics, total = self._grad_fn(data, h0, z0)
        # Drop the whole update if any gradient is non-finite (the reference's guard; it raises
        # after 1000 consecutive drops instead of training on NaN).
        grads = [p.grad for p in self.params if p.grad is not None]
        n_bad = int((~torch.isfinite(torch.stack(torch._foreach_norm(grads)))).sum()) if grads else 0
        if n_bad:
            self.nf_streak += 1
            self.nf_skipped += 1
            if self.nf_streak > MAX_CONSECUTIVE_NONFINITE:
                raise RuntimeError(f"{self.nf_streak} consecutive updates had non-finite gradients")
        else:
            self.nf_streak = 0
            clip_grad_agc_(self.params, self.cfg.agc, self.cfg.pmin)
            self.opt.step()
            self.sched.step()
        self.opt.zero_grad(set_to_none=True)
        # clone(): under --compile these are CUDA-graph outputs that the next update overwrites.
        out = {f"loss/{k}": v.detach().clone() for k, v in losses.items()}
        out.update({k: v.detach().clone() for k, v in metrics.items()})
        out["loss/total"] = total.detach().clone()
        return out


# ============================================================================ replay


class Replay:
    """Ring buffer of transitions laid out (time, env), plus the reference's online queue.

    Capacity is replay_size // env_num rows per env (the reference's 2-D storage does the same).
    z is stored as class indices (uint8) since it is exactly one-hot; h as float32.

    Sampling: if at least batch_size fresh windows wait in the online queue, the batch is those
    windows (oldest first); otherwise batch_size windows are drawn uniformly over (env, start).
    The online queue holds the most recent 2 * batch_size non-overlapping windows of 65 rows per
    env; it lets training see new data right after collection.
    """

    def __init__(self, cfg: Config, rng: np.random.Generator):
        self.cfg, self.rng = cfg, rng
        self.E = cfg.env_num
        self.L = cfg.batch_length + 1
        self.cap = cfg.replay_size // cfg.env_num
        c, E, s = self.cap, self.E, cfg.obs_size
        self.data = {
            "image": np.zeros((c, E, 3, s, s), np.uint8),
            "action": np.zeros((c, E, cfg.action_dim), np.float32),
            "reward": np.zeros((c, E), np.float32),
            "is_first": np.zeros((c, E), bool),
            "is_last": np.zeros((c, E), bool),
            "is_terminal": np.zeros((c, E), bool),
            "h": np.zeros((c, E, cfg.deter), np.float32),
            "z": np.zeros((c, E, cfg.stoch), np.uint8),
        }
        self.rows = 0                                        # rows written (all envs advance together)
        self._pending = [0] * E
        self._pending_start = [0] * E
        self.queue = collections.deque(maxlen=2 * cfg.batch_size)

    def add(self, step: dict):
        """step: dict of (E, ...) numpy arrays for one row."""
        i = self.rows % self.cap
        for k, v in step.items():
            self.data[k][i] = v
        if self.cfg.online_queue:
            for e in range(self.E):
                if self._pending[e] == 0:
                    self._pending_start[e] = self.rows
                self._pending[e] += 1
                if self._pending[e] == self.L:
                    self.queue.append((e, self._pending_start[e]))
                    self._pending[e] = 0
        self.rows += 1

    def sample(self, device):
        B = self.cfg.batch_size
        if self.cfg.online_queue and len(self.queue) >= B:
            windows = [self.queue.popleft() for _ in range(B)]
            envs = np.array([w[0] for w in windows])
            starts = np.array([w[1] for w in windows])
        else:
            size = min(self.rows, self.cap)
            oldest = self.rows - size
            envs = self.rng.integers(0, self.E, size=B)
            starts = oldest + self.rng.integers(0, size - self.L + 1, size=B)
        rows = (starts[:, None] + np.arange(self.L)[None]) % self.cap               # (B, L)
        get = {k: v[rows, envs[:, None]] for k, v in self.data.items()}            # (B, L, ...)
        t = lambda x, dtype=None: torch.as_tensor(x, device=device, dtype=dtype)  # noqa: E731
        z0 = F.one_hot(t(get["z"][:, 0]).long(), self.cfg.discrete).float()
        data = {
            "image": channels_last_view(t(get["image"][:, 1:])).float() / 255.0,
            "action": t(get["action"][:, :-1]),                  # the action BEFORE each frame
            "reward": t(get["reward"][:, 1:, None]),
            "is_first": t(get["is_first"][:, 1:]),
            "is_last": t(get["is_last"][:, 1:, None], torch.float32),
            "is_terminal": t(get["is_terminal"][:, 1:, None], torch.float32),
        }
        return data, t(get["h"][:, 0]), z0


# ============================================================================ schedule helpers


class Every:
    """Returns how many multiples of `every` have passed since the last call (1 on the first call)."""

    def __init__(self, every):
        self._every, self._last = every, None

    def __call__(self, step):
        if not self._every:
            return 0
        if self._last is None:
            self._last = step
            return 1
        count = int((step - self._last) / self._every)
        self._last += self._every * count
        return count


class Once:
    def __init__(self):
        self._once = True

    def __call__(self):
        if self._once:
            self._once = False
            return True
        return False


# ============================================================================ loops


def channels_last_view(image):
    """Same values and shape (..., C, H, W), but stored channels-last, as a view.

    The reference keeps frames as (..., H, W, C) and permutes them to channels-first, so its
    convolutions see a channels-last memory layout. Convolution kernels pick a different algorithm
    for that layout, which changes float rounding (2.7e-6 relative in the gradients, measured on
    CPU). Matching the layout makes the port bit-identical to the reference.
    """
    return image.movedim(-3, -1).contiguous().movedim(-1, -3)


def _stamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _to_torch_obs(obs: dict, device) -> dict:
    return {"image": channels_last_view(torch.as_tensor(obs["image"], device=device)),
            "is_first": torch.as_tensor(obs["is_first"], device=device)}


def evaluate(agent: Agent, envs) -> np.ndarray:
    """One greedy episode per env, run together; returns the per-env returns."""
    n = len(envs)
    done = np.ones(n, bool)
    once_done = np.zeros(n, bool)
    returns = np.zeros(n, np.float64)
    state = agent.initial_state(n)
    act = state[2].clone()
    while not once_done.all():
        obs, reward, done = step_envs(envs, act.float().cpu().numpy(), done)
        act, state = agent.act(_to_torch_obs(obs, agent.device), state, greedy=True)
        returns += reward * ~once_done
        once_done |= done
    return returns


def train(cfg: Config, log=print, save_path=None) -> Agent:
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = torch.device(cfg.device)

    agent = Agent(cfg).to(device)
    n_params = sum(p.numel() for p in agent.params)
    log(f"{_stamp()} params {n_params:,} config {cfg}")
    envs = [CartpoleEnv(cfg.seed + i, cfg.action_repeat, cfg.time_limit, cfg.obs_size, cfg.task) for i in range(cfg.env_num)]
    eval_envs = [CartpoleEnv(cfg.seed + i, cfg.action_repeat, cfg.time_limit, cfg.obs_size, cfg.task)
                 for i in range(cfg.eval_episodes)]
    replay = Replay(cfg, rng)

    E, repeat = cfg.env_num, cfg.action_repeat
    updates_needed = Every(cfg.batch_size * cfg.batch_length / cfg.train_ratio * repeat)
    should_pretrain, should_log, should_eval = Once(), Every(cfg.log_every), Every(cfg.eval_every)

    step, n_updates = 0, 0
    done = np.ones(E, bool)
    returns = np.zeros(E, np.float64)
    lengths = np.zeros(E, np.int64)
    finished = []
    metrics = collections.defaultdict(list)
    state = agent.initial_state(E)
    act = state[2].clone()
    t0 = t_log = time.time()
    step_log = 0
    while step < cfg.steps:
        if should_eval(step) and cfg.eval_episodes > 0:
            t_eval = time.time()
            rets = evaluate(agent, eval_envs)
            log(f"{_stamp()} step {step} eval_return {rets.mean():.1f} episodes "
                f"{[round(float(r), 1) for r in rets]} eval_s {time.time() - t_eval:.0f} "
                f"elapsed_s {time.time() - t0:.0f}")
        for i in range(E):
            if done[i] and lengths[i] > 0:
                finished.append(returns[i])
                returns[i] = lengths[i] = 0
        step += int((~done).sum()) * repeat
        lengths += ~done

        obs, reward, done = step_envs(envs, act.float().cpu().numpy(), done)
        act, state = agent.act(_to_torch_obs(obs, device), state, greedy=False)
        replay.add({
            "image": obs["image"], "reward": reward, "is_first": obs["is_first"],
            "is_last": obs["is_last"], "is_terminal": obs["is_terminal"],
            # The action stored with the last frame of an episode is zeroed (never executed).
            "action": (act * torch.as_tensor(~done, device=device).unsqueeze(-1)).float().cpu().numpy(),
            "h": state[0].float().cpu().numpy(),
            "z": state[1].argmax(-1).to(torch.uint8).cpu().numpy(),
        })
        returns += reward

        if step // (E * repeat) > cfg.batch_length + 1:
            n = cfg.pretrain if should_pretrain() else updates_needed(step)
            for _ in range(n):
                for k, v in agent.update(*replay.sample(device)).items():
                    metrics[k].append(v)
            n_updates += n
            if should_log(step):
                now = time.time()
                fps = (step - step_log) / max(now - t_log, 1e-9)
                t_log, step_log = now, step
                parts = [f"{k} {torch.stack(v).float().mean().item():.3g}" for k, v in sorted(metrics.items())]
                train_ret = f"{np.mean(finished):.1f}" if finished else "nan"
                log(f"{_stamp()} step {step} updates {n_updates} env_steps_per_s {fps:.2f} "
                    f"train_return {train_ret} skipped {agent.nf_skipped} " + " ".join(parts))
                metrics.clear()
                finished = []
    log(f"{_stamp()} done step {step} updates {n_updates} elapsed_s {time.time() - t0:.0f}")
    if save_path:
        torch.save({"agent": agent.state_dict(), "config": cfg}, save_path)
        log(f"{_stamp()} saved {save_path}")
    return agent


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--task", default=None, help="dm_control cartpole task: swingup (default) or balance")
    p.add_argument("--env-num", type=int, default=None)
    p.add_argument("--eval-episodes", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--log-every", type=int, default=None)
    p.add_argument("--replay-size", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default=None, help="autocast dtype: bfloat16 (default) or float32")
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None,
                   help="torch.compile the update step (default: on for CUDA, off for CPU)")
    p.add_argument("--small", action="store_true", help="tiny model for smoke tests")
    p.add_argument("--save", default=None, help="path for the final agent checkpoint")
    a = p.parse_args()
    over = {k: v for k, v in dict(
        seed=a.seed, task=a.task, steps=a.steps, env_num=a.env_num, eval_episodes=a.eval_episodes,
        eval_every=a.eval_every, log_every=a.log_every, replay_size=a.replay_size,
        device=a.device, compute_dtype=a.dtype,
        compile=a.compile if a.compile is not None else a.device.startswith("cuda")).items()
        if v is not None}
    cfg = small_config(**over) if a.small else Config(**over)
    torch.set_float32_matmul_precision("high")   # as the reference's train.py
    train(cfg, log=lambda s: print(s, flush=True), save_path=a.save)


if __name__ == "__main__":
    main()
