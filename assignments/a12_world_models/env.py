"""dm_control cartpole-balance from 64x64 pixels. Provided, not a hole.

This wraps the DeepMind Control Suite cartpole-balance task so the rest of the assignment sees a
single continuous force action and 64x64x3 image observations, the regime where DreamerV3's
"behaviors learned in imagination" demonstrably transfers to the real environment.

The dm_control / MuJoCo dependency is heavy and optional. It is imported lazily inside the functions
that need it (and inside viz.py), so the graded mechanism tests (shapes, gradcheck, symlog/two-hot,
straight-through, KL, lambda-returns, the continuous actor, the differentiable imagination) run on
CPU without dm_control installed. MUJOCO_GL=egl selects headless GPU rendering; set it before any
dm_control import.

Task and conventions:
- cartpole-balance: the pole starts near upright; the dense reward is in [0, 1] (1 when balanced),
  there is no early termination, and an episode is 1000 environment steps.
- Action: a single force in [-1, 1]. The actor emits a Tanh-Normal sample already in that range.
- action_repeat = 2: each agent step applies the same force for 2 environment steps and sums the
  reward, so one episode is ~500 agent steps. This is the standard dm_control pixel setup and keeps
  the imagined horizon short relative to the dynamics.
- An observation is rendered from camera 0 at 64x64 and returned as a (3, 64, 64) float32 array in
  [0, 1] (channels-first for torch convolutions).

The collected replay is a list of episode dicts with arrays:
  obs (T, 3, 64, 64), actions (T, 1) float, rewards (T,) float, conts (T,) float (1 - done).
Actions are stored as floats (continuous), not integer indices.
"""

import os

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

OBS_SIZE = 64
ACTION_REPEAT = 2
EPISODE_AGENT_STEPS = 500   # 1000 env steps / action_repeat


def make_env(seed: int = 0):
    """Load the cartpole-balance task. Imports dm_control lazily (heavy optional dependency)."""
    from dm_control import suite
    return suite.load("cartpole", "balance", task_kwargs={"random": seed})


def render(env, obs_size: int = OBS_SIZE) -> np.ndarray:
    """Render the current physics state to a (3, obs_size, obs_size) float32 image in [0, 1]."""
    img = env.physics.render(obs_size, obs_size, camera_id=0)
    return np.transpose(img.astype(np.float32) / 255.0, (2, 0, 1))


def random_return(env, n: int = 10, max_agent_steps: int = EPISODE_AGENT_STEPS) -> float:
    """Mean real-environment return of a uniform-random force policy over n episodes (max ~500).

    This is the baseline the imagination-trained policy must clear. Measured ~214 on this task.
    """
    tots = []
    for _ in range(n):
        env.reset()
        tot = 0.0
        for _ in range(max_agent_steps):
            ts = env.step(np.random.uniform(-1.0, 1.0, 1))
            tot += ts.reward
            if ts.last():
                break
        tots.append(tot)
    return float(np.mean(tots))


def run_episode(env, model, actor, explore, device, max_agent_steps: int = EPISODE_AGENT_STEPS):
    """Run one real episode with the RECURRENT policy and return (episode_dict, total_return).

    The policy carries the RSSM state (h, z) across the real episode: reset to the zero initial
    state, and at each step advance h with the PREVIOUS action, take the posterior from the encoded
    current frame, and sample the actor. explore=True samples the Tanh-Normal; explore=False takes
    the greedy mean action (used for the reported greedy return).

    The whole rollout runs under no_grad: this is data collection / evaluation, not a training step.
    """
    import torch

    with torch.no_grad():
        env.reset()
        h, z = model.rssm.initial_state(1, device=device, dtype=torch.float32)
        a_prev = torch.zeros(1, 1, device=device)
        obs_l, act_l, rew_l, cont_l = [], [], [], []
        total = 0.0
        last = None
        for _ in range(max_agent_steps):
            img = render(env)
            t = torch.as_tensor(img[None], device=device)
            h = model.rssm.forward_h(h, z, a_prev)             # advance with the previous action
            _, z, _ = model.rssm.posterior(h, model.encoder(t))
            a, _ = actor.sample(h, z, greedy=not explore)      # (1, 1) in [-1, 1]
            force = a.cpu().numpy().reshape(1)
            r = 0.0
            for _ in range(ACTION_REPEAT):
                last = env.step(force)
                r += last.reward
                if last.last():
                    break
            obs_l.append(img)
            act_l.append(force.astype(np.float32))
            rew_l.append(r)
            cont_l.append(0.0 if last.last() else 1.0)
            total += r
            a_prev = a
            if last.last():
                break
    ep = {
        "obs": np.stack(obs_l).astype(np.float32),       # (T, 3, 64, 64)
        "actions": np.stack(act_l).astype(np.float32),   # (T, 1) continuous force
        "rewards": np.array(rew_l, np.float32),          # (T,)
        "conts": np.array(cont_l, np.float32),           # (T,)
    }
    return ep, total
