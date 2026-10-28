"""dm_control cartpole (swingup by default, or balance) from 64x64 pixels. Provided, not a hole.

Ported from r2dreamer's `envs/dmc.py` plus the wrappers it applies (`NormalizeActions`,
`TimeLimit`, `Dtype`) and the batched stepping of `envs/parallel.py`. Differences from the
reference are listed where they occur; none changes what the agent sees.

Conventions:
- Action: one force. The policy's Gaussian sample can leave [-1, 1]; `step` clips it. MuJoCo
  clamps the control to the actuator's ctrlrange [-1, 1] anyway, and cartpole's control-cost
  term is 0 beyond |a| = 1, so clipping here gives the same physics state and the same reward as
  the reference's unclipped call (checked over 1000 random steps with 611 out-of-range actions).
- action_repeat = 2: each agent step applies the force for 2 physics steps and sums the reward.
  An episode is 1000 physics steps = 500 agent steps, so the maximum return is 1000.
- Observation: a dict with `image` (3, 64, 64) uint8 rendered from camera 0 (channels-first; the
  reference keeps (64, 64, 3)), and flags `is_first`, `is_last`, `is_terminal`. The reference also
  returns the proprioceptive state; its DreamerV3 pixel config ignores it, so it is dropped.
- `reset` returns reward 0 with is_first = True. After `step`, the reward is the one earned by the
  action just applied, attached to the frame that action produced.
- is_terminal is 1 - discount at a non-first step. cartpole never terminates (discount
  stays 1); the episode ends only by the time limit, which sets is_last but not is_terminal.

dm_control is imported lazily so that the modules that do not step the env load without it.
MUJOCO_GL=egl selects headless GPU rendering; it must be set before the first dm_control import.

The reference runs each env in its own process (about 0.9 GB each). Here all envs step in the
training process, one after another. Same seeds, same per-env call sequence, less memory.
"""

import os

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

OBS_SIZE = 64
ACTION_REPEAT = 2
TIME_LIMIT = 1000                                  # physics steps
EPISODE_AGENT_STEPS = TIME_LIMIT // ACTION_REPEAT  # 500


def make_env(seed: int = 0, task: str = "swingup"):
    """The raw dm_control cartpole task ("swingup" by default) with its own random seed."""
    from dm_control import suite
    return suite.load("cartpole", task, task_kwargs={"random": seed})


def render(env, obs_size: int = OBS_SIZE) -> np.ndarray:
    """Render camera 0 to a (3, obs_size, obs_size) uint8 array."""
    img = env.physics.render(obs_size, obs_size, camera_id=0)
    return np.ascontiguousarray(np.transpose(img, (2, 0, 1)))


class CartpoleEnv:
    """A cartpole task with action repeat, a time limit, and dict observations."""

    def __init__(self, seed: int, action_repeat: int = ACTION_REPEAT, time_limit: int = TIME_LIMIT,
                 obs_size: int = OBS_SIZE, task: str = "swingup"):
        self._env = make_env(seed, task)
        self._repeat = action_repeat
        self._duration = time_limit // action_repeat
        self._size = obs_size
        self._step = None

    def _obs(self, time_step, first: bool) -> dict:
        return {
            "image": render(self._env, self._size),
            "is_first": bool(first),
            "is_last": bool(time_step.last()),
            "is_terminal": False if first else bool(time_step.discount == 0),
        }

    def reset(self) -> dict:
        self._step = 0
        return self._obs(self._env.reset(), first=True)

    def step(self, action):
        """Apply `action` (clipped to [-1, 1]) for action_repeat physics steps.

        Returns (obs, reward, done); reward is float32, the sum over the repeated steps.
        """
        assert self._step is not None, "reset() first"
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(1), -1.0, 1.0)
        reward = 0.0
        for _ in range(self._repeat):
            time_step = self._env.step(action)
            reward += time_step.reward or 0.0
            if time_step.last():
                break
        obs = self._obs(time_step, first=False)
        done = time_step.last()
        self._step += 1
        if self._step >= self._duration:   # the reference's TimeLimit wrapper; dm_control's own
            done = True                    # limit fires on the same step for cartpole
            obs["is_last"] = True
            self._step = None
        return obs, np.float32(reward), bool(done)


def step_envs(envs, actions: np.ndarray, done: np.ndarray):
    """Step a list of envs together, resetting those flagged `done` instead of stepping them.

    This is the reference's `ParallelEnv.step`. A reset env reports reward 0 and done False.
    Args:
        actions: (E, A) float. done: (E,) bool from the previous call.
    Returns:
        obs dict of stacked arrays (image (E, 3, 64, 64) uint8, flags (E,) bool), reward (E,)
        float32, done (E,) bool.
    """
    obs_l, rew_l, done_l = [], [], []
    for env, a, d in zip(envs, actions, done):
        if d:
            obs_l.append(env.reset())
            rew_l.append(0.0)
            done_l.append(False)
        else:
            o, r, dn = env.step(a)
            obs_l.append(o)
            rew_l.append(r)
            done_l.append(dn)
    obs = {k: np.stack([o[k] for o in obs_l]) for k in obs_l[0]}
    return obs, np.asarray(rew_l, np.float32), np.asarray(done_l, bool)


def baseline_returns(env: CartpoleEnv, action_fn, n: int = 10) -> np.ndarray:
    """Per-episode returns of a fixed policy, same protocol as the agent's evaluation.

    action_fn() returns a (1,) force for the next agent step.
    """
    tots = []
    for _ in range(n):
        env.reset()
        tot, done = 0.0, False
        while not done:
            _, r, done = env.step(action_fn())
            tot += float(r)
        tots.append(tot)
    return np.array(tots)


def random_return(env: CartpoleEnv, n: int = 10) -> float:
    """Mean return of a uniform-random force policy (about 43 on swingup, 310 on balance)."""
    return float(np.mean(baseline_returns(env, lambda: np.random.uniform(-1.0, 1.0, 1), n)))


def zero_force_return(env: CartpoleEnv, n: int = 10) -> float:
    """Mean return of the do-nothing policy (0 on swingup, about 750 on balance)."""
    return float(np.mean(baseline_returns(env, lambda: np.zeros(1), n)))
