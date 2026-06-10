"""A deterministic 2D point-mass reacher and a scripted expert. Provided, not a hole.

The point mass lives in the unit square [0, 1]^2. A step adds a velocity action a = (vx, vy)
clipped to [-v_max, v_max]: p <- clip(p + a, 0, 1). Each episode picks one of four fixed goals
(the corners, inset from the edge). The scripted expert moves straight toward the goal at speed
v_max with small Gaussian jitter and stops within eps of the goal.

The conditioning vector c is what the policy sees. Two modes:

  goal_conditioned=True (default): c carries the goal (one-hot of the 4 goals, or the 2D goal
    coordinate), so the expert action is a deterministic function of (state, goal) up to jitter.
    p(a | c) is effectively unimodal; a plain regressor fits it as well as a generative head.

  goal_conditioned=False: the goal is hidden from c (c is the state only) while the goal still
    varies across episodes. From a fixed state the demonstrated action points to one of several
    goals, so p(a | state) is multimodal. A unimodal regressor averages the modes and aims
    between goals; a generative head samples one coherent mode. This is the setting that
    motivates a flow-matching action head over plain regression.

No external RL/robot dependency; pure NumPy.
"""

import numpy as np

V_MAX = 0.05
EPS = 0.05
N_GOALS = 4
# Four corners, inset from the edge so the point mass does not clip against the wall at the goal.
GOALS = np.array(
    [[0.15, 0.15], [0.85, 0.15], [0.15, 0.85], [0.85, 0.85]],
    dtype=np.float32,
)


def expert_action(p, goal, *, v_max=V_MAX, jitter=0.0, rng=None):
    """The scripted expert's next-step velocity from state p toward goal.

    Move straight at speed v_max, but never overshoot: if the goal is within one step, step
    exactly to it. Optional Gaussian jitter (std `jitter`) is added before clipping. p, goal are
    (..., 2); returns (..., 2).
    """
    p = np.asarray(p, dtype=np.float32)
    goal = np.asarray(goal, dtype=np.float32)
    delta = goal - p
    dist = np.linalg.norm(delta, axis=-1, keepdims=True)
    # Unit direction toward the goal, guarded against a zero-length step at the goal.
    safe = np.maximum(dist, 1e-8)
    step = np.minimum(dist, v_max)              # do not overshoot the goal
    a = delta / safe * step
    if jitter > 0.0 and rng is not None:
        a = a + rng.normal(0.0, jitter, size=a.shape).astype(np.float32)
    return np.clip(a, -v_max, v_max).astype(np.float32)


def step_env(p, a, *, v_max=V_MAX):
    """One environment step: clip the action, advance, clip into the unit square."""
    a = np.clip(np.asarray(a, dtype=np.float32), -v_max, v_max)
    return np.clip(np.asarray(p, dtype=np.float32) + a, 0.0, 1.0).astype(np.float32)


def goal_cond(goal_idx, *, repr="onehot"):
    """Encode a goal index (or array of indices) as the conditioning the policy reads.

    repr="onehot": (..., N_GOALS) one-hot. repr="coord": (..., 2) goal coordinate.
    """
    goal_idx = np.asarray(goal_idx)
    if repr == "onehot":
        oh = np.zeros(goal_idx.shape + (N_GOALS,), dtype=np.float32)
        np.put_along_axis(oh, goal_idx[..., None], 1.0, axis=-1)
        return oh
    if repr == "coord":
        return GOALS[goal_idx]
    raise ValueError(f"unknown goal repr {repr!r}")


def make_condition(p, goal_idx, *, goal_conditioned=True, repr="onehot"):
    """Build the conditioning vector c the policy sees.

    goal_conditioned=True: concat the state p (2D) with the goal encoding. The goal is visible,
      so p(a | c) is unimodal.
    goal_conditioned=False: c is the state p only, padded with zeros where the goal encoding
      would go so the conditioning width is fixed across modes. The goal is hidden, so
      p(a | c) is multimodal across the goals consistent with that state.

    Returns c of shape (..., 2 + cond_width), where cond_width is N_GOALS for onehot or 2 for
    coord. The padding keeps the policy input width identical in both modes.
    """
    p = np.asarray(p, dtype=np.float32)
    cond_width = N_GOALS if repr == "onehot" else 2
    if goal_conditioned:
        g = goal_cond(goal_idx, repr=repr)
    else:
        g = np.zeros(np.asarray(goal_idx).shape + (cond_width,), dtype=np.float32)
    return np.concatenate([p, g], axis=-1).astype(np.float32)


def cond_dim(repr="onehot"):
    """Width of the conditioning vector c for a given goal encoding."""
    return 2 + (N_GOALS if repr == "onehot" else 2)


def sample_start(goal_idx, rng):
    """Sample a start position far from GOALS[goal_idx], matching the demo distribution.

    Biased toward the opposite corner so the episode is a real multi-step trajectory. Rollout
    evaluation must use this same distribution: a start sampled uniformly over the whole square
    lands the point mass in near-goal states the demos never visited, which is out of distribution
    and makes success rates read low for reasons unrelated to the mechanism.
    """
    g = GOALS[goal_idx]
    opp = 1.0 - g
    p0 = (opp + rng.uniform(-0.2, 0.2, size=2)).astype(np.float32)
    return np.clip(p0, 0.02, 0.98).astype(np.float32)


def rollout_expert(goal_idx, p0, *, max_steps=30, v_max=V_MAX, eps=EPS, jitter=0.0, rng=None):
    """Roll the scripted expert from p0 toward GOALS[goal_idx]. Returns (states, actions).

    states is (L+1, 2) (including the start), actions is (L, 2) of the expert's per-step
    velocities. Stops early once within eps of the goal. L <= max_steps.
    """
    goal = GOALS[goal_idx]
    p = np.asarray(p0, dtype=np.float32).copy()
    states = [p.copy()]
    actions = []
    for _ in range(max_steps):
        a = expert_action(p, goal, v_max=v_max, jitter=jitter, rng=rng)
        actions.append(a.copy())
        p = step_env(p, a, v_max=v_max)
        states.append(p.copy())
        if np.linalg.norm(p - goal) < eps:
            break
    return np.stack(states, axis=0), np.stack(actions, axis=0)


def collect_demos(n=200, seed=0, *, max_steps=40, jitter=0.003, repr="onehot",
                  goal_conditioned=True, chunk=None):
    """Collect n scripted demonstration trajectories.

    Each trajectory picks a random goal and a random start, then rolls the expert. Returns a
    dict of stacked PER-STEP arrays padded/truncated to a common length T (the median episode
    length), with a validity mask:

      states   (n, T, 2)      point-mass positions
      actions  (n, T, 2)      expert next-step velocities
      cond     (n, T, C)      conditioning c per step (state + goal encoding, masked per mode)
      goal_idx (n,)           the goal of each trajectory
      mask     (n, T)         1 where the step is a real expert step, 0 where padded

    chunk is unused here; the training loop builds chunks from actions with chunk_actions.
    """
    rng = np.random.default_rng(seed)
    raw = []
    for _ in range(n):
        gi = int(rng.integers(0, N_GOALS))
        # Start far from the goal (biased toward the opposite half of the square) so every episode
        # has a real multi-step trajectory to imitate. This keeps the chunked targets long enough
        # for the H=16 ablation: crossing the square between opposite inset corners at v_max=0.05
        # takes ~20 steps.
        p0 = sample_start(gi, rng)
        states, actions = rollout_expert(gi, p0, max_steps=max_steps, jitter=jitter, rng=rng)
        raw.append((gi, states, actions))
    lengths = np.array([a.shape[0] for _, _, a in raw])
    # Pad to the longest episode (capped at max_steps) so the per-step arrays are rectangular and
    # the validity mask marks the real steps. A fixed long T supports chunk sizes up to ~max_steps.
    T = int(min(lengths.max(), max_steps))
    T = max(T, 1)

    S = np.zeros((n, T, 2), dtype=np.float32)
    A = np.zeros((n, T, 2), dtype=np.float32)
    M = np.zeros((n, T), dtype=np.float32)
    GI = np.zeros((n,), dtype=np.int64)
    for i, (gi, states, actions) in enumerate(raw):
        L = min(actions.shape[0], T)
        S[i, :L] = states[:L]
        A[i, :L] = actions[:L]
        M[i, :L] = 1.0
        GI[i] = gi
    gi_full = np.repeat(GI[:, None], T, axis=1)          # (n, T) goal per step
    C = make_condition(S, gi_full, goal_conditioned=goal_conditioned, repr=repr)
    return {
        "states": S,
        "actions": A,
        "cond": C,
        "goal_idx": GI,
        "mask": M,
        "T": T,
        "repr": repr,
        "goal_conditioned": goal_conditioned,
    }
