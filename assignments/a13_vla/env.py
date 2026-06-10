"""The dm_control reacher env, the analytic expert, and filtered pixel demos. Provided, not a hole.

This is the authentic-control half of the capstone: a 2-link reacher (dm_control 'reacher' easy),
controlled from a 64x64 RGB camera image. The action is a 2D joint torque in [-1, 1]; the policy
never sees joint angles or the target position directly, only the rendered frame. dm_control is
imported lazily inside the functions that need it so the CPU mechanism tests (flow, bc, ddpm) run
without the robot library installed. Set MUJOCO_GL=egl for headless rendering.

The expert is analytic inverse kinematics plus a PD controller in joint space - it has privileged
access to the target world position and the joint state, which the learned policy does not. Demos
are filtered: only episodes that actually reach the target (reward > 0.5) are kept, truncated at the
reach step, so behavior cloning sees clean successful trajectories. The expert reaches on ~83% of
random seeds in ~20 steps.

The retained point-mass functions at the bottom (GOALS, expert_action, make_condition, sample_start,
collect_demos_pointmass) drive the self-contained 2D multimodal side-demo in viz.py: the lesson that
a regressor averages multimodal actions while a generative head samples a mode. That lesson is hard
to show from pixels (the image determines the target, so p(action | image) is effectively unimodal),
so it stays a small NumPy side-demo with no robot dependency.
"""

import numpy as np

# Reacher 2-link geometry (both links length 0.12). Used by the analytic IK expert.
L0 = L1 = 0.12


# --------------------------------------------------------------------------------------------------
# dm_control reacher wrapper (lazy dm_control import inside the functions).
# --------------------------------------------------------------------------------------------------

def make_reacher(seed=0):
    """Load the dm_control 'reacher' easy task with a fixed random seed. Lazy dm_control import.

    Returns the dm_control Environment. The action spec is 2D torque in [-1, 1]; reward is 1 while
    the finger overlaps the target and 0 otherwise (we threshold at 0.5 to detect a reach).
    """
    from dm_control import suite
    return suite.load("reacher", "easy", task_kwargs={"random": int(seed)})


def render_obs(env):
    """Render the conditioning observation: the 64x64 RGB frame normalized to [0, 1], (3, 64, 64).

    The learned policy reads only this image. camera_id=0 is the fixed overhead camera.
    """
    img = env.physics.render(64, 64, camera_id=0)          # (64, 64, 3) uint8
    return (img.astype(np.float32) / 255.0).transpose(2, 0, 1)   # (3, 64, 64) in [0, 1]


def ik(tx, ty):
    """Analytic 2-link inverse kinematics: target (tx, ty) -> (shoulder, wrist) joint angles.

    Clamps the reach radius to the arm's workspace and the elbow cosine to [-1, 1] so arccos is
    always defined. Returns the elbow-down solution.
    """
    r = min(np.hypot(tx, ty), L0 + L1 - 0.002)
    r = max(r, 0.002)
    c2 = np.clip((r * r - L0 * L0 - L1 * L1) / (2 * L0 * L1), -1.0, 1.0)
    th2 = np.arccos(c2)
    th1 = np.arctan2(ty, tx) - np.arctan2(L1 * np.sin(th2), L0 + L1 * np.cos(th2))
    return np.array([th1, th2])


def expert_torque(physics, kp=12.0, kd=0.8):
    """The analytic IK + PD expert torque from privileged state. Returns a 2D torque in [-1, 1].

    Solve IK for the target world position, take the wrapped joint-angle error to the IK solution,
    and apply a PD law kp*err - kd*qvel clipped to the torque limits. physics is env.physics.
    """
    tgt = physics.named.data.geom_xpos["target"][:2]
    th = ik(tgt[0], tgt[1])
    err = (th - physics.data.qpos[:2] + np.pi) % (2 * np.pi) - np.pi
    return np.clip(kp * err - kd * physics.data.qvel[:2], -1.0, 1.0)


def random_torque(rng):
    """A random-torque policy: a 2D torque sampled uniformly in [-1, 1]. The rollout floor."""
    return rng.uniform(-1.0, 1.0, size=2).astype(np.float32)


def reached(timestep):
    """Whether a step reached the target: reward present and above 0.5 (finger overlaps target)."""
    return timestep.reward is not None and timestep.reward > 0.5


# --------------------------------------------------------------------------------------------------
# Filtered pixel-demo collection.
# --------------------------------------------------------------------------------------------------

def collect_one_episode(seed, *, max_steps=80, kp=12.0, kd=0.8):
    """Roll the analytic expert in one reacher episode, recording (image, torque) per step.

    Returns (obs (L, 3, 64, 64), act (L, 2), reach_step or None). reach_step is the first step at
    which the finger overlaps the target; the episode is recorded up to and including that step.
    If the expert never reaches within max_steps, reach_step is None and the full rollout is
    returned (the caller filters these out).
    """
    env = make_reacher(seed=seed)
    env.reset()
    obs, act = [], []
    reach_step = None
    for step in range(max_steps):
        o = render_obs(env)
        a = expert_torque(env.physics, kp=kp, kd=kd).astype(np.float32)
        ts = env.step(a)
        obs.append(o)
        act.append(a)
        if reached(ts):
            reach_step = step
            break
    return np.stack(obs, 0), np.stack(act, 0), reach_step


def collect_demos(n_success=200, *, seed0=0, max_steps=80, pad_T=None, max_episodes=None):
    """Collect filtered successful pixel demonstrations for behavior cloning.

    Run the analytic expert on consecutive random seeds, KEEP only the episodes that reach the
    target (truncated at the reach step), and stop once n_success successful episodes are gathered.
    Per-step arrays are padded to a common length T (the max kept episode length, or pad_T) with a
    validity mask so the chunk builder never mixes in padded steps.

    Returns a dict:
      obs    (n_success, T, 3, 64, 64)  rendered frames, [0, 1]
      act    (n_success, T, 2)          expert torques
      mask   (n_success, T)             1 where the step is a real expert step, 0 where padded
      T      int                        the padded length
      n_tried int, reach_frac float     collection statistics

    dm_control is imported lazily inside collect_one_episode.
    """
    kept = []
    tried = 0
    successes = 0
    cap = max_episodes if max_episodes is not None else 100000
    s = seed0
    while successes < n_success and tried < cap:
        o, a, reach_step = collect_one_episode(s, max_steps=max_steps)
        tried += 1
        s += 1
        if reach_step is None:
            continue
        L = reach_step + 1
        kept.append((o[:L], a[:L]))
        successes += 1
    if not kept:
        raise RuntimeError("no successful demos collected; check MUJOCO_GL=egl and dm_control")

    lengths = np.array([a.shape[0] for _, a in kept])
    T = int(pad_T if pad_T is not None else lengths.max())
    n = len(kept)
    O = np.zeros((n, T, 3, 64, 64), dtype=np.float32)
    A = np.zeros((n, T, 2), dtype=np.float32)
    M = np.zeros((n, T), dtype=np.float32)
    for i, (o, a) in enumerate(kept):
        Li = min(a.shape[0], T)
        O[i, :Li] = o[:Li]
        A[i, :Li] = a[:Li]
        M[i, :Li] = 1.0
    return {
        "obs": O,
        "act": A,
        "mask": M,
        "T": T,
        "n_tried": tried,
        "reach_frac": successes / tried if tried else 0.0,
    }


def rollout_policy(action_fn, *, n=64, seed0=1000, max_steps=80, chunk=1):
    """Roll a pixel policy in the reacher and return the reach-success fraction.

    Each episode loads a fresh reacher seed, renders the frame, calls action_fn(obs (1,3,64,64)) ->
    chunk (1, H, 2), executes the H torques open-loop, then re-queries. Success = the finger reaches
    the target (reward > 0.5) at some step before max_steps. action_fn owns the policy (a trained
    flow/BC head with its encoder, or the random/expert baseline). dm_control is imported lazily.
    """
    successes = 0
    for i in range(n):
        env = make_reacher(seed=seed0 + i)
        env.reset()
        steps = 0
        hit = False
        while steps < max_steps and not hit:
            o = render_obs(env)[None]                      # (1, 3, 64, 64)
            chunk_a = action_fn(o)[0]                       # (H, 2)
            for h in range(chunk):
                a = np.clip(np.asarray(chunk_a[h], dtype=np.float32), -1.0, 1.0)
                ts = env.step(a)
                steps += 1
                if reached(ts):
                    hit = True
                    break
                if steps >= max_steps:
                    break
        successes += int(hit)
    return successes / n


def random_reach_success(*, n=64, seed0=1000, max_steps=80):
    """Reach-success of a random-torque policy: the rollout floor BC must clearly beat."""
    rng = np.random.default_rng(0)
    successes = 0
    for i in range(n):
        env = make_reacher(seed=seed0 + i)
        env.reset()
        hit = False
        for _ in range(max_steps):
            ts = env.step(random_torque(rng))
            if reached(ts):
                hit = True
                break
        successes += int(hit)
    return successes / n


# --------------------------------------------------------------------------------------------------
# Retained point-mass side-demo (the multimodal regression-vs-generative lesson). Pure NumPy.
# --------------------------------------------------------------------------------------------------
#
# This 2D point mass is NOT the robot. It exists only to show one lesson cleanly: when the same
# observation admits several valid expert actions (the goal is hidden, so the action could point to
# any of four goals), a regressor trained with MSE averages the modes and aims nowhere, while a
# generative head samples one coherent mode. The reacher cannot show this from pixels, because the
# image fixes the target and p(action | image) is effectively unimodal. So the lesson lives here.

V_MAX = 0.05
EPS = 0.05
N_GOALS = 4
GOALS = np.array(
    [[0.15, 0.15], [0.85, 0.15], [0.15, 0.85], [0.85, 0.85]],
    dtype=np.float32,
)


def expert_action(p, goal, *, v_max=V_MAX, jitter=0.0, rng=None):
    """Point-mass scripted expert: next-step velocity from p toward goal, not overshooting."""
    p = np.asarray(p, dtype=np.float32)
    goal = np.asarray(goal, dtype=np.float32)
    delta = goal - p
    dist = np.linalg.norm(delta, axis=-1, keepdims=True)
    safe = np.maximum(dist, 1e-8)
    step = np.minimum(dist, v_max)
    a = delta / safe * step
    if jitter > 0.0 and rng is not None:
        a = a + rng.normal(0.0, jitter, size=a.shape).astype(np.float32)
    return np.clip(a, -v_max, v_max).astype(np.float32)


def step_pointmass(p, a, *, v_max=V_MAX):
    """One point-mass step: clip the action, advance, clip into the unit square."""
    a = np.clip(np.asarray(a, dtype=np.float32), -v_max, v_max)
    return np.clip(np.asarray(p, dtype=np.float32) + a, 0.0, 1.0).astype(np.float32)


def goal_cond(goal_idx, *, repr="onehot"):
    """Encode a point-mass goal index as one-hot (N_GOALS) or coordinate (2)."""
    goal_idx = np.asarray(goal_idx)
    if repr == "onehot":
        oh = np.zeros(goal_idx.shape + (N_GOALS,), dtype=np.float32)
        np.put_along_axis(oh, goal_idx[..., None], 1.0, axis=-1)
        return oh
    if repr == "coord":
        return GOALS[goal_idx]
    raise ValueError(f"unknown goal repr {repr!r}")


def make_condition(p, goal_idx, *, goal_conditioned=True, repr="onehot"):
    """Build the point-mass conditioning c: state concatenated with the goal encoding (or zeros).

    goal_conditioned=True makes p(a|c) unimodal (goal visible); False hides the goal so p(a|c) is
    multimodal across the goals consistent with the state. The width is fixed across modes.
    """
    p = np.asarray(p, dtype=np.float32)
    cond_width = N_GOALS if repr == "onehot" else 2
    if goal_conditioned:
        g = goal_cond(goal_idx, repr=repr)
    else:
        g = np.zeros(np.asarray(goal_idx).shape + (cond_width,), dtype=np.float32)
    return np.concatenate([p, g], axis=-1).astype(np.float32)


def cond_dim(repr="onehot"):
    """Width of the point-mass conditioning vector for a given goal encoding."""
    return 2 + (N_GOALS if repr == "onehot" else 2)


def sample_start(goal_idx, rng):
    """Sample a point-mass start far from GOALS[goal_idx] (biased to the opposite corner)."""
    g = GOALS[goal_idx]
    opp = 1.0 - g
    p0 = (opp + rng.uniform(-0.2, 0.2, size=2)).astype(np.float32)
    return np.clip(p0, 0.02, 0.98).astype(np.float32)


def rollout_expert(goal_idx, p0, *, max_steps=30, v_max=V_MAX, eps=EPS, jitter=0.0, rng=None):
    """Roll the point-mass scripted expert from p0 toward GOALS[goal_idx]. Returns (states, actions)."""
    goal = GOALS[goal_idx]
    p = np.asarray(p0, dtype=np.float32).copy()
    states = [p.copy()]
    actions = []
    for _ in range(max_steps):
        a = expert_action(p, goal, v_max=v_max, jitter=jitter, rng=rng)
        actions.append(a.copy())
        p = step_pointmass(p, a, v_max=v_max)
        states.append(p.copy())
        if np.linalg.norm(p - goal) < eps:
            break
    return np.stack(states, axis=0), np.stack(actions, axis=0)


def collect_demos_pointmass(n=400, seed=0, *, max_steps=40, jitter=0.003, repr="onehot",
                            goal_conditioned=True):
    """Collect point-mass demos for the multimodal side-demo. Per-step arrays padded to T.

    Returns states/actions/cond/goal_idx/mask, matching the side-demo's build_batch_pointmass.
    """
    rng = np.random.default_rng(seed)
    raw = []
    for _ in range(n):
        gi = int(rng.integers(0, N_GOALS))
        p0 = sample_start(gi, rng)
        states, actions = rollout_expert(gi, p0, max_steps=max_steps, jitter=jitter, rng=rng)
        raw.append((gi, states, actions))
    lengths = np.array([a.shape[0] for _, _, a in raw])
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
    gi_full = np.repeat(GI[:, None], T, axis=1)
    C = make_condition(S, gi_full, goal_conditioned=goal_conditioned, repr=repr)
    return {"states": S, "actions": A, "cond": C, "goal_idx": GI, "mask": M, "T": T,
            "repr": repr, "goal_conditioned": goal_conditioned}
