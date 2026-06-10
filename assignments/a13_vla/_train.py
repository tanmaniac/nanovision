"""Shared training and rollout helpers for the tests and viz. Provided, not a hole.

These wrap the three action heads (flow, BC, DDPM) so the tests and viz drive them the same way:
build chunked conditioning batches from the scripted demos, train one head, and roll a trained
policy in the env to measure success rate. Kept out of the test files so the overfit tests and the
viz ablation share one code path.
"""

import numpy as np
import torch

import env as ENV
from bc import BCPolicy, bc_loss, chunk_actions
from ddpm import DDPMHead, ddpm_loss, ddpm_sample, make_schedule
from flow import FlowHead, flow_loss, flow_sample


def build_batch(demos, H, *, device="cpu"):
    """Turn a demos dict into a flat chunk-training batch.

    For each trajectory, chunk the per-step actions into overlapping H-step windows and pair each
    window with the conditioning at its first step. Only windows whose H steps are ALL real expert
    steps (mask=1) are kept, so a chunk never mixes in padded zeros. Returns
    (a_chunk (N, H, 2), c (N, C)) on `device`.
    """
    actions = torch.from_numpy(demos["actions"]).float()        # (n, T, 2)
    cond = torch.from_numpy(demos["cond"]).float()              # (n, T, C)
    mask = torch.from_numpy(demos["mask"]).float()             # (n, T)
    T = actions.shape[1]
    chunks = chunk_actions(actions, H)                          # (n, T-H+1, H, 2)
    mask_chunks = chunk_actions(mask.unsqueeze(-1), H)          # (n, T-H+1, H, 1)
    c_starts = cond[:, : T - H + 1]                            # (n, T-H+1, C)
    valid = mask_chunks.squeeze(-1).min(dim=-1).values > 0.5    # all H steps real
    a_flat = chunks[valid]                                      # (N, H, 2)
    c_flat = c_starts[valid]                                    # (N, C)
    return a_flat.to(device), c_flat.to(device)


def train_flow(a_chunk, c, cfg, *, steps=2000, lr=3e-3, seed=0, device="cpu", batch=256):
    torch.manual_seed(seed)
    head = FlowHead(cfg, cond_in=c.shape[-1]).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    g = torch.Generator(device=device).manual_seed(seed)
    N = a_chunk.shape[0]
    for _ in range(steps):
        idx = torch.randint(0, N, (min(batch, N),), generator=g, device=device)
        opt.zero_grad()
        loss = flow_loss(head, a_chunk[idx], c[idx], generator=g)
        loss.backward()
        opt.step()
    return head


def train_bc(a_chunk, c, cfg, *, steps=2000, lr=3e-3, seed=0, device="cpu", batch=256):
    torch.manual_seed(seed)
    policy = BCPolicy(cfg, cond_in=c.shape[-1]).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    g = torch.Generator(device=device).manual_seed(seed)
    N = a_chunk.shape[0]
    for _ in range(steps):
        idx = torch.randint(0, N, (min(batch, N),), generator=g, device=device)
        opt.zero_grad()
        loss = bc_loss(policy, a_chunk[idx], c[idx])
        loss.backward()
        opt.step()
    return policy


def train_ddpm(a_chunk, c, cfg, *, steps=2000, lr=3e-3, seed=0, device="cpu", batch=256):
    torch.manual_seed(seed)
    head = DDPMHead(cfg, cond_in=c.shape[-1], T=cfg.ddpm_T).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    abar = make_schedule(cfg.ddpm_T).to(device)
    g = torch.Generator(device=device).manual_seed(seed)
    N = a_chunk.shape[0]
    for _ in range(steps):
        idx = torch.randint(0, N, (min(batch, N),), generator=g, device=device)
        opt.zero_grad()
        loss = ddpm_loss(head, a_chunk[idx], c[idx], abar, generator=g)
        loss.backward()
        opt.step()
    return head


def _policy_action_fn(head, kind, cfg, *, device="cpu"):
    """Return a callable c_np (B, C) -> chunk_np (B, H, 2) for a trained head."""
    abar = make_schedule(cfg.ddpm_T).to(device) if kind == "ddpm" else None

    def fn(c_np):
        c = torch.from_numpy(np.asarray(c_np, dtype=np.float32)).to(device)
        with torch.no_grad():
            if kind == "flow":
                ch = flow_sample(head, c, cfg.chunk, cfg.n_flow_steps)
            elif kind == "bc":
                ch = head(c)
            elif kind == "ddpm":
                ch = ddpm_sample(head, c, cfg.chunk, abar)
            else:
                raise ValueError(kind)
        return ch.cpu().numpy()

    return fn


def rollout_success(head, kind, cfg, *, n=64, max_steps=60, seed=0, device="cpu",
                    goal_conditioned=True):
    """Roll a trained policy open-loop in the env and return the success fraction.

    Each episode samples a goal and start, builds the conditioning the head was trained on, queries
    an H-step chunk, executes it open-loop, then re-queries. Success = the point mass ends within
    eps of its goal.
    """
    rng = np.random.default_rng(seed)
    fn = _policy_action_fn(head, kind, cfg, device=device)
    H = cfg.chunk
    successes = 0
    for _ in range(n):
        gi = int(rng.integers(0, cfg.n_goals))
        p = ENV.sample_start(gi, rng)
        goal = ENV.GOALS[gi]
        steps_taken = 0
        while steps_taken < max_steps:
            c = ENV.make_condition(p, gi, goal_conditioned=goal_conditioned, repr=cfg.repr)
            chunk = fn(c[None])[0]                       # (H, 2)
            for h in range(H):
                p = ENV.step_env(p, chunk[h], v_max=cfg.v_max)
                steps_taken += 1
                if np.linalg.norm(p - goal) < cfg.eps or steps_taken >= max_steps:
                    break
            if np.linalg.norm(p - goal) < cfg.eps:
                break
        if np.linalg.norm(p - goal) < cfg.eps:
            successes += 1
    return successes / n
