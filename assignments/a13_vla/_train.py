"""Training and rollout helpers for the pixel reacher and the point-mass side-demo. Provided.

The pixel path is the capstone: encode the 64x64 frame to a conditioning vector, train an action
head (flow or BC) on filtered expert demos by behavior cloning, and roll the trained policy in the
real reacher to measure reach success. The encoder and the head train jointly under one optimizer -
there is no separate representation-learning step. Each head gets its own encoder instance so the
flow-vs-BC comparison is clean.

The point-mass helpers at the bottom drive the retained 2D multimodal side-demo and never touch
dm_control. This file imports dm_control only transitively through env.rollout_policy at rollout
time; building and training batches needs only torch.
"""

import numpy as np
import torch

import env as ENV
from bc import BCPolicy, bc_loss, chunk_actions
from ddpm import DDPMHead, ddpm_loss, ddpm_sample, make_schedule
from flow import FlowHead, flow_loss, flow_sample
from nets import Encoder


# --------------------------------------------------------------------------------------------------
# Pixel reacher: batch building, joint encoder+head training, real-env rollout.
# --------------------------------------------------------------------------------------------------

def build_pixel_batch(demos, H, *, device="cpu"):
    """Turn the filtered pixel demos into a flat chunk-training batch.

    For each demo, chunk the per-step torques into overlapping H-step windows and pair each window
    with the FRAME at its first step. Only windows whose H steps are all real expert steps (mask=1)
    are kept. Returns (obs (N, 3, 64, 64), a_chunk (N, H, 2)) on `device`.
    """
    obs = torch.from_numpy(demos["obs"]).float()                # (n, T, 3, 64, 64)
    act = torch.from_numpy(demos["act"]).float()                # (n, T, 2)
    mask = torch.from_numpy(demos["mask"]).float()             # (n, T)
    n, T = act.shape[0], act.shape[1]
    chunks = chunk_actions(act, H)                              # (n, T-H+1, H, 2)
    mask_chunks = chunk_actions(mask.unsqueeze(-1), H)          # (n, T-H+1, H, 1)
    obs_starts = obs[:, : T - H + 1]                           # (n, T-H+1, 3, 64, 64)
    valid = mask_chunks.squeeze(-1).min(dim=-1).values > 0.5    # (n, T-H+1) all H steps real
    a_flat = chunks[valid]                                      # (N, H, 2)
    o_flat = obs_starts[valid]                                  # (N, 3, 64, 64)
    return o_flat.to(device), a_flat.to(device)


def train_pixel_flow(obs, a_chunk, cfg, *, steps=3000, lr=3e-4, seed=0, device="cpu", batch=128):
    """Train the flow head and its image encoder jointly by conditional flow matching.

    Returns (encoder, head). c = encoder(obs); flow_loss(head, a_chunk, c). One Adam over both.
    """
    torch.manual_seed(seed)
    enc = Encoder(cfg).to(device)
    head = FlowHead(cfg, cond_in=cfg.embed_dim).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=lr)
    g = torch.Generator(device=device).manual_seed(seed)
    N = a_chunk.shape[0]
    for _ in range(steps):
        idx = torch.randint(0, N, (min(batch, N),), generator=g, device=device)
        opt.zero_grad()
        c = enc(obs[idx])
        loss = flow_loss(head, a_chunk[idx], c, generator=g)
        loss.backward()
        opt.step()
    return enc, head


def train_pixel_bc(obs, a_chunk, cfg, *, steps=3000, lr=3e-4, seed=0, device="cpu", batch=128):
    """Train the BC regressor and its image encoder jointly. Returns (encoder, policy)."""
    torch.manual_seed(seed)
    enc = Encoder(cfg).to(device)
    policy = BCPolicy(cfg, cond_in=cfg.embed_dim).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(policy.parameters()), lr=lr)
    g = torch.Generator(device=device).manual_seed(seed)
    N = a_chunk.shape[0]
    for _ in range(steps):
        idx = torch.randint(0, N, (min(batch, N),), generator=g, device=device)
        opt.zero_grad()
        c = enc(obs[idx])
        loss = bc_loss(policy, a_chunk[idx], c)
        loss.backward()
        opt.step()
    return enc, policy


def pixel_action_fn(enc, head, kind, cfg, *, device="cpu"):
    """Return a callable obs_np (B, 3, 64, 64) -> chunk_np (B, H, 2) for a trained pixel policy."""
    enc.eval()
    head.eval()

    def fn(obs_np):
        o = torch.from_numpy(np.asarray(obs_np, dtype=np.float32)).to(device)
        with torch.no_grad():
            c = enc(o)
            if kind == "flow":
                ch = flow_sample(head, c, cfg.chunk, cfg.n_flow_steps)
            elif kind == "bc":
                ch = head(c)
            else:
                raise ValueError(kind)
        return ch.cpu().numpy()

    return fn


def pixel_rollout_success(enc, head, kind, cfg, *, n=48, seed0=1000, device="cpu"):
    """Roll a trained pixel policy in the real reacher and return the reach-success fraction."""
    fn = pixel_action_fn(enc, head, kind, cfg, device=device)
    return ENV.rollout_policy(fn, n=n, seed0=seed0, max_steps=cfg.max_steps, chunk=cfg.chunk)


# --------------------------------------------------------------------------------------------------
# Point-mass side-demo: state-conditioned batch building and training (no dm_control).
# --------------------------------------------------------------------------------------------------

def build_batch_pointmass(demos, H, *, device="cpu"):
    """Chunk point-mass demos into (a_chunk (N, H, 2), c (N, C)) using the per-step state+goal cond."""
    actions = torch.from_numpy(demos["actions"]).float()
    cond = torch.from_numpy(demos["cond"]).float()
    mask = torch.from_numpy(demos["mask"]).float()
    T = actions.shape[1]
    chunks = chunk_actions(actions, H)
    mask_chunks = chunk_actions(mask.unsqueeze(-1), H)
    c_starts = cond[:, : T - H + 1]
    valid = mask_chunks.squeeze(-1).min(dim=-1).values > 0.5
    return chunks[valid].to(device), c_starts[valid].to(device)


def train_bc(a_chunk, c, cfg, *, steps=2000, lr=3e-3, seed=0, device="cpu", batch=256):
    """Train the BC regressor on the point-mass state conditioning (side-demo)."""
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


def train_flow(a_chunk, c, cfg, *, steps=2000, lr=3e-3, seed=0, device="cpu", batch=256):
    """Train the flow head on the point-mass state conditioning (side-demo)."""
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
