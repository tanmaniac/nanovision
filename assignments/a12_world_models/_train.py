"""The DreamerV3 collect-fit-imagine training loop for cartpole-from-pixels. Provided, not a hole.

This is the data iteration, optimizer, logging, critic-EMA bookkeeping, and the recurrent eval
policy. The holed mechanism (kl_loss, the ELBO assembly, lambda-returns, the differentiable
imagine_dynamics, the dynamics-backprop actor loss, the critic loss) is called from here.

The loop is the actual DreamerV3 procedure: collect one real episode with the current actor, refit
the world model on replay, then train the actor and critic on imagined rollouts, and repeat. The
actor is the continuous Tanh-Normal ContActor trained by dynamics backprop; the imagined return is
differentiable through the world model. A short critic warmup (no actor updates for the first few
iterations) lets the value function settle before the actor starts pulling on it.

dm_control is imported only inside env.py (lazily), so importing this module does not require it;
only dreamer_train (which actually steps the env) does.
"""

import copy

import numpy as np
import torch
import torch.nn as nn

from actor_critic import (
    ContActor,
    Critic,
    ReturnNormalizer,
    actor_loss_dynbackprop,
    critic_loss,
    imagine_dynamics,
)
from nets import twohot_decode
from world_model import WorldModel

SEQ_LEN = 50
BATCH = 16
IMAG_STARTS = 400          # number of posterior states sampled as imagination starts per update
WARMUP_ITERS = 5           # train only the critic (and world model) for these many iterations
UPDATES_PER_ITER = 100     # gradient updates per collected episode


def batch_to_tensors(batch_np, device="cpu", dtype=torch.float32):
    """Convert a replay dict of NumPy arrays to tensors on the device. Actions are float (continuous)."""
    return {
        "obs": torch.as_tensor(batch_np["obs"], dtype=dtype, device=device),
        "actions": torch.as_tensor(batch_np["actions"], dtype=dtype, device=device),  # continuous
        "rewards": torch.as_tensor(batch_np["rewards"], dtype=dtype, device=device),
        "conts": torch.as_tensor(batch_np["conts"], dtype=dtype, device=device),
    }


def sample_batch(buffer, device):
    """Sample BATCH length-SEQ_LEN subsequences from the episode buffer. Returns a tensor dict."""
    ob, ac, rw, ct = [], [], [], []
    while len(ob) < BATCH:
        ep = buffer[np.random.randint(len(buffer))]
        T = ep["obs"].shape[0]
        if T <= SEQ_LEN:
            continue
        s = np.random.randint(0, T - SEQ_LEN)
        ob.append(ep["obs"][s:s + SEQ_LEN])
        ac.append(ep["actions"][s:s + SEQ_LEN])
        rw.append(ep["rewards"][s:s + SEQ_LEN])
        ct.append(ep["conts"][s:s + SEQ_LEN])
    return batch_to_tensors(
        {"obs": np.stack(ob), "actions": np.stack(ac), "rewards": np.stack(rw), "conts": np.stack(ct)},
        device=device,
    )


def _encode_starts(model, batch, device):
    """Encode a replay batch to a flat pool of DETACHED posterior states (h, z) for imagination."""
    with torch.no_grad():
        embeds = model.encode_seq(batch["obs"])
        h0, z0 = model.rssm.initial_state(BATCH, device=device, dtype=torch.float32)
        hs, zs, _, _ = model.rssm.observe(embeds, batch["actions"], h0, z0)
    h_flat = hs.reshape(-1, model.cfg.h_dim).detach()
    z_flat = zs.reshape(-1, model.cfg.n_cat * model.cfg.n_cls).detach()
    return h_flat, z_flat


def dreamer_train(cfg, *, iters=400, seed_episodes=8, buffer_cap=200, model_lr=1e-4,
                  actor_lr=4e-5, critic_lr=4e-5, critic_ema=0.98, seed=0, device="cuda",
                  log=True, eval_every=5, eval_episodes=5):
    """Run the collect-fit-imagine loop. Returns (model, actor, critic, env, history).

    history records the greedy real-env return at each eval, the env-step count, and the random
    baseline so viz can draw the learning curve. The training never calls the env for gradients; the
    actor learns purely from imagined rollouts and is then evaluated greedily in the real env.
    """
    from env import run_episode, random_return, make_env

    torch.manual_seed(seed)
    np.random.seed(seed)
    env = make_env(seed)
    rand_ret = random_return(env)

    model = WorldModel(cfg).to(device)
    actor = ContActor(cfg).to(device)
    critic = Critic(cfg).to(device)
    critic_slow = copy.deepcopy(critic).to(device)
    for p in critic_slow.parameters():
        p.requires_grad_(False)
    opt_wm = torch.optim.Adam(model.parameters(), model_lr)
    opt_a = torch.optim.Adam(actor.parameters(), actor_lr)
    opt_c = torch.optim.Adam(critic.parameters(), critic_lr)
    norm = ReturnNormalizer(cfg.ret_ema_decay)
    bins = critic.bins

    buffer = [run_episode(env, model, actor, explore=True, device=device)[0] for _ in range(seed_episodes)]
    env_steps = sum(e["obs"].shape[0] for e in buffer)

    hist = {"env_steps": [], "greedy_return": [], "imag_return": [], "random": rand_ret}
    for it in range(iters):
        ep, collect_ret = run_episode(env, model, actor, explore=True, device=device)
        buffer.append(ep)
        env_steps += ep["obs"].shape[0]
        if len(buffer) > buffer_cap:
            buffer = buffer[-buffer_cap:]

        last_imag = 0.0
        for _ in range(UPDATES_PER_ITER):
            # World-model update on a replay batch.
            total, parts = model.loss(sample_batch(buffer, device))
            opt_wm.zero_grad()
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 100.0)
            opt_wm.step()

            # Imagination starts: detached posterior states from a fresh batch.
            h_pool, z_pool = _encode_starts(model, sample_batch(buffer, device), device)
            idx = torch.randint(0, h_pool.shape[0], (IMAG_STARTS,), device=device)

            returns, ents, H_h, H_z = imagine_dynamics(
                model, actor, critic, h_pool[idx], z_pool[idx], cfg
            )
            ret_range = norm.update(returns.detach())

            a_loss = actor_loss_dynbackprop(returns, ents, cfg.ent_coef, ret_range)
            if it >= WARMUP_ITERS:
                opt_a.zero_grad()
                a_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(actor.parameters(), 100.0)
                opt_a.step()
            last_imag = float(returns.mean())

            # Critic update: regress the detached lambda-returns, with a slow-EMA self-target reg.
            c_logits = critic.logits(H_h[:, :-1].detach(), H_z[:, :-1].detach())
            c_loss = critic_loss(c_logits, returns.detach(), bins)
            with torch.no_grad():
                slow = torch.softmax(critic_slow.logits(H_h[:, :-1].detach(), H_z[:, :-1].detach()), dim=-1)
            reg = critic_loss(c_logits, twohot_decode(slow, bins), bins)
            opt_c.zero_grad()
            (c_loss + reg).backward()
            opt_c.step()
            with torch.no_grad():
                for ps, pf in zip(critic_slow.parameters(), critic.parameters()):
                    ps.mul_(critic_ema).add_(pf, alpha=1.0 - critic_ema)

        if it % eval_every == 0 or it == iters - 1:
            gret = float(np.mean([
                run_episode(env, model, actor, explore=False, device=device)[1]
                for _ in range(eval_episodes)
            ]))
            hist["env_steps"].append(env_steps)
            hist["greedy_return"].append(gret)
            hist["imag_return"].append(last_imag)
            if log:
                tag = "  [warmup]" if it < WARMUP_ITERS else ""
                print(f"[it {it:3d} | env_steps {env_steps:6d}] greedy_return={gret:6.1f} "
                      f"(random {rand_ret:.0f}) collect_return={collect_ret:6.1f} "
                      f"recon={float(parts['recon']):.3f} imag_ret={last_imag:.2f}{tag}", flush=True)

    return model, actor, critic, env, hist


def smoke_train(cfg, *, device="cpu", seed=0):
    """A tiny end-to-end smoke run: a few updates on a synthetic buffer, no env. Used by the test.

    Builds a small fake replay buffer of random frames so the whole pipeline (world-model loss,
    imagine_dynamics, dynamics-backprop actor loss, critic loss, EMA) runs a handful of steps on CPU
    without dm_control. It is not a learning run; it only checks the loop executes without error.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = WorldModel(cfg).to(device)
    actor = ContActor(cfg).to(device)
    critic = Critic(cfg).to(device)
    critic_slow = copy.deepcopy(critic).to(device)
    for p in critic_slow.parameters():
        p.requires_grad_(False)
    opt_wm = torch.optim.Adam(model.parameters(), 1e-4)
    opt_a = torch.optim.Adam(actor.parameters(), 4e-5)
    opt_c = torch.optim.Adam(critic.parameters(), 4e-5)
    norm = ReturnNormalizer(cfg.ret_ema_decay)
    bins = critic.bins

    T = SEQ_LEN + 5
    ep = {
        "obs": np.random.rand(T, cfg.obs_ch, cfg.obs_size, cfg.obs_size).astype(np.float32),
        "actions": np.random.uniform(-1, 1, (T, cfg.action_dim)).astype(np.float32),
        "rewards": np.random.rand(T).astype(np.float32),
        "conts": np.ones(T, np.float32),
    }
    buffer = [ep, ep]

    for step in range(3):
        total, _ = model.loss(sample_batch(buffer, device))
        opt_wm.zero_grad()
        total.backward()
        opt_wm.step()

        h_pool, z_pool = _encode_starts(model, sample_batch(buffer, device), device)
        idx = torch.randint(0, h_pool.shape[0], (32,), device=device)
        returns, ents, H_h, H_z = imagine_dynamics(model, actor, critic, h_pool[idx], z_pool[idx], cfg)
        ret_range = norm.update(returns.detach())
        a_loss = actor_loss_dynbackprop(returns, ents, cfg.ent_coef, ret_range)
        opt_a.zero_grad()
        a_loss.backward(retain_graph=True)
        opt_a.step()

        c_logits = critic.logits(H_h[:, :-1].detach(), H_z[:, :-1].detach())
        c_loss = critic_loss(c_logits, returns.detach(), bins)
        with torch.no_grad():
            slow = torch.softmax(critic_slow.logits(H_h[:, :-1].detach(), H_z[:, :-1].detach()), dim=-1)
        reg = critic_loss(c_logits, twohot_decode(slow, bins), bins)
        opt_c.zero_grad()
        (c_loss + reg).backward()
        opt_c.step()
        with torch.no_grad():
            for ps, pf in zip(critic_slow.parameters(), critic.parameters()):
                ps.mul_(0.98).add_(pf, alpha=0.02)
    return True
