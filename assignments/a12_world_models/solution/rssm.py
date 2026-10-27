"""The recurrent state-space model (RSSM): a block GRU state h and a categorical latent z.

Ported from r2dreamer's `rssm.py` (classes `Deter` and `RSSM`). The model state at step t is
(h_t, z_t): h_t is a 2048-wide deterministic vector carried by a block-diagonal GRU, z_t is 32
categorical variables with 16 classes each, stored one-hot as (32, 16).

One step with an observation (`obs_step`, the posterior path):
    h_t = f(h_{t-1}, z_{t-1}, a_{t-1})           forward_h, the block GRU
    z_t ~ q(z_t | h_t, e_t)                      posterior, e_t = encoder(o_t)
One step without an observation (`img_step`, the prior path used in imagination):
    h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
    z_t ~ p(z_t | h_t)                           prior
The feature every head reads is concat(flatten(z_t), h_t).

Episode starts: when is_first is set for a step, h, z and the previous action are zeroed before
the GRU step, so each episode starts from the same zero state (the reference's `initial: learned`
config key is unused; its `initial()` returns zeros).
"""

import torch
from torch import Tensor, nn

from nets import BlockLinear, categorical_kl, categorical_sample, rms_norm, unimix_logits, weight_init_


class BlockGRU(nn.Module):
    """h_t = f(h_{t-1}, z_{t-1}, a_{t-1}): the reference's `Deter` module.

    h, z and a are each projected to `hidden` units (Linear, RMSNorm, SiLU). The concatenated
    projection is copied to each of the G blocks next to that block's slice of h, passed through
    a block-linear layer, and a final block-linear layer produces GRU-style reset, candidate and
    update gates, block by block. Blocks only see their own slice of h, which makes the 2048-wide
    recurrence about G times cheaper than a dense GRU.
    """

    def __init__(self, deter: int, z_dim: int, act_dim: int, hidden: int, blocks: int, dyn_layers: int):
        super().__init__()
        self.blocks = int(blocks)
        self.in_h = nn.Sequential(nn.Linear(deter, hidden, bias=True), rms_norm(hidden), nn.SiLU())
        self.in_z = nn.Sequential(nn.Linear(z_dim, hidden, bias=True), rms_norm(hidden), nn.SiLU())
        self.in_a = nn.Sequential(nn.Linear(act_dim, hidden, bias=True), rms_norm(hidden), nn.SiLU())
        self.hid = nn.Sequential()
        in_ch = (3 * hidden + deter // self.blocks) * self.blocks
        for i in range(dyn_layers):
            self.hid.add_module(f"block_linear{i}", BlockLinear(in_ch, deter, self.blocks))
            self.hid.add_module(f"norm{i}", rms_norm(deter))
            self.hid.add_module(f"act{i}", nn.SiLU())
            in_ch = deter
        self.gates = BlockLinear(in_ch, 3 * deter, self.blocks)

    def _flat2group(self, x: Tensor) -> Tensor:
        return x.reshape(*x.shape[:-1], self.blocks, -1)

    def _group2flat(self, x: Tensor) -> Tensor:
        return x.reshape(*x.shape[:-2], -1)

    def forward(self, h: Tensor, z: Tensor, a: Tensor) -> Tensor:
        """h: (B, D), z: (B, S, K), a: (B, A) -> new h (B, D)."""
        B = a.shape[0]
        z = z.reshape(B, -1)
        # Divide by max(|a|, 1): for a 1-D action this equals clipping to [-1, 1], the range the
        # environment applies. The policy's Gaussian sample is not squashed, so this matters.
        a = a / torch.clip(torch.abs(a), min=1.0).detach()
        x = torch.cat([self.in_h(h), self.in_z(z), self.in_a(a)], -1)          # (B, 3U)
        x = x.unsqueeze(-2).expand(-1, self.blocks, -1)                         # (B, G, 3U)
        x = self._group2flat(torch.cat([self._flat2group(h), x], -1))           # (B, G*(D/G + 3U))
        x = self.gates(self.hid(x))                                             # (B, 3D)
        reset, cand, update = (self._group2flat(g) for g in torch.chunk(self._flat2group(x), 3, dim=-1))
        reset = torch.sigmoid(reset)
        cand = torch.tanh(reset * cand)
        # The -1 bias makes the update gate start near sigmoid(-1) = 0.27, so h changes slowly.
        update = torch.sigmoid(update - 1)
        return update * cand + (1 - update) * h


class RSSM(nn.Module):
    def __init__(self, cfg, embed_dim: int):
        super().__init__()
        self.stoch, self.discrete, self.deter = cfg.stoch, cfg.discrete, cfg.deter
        self.unimix = cfg.unimix
        self.z_dim = cfg.stoch * cfg.discrete
        self.feat_dim = self.z_dim + cfg.deter
        self.gru = BlockGRU(cfg.deter, self.z_dim, cfg.action_dim, cfg.hidden, cfg.blocks, cfg.dyn_layers)

        # Posterior logits from (h, embed); the reference's `_obs_net`.
        self.post_net = nn.Sequential()
        inp = cfg.deter + embed_dim
        for i in range(cfg.obs_layers):
            self.post_net.add_module(f"linear{i}", nn.Linear(inp, cfg.hidden, bias=True))
            self.post_net.add_module(f"norm{i}", rms_norm(cfg.hidden))
            self.post_net.add_module(f"act{i}", nn.SiLU())
            inp = cfg.hidden
        self.post_net.add_module("logits", nn.Linear(inp, self.z_dim, bias=True))

        # Prior logits from h alone; the reference's `_img_net`.
        self.prior_net = nn.Sequential()
        inp = cfg.deter
        for i in range(cfg.img_layers):
            self.prior_net.add_module(f"linear{i}", nn.Linear(inp, cfg.hidden, bias=True))
            self.prior_net.add_module(f"norm{i}", rms_norm(cfg.hidden))
            self.prior_net.add_module(f"act{i}", nn.SiLU())
            inp = cfg.hidden
        self.prior_net.add_module("logits", nn.Linear(inp, self.z_dim))
        self.apply(weight_init_)

    def initial(self, batch_size: int, device=None):
        """Zero state: h (B, D), z (B, S, K)."""
        h = torch.zeros(batch_size, self.deter, dtype=torch.float32, device=device)
        z = torch.zeros(batch_size, self.stoch, self.discrete, dtype=torch.float32, device=device)
        return h, z

    def _logits(self, x: Tensor) -> Tensor:
        return x.reshape(*x.shape[:-1], self.stoch, self.discrete)

    def forward_h(self, h: Tensor, z: Tensor, a: Tensor) -> Tensor:
        """Advance the deterministic state one step with the previous latent and action."""
        return self.gru(h, z, a)

    def prior(self, h: Tensor):
        """z ~ p(z | h). Returns (z one-hot (..., S, K), logits (..., S, K))."""
        logits = self._logits(self.prior_net(h))
        return categorical_sample(logits, self.unimix), logits

    def posterior(self, h: Tensor, embed: Tensor):
        """z ~ q(z | h, e). Returns (z one-hot (..., S, K), logits (..., S, K))."""
        logits = self._logits(self.post_net(torch.cat([h, embed], dim=-1)))
        return categorical_sample(logits, self.unimix), logits

    def obs_step(self, h, z, prev_action, embed, is_first):
        """One posterior step. is_first: (B,) bool; where set, the state and action are zeroed first."""
        reset = is_first.reshape(-1, 1)
        h = torch.where(reset, torch.zeros_like(h), h)
        z = torch.where(reset.unsqueeze(-1), torch.zeros_like(z), z)
        prev_action = torch.where(reset, torch.zeros_like(prev_action), prev_action)
        h = self.forward_h(h, z, prev_action)
        z, logits = self.posterior(h, embed)
        return h, z, logits

    def img_step(self, h, z, prev_action):
        """One prior step (no observation)."""
        h = self.forward_h(h, z, prev_action)
        z, _ = self.prior(h)
        return h, z

    def observe(self, embed, actions, h0, z0, is_first):
        """Posterior rollout over a sequence.

        Args:
            embed: (B, T, E). actions: (B, T, A), actions[:, t] is the action taken BEFORE frame t.
            h0, z0: (B, D), (B, S, K) state before the first step. is_first: (B, T) bool.
        Returns:
            hs (B, T, D), zs (B, T, S, K), post_logits (B, T, S, K).
        """
        h, z = h0, z0
        hs, zs, logits = [], [], []
        for t in range(actions.shape[1]):
            h, z, logit = self.obs_step(h, z, actions[:, t], embed[:, t], is_first[:, t])
            hs.append(h)
            zs.append(z)
            logits.append(logit)
        return torch.stack(hs, 1), torch.stack(zs, 1), torch.stack(logits, 1)

    def imagine_with_action(self, h, z, actions):
        """Prior rollout under a given action sequence (B, T, A). Returns hs, zs."""
        hs, zs = [], []
        for t in range(actions.shape[1]):
            h, z = self.img_step(h, z, actions[:, t])
            hs.append(h)
            zs.append(z)
        return torch.stack(hs, 1), torch.stack(zs, 1)

    def get_feat(self, h: Tensor, z: Tensor) -> Tensor:
        """concat(flatten(z), h): (..., S*K + D)."""
        return torch.cat([z.reshape(*z.shape[:-2], self.z_dim), h], -1)


def kl_loss(post_logits: Tensor, prior_logits: Tensor, free: float, unimix: float):
    """DreamerV3 KL terms with free bits. Returns (dyn_loss, rep_loss), each (B, T).

    dyn = max(free, KL(sg(q) || p)) trains the prior toward the posterior.
    rep = max(free, KL(q || sg(p))) trains the posterior (and encoder) toward the prior.
    The KL is taken between the unimix distributions actually sampled from, summed over the 32
    categoricals. torch.clip has zero gradient below the floor, so a term already under 1 nat
    stops pushing.
    """
    post = unimix_logits(post_logits, unimix)
    prior = unimix_logits(prior_logits, unimix)
    rep = categorical_kl(post, prior.detach()).sum(-1)
    dyn = categorical_kl(post.detach(), prior).sum(-1)
    return torch.clip(dyn, min=free), torch.clip(rep, min=free)
