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

Holes in this file (each raises NotImplementedError until written): RSSM.prior, RSSM.posterior,
RSSM.obs_step, RSSM.img_step and kl_loss. The block GRU, the logit networks and the sequence
loops (observe, imagine_with_action) are provided.

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
        """z ~ p(z | h), the dynamics predictor used when there is no observation.

        logits = self.prior_net(h), reshaped to (..., S, K) with self._logits; z is a
        straight-through one-hot sample from the unimix categorical (nets.categorical_sample with
        self.unimix).

        Args:
            h: (..., D) deterministic state.
        Returns:
            z (..., S, K) one-hot sample, logits (..., S, K) raw (before unimix).
        """
        raise NotImplementedError("implement the prior: the categorical latent predicted from h alone")

    def posterior(self, h: Tensor, embed: Tensor):
        """z ~ q(z | h, e), the representation model that sees the current frame's embedding.

        logits = self.post_net(concat([h, embed], -1)) (h first), reshaped to (..., S, K); z is a
        straight-through one-hot sample from the unimix categorical, as in prior.

        Args:
            h: (..., D). embed: (..., E) encoder output for the current frame.
        Returns:
            z (..., S, K) one-hot sample, logits (..., S, K) raw (before unimix).
        """
        raise NotImplementedError("implement the posterior: the categorical latent inferred from h and the image embedding")

    def obs_step(self, h, z, prev_action, embed, is_first):
        """One filtering step with an observation:

            h_t = forward_h(h_{t-1}, z_{t-1}, a_{t-1});   z_t ~ q(z_t | h_t, e_t)

        Episode starts: for every row where is_first is set, replace h_{t-1}, z_{t-1} and a_{t-1}
        by zeros BEFORE the GRU step (torch.where with the flag broadcast to each shape), so every
        episode starts from the same zero state no matter what the previous episode left behind.
        Rows without the flag are untouched.

        Args:
            h: (B, D). z: (B, S, K). prev_action: (B, A). embed: (B, E). is_first: (B,) bool.
        Returns:
            h_t (B, D), z_t (B, S, K) posterior sample, posterior logits (B, S, K).
        """
        raise NotImplementedError("implement one observation step of the RSSM")

    def img_step(self, h, z, prev_action):
        """One imagination step without an observation:

            h_t = forward_h(h_{t-1}, z_{t-1}, a_{t-1});   z_t ~ p(z_t | h_t)

        No is_first handling: imagined rollouts never cross an episode start.

        Args:
            h: (B, D). z: (B, S, K). prev_action: (B, A).
        Returns:
            h_t (B, D), z_t (B, S, K) prior sample.
        """
        raise NotImplementedError("implement one imagination step of the RSSM")

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
    """DreamerV3 KL terms with free bits (DreamerV3 v2, eq. 3). Returns (dyn_loss, rep_loss).

        dyn = max(free, KL(sg(q) || p))     trains the prior p toward the posterior q
        rep = max(free, KL(q || sg(p)))     trains the posterior (and the encoder) toward the prior

    sg is stop-gradient (.detach()). q and p are the unimix distributions actually sampled from:
    pass both logit tensors through nets.unimix_logits first, then use nets.categorical_kl, which
    takes (left, right) logits and returns KL(left || right) summed over the class axis. Sum the
    result over the S = 32 categoricals (the second-to-last axis of the logits), so each term is
    one number per time step. The two terms have the same value and differ only in where their
    gradient goes.

    Free bits: torch.clip(kl, min=free) with free = 1 nat. Below the floor the clip has zero
    gradient, so a KL already under 1 nat stops pulling the two distributions together.
    The weights beta_dyn = 1.0 and beta_rep = 0.1 are applied later, by cfg.loss_scales.

    Args:
        post_logits, prior_logits: (B, T, S, K) raw logits. free: floor in nats. unimix: ratio.
    Returns:
        dyn_loss, rep_loss: each (B, T).
    """
    raise NotImplementedError("implement the balanced KL loss with free bits")
