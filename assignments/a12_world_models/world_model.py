"""The world-model ELBO: reconstruction + reward + continuation, minus a balanced KL.

The world model is a sequence VAE. The posterior q(z_t | h_t, o_t) sees each frame; the decoder
reconstructs the frame from (h_t, z_t); the reward and continuation heads predict r_t and c_t.
The KL trains the prior p(z_t | h_t) toward the posterior so imagination (prior only) stays
grounded.

DreamerV3 splits the KL into two separately-weighted terms with stop-gradient on the opposite
side (arXiv:2301.04104 eq. 2-3), NOT DreamerV2's single 0.8/0.2 balance:
  dynamics loss      L_dyn = max(free_bits, KL[sg(q) || p])   trains the prior toward q,
  representation loss L_rep = max(free_bits, KL[q || sg(p)])  trains the posterior toward p.
The per-head KLs are summed over the n_cat heads first (the factorized posterior makes the joint
KL the sum of per-head KLs), then the free-bits max clips that single summed scalar per term at
1 nat. The total is 0.5 * L_dyn + 0.1 * L_rep; the 5:1 ratio moves the prior faster.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from nets import Decoder, Encoder, symlog, twohot_loss, value_bins
from rssm import RSSMCell


def _categorical_kl(logits_q: Tensor, logits_p: Tensor) -> Tensor:
    """KL[q || p] for n_cat independent categoricals, summed over heads. Returns (B,).

    logits are (B, n_cat, n_cls). Per head, KL = sum_c q_c (log q_c - log p_c); the joint KL of
    the factorized distribution is the sum over heads.
    """
    logq = F.log_softmax(logits_q, dim=-1)
    logp = F.log_softmax(logits_p, dim=-1)
    q = logq.exp()
    per_head = (q * (logq - logp)).sum(dim=-1)   # (B, n_cat)
    return per_head.sum(dim=-1)                   # (B,)


def kl_loss(post_logits: Tensor, prior_logits: Tensor, free_bits: float,
            beta_dyn: float, beta_rep: float, n_cat: int, n_cls: int):
    """DreamerV3 two-term KL with free bits and per-term weighting.

    Args:
        post_logits, prior_logits: (..., n_cat*n_cls) posterior q and prior p logits.
        free_bits: nats floor on the SUMMED-over-heads KL per term (clip on the total, not per-head).
        beta_dyn, beta_rep: weights on the dynamics and representation terms (0.5, 0.1).
        n_cat, n_cls: categorical layout.
    Returns:
        scalar weighted KL, and a dict {"dyn": L_dyn, "rep": L_rep} of the clipped per-term means.
    """
    raise NotImplementedError(
        "implement kl_loss (DreamerV3 eq. 2-3): reshape to (-1, n_cat, n_cls); use the provided "
        "_categorical_kl (it sums per-head KL over heads). Dynamics term clips "
        "max(free_bits, KL[sg(q) || p]); representation term clips max(free_bits, KL[q || sg(p)]) "
        "- the free-bits max is on the SUMMED scalar per term, NOT per head. Average over batch, "
        "return beta_dyn*L_dyn + beta_rep*L_rep and a dict {'dyn': L_dyn, 'rep': L_rep}"
    )


class WorldModel(nn.Module):
    """Encoder, RSSM cell, decoder, and two-hot reward / Bernoulli-logit continuation heads."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.rssm = RSSMCell(cfg)
        state_dim = cfg.h_dim + cfg.n_cat * cfg.n_cls
        self.reward_head = nn.Sequential(
            nn.Linear(state_dim, cfg.h_dim), nn.SiLU(), nn.Linear(cfg.h_dim, cfg.n_bins)
        )
        self.cont_head = nn.Sequential(
            nn.Linear(state_dim, cfg.h_dim), nn.SiLU(), nn.Linear(cfg.h_dim, 1)
        )
        self.register_buffer("bins", value_bins(cfg))

    def encode_seq(self, obs: Tensor) -> Tensor:
        """(B, T, 3, S, S) -> (B, T, embed_dim) by folding time into the batch for the CNN."""
        B, T = obs.shape[:2]
        e = self.encoder(obs.reshape(B * T, *obs.shape[2:]))
        return e.reshape(B, T, -1)

    def loss(self, batch, greedy: bool = False):
        """The full world-model ELBO on a batch of sequences.

        batch holds obs (B, T, 3, S, S), actions (B, T), rewards (B, T), conts (B, T) as tensors.
        Encode the frames, run the posterior over the sequence, decode the states, predict reward
        and continuation, and add the balanced KL.

        Reconstruction is MSE against symlog(obs) (the symexp is applied at viz time). Reward uses
        the two-hot cross-entropy over value-space bins. Continuation is a Bernoulli-logit BCE.

        Returns (total_loss, parts_dict).
        """
        raise NotImplementedError(
            "implement the ELBO assembly: encode_seq(obs) -> embeds; rssm.observe(...) -> "
            "hs, zs, prior_logits, post_logits; states = cat([hs, zs]). Reconstruction = "
            "mse(decoder(states), symlog(obs)); reward = twohot_loss(reward_head(states), rewards, "
            "bins).mean(); continuation = BCE-with-logits(cont_head(states), conts); add "
            "kl_loss(post_logits, prior_logits, free_bits, kl_dyn_scale, kl_rep_scale, n_cat, "
            "n_cls). total = recon + reward + cont + kl; return (total, parts) with the parts "
            "dict carrying recon/reward/cont/kl/kl_dyn/kl_rep for logging"
        )

    def encode_start(self, obs0: Tensor):
        """Encode one real frame to a posterior start state (h, z) for imagination. Provided.

        obs0 is (B, 3, S, S). Runs one posterior step from the zero initial state with a zero
        previous action, returning (h, z) to seed imagine().
        """
        B = obs0.shape[0]
        embed = self.encoder(obs0)
        h0, z0 = self.rssm.initial_state(B, device=obs0.device, dtype=obs0.dtype)
        a0 = torch.zeros(B, dtype=torch.long, device=obs0.device)
        h = self.rssm.forward_h(h0, z0, a0)
        _, z, _ = self.rssm.posterior(h, embed)
        return h, z
