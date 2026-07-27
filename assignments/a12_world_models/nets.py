"""Scalar-target encodings and the straight-through categorical sampler.

DreamerV3 makes one fixed hyperparameter set work across reward scales with two ideas this file
implements. symlog compresses large magnitudes while staying near the identity around zero, with an
exact inverse symexp. Two-hot encoding turns scalar regression into classification over a fixed set
of bins, so the loss is a cross-entropy that cannot blow up on an outlier the way a squared error
does.

The bins live in SYMLOG space. twohot_encode pushes the target through symlog before splitting it
over the bins, and twohot_decode applies symexp once after the bin expectation, matching the
canonical DreamerV3 DiscDist. Keeping the bins in symlog space also keeps the decode bounded.
symlog/symexp are also applied to the reconstruction MSE targets in world_model.py.

The straight-through estimator lets a discrete categorical sample carry gradients: the forward
pass uses a hard one-hot, the backward pass uses the (unimix-blended) softmax probabilities.

The math is in the symlog and two-hot encoding, and the categorical latents and the straight-through
estimator, sections of the README.

Also provided here: the CNN encoder and decoder (conv plumbing, not the lesson) and twohot_loss.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def symlog(x: Tensor) -> Tensor:
    """symlog: compresses large magnitudes, ~identity near 0. See the symlog and two-hot encoding section of the README."""
    raise NotImplementedError("implement symlog, the signed logarithmic compressor")


def symexp(x: Tensor) -> Tensor:
    """symexp: the exact inverse of symlog. See the symlog and two-hot encoding section of the README."""
    raise NotImplementedError("implement symexp, the inverse of symlog")


def value_bins(cfg) -> Tensor:
    """The two-hot bin positions in symlog space: linspace(bin_lo, bin_hi, n_bins).

    The bins span [-20, 20] in symlog space. twohot_encode pushes the target through symlog onto
    these bins; twohot_decode applies symexp after the expectation. Returns a (n_bins,) tensor.
    """
    return torch.linspace(cfg.bin_lo, cfg.bin_hi, cfg.n_bins)


def twohot_encode(y: Tensor, bins: Tensor) -> Tensor:
    """Soft two-hot label of target y over symlog-space bins.

    The target is pushed through symlog before it is split across the two bracketing bins, so it
    pairs with the symexp in twohot_decode and the round-trip is exact.

    Args:
        y: (...) raw target values.
        bins: (n_bins,) monotonically increasing symlog-space bin positions.
    Returns:
        (..., n_bins) two-hot soft labels summing to 1 along the last axis.

    See the symlog and two-hot encoding section of the README.
    """
    raise NotImplementedError(
        "implement twohot_encode: ys = symlog(y), clamp into [bins[0], bins[-1]], find the "
        "bracketing bins b_lo <= ys <= b_hi (searchsorted), put weight (ys - b_lo)/(b_hi - b_lo) "
        "on hi and the complement on lo; return (..., n_bins) summing to 1 along the last axis"
    )


def twohot_decode(probs: Tensor, bins: Tensor) -> Tensor:
    """Decode a bin distribution back to a scalar in value space.

    The bins are in symlog space, so the expectation is taken over the symlog-space bins and then
    inverted with symexp. Keeping the expectation in symlog space keeps the decode bounded even when
    a little probability mass lands on the extreme bins.

    Args:
        probs: (..., n_bins) bin probabilities (already softmaxed).
        bins: (n_bins,) symlog-space bin positions.
    Returns:
        (...) the decoded scalar in value space.

    See the symlog and two-hot encoding section of the README.
    """
    raise NotImplementedError(
        "implement twohot_decode: symexp(sum(probs * bins)) over the last axis"
    )


def twohot_loss(logits: Tensor, target: Tensor, bins: Tensor) -> Tensor:
    """Cross-entropy of log_softmax(logits) against the two-hot label of target.

    Provided wrapper used by the reward, critic, and continuation heads. The symexp lives in the
    bin construction, not here, so this takes the raw target directly.
    """
    logp = F.log_softmax(logits, dim=-1)
    label = twohot_encode(target, bins)
    return -(label * logp).sum(dim=-1)


def categorical_sample(logits: Tensor, unimix: float, n_cat: int, n_cls: int, greedy: bool = False):
    """Straight-through sample from n_cat categoricals, each over n_cls classes.

    The class probabilities are blended with a uniform distribution (the unimix floor) so no logit
    can be driven to -infinity, and the returned sample is a straight-through estimate: a hard
    one-hot in the forward pass whose gradient is that of the blended probabilities.

    The greedy path keeps the graph deterministic for gradcheck and the shape/straight-through
    tests; training uses the sampled path under a fixed global seed.

    Args:
        logits: (B, n_cat * n_cls) raw logits.
        unimix: uniform mixture weight (DreamerV3 default 0.01).
        n_cat, n_cls: categorical head count and class count.
        greedy: if True, take argmax instead of sampling.
    Returns:
        z: (B, n_cat * n_cls) straight-through one-hot sample.
        probs: (B, n_cat, n_cls) the unimix-blended probabilities (used by the KL).

    See the logits, one-hot samples, and the straight-through estimator section of the README.
    """
    raise NotImplementedError(
        "implement categorical_sample: reshape logits to (B, n_cat, n_cls), softmax, blend "
        "probs = (1-unimix)*softmax + unimix/n_cls; draw a one-hot (argmax when greedy, else "
        "torch.multinomial); return z = (onehot - probs).detach() + probs flattened to "
        "(B, n_cat*n_cls), and the blended probs (B, n_cat, n_cls)"
    )


class Encoder(nn.Module):
    """CNN that maps an obs (B, 3, 64, 64) to an embedding (B, embed_dim). Provided plumbing.

    Four stride-2 convs with channel widths 32/64/128/256 take 64x64 down to 4x4, then a linear
    projects the flattened 256*4*4 features to embed_dim. This is the standard DreamerV3 64x64
    pixel encoder.
    """

    def __init__(self, cfg):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cfg.obs_ch, 32, 4, stride=2, padding=1), nn.SiLU(),   # S -> S/2
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.SiLU(),           # S/2 -> S/4
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.SiLU(),          # S/4 -> S/8
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.SiLU(),         # S/8 -> S/16
        )
        feat_hw = cfg.obs_size // 16     # 64 -> 4; the four stride-2 convs each halve the side
        self.proj = nn.Linear(256 * feat_hw * feat_hw, cfg.embed_dim)

    def forward(self, obs: Tensor) -> Tensor:
        return self.proj(self.conv(obs).flatten(1))


class Decoder(nn.Module):
    """Mirror CNN that maps a state (B, h_dim + n_cat*n_cls) to an obs (B, 3, S, S). Provided.

    A linear lifts the state to 256 x (S/16) x (S/16), then four stride-2 transposed convs
    (256/128/64/32 -> 3) bring it back up to S x S, mirroring the encoder.
    """

    def __init__(self, cfg):
        super().__init__()
        self.feat_hw = cfg.obs_size // 16
        in_dim = cfg.h_dim + cfg.n_cat * cfg.n_cls
        self.proj = nn.Linear(in_dim, 256 * self.feat_hw * self.feat_hw)
        self.deconv = nn.Sequential(
            nn.SiLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.SiLU(),   # S/16 -> S/8
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.SiLU(),    # S/8 -> S/4
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.SiLU(),     # S/4 -> S/2
            nn.ConvTranspose2d(32, cfg.obs_ch, 4, stride=2, padding=1),        # S/2 -> S
        )

    def forward(self, state: Tensor) -> Tensor:
        x = self.proj(state).view(-1, 256, self.feat_hw, self.feat_hw)
        return self.deconv(x)
