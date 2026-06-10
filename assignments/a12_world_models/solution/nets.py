"""Scalar-target encodings and the straight-through categorical sampler. Answer key.

DreamerV3 makes one fixed hyperparameter set work across reward scales with two ideas this file
implements. symlog compresses large magnitudes: symlog(x) = sign(x) * log(|x| + 1), with exact
inverse symexp(x) = sign(x) * (exp(|x|) - 1). Two-hot encoding turns scalar regression into
classification over a fixed set of bins, so the loss is a cross-entropy that cannot blow up on an
outlier the way a squared error does.

The bins live in SYMLOG space: bins = linspace(-20, 20, n_bins). twohot_encode pushes the target
through symlog before splitting it over the bins, and twohot_decode applies symexp once after the
bin expectation, matching the canonical DreamerV3 DiscDist (transfwd=symlog, transbwd=symexp). The
round-trip is exact because a clean two-hot label has its expectation at symlog(y) and
symexp(symlog(y)) = y. Symlog-space bins also keep the decode bounded; value-space bins would put
the outer buckets at symexp(20) ~ 5e8, so any residual softmax tail mass there would dominate the
expectation. symlog/symexp are also applied to the reconstruction MSE targets in world_model.py.

The straight-through estimator lets a discrete categorical sample carry gradients: the forward
pass uses a hard one-hot, the backward pass uses the (unimix-blended) softmax probabilities.

Also provided here: the CNN encoder and decoder (conv plumbing, not the lesson) and twohot_loss.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def symlog(x: Tensor) -> Tensor:
    """symlog(x) = sign(x) * log(|x| + 1). Compresses large magnitudes, ~identity near 0."""
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x: Tensor) -> Tensor:
    """symexp(x) = sign(x) * (exp(|x|) - 1). Exact inverse of symlog."""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


def value_bins(cfg) -> Tensor:
    """The two-hot bin positions in symlog space: linspace(bin_lo, bin_hi, n_bins).

    The bins span [-20, 20] in symlog space. twohot_encode pushes the target through symlog onto
    these bins; twohot_decode applies symexp after the expectation. Returns a (n_bins,) tensor.
    """
    return torch.linspace(cfg.bin_lo, cfg.bin_hi, cfg.n_bins)


def twohot_encode(y: Tensor, bins: Tensor) -> Tensor:
    """Soft two-hot label of target y over symlog-space bins.

    Push y through symlog, clamp into the bin range, and split it across the two bracketing bins
    b_lo <= symlog(y) <= b_hi with weight (b_hi - symlog(y))/(b_hi - b_lo) on lo and the complement
    on hi. The symlog here pairs with the symexp in twohot_decode, so the round-trip is exact.

    Args:
        y: (...) raw target values.
        bins: (n_bins,) monotonically increasing symlog-space bin positions.
    Returns:
        (..., n_bins) two-hot soft labels summing to 1 along the last axis.
    """
    n = bins.shape[0]
    ys = symlog(y)
    yc = torch.clamp(ys, bins[0], bins[-1])
    # Upper bin index: first bin >= symlog(y). searchsorted on the monotone bins.
    hi = torch.searchsorted(bins, yc.contiguous(), right=False)
    hi = torch.clamp(hi, 1, n - 1)
    lo = hi - 1
    b_lo = bins[lo]
    b_hi = bins[hi]
    w_hi = (yc - b_lo) / (b_hi - b_lo)
    w_lo = 1.0 - w_hi
    out = torch.zeros(*y.shape, n, dtype=bins.dtype, device=bins.device)
    out.scatter_(-1, lo.unsqueeze(-1), w_lo.unsqueeze(-1))
    out.scatter_(-1, hi.unsqueeze(-1), w_hi.unsqueeze(-1))
    return out


def twohot_decode(probs: Tensor, bins: Tensor) -> Tensor:
    """Decode a bin distribution: symexp of the expectation symexp(sum(probs * bins)).

    The bins are in symlog space, so take the expected symlog value and invert with symexp. Keeping
    the expectation in symlog space (bins at +/- 20, not +/- 5e8) keeps the decode bounded even when
    a little probability mass lands on the extreme bins.

    Args:
        probs: (..., n_bins) bin probabilities (already softmaxed).
        bins: (n_bins,) symlog-space bin positions.
    Returns:
        (...) the decoded scalar in value space.
    """
    return symexp((probs * bins).sum(dim=-1))


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

    Reshape logits to (B, n_cat, n_cls), softmax, blend with a uniform distribution
    (unimix floor): probs = (1 - unimix) * softmax + unimix / n_cls. Draw a one-hot sample
    (argmax when greedy, else multinomial), then return the straight-through estimate
    z = (onehot - probs).detach() + probs flattened to (B, n_cat * n_cls), plus the blended probs.

    The greedy path keeps the graph deterministic for gradcheck and the shape/straight-through
    tests; training uses the multinomial path under a fixed global seed.

    Args:
        logits: (B, n_cat * n_cls) raw logits.
        unimix: uniform mixture weight (DreamerV3 default 0.01).
        n_cat, n_cls: categorical head count and class count.
        greedy: if True, take argmax instead of sampling.
    Returns:
        z: (B, n_cat * n_cls) straight-through one-hot sample.
        probs: (B, n_cat, n_cls) the unimix-blended probabilities (used by the KL).
    """
    B = logits.shape[0]
    logits = logits.view(B, n_cat, n_cls)
    soft = F.softmax(logits, dim=-1)
    probs = (1.0 - unimix) * soft + unimix / n_cls
    if greedy:
        idx = probs.argmax(dim=-1)
    else:
        flat = probs.reshape(B * n_cat, n_cls)
        idx = torch.multinomial(flat, 1).reshape(B, n_cat)
    onehot = F.one_hot(idx, n_cls).to(probs.dtype)
    z = (onehot - probs).detach() + probs           # straight-through
    z = z.reshape(B, n_cat * n_cls)
    return z, probs


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
