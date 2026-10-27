"""Building blocks: squashing functions, distributions, and the networks of the DreamerV3 agent.

Holes in this file (each raises NotImplementedError until written): symlog, symexp,
unimix_logits, categorical_sample, twohot_encode, twohot_decode. Everything else is provided.
The README sections on symlog, two-hot targets and categorical latents give the background.

Ported from r2dreamer (commit becf338): `distributions.py` (symlog, unimix, straight-through
categorical, two-hot over symexp bins, bounded normal actor), `networks.py` (block-linear layer,
conv encoder/decoder, MLP heads) and `tools.weight_init_`. The numerics match the reference
operation for operation, so a state dict copied from the reference gives the same outputs.

Image layout is channels-first: the encoder reads (..., 3, 64, 64) floats in [0, 1] and the
decoder writes the same shape. The reference stores frames channels-last and permutes inside the
encoder and decoder; the convolutions see identical NCHW tensors in both.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch import distributions as td
from torch.nn import init as nn_init


# ----------------------------------------------------------------------------- squashing


def symlog(x: Tensor) -> Tensor:
    """symlog(x) = sign(x) * log(1 + |x|), elementwise. Same shape as x.

    Near 0 it behaves like x (slope 1); for large |x| it grows like log|x|, so targets spanning
    many orders of magnitude land in a small range. Use torch.log1p for accuracy near 0.
    """
    raise NotImplementedError("implement symlog, the signed logarithmic compressor")


def symexp(x: Tensor) -> Tensor:
    """symexp(x) = sign(x) * (exp(|x|) - 1), elementwise: the exact inverse of symlog.

    Use torch.expm1 for accuracy near 0. symexp_bins below builds the two-hot bin centers with it.
    """
    raise NotImplementedError("implement symexp, the inverse of symlog")


# ----------------------------------------------------------------------------- categoricals


def unimix_logits(logits: Tensor, ratio: float) -> Tensor:
    """Mix `ratio` of a uniform distribution into softmax(logits); return log-probabilities.

        p = (1 - ratio) * softmax(logits) + ratio / K,     return log(p)

    over the last axis (K classes). Compute the softmax in float32 (logits.float()). With ratio
    0.01 and K = 16 every class keeps probability >= 0.01 / 16 = 6.25e-4, so log p >= -7.4 and
    the KL between two such distributions stays finite.

    Args:
        logits: (..., K). ratio: the mixing weight (cfg.unimix = 0.01).
    Returns:
        (..., K) float32 log-probabilities of the mixed distribution.
    """
    raise NotImplementedError("implement the uniform-mixture logits")


def categorical_sample(logits: Tensor, unimix: float) -> Tensor:
    """Straight-through one-hot sample from the unimix categorical over the last axis.

    Steps, matching the reference so that the random-number stream and rounding are identical:

    1. logp = unimix_logits(logits, unimix), then renormalize it,
       logp = logp - logsumexp(logp, -1) (torch.distributions.Categorical does this to its logits
       and the reference samples through it; the step changes values only by rounding).
    2. probs = softmax(logp, -1).
    3. Draw one class per categorical with torch.multinomial(probs.reshape(-1, K), 1), reshape back
       to probs.shape[:-1], and one-hot it to (..., K) in probs.dtype.
    4. Return value.detach() + (probs - probs.detach()), with the parentheses as written.

    The forward value of step 4 is the hard one-hot, because (probs - probs.detach()) is exactly 0.
    The backward pass sees only the probs term, so d(sample)/d(logits) = d(probs)/d(logits): the
    straight-through estimator. Without the parentheses, value + probs - probs can round to a
    value that is not exactly one-hot.

    Args:
        logits: (..., K) raw logits; for z they are (B, S, K) with S = 32 categoricals.
        unimix: uniform mixing weight.
    Returns:
        (..., K) float one-hot sample carrying the straight-through gradient.
    """
    raise NotImplementedError("implement straight-through categorical sampling")


def categorical_kl(logits_left: Tensor, logits_right: Tensor) -> Tensor:
    """KL(left || right) between categoricals over the last axis. Returns shape (...)."""
    logprob_left = torch.log_softmax(logits_left, -1)
    logprob_right = torch.log_softmax(logits_right, -1)
    prob = torch.softmax(logits_left, -1)
    return (prob * (logprob_left - logprob_right)).sum(-1)


def categorical_entropy(logits: Tensor, unimix: float) -> Tensor:
    """Entropy of the unimix categorical, summed over the variable axis (-2). For logging."""
    logp = unimix_logits(logits, unimix)
    return -(logp.exp() * logp).sum((-2, -1))


# ----------------------------------------------------------------------------- two-hot


def symexp_bins(n: int, device=None) -> Tensor:
    """n bin centers, symexp of an even grid on [-20, 20]; exactly antisymmetric.

    The bins live in raw (symexp) space and the two-hot weights are linear in raw space between
    the two neighboring bins. This is the r2dreamer / DreamerV3 `symexp_twohot` layout and
    matches DreamerV3 paper v2 (eq. 10); paper v1 used bins spaced linearly in symlog space.
    """
    if n % 2 == 1:
        half = torch.linspace(-20, 0, (n - 1) // 2 + 1, dtype=torch.float32, device=device)
        half = symexp(half)
        return torch.cat([half, -half[:-1].flip(dims=(0,))], 0)
    half = torch.linspace(-20, 0, n // 2, dtype=torch.float32, device=device)
    half = symexp(half)
    return torch.cat([half, -half.flip(dims=(0,))], 0)


def twohot_encode(y: Tensor, bins: Tensor) -> Tensor:
    """Two-hot label of scalar targets y over increasing bin centers.

    Let b_lo be the largest bin <= y and b_hi the smallest bin > y. The label puts
    (b_hi - y) / (b_hi - b_lo) on b_lo and (y - b_lo) / (b_hi - b_lo) on b_hi, so the label's
    expected bin value, sum_i label_i * bins_i, equals y exactly. The weights are linear in RAW
    space between the two bins (the bins themselves are symexp-spaced, see symexp_bins).

    Reference conventions, kept for bit-identical results:
    - Indices: below = (number of bins <= y) - 1, above = N - (number of bins > y), both clamped
      to [0, N - 1]. Compare in the bins' dtype: (bins <= y.unsqueeze(-1)).sum(-1).
    - If below == above (y lies exactly on a bin, or outside the range), both distances are set
      to 1, so the label is 0.5 + 0.5 on that one bin. A target outside [bins[0], bins[-1]] goes
      entirely to the end bin.
    - Otherwise dist_to_below = |bins[below] - y|, dist_to_above = |bins[above] - y|, and
      weight_below = dist_to_above / total, weight_above = dist_to_below / total.
    - Build the label as one_hot(below) * weight_below + one_hot(above) * weight_above, float32.

    Args:
        y: (...,) float32 targets.
        bins: (N,) increasing bin centers.
    Returns:
        (..., N) float32 label; each row sums to 1 and has at most two nonzero entries.
    """
    raise NotImplementedError("implement two-hot encoding")


def twohot_log_prob(logits: Tensor, target: Tensor, bins: Tensor) -> Tensor:
    """Log-likelihood of target under the two-hot distribution: sum_i label_i * log softmax_i.

    Args:
        logits: (..., N). target: (..., 1) float32. bins: (N,).
    Returns:
        (...,) log-likelihood; its negative is the cross-entropy loss.
    """
    logits = logits.float()
    label = twohot_encode(target.squeeze(-1).detach(), bins)
    log_pred = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    return (label * log_pred).sum(dim=-1)


def twohot_decode(logits: Tensor, bins: Tensor) -> Tensor:
    """Expected value of the bin distribution, sum_i softmax(logits)_i * bins_i, shape (..., 1).

    Compute probs = softmax(logits.float(), -1). The symexp bins reach about +-4.85e8 at the ends.
    Summing p_i * bins_i directly adds large terms of opposite sign and loses precision, so the
    reference uses the antisymmetry bins[N-1-i] = -bins[i] and sums over the lower half only:

        m = N // 2
        value = sum_{i < m} (p_i - p_{N-1-i}) * bins_i     (+ p_m * bins_m when N is odd)

    p_{N-1-i} for i < m is probs[..., N - m:] flipped on the last axis. Keep the last axis with
    keepdim=True so the result is (..., 1).

    Args:
        logits: (..., N) raw logits. bins: (N,) antisymmetric bin centers from symexp_bins.
    Returns:
        (..., 1) float32 decoded scalar.
    """
    raise NotImplementedError("implement two-hot decoding")


# ----------------------------------------------------------------------------- other distributions


def actor_dist(out: Tensor, min_std: float, max_std: float) -> td.Distribution:
    """Gaussian policy from the actor's raw output (..., 2A): mean tanh(mu), std in [min, max].

    The sample is not squashed: it can leave [-1, 1]. The environment clips it and the RSSM
    rescales it (see BlockGRU.forward). This is the reference's `bounded_normal`.
    """
    mean, std = torch.chunk(out, 2, dim=-1)
    std = (max_std - min_std) * torch.sigmoid(std + 2.0) + min_std
    return td.Independent(td.Normal(torch.tanh(mean.float()), std.float()), 1)


def bernoulli_dist(logits: Tensor) -> td.Distribution:
    """Continuation flag distribution, event shape (1,)."""
    return td.Independent(td.Bernoulli(logits=logits.float()), 1)


# ----------------------------------------------------------------------------- init and layers


def weight_init_(m: nn.Module, fan_type: str = "in") -> None:
    """Truncated normal with std 1.1368 / sqrt(fan_in), zero bias, RMSNorm scale 1."""
    if isinstance(m, nn.RMSNorm):
        with torch.no_grad():
            m.weight.fill_(1.0)
        return
    weight = getattr(m, "weight", None)
    if weight is None or weight.numel() == 0:
        return
    in_num, out_num = nn_init._calculate_fan_in_and_fan_out(weight)
    with torch.no_grad():
        fan = {"avg": (in_num + out_num) / 2, "in": in_num, "out": out_num}[fan_type]
        std = 1.1368 * np.sqrt(1 / fan)
        nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
        bias = getattr(m, "bias", None)
        if bias is not None:
            bias.fill_(0.0)


def rms_norm(dim: int) -> nn.RMSNorm:
    """RMSNorm with float32 scale and eps 1e-4, as used everywhere in the reference."""
    return nn.RMSNorm(dim, eps=1e-04, dtype=torch.float32)


class BlockLinear(nn.Module):
    """Block-diagonal linear layer: G independent (in/G -> out/G) maps on contiguous slices.

    The weight is stored as (out/G, in/G, G). torch's fan-in rule multiplies dims 1 and 2, so
    weight_init_ sees fan_in = in (the full width, not in/G). The reference relies on this layout
    and it is kept.
    """

    def __init__(self, in_ch: int, out_ch: int, blocks: int):
        super().__init__()
        self.in_ch, self.out_ch, self.blocks = int(in_ch), int(out_ch), int(blocks)
        self.weight = nn.Parameter(torch.empty(self.out_ch // self.blocks, self.in_ch // self.blocks, self.blocks))
        self.bias = nn.Parameter(torch.zeros(self.out_ch))

    def forward(self, x: Tensor) -> Tensor:
        batch_shape = x.shape[:-1]
        x = x.view(*batch_shape, self.blocks, self.in_ch // self.blocks)      # (..., G, I/G)
        x = torch.einsum("...gi,oig->...go", x, self.weight)                   # (..., G, O/G)
        return x.reshape(*batch_shape, self.out_ch) + self.bias


class Conv2dSamePad(nn.Conv2d):
    """Conv2d with TensorFlow 'SAME' padding (output size = ceil(input / stride))."""

    def _calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max(((i + s - 1) // s - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: Tensor) -> Tensor:
        ih, iw = x.size()[-2:]
        pad_h = self._calc_same_pad(ih, self.kernel_size[0], self.stride[0], self.dilation[0])
        pad_w = self._calc_same_pad(iw, self.kernel_size[1], self.stride[1], self.dilation[1])
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class RMSNorm2D(nn.RMSNorm):
    """RMSNorm over the channel axis of an NCHW tensor."""

    def __init__(self, ch: int):
        super().__init__(ch, eps=1e-04, dtype=torch.float32)

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ConvEncoder(nn.Module):
    """Four (conv 5x5 stride 1, maxpool 2, RMSNorm, SiLU) stages: 64x64x3 -> 4x4x64 -> 1024 vector."""

    def __init__(self, cfg):
        super().__init__()
        depths = [cfg.cnn_depth * m for m in cfg.cnn_mults]
        layers, in_dim, h = [], 3, cfg.obs_size
        for depth in depths:
            layers += [Conv2dSamePad(in_dim, depth, cfg.kernel_size, stride=1, bias=True),
                       nn.MaxPool2d(2, 2), RMSNorm2D(depth), nn.SiLU()]
            in_dim, h = depth, h // 2
        self.layers = nn.Sequential(*layers)
        self.out_dim = depths[-1] * h * h
        self.apply(weight_init_)

    def forward(self, image: Tensor) -> Tensor:
        """image: (..., 3, H, W) float in [0, 1] -> (..., out_dim)."""
        x = image - 0.5
        lead = x.shape[:-3]
        x = self.layers(x.reshape(-1, *x.shape[-3:]))
        x = x.reshape(x.shape[0], -1)
        return x.reshape(*lead, x.shape[-1])


class ConvDecoder(nn.Module):
    """(h, z) -> image. A 4x4x64 map is built from h (block-linear) plus z (MLP), then upsampled.

    Each upsampling stage is (nearest x2, conv 5x5, RMSNorm, SiLU); the last stage outputs 3
    channels through a sigmoid, so the reconstruction is in [0, 1].
    """

    def __init__(self, cfg):
        super().__init__()
        depths = [cfg.cnn_depth * m for m in cfg.cnn_mults]
        minres = cfg.obs_size // 2 ** len(depths)
        self.min_shape = (minres, minres, depths[-1])                          # (H_feat, W_feat, C_feat)
        self.bspace = cfg.dec_bspace
        flat = math.prod(self.min_shape)
        z_dim = cfg.stoch * cfg.discrete
        self.sp0 = BlockLinear(cfg.deter, flat, self.bspace)
        self.sp1 = nn.Sequential(nn.Linear(z_dim, 2 * cfg.units), rms_norm(2 * cfg.units), nn.SiLU())
        self.sp2 = nn.Linear(2 * cfg.units, flat)
        self.sp_norm = nn.Sequential(rms_norm(depths[-1]), nn.SiLU())
        layers, in_dim = [], depths[-1]
        for depth in reversed(depths[:-1]):
            layers += [nn.Upsample(scale_factor=2, mode="nearest"),
                       Conv2dSamePad(in_dim, depth, cfg.kernel_size, stride=1, bias=True),
                       RMSNorm2D(depth), nn.SiLU()]
            in_dim = depth
        layers += [nn.Upsample(scale_factor=2, mode="nearest"),
                   Conv2dSamePad(in_dim, 3, cfg.kernel_size, stride=1, bias=True)]
        self.layers = nn.Sequential(*layers)
        self.apply(weight_init_)

    def forward(self, h: Tensor, z: Tensor) -> Tensor:
        """h: (..., D), z: (..., S, K) -> (..., 3, H, W) in [0, 1]."""
        lead = h.shape[:-1]
        x0 = h.reshape(lead.numel(), h.shape[-1])
        x1 = z.reshape(lead.numel(), -1)
        H, W, C = self.min_shape
        # From h: block-linear, laid out so each block fills C/G channels of every pixel.
        x0 = self.sp0(x0).reshape(-1, self.bspace, H, W, C // self.bspace)
        x0 = x0.permute(0, 2, 3, 1, 4).reshape(-1, H, W, C)
        # From z: a two-layer MLP straight to the 4x4xC map.
        x1 = self.sp2(self.sp1(x1)).reshape(-1, H, W, C)
        x = self.sp_norm(x0 + x1).permute(0, 3, 1, 2)                          # (N, C, H, W)
        x = torch.sigmoid(self.layers(x))
        return x.reshape(*lead, *x.shape[1:])


class MLP(nn.Module):
    """`layers` x (Linear, RMSNorm, SiLU)."""

    def __init__(self, inp_dim: int, layers: int, units: int):
        super().__init__()
        self.layers = nn.Sequential()
        for i in range(layers):
            self.layers.add_module(f"linear{i}", nn.Linear(inp_dim, units, bias=True))
            self.layers.add_module(f"norm{i}", rms_norm(units))
            self.layers.add_module(f"act{i}", nn.SiLU())
            inp_dim = units
        self.out_dim = units

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class MLPHead(nn.Module):
    """MLP followed by a linear output layer whose initial weights are multiplied by `outscale`.

    outscale 0 (reward, critic) makes the head output all-zero logits at init, so the two-hot
    prediction starts at exactly 0.
    """

    def __init__(self, inp_dim: int, out_dim: int, layers: int, units: int, outscale: float = 1.0):
        super().__init__()
        self.mlp = MLP(inp_dim, layers, units)
        self.last = nn.Linear(self.mlp.out_dim, out_dim, bias=True)
        self.mlp.apply(weight_init_)
        self.last.apply(weight_init_)
        if outscale != 1.0:
            with torch.no_grad():
                self.last.weight.mul_(outscale)

    def forward(self, x: Tensor) -> Tensor:
        return self.last(self.mlp(x))
