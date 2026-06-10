"""Exact, training-free checks of symlog/symexp and the value-space two-hot encoding."""

import torch

from config import WorldModelConfig
from nets import (
    symexp,
    symlog,
    twohot_decode,
    twohot_encode,
    twohot_loss,
    value_bins,
)


def test_symexp_inverts_symlog():
    # Includes large magnitudes where the compression is strong.
    x = torch.tensor([0.0, 1e-4, 0.5, -1.3, 7.0, -50.0, 1234.5, -9876.0], dtype=torch.float64)
    back = symexp(symlog(x))
    assert torch.allclose(back, x, atol=1e-6, rtol=1e-6), (back - x).abs().max()


def test_twohot_is_exact_inverse_in_range():
    cfg = WorldModelConfig()
    bins = value_bins(cfg).double()
    # Raw targets well inside the bin range (the toy's returns are O(1)).
    y = torch.tensor([0.0, 0.37, -0.9, 2.5, -4.2, 10.0, -15.0], dtype=torch.float64)
    enc = twohot_encode(y, bins)
    dec = twohot_decode(enc, bins)
    assert torch.allclose(dec, y, atol=1e-5), (dec - y).abs().max()


def test_twohot_label_has_two_nonzero_entries_summing_to_one():
    cfg = WorldModelConfig()
    bins = value_bins(cfg).double()
    y = torch.tensor([0.3, -2.1, 5.0], dtype=torch.float64)
    enc = twohot_encode(y, bins)
    assert torch.allclose(enc.sum(-1), torch.ones(3, dtype=torch.float64))
    # At most two nonzero per row (exactly two unless y lands on a bin).
    nnz = (enc > 0).sum(-1)
    assert (nnz <= 2).all() and (nnz >= 1).all(), nnz


def test_twohot_loss_minimized_at_matching_logits():
    cfg = WorldModelConfig()
    bins = value_bins(cfg).double()
    y = torch.tensor([1.7], dtype=torch.float64)
    label = twohot_encode(y, bins)
    # Logits that reproduce the two-hot label as a softmax give a lower loss than mismatched logits.
    # Build matching logits by taking log of the (clamped) label.
    matched = torch.log(label.clamp_min(1e-12))
    mismatched = torch.zeros_like(label)  # uniform softmax, far from the sharp two-hot target
    loss_match = twohot_loss(matched, y, bins)
    loss_mismatch = twohot_loss(mismatched, y, bins)
    assert loss_match < loss_mismatch


def test_bins_in_symlog_space():
    # The bins are linspace(-20, 20, n) in symlog space: evenly spaced and symmetric about 0.
    cfg = WorldModelConfig()
    bins = value_bins(cfg)
    assert bins.shape == (cfg.n_bins,)
    assert torch.all(bins[1:] > bins[:-1]), "bins must be strictly increasing"
    assert torch.allclose(bins[0], -bins[-1], rtol=1e-5), "bins must be symmetric about 0"
    assert torch.allclose(bins[0], torch.tensor(cfg.bin_lo)) and \
        torch.allclose(bins[-1], torch.tensor(cfg.bin_hi))
    # Evenly spaced (symlog space), so the gaps are constant.
    gaps = bins[1:] - bins[:-1]
    assert torch.allclose(gaps, gaps[0].expand_as(gaps), atol=1e-5)
