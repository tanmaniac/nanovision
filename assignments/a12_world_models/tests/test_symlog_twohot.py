"""symlog / symexp are inverses; two-hot encode splits a target between its two neighboring bins
with linear weights, and decode returns the expected bin value."""

import math

import torch

from nets import symexp, symexp_bins, symlog, twohot_decode, twohot_encode


def test_symlog_symexp_inverse():
    x = torch.cat([torch.linspace(-1e6, 1e6, 101, dtype=torch.float64),
                   torch.linspace(-3, 3, 61, dtype=torch.float64)])
    torch.testing.assert_close(symexp(symlog(x)), x, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(symlog(symexp(x[-61:])), x[-61:], rtol=1e-12, atol=1e-12)


def test_symlog_values():
    x = torch.tensor([0.0, 1e-8, -1e-8, torch.e - 1, -(torch.e - 1), 1e4], dtype=torch.float64)
    expect = torch.tensor([0.0, 1e-8, -1e-8, 1.0, -1.0, math.log1p(1e4)], dtype=torch.float64)
    torch.testing.assert_close(symlog(x), expect, rtol=1e-12, atol=1e-15)  # slope 1 at 0, odd


def test_symexp_bins_antisymmetric():
    for n in (255, 31, 30):
        b = symexp_bins(n)
        assert b.shape == (n,)
        assert torch.all(b[1:] >= b[:-1])       # an even n repeats the bin at 0
        assert torch.equal(b, -b.flip(0))
    assert torch.all(symexp_bins(255)[1:] > symexp_bins(255)[:-1])
    assert torch.isclose(symexp_bins(255)[-1], symexp(torch.tensor(20.0)))


def test_twohot_hand_example():
    bins = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    lab = twohot_encode(torch.tensor([0.25, -1.5, 1.0, 7.0, -9.0]), bins)
    expect = torch.tensor([
        [0, 0, 0.75, 0.25, 0],      # between bins 0 and 1, linear weights
        [0.5, 0.5, 0, 0, 0],
        [0, 0, 0, 1.0, 0],          # exactly on a bin
        [0, 0, 0, 0, 1.0],          # above the range: all on the last bin
        [1.0, 0, 0, 0, 0],          # below the range: all on the first bin
    ])
    torch.testing.assert_close(lab, expect)


def test_twohot_two_neighbors_and_weights():
    bins = symexp_bins(255)
    y = torch.cat([torch.randn(200) * 3, torch.randn(200) * 300])
    lab = twohot_encode(y, bins)
    assert lab.shape == (400, 255)
    torch.testing.assert_close(lab.sum(-1), torch.ones(400))
    for i in range(len(y)):
        nz = torch.nonzero(lab[i]).flatten()
        k = int(torch.searchsorted(bins, y[i], right=True)) - 1     # bins[k] <= y < bins[k + 1]
        assert set(nz.tolist()) <= {k, k + 1}
        w_hi = (y[i] - bins[k]) / (bins[k + 1] - bins[k])
        torch.testing.assert_close(lab[i, k + 1], w_hi, atol=1e-5, rtol=1e-5)


def test_twohot_round_trip():
    bins = symexp_bins(255)
    y = torch.cat([torch.linspace(-2000, 2000, 401), torch.tensor([0.0, 1e-3, -1e-3, 1.0])])
    lab = twohot_encode(y, bins)
    back = twohot_decode(torch.log(lab), bins)       # softmax(log(label)) = label
    assert back.shape == (len(y), 1)
    torch.testing.assert_close(back.squeeze(-1), y, rtol=1e-5, atol=1e-5)


def test_twohot_decode_is_expected_value():
    for bins in (torch.tensor([-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0]), torch.tensor([-3.0, -1.0, 1.0, 3.0])):
        logits = torch.randn(5, 3, len(bins))
        expect = (torch.softmax(logits.double(), -1) * bins.double()).sum(-1, keepdim=True)
        out = twohot_decode(logits, bins)
        assert out.shape == (5, 3, 1)
        torch.testing.assert_close(out.double(), expect, rtol=1e-5, atol=1e-6)
