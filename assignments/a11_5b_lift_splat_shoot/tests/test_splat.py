"""Mechanism C: the sort+cumsum splat and the pillar index.

cumsum_pool must sum features that share a bin with no scatter_add (the cumsum trick is the
point). The test compares it against a scatter_add ORACLE - scatter_add is allowed here in the
test, forbidden in lift_splat.py.
"""

import torch

from config import LSSConfig

from nanovision.lift_splat import cumsum_pool, pillar_index


def _scatter_oracle(feats, idx, n_bins):
    # Reference pooled sum via scatter_add; lift_splat.py may NOT use this.
    out = feats.new_zeros(n_bins, feats.shape[1])
    keep = idx >= 0
    out.index_add_(0, idx[keep], feats[keep])
    return out


def test_same_pillar_sum():
    feats = torch.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    idx = torch.tensor([5, 5])
    out = cumsum_pool(feats, idx, n_bins=8)
    assert torch.allclose(out[5], feats[0] + feats[1])
    # No other bin received anything.
    other = torch.cat([out[:5], out[6:]], dim=0)
    assert other.abs().max() == 0.0


def test_out_of_bounds_dropped():
    feats = torch.tensor([[1.0, 1.0], [9.0, 9.0]])
    idx = torch.tensor([-1, 3])
    out = cumsum_pool(feats, idx, n_bins=5)
    assert torch.allclose(out[3], feats[1])
    assert out.sum() == feats[1].sum()           # the idx=-1 point contributed nothing


def test_matches_scatter_oracle():
    torch.manual_seed(0)
    N, C, n_bins = 50, 3, 10
    feats = torch.randn(N, C)
    idx = torch.randint(-1, n_bins, (N,))
    out = cumsum_pool(feats, idx, n_bins)
    ref = _scatter_oracle(feats, idx, n_bins)
    assert torch.allclose(out, ref, atol=1e-5)


def test_gradcheck():
    torch.manual_seed(1)
    N, C, n_bins = 12, 2, 5
    idx = torch.randint(-1, n_bins, (N,))        # fixed permutation given idx
    feats = torch.randn(N, C, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda f: cumsum_pool(f, idx, n_bins).sum(), (feats,)
    )


def test_pillar_index_center_and_oob():
    cfg = LSSConfig()
    grid = cfg.bev_grid()
    # A point inside the grid maps to the expected cell; a point past x_max -> -1.
    # ego (x, y) = (2.5, 0.5): ix = floor((2.5-0)/1)=2, iy = floor((0.5-(-8))/1)=8 -> 2*16+8.
    pts = torch.tensor([[2.5, 0.5], [99.0, 0.0]])
    idx = pillar_index(pts, grid)
    assert idx[0].item() == 2 * grid.ny + 8
    assert idx[1].item() == -1
