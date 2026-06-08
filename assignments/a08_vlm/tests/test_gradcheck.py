"""Float64 gradcheck of the projector and the resampler forward passes."""

import torch

from projector import MLPProjector
from resampler import PerceiverResampler


def test_projector_gradcheck():
    torch.manual_seed(0)
    proj = MLPProjector(dim_v=5, dim_l=6).double()
    feats = torch.randn(2, 4, 5, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(proj, (feats,))


def test_resampler_gradcheck():
    torch.manual_seed(0)
    res = PerceiverResampler(dim_v=5, dim_l=6, n_queries=3, n_heads=2).double()
    feats = torch.randn(2, 4, 5, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(res, (feats,))
