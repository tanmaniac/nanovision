"""Pillar extrusion: shape and gradient correctness."""

import torch
from torch import nn

from occupancy import bev_to_voxel


def test_shape():
    B, C, Y, X, n_z = 2, 16, 32, 32, 8
    bev = torch.randn(B, C, Y, X)
    conv = nn.Conv2d(C, C * n_z, 1)
    out = bev_to_voxel(bev, n_z, conv)
    assert out.shape == (B, C, n_z, Y, X)


def test_gradcheck():
    torch.manual_seed(0)
    B, C, Y, X, n_z = 1, 2, 4, 4, 2
    bev = torch.randn(B, C, Y, X, dtype=torch.float64, requires_grad=True)
    conv = nn.Conv2d(C, C * n_z, 1).double()
    assert torch.autograd.gradcheck(lambda b: bev_to_voxel(b, n_z, conv), (bev,))
