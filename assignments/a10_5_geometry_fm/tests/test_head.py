"""PointmapHead output shapes and the confidence parameterization (C >= 1)."""

import torch

from head import PointmapHead


def test_head_shapes_and_confidence():
    torch.manual_seed(0)
    B, dim, grid = 3, 64, 4
    head = PointmapHead(dim, grid)
    tokens = torch.randn(B, grid * grid, dim)
    pts, conf = head(tokens)
    assert pts.shape == (B, grid, grid, 3)
    assert conf.shape == (B, grid, grid)
    # C = 1 + exp(logit) is strictly >= 1.
    assert (conf >= 1.0).all()


def test_head_confidence_can_grow():
    # With a large positive logit the confidence should exceed 1 by a wide margin,
    # confirming it is not clamped to 1.
    torch.manual_seed(1)
    head = PointmapHead(8, 2)
    tokens = torch.randn(2, 4, 8)
    _, conf = head(tokens)
    assert conf.max() > 1.0
