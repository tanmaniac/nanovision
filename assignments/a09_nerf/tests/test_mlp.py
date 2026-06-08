"""NeRFMLP forward shapes and output ranges (density >= 0, color in [0, 1])."""

import torch

from config import NeRFConfig
from model import NeRFMLP


def test_forward_shapes_and_ranges():
    torch.manual_seed(0)
    cfg = NeRFConfig()
    model = NeRFMLP(
        pos_L=cfg.pos_L,
        dir_L=cfg.dir_L,
        hidden=cfg.hidden,
        n_layers=cfg.n_layers,
        include_input=cfg.include_input,
        scene_bound=cfg.scene_bound,
    )
    R, N = 7, cfg.n_samples
    positions = torch.randn(R, N, 3)
    directions = torch.randn(R, 3)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    sigma, rgb = model(positions, directions)
    assert sigma.shape == (R, N)
    assert rgb.shape == (R, N, 3)
    assert (sigma >= 0).all()
    assert (rgb >= 0).all() and (rgb <= 1).all()
