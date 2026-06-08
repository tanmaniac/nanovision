"""The 2D velocity field, a small MLP over (x, t). Provided.

v_theta(x, t) takes x (B, D) and a scalar timestep t (B,) and returns a velocity (B, D).
The timestep is embedded with the same sinusoidal construction as the transformer's
positional encoding (and A5's diffusion time embedding), then concatenated with x before
the MLP. This is the only network A6 needs for the 2D toy; the image demo reuses A5's
U-Net through nanovision.unet.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def sinusoidal_embedding(t: Tensor, dim: int) -> Tensor:
    """Sinusoidal embedding of a scalar t (B,) into (B, dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device).float() / half)
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class VelocityMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.time_dim = cfg.time_dim
        w = cfg.mlp_width
        layers = [nn.Linear(cfg.data_dim + cfg.time_dim, w), nn.SiLU()]
        for _ in range(cfg.mlp_depth - 1):
            layers += [nn.Linear(w, w), nn.SiLU()]
        layers += [nn.Linear(w, cfg.data_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        temb = sinusoidal_embedding(t, self.time_dim)
        return self.net(torch.cat([x, temb], dim=-1))
