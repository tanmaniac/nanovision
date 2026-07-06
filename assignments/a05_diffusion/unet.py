"""A small time-embedded U-Net for diffusion on 16x16 images.

Standard CNN plumbing (GroupNorm ResNet blocks, a bottleneck self-attention, strided
down/up sampling with skip connections) is provided. You implement the two
diffusion-specific pieces: `timestep_embedding` (the sinusoidal embedding from the
transformer, now indexed by the scalar diffusion timestep instead of sequence position)
and the AdaGN time-injection line in ResBlock.forward (how the timestep+class signal
enters the network; A7's adaLN-Zero generalizes exactly this).

forward(x, t, labels): x is (B, C, H, W), t is (B,) integer timesteps, labels is (B,) class
ids or None (None = unconditional). The class embedding table has one extra null row at
index num_classes for classifier-free guidance. Output is (B, C, H, W).
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def timestep_embedding(t: Tensor, dim: int) -> Tensor:
    """Sinusoidal embedding of a scalar timestep t (B,) into (B, dim).

    Same sin/cos construction as the transformer's positional encoding, indexed by the
    diffusion timestep value rather than a sequence position (pad one column if dim is odd).
    See the network section of the README.
    """
    raise NotImplementedError("implement the sinusoidal timestep embedding")


def _groups(c: int) -> int:
    return 8 if c % 8 == 0 else 1


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, tdim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(cin), cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.temb_proj = nn.Linear(tdim, cout)
        self.norm2 = nn.GroupNorm(_groups(cout), cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x: Tensor, temb: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        # AdaGN time injection: add the time+class embedding `temb` into h as a per-channel
        # shift, broadcast over H and W. Replace the next line with that. See the network
        # section of the README.
        raise NotImplementedError("implement the AdaGN time injection")
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SelfAttn2d(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.norm = nn.GroupNorm(_groups(c), c)
        self.qkv = nn.Conv2d(c, c * 3, 1)
        self.proj = nn.Conv2d(c, c, 1)
        self.scale = c ** -0.5

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        q = q.reshape(B, C, H * W).transpose(1, 2)            # (B, N, C)
        k = k.reshape(B, C, H * W)                            # (B, C, N)
        v = v.reshape(B, C, H * W).transpose(1, 2)            # (B, N, C)
        attn = torch.softmax(q @ k * self.scale, dim=-1)      # (B, N, N)
        out = (attn @ v).transpose(1, 2).reshape(B, C, H, W)
        return x + self.proj(out)


class TimeEmbeddedUNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        c = cfg.base_width
        self.base = c
        self.tdim = c * 4
        self.num_classes = cfg.num_classes
        self.null_index = cfg.num_classes
        self.time_mlp = nn.Sequential(nn.Linear(c, self.tdim), nn.SiLU(),
                                      nn.Linear(self.tdim, self.tdim))
        self.class_emb = nn.Embedding(cfg.num_classes + 1, self.tdim)
        self.in_conv = nn.Conv2d(cfg.channels, c, 3, padding=1)
        self.d1 = ResBlock(c, c, self.tdim)
        self.down1 = nn.Conv2d(c, c, 3, stride=2, padding=1)          # 16 -> 8
        self.d2 = ResBlock(c, 2 * c, self.tdim)
        self.down2 = nn.Conv2d(2 * c, 2 * c, 3, stride=2, padding=1)  # 8 -> 4
        self.m1 = ResBlock(2 * c, 2 * c, self.tdim)
        self.attn = SelfAttn2d(2 * c)
        self.m2 = ResBlock(2 * c, 2 * c, self.tdim)
        self.up2 = nn.ConvTranspose2d(2 * c, 2 * c, 4, stride=2, padding=1)  # 4 -> 8
        self.u2 = ResBlock(2 * c + 2 * c, 2 * c, self.tdim)
        self.up1 = nn.ConvTranspose2d(2 * c, c, 4, stride=2, padding=1)      # 8 -> 16
        self.u1 = ResBlock(c + c, c, self.tdim)
        self.out_norm = nn.GroupNorm(_groups(c), c)
        self.out_conv = nn.Conv2d(c, cfg.channels, 3, padding=1)

    def forward(self, x: Tensor, t: Tensor, labels: Tensor | None = None) -> Tensor:
        temb = self.time_mlp(timestep_embedding(t, self.base))
        if labels is not None:
            temb = temb + self.class_emb(labels)
        x = self.in_conv(x)
        h1 = self.d1(x, temb)                       # 16x16 x c
        h = self.down1(h1)                          # 8x8 x c
        h2 = self.d2(h, temb)                       # 8x8 x 2c
        h = self.down2(h2)                          # 4x4 x 2c
        h = self.m2(self.attn(self.m1(h, temb)), temb)
        h = self.up2(h)                             # 8x8 x 2c
        h = self.u2(torch.cat([h, h2], dim=1), temb)
        h = self.up1(h)                             # 16x16 x c
        h = self.u1(torch.cat([h, h1], dim=1), temb)
        return self.out_conv(F.silu(self.out_norm(h)))
