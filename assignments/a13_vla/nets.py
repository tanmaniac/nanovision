"""The image encoder that turns a camera frame into the conditioning vector. Provided plumbing.

A vision-language-action policy reads pixels, not privileged state. Here the observation is a
64x64 RGB render of the reacher; this CNN maps it to a fixed-width embedding that the action heads
consume as their conditioning c. The action heads (flow, BC, DDPM) are unchanged from the
state-conditioned version: they take a vector c of width cond_in and do not care that it now comes
from a convolutional encoder instead of a state-plus-goal concatenation.

The encoder trains jointly with the action head under behavior cloning - there is no separate
representation-learning step. The conv stack is the standard DreamerV3 64x64 pixel encoder (four
stride-2 convs, 32/64/128/256 channels, 64x64 -> 4x4), reused from the world-models assignment.
This file is pure torch and has no dm_control dependency, so the CPU mechanism tests import it
without the robot library installed.
"""

import torch
from torch import Tensor, nn


class Encoder(nn.Module):
    """CNN that maps an obs (B, obs_ch, obs_size, obs_size) to an embedding (B, embed_dim).

    Four stride-2 convs with channel widths 32/64/128/256 take 64x64 down to 4x4, then a linear
    projects the flattened 256*4*4 features to embed_dim. The embedding is the conditioning vector
    c the action heads read; embed_dim is the heads' cond_in.
    """

    def __init__(self, cfg):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cfg.obs_ch, 32, 4, stride=2, padding=1), nn.SiLU(),   # S -> S/2
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.SiLU(),           # S/2 -> S/4
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.SiLU(),          # S/4 -> S/8
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.SiLU(),         # S/8 -> S/16
        )
        feat_hw = cfg.obs_size // 16     # 64 -> 4; the four stride-2 convs each halve the side
        self.proj = nn.Linear(256 * feat_hw * feat_hw, cfg.embed_dim)
        self.embed_dim = cfg.embed_dim

    def forward(self, obs: Tensor) -> Tensor:
        return self.proj(self.conv(obs).flatten(1))
