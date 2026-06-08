"""A tiny diffusion transformer (DiT, Peebles and Xie 2023) over VAE latents.

The latent (B, C, H, W) is patchified into N tokens, linearly embedded, and run through a
stack of transformer blocks. Conditioning on the diffusion timestep and the class label
enters through adaLN-Zero: a small MLP regresses per-block (shift, scale, gate) from the
conditioning vector, modulating each LayerNorm and gating each residual branch. The gate
and the output head are zero-initialized, so at init every block is the identity and the
whole DiT predicts zeros - the standard DiT initialization that keeps early training
stable, the same idea as zero-init residual branches in ResNets.

modulate, patchify, unpatchify, and DiTBlock.forward are the holes. timestep_embedding,
the DiTBlock wiring, and the DiT module are provided.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from nanovision.attention import MultiHeadAttention


def timestep_embedding(t: Tensor, dim: int) -> Tensor:
    """Sinusoidal embedding of a scalar timestep t (B,) into (B, dim).

    Same cos-then-sin, log-spaced-frequency construction as the diffusion U-Net's time
    embedding, indexed here by the continuous flow-matching time in [0, 1].
    """
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device).float() / half)
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """adaLN affine: x * (1 + scale) + shift.

    x is (B, N, d); shift and scale arrive as (B, d) and MUST be unsqueezed to (B, 1, d)
    here so they broadcast over the N token axis. Without the unsqueeze the (B, d) tensors
    do not broadcast against (B, N, d).
    """
    raise NotImplementedError("implement the adaLN modulate affine")


def patchify(z: Tensor, p: int) -> Tensor:
    """(B, C, H, W) -> (B, N, p*p*C), N = (H/p)(W/p), row-major patch order.

    Pure reshape/permute, no projection. Patch (i, j) occupies row-major position
    i*(W/p) + j, and within a patch the values are ordered (C, p, p) flattened.
    unpatchify must be the exact inverse.
    """
    raise NotImplementedError("implement patchify")


def unpatchify(tokens: Tensor, p: int, C: int, H: int, W: int) -> Tensor:
    """Exact inverse of patchify: (B, N, p*p*C) -> (B, C, H, W)."""
    raise NotImplementedError("implement unpatchify")


class DiTBlock(nn.Module):
    """A transformer block with adaLN-Zero conditioning.

    The conditioning vector c (B, d) is mapped by adaLN_modulation to six (B, d) tensors:
    (shift, scale, gate) for the attention sub-layer and (shift, scale, gate) for the MLP.
    The final Linear in adaLN_modulation has zero weight AND zero bias, so at init every
    gate is 0 and both residual branches contribute nothing - the block is the identity
    regardless of c.
    """

    def __init__(self, d: int, n_heads: int, mlp_ratio: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False)
        self.attn = MultiHeadAttention(d, n_heads)         # bidirectional, no mask
        hidden = mlp_ratio * d
        self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        # Zero-init the modulation Linear (weight and bias) so the block is identity at init.
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        """adaLN-Zero block: regress 6 modulation params from c, then gated attn and MLP.

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp =
            self.adaLN_modulation(c).chunk(6, dim=-1)   # each (B, d)

        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        """
        raise NotImplementedError("implement the adaLN-Zero DiTBlock forward")


class DiT(nn.Module):
    """The full DiT: patch-embed, conditioned transformer blocks, final adaLN, output head.

    forward(z, t, y): z is the latent (B, C, H, W), t (B,) the continuous flow time, y (B,)
    the class label. Returns the predicted velocity, same shape as z.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        C, p, d = cfg.latent_dim, cfg.patch_size, cfg.d_model
        self.C, self.p = C, p
        self.latent_hw = cfg.image_size // cfg.f                 # spatial size of the latent
        n = (self.latent_hw // p) ** 2                           # number of tokens
        self.patch_embed = nn.Linear(p * p * C, d)
        self.pos_embed = nn.Parameter(torch.zeros(1, n, d))
        nn.init.normal_(self.pos_embed, std=0.02)
        self.class_embed = nn.Embedding(cfg.num_classes, d)
        self.time_mlp = nn.Sequential(nn.Linear(cfg.time_dim, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList(
            DiTBlock(d, cfg.n_heads, cfg.mlp_ratio) for _ in range(cfg.n_blocks)
        )
        # Final adaLN over the last layer, then a linear output head. Both zero-init, so the
        # DiT predicts zeros at init (combined with the zero-init block gates).
        self.norm_final = nn.LayerNorm(d, elementwise_affine=False)
        self.adaLN_final = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        nn.init.zeros_(self.adaLN_final[-1].weight)
        nn.init.zeros_(self.adaLN_final[-1].bias)
        self.head = nn.Linear(d, p * p * C)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, z: Tensor, t: Tensor, y: Tensor) -> Tensor:
        _, C, H, W = z.shape
        p = self.p
        tokens = self.patch_embed(patchify(z, p)) + self.pos_embed
        c = self.time_mlp(timestep_embedding(t, self.cfg.time_dim)) + self.class_embed(y)
        for block in self.blocks:
            tokens = block(tokens, c)
        shift, scale = self.adaLN_final(c).chunk(2, dim=-1)
        tokens = modulate(self.norm_final(tokens), shift, scale)
        tokens = self.head(tokens)
        return unpatchify(tokens, p, C, H, W)
