"""Provided scaffolding for A3.5: tubelet helpers and a minimal video ViT encoder.

None of this file is a hole. It gives the reconstruction-target helpers (the video
analog of A3's patchify/per_patch_normalize) and the encoder that wraps the
student's TubeletEmbedding with a learned spatiotemporal positional embedding and the
A1 transformer stack. The same file ships at the top level and in solution/ unchanged.

The encoder uses JOINT space-time self-attention: the A1 TransformerEncoder runs over
all N = T' * S' tubelet tokens at once. That is fine here because N is tiny (48); at
real video scale this is O(N^2) in T'*S', which is exactly what factorized space-time
attention (TimeSformer, ViViT) exists to avoid. See the README.

Token / flatten convention (everything downstream depends on it): tubelet tokens are
ordered temporal-outermost, idx = t' * S' + s with S' = (H/p)*(W/p) and s = h'*W' + w'.
TubeletEmbedding's Conv3d output (B, D, T', H', W') flattened with flatten(2) gives the
same order, and tubeletify below matches it, so the positional embedding, the tube
mask, and the reconstruction target all line up.
"""

import torch
from torch import Tensor, nn

from nanovision.primitives import LayerNorm
from nanovision.transformer import TransformerEncoder, TubeletEmbedding


def tubeletify(video: Tensor, tubelet_t: int, patch: int) -> Tensor:
    """Cut a video into non-overlapping space-time tubelets, each flattened.

    video: (B, C, T, H, W). Returns (B, N, t*p*p*C) with N = (T/t)*(H/p)*(W/p),
    tubelets in temporal-outermost order. The per-tubelet vector is laid out
    (C, t, p_row, p_col) flattened, matching TubeletEmbedding's Conv3d weight layout
    so the embed-equals-strided-conv identity holds. This is the MAE target space.
    """
    B, C, T, H, W = video.shape
    t, p = tubelet_t, patch
    tp, gh, gw = T // t, H // p, W // p
    x = video.reshape(B, C, tp, t, gh, p, gw, p)
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)          # (B, T', gh, gw, C, t, p, p)
    x = x.reshape(B, tp * gh * gw, C * t * p * p)   # (B, N, C*t*p*p)
    return x


def untubeletify(tubelets: Tensor, tubelet_t: int, patch: int, channels: int,
                 t_prime: int, grid: int) -> Tensor:
    """Inverse of tubeletify. tubelets: (B, N, t*p*p*C) -> video (B, C, T, H, W).

    t_prime is the number of temporal tokens (T = t_prime * tubelet_t) and grid is
    the spatial tokens per side (H = grid * patch, assuming a square frame).
    """
    B, N, _ = tubelets.shape
    t, p, C, tp, gh = tubelet_t, patch, channels, t_prime, grid
    gw = N // (tp * gh)
    x = tubelets.reshape(B, tp, gh, gw, C, t, p, p)
    x = x.permute(0, 4, 1, 5, 2, 6, 3, 7)          # (B, C, T', t, gh, p, gw, p)
    x = x.reshape(B, C, tp * t, gh * p, gw * p)
    return x


def per_tubelet_normalize(target: Tensor, eps: float = 1e-6) -> Tensor:
    """Normalize each tubelet vector to zero mean and unit variance.

    target: (B, N, t*p*p*C). The video MAE computes its loss against per-tubelet-
    normalized pixels, the temporal analog of A3's per-patch normalization. The
    statistics are taken over the pixel dimension of each tubelet independently.
    """
    mean = target.mean(dim=-1, keepdim=True)
    var = target.var(dim=-1, keepdim=True, unbiased=False)
    return (target - mean) / torch.sqrt(var + eps)


class VideoViTEncoder(nn.Module):
    """Minimal video ViT: TubeletEmbedding, learned spatiotemporal PE, A1 stack.

    forward(video): video is (B, C, T, H, W); returns (B, N, dim) tubelet-token
        features after a final LayerNorm.
    forward_tokens(tokens): tokens is (B, L, dim) already-embedded tubelet tokens
        (used by the MAE, which masks before the transformer); returns (B, L, dim).

    embed and pos_embed are exposed so the MAE can add the encoder PE before masking.
    pos_embed is a single learned table (1, N, dim) over all tubelet tokens.
    """

    def __init__(self, img_size: int = 16, patch: int = 4, tubelet_t: int = 2,
                 n_frames: int = 6, in_chans: int = 3, dim: int = 64, depth: int = 4,
                 n_heads: int = 4, mlp_ratio: float = 4.0):
        super().__init__()
        self.grid = img_size // patch
        self.t_prime = n_frames // tubelet_t
        self.n_tokens = self.t_prime * self.grid * self.grid
        self.dim = dim
        self.tubelet = TubeletEmbedding(in_chans, dim, patch, tubelet_t)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_tokens, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.encoder = TransformerEncoder(
            dim, n_heads, depth, mlp_ratio=mlp_ratio, norm="layer", ffn="mlp", pos="none"
        )
        self.norm = LayerNorm(dim)

    def embed(self, video: Tensor) -> Tensor:
        """video: (B, C, T, H, W) -> tubelet tokens (B, N, dim), no PE added yet."""
        return self.tubelet(video)

    def forward_tokens(self, tokens: Tensor) -> Tensor:
        """Run the transformer stack + final norm on an embedded sequence."""
        return self.norm(self.encoder(tokens))

    def forward(self, video: Tensor) -> Tensor:
        tokens = self.embed(video) + self.pos_embed
        return self.forward_tokens(tokens)
