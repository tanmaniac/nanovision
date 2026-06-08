"""Provided scaffolding for A3: a minimal ViT encoder plus patch helpers and the
DINO multi-crop augmentation and projection head.

None of this file is a hole. The learner built PatchEmbed in A2; here the patch
embed, positional embedding, and the transformer stack are given so A3 can focus
on the self-supervised mechanisms (random masking, the MAE loss, the DINO loss,
the EMA teacher, centering, and the collapse instrument). The same file ships
at the top level and in solution/ unchanged.

Shapes: images are (B, C, H, W); patchify gives (B, N, p*p*C) with
N = (H/p) * (W/p); the ViT encoder maps a token sequence (B, L, dim) to (B, L, dim).
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from nanovision.primitives import LayerNorm
from nanovision.transformer import TransformerEncoder


def patchify(img: Tensor, patch: int) -> Tensor:
    """Cut an image into non-overlapping patches and flatten each to a vector.

    img: (B, C, H, W). Returns (B, N, p*p*C) with N = (H/p) * (W/p), patches in
    row-major order. The per-patch vector is laid out (C, p_row, p_col) flattened,
    so unpatchify inverts it exactly. This is the MAE reconstruction target space.
    """
    B, C, H, W = img.shape
    p = patch
    gh, gw = H // p, W // p
    x = img.reshape(B, C, gh, p, gw, p)
    x = x.permute(0, 2, 4, 1, 3, 5)          # (B, gh, gw, C, p, p)
    x = x.reshape(B, gh * gw, C * p * p)     # (B, N, C*p*p)
    return x


def unpatchify(patches: Tensor, patch: int, channels: int, grid: int) -> Tensor:
    """Inverse of patchify. patches: (B, N, p*p*C) -> img (B, C, H, W).

    grid is the number of patches per side (so N = grid*grid and H = grid*patch).
    """
    B, N, _ = patches.shape
    p, C, gh = patch, channels, grid
    gw = N // gh
    x = patches.reshape(B, gh, gw, C, p, p)
    x = x.permute(0, 3, 1, 4, 2, 5)          # (B, C, gh, p, gw, p)
    x = x.reshape(B, C, gh * p, gw * p)
    return x


def per_patch_normalize(target: Tensor, eps: float = 1e-6) -> Tensor:
    """Normalize each patch vector to zero mean and unit variance.

    target: (B, N, p*p*C). MAE computes its loss against per-patch-normalized
    pixels, which He et al. (2022) found improves the learned representation. The
    statistics are taken over the pixel dimension of each patch independently.
    """
    mean = target.mean(dim=-1, keepdim=True)
    var = target.var(dim=-1, keepdim=True, unbiased=False)
    return (target - mean) / torch.sqrt(var + eps)


class ViTEncoder(nn.Module):
    """Minimal ViT encoder: Conv2d patch embed, learned PE, transformer stack.

    Built for fixed 32x32 inputs with patch 4 (so 64 patch tokens). There is no
    CLS token here; the encoder returns one token per patch. A learned positional
    embedding of shape (1, N, dim) is added to the patch tokens. The transformer is
    the A1 classic-ViT config (LayerNorm + GELU MLP + absolute PE, non-causal).

    forward(x): x is (B, C, H, W); returns (B, N, dim) patch-token features
        (after a final LayerNorm).
    forward_tokens(tokens): tokens is (B, L, dim) already-embedded patch tokens
        (used by MAE, which masks before the transformer); returns (B, L, dim).

    The patch embed and the per-token positional embedding are exposed so MAE can
    add the encoder PE before masking. `pos_embed` is (1, N, dim).
    """

    def __init__(self, img_size: int = 32, patch: int = 4, in_chans: int = 3,
                 dim: int = 64, depth: int = 4, n_heads: int = 4, mlp_ratio: float = 4.0):
        super().__init__()
        self.grid = img_size // patch
        self.n_patches = self.grid * self.grid
        self.patch = patch
        self.dim = dim
        self.proj = nn.Conv2d(in_chans, dim, kernel_size=patch, stride=patch)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.encoder = TransformerEncoder(
            dim, n_heads, depth, mlp_ratio=mlp_ratio, norm="layer", ffn="mlp", pos="none"
        )
        self.norm = LayerNorm(dim)

    def embed(self, x: Tensor) -> Tensor:
        """x: (B, C, H, W) -> patch tokens (B, N, dim), no PE added yet."""
        x = self.proj(x)                 # (B, dim, grid, grid)
        x = x.flatten(2).transpose(1, 2)  # (B, N, dim)
        return x

    def _pos_embed_for(self, n_tokens: int) -> Tensor:
        """Return a (1, n_tokens, dim) PE, bicubically resized when a crop has a
        different patch-grid size than the training grid (DINO local crops are
        smaller than global crops, so they have fewer patch tokens).
        """
        if n_tokens == self.n_patches:
            return self.pos_embed
        old = self.grid
        new = int(round(n_tokens ** 0.5))
        pe = self.pos_embed.reshape(1, old, old, self.dim).permute(0, 3, 1, 2)
        pe = F.interpolate(pe, size=(new, new), mode="bicubic", align_corners=False)
        return pe.permute(0, 2, 3, 1).reshape(1, new * new, self.dim)

    def forward_tokens(self, tokens: Tensor) -> Tensor:
        """Run the transformer stack + final norm on an embedded sequence."""
        return self.norm(self.encoder(tokens))

    def forward(self, x: Tensor) -> Tensor:
        tokens = self.embed(x)
        tokens = tokens + self._pos_embed_for(tokens.shape[1])
        return self.forward_tokens(tokens)


class DINOHead(nn.Module):
    """DINO projection head: MLP -> L2 normalize -> weight-normalized linear to K.

    Maps a backbone feature (B, dim) to logits over K prototypes (B, K). The last
    linear is weight-normalized (its direction is learned, its norm fixed to 1) as
    in the DINO reference, which stabilizes training. L2-normalizing the hidden
    feature before the prototype layer keeps the prototype logits bounded.

    forward(x): x is (B, dim) or (N, dim); returns (B, K) prototype logits.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256, bottleneck: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck),
        )
        last = nn.Linear(bottleneck, out_dim, bias=False)
        nn.init.trunc_normal_(last.weight, std=0.02)
        # Weight-normalize the prototype layer; freeze the norm at 1 (DINO does
        # this to keep the prototypes on the unit sphere early in training).
        self.last = nn.utils.parametrizations.weight_norm(last)
        self.last.parametrizations.weight.original0.data.fill_(1)
        self.last.parametrizations.weight.original0.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last(x)


class DINOModel(nn.Module):
    """A ViT backbone followed by a DINO projection head.

    The backbone here mean-pools its patch tokens to a single (B, dim) image
    feature (there is no CLS token), then the head maps it to (B, K) prototype
    logits. Student and teacher are two instances of this module; the teacher is an
    EMA of the student (see dino.ema_update).

    forward(x): x is (B, C, H, W); returns (B, K) prototype logits.
    """

    def __init__(self, img_size: int = 32, patch: int = 4, in_chans: int = 3,
                 dim: int = 64, depth: int = 4, n_heads: int = 4, mlp_ratio: float = 4.0,
                 out_dim: int = 256, head_hidden: int = 256):
        super().__init__()
        self.backbone = ViTEncoder(img_size, patch, in_chans, dim, depth, n_heads, mlp_ratio)
        self.head = DINOHead(dim, out_dim, hidden_dim=head_hidden)

    def forward(self, x: Tensor) -> Tensor:
        tokens = self.backbone(x)        # (B, N, dim)
        feat = tokens.mean(dim=1)         # mean-pool to (B, dim)
        return self.head(feat)


def multi_crop(img: Tensor, n_global: int = 2, n_local: int = 4,
               global_size: int = 32, local_size: int = 16) -> tuple[list[Tensor], list[Tensor]]:
    """Produce DINO global and local crops from a batch of images.

    img: (B, C, H, W). Returns (global_crops, local_crops). Each global crop is a
    full-resolution (B, C, global_size, global_size) view; each local crop is a
    smaller random region resized up to (B, C, local_size, local_size). The teacher
    sees only the global crops; the student sees all crops. The asymmetry (local
    crops are smaller and only the student gets them) forces the model to match a
    small region's representation to the global one - the local-to-global objective.

    This is a deterministic-given-seed light version: global crops are random small
    jitters of the full image, local crops are random sub-windows. It is enough to
    drive the overfit and collapse tests without a real augmentation stack.
    """
    B, C, H, W = img.shape

    def random_resized(src: Tensor, out: int, area_lo: float, area_hi: float) -> Tensor:
        area = (area_lo + (area_hi - area_lo) * torch.rand(1).item()) * H * W
        side = int(round(area ** 0.5))
        side = max(4, min(side, H))
        top = torch.randint(0, H - side + 1, (1,)).item()
        left = torch.randint(0, W - side + 1, (1,)).item()
        crop = src[:, :, top:top + side, left:left + side]
        return F.interpolate(crop, size=(out, out), mode="bilinear", align_corners=False)

    globals_ = [random_resized(img, global_size, 0.5, 1.0) for _ in range(n_global)]
    locals_ = [random_resized(img, local_size, 0.1, 0.5) for _ in range(n_local)]
    return globals_, locals_
