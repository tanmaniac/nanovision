"""GeometryFM: the DUSt3R-style two-view pointmap regressor. Provided.

Pipeline:
  1. A single ViT encoder runs on both images (Siamese: shared weights), giving per-patch
     tokens f1, f2 each (B, N, dim) with N = (img_size / patch)^2.
  2. Two decoders, one per view. The view-1 decoder runs self-attention over f1 and
     cross-attends to f2 (the other view's tokens as memory); the view-2 decoder runs over f2
     and cross-attends to f1. This cross-view attention is how view 2's points get placed in
     view 1's frame: each view's prediction is informed by what the other view sees.
  3. Two PointmapHeads map the decoded tokens to per-patch pointmaps + confidence, both in
     camera 1's frame.

The decoders are built from non-causal TransformerBlock(causal=False, cross_attn=True). A
causal mask over spatial patch tokens would be wrong (patch 0 could attend only to itself);
the patch grid has no temporal order, so self-attention is bidirectional. TransformerDecoder
is NOT used because it hardcodes causal=True.

The wiring is provided; the lesson is the pointmap head, the loss, and the cross-view
structure, not the plumbing. forward(img1, img2, use_cross=True) returns
(pts1, conf1, pts2, conf2). With use_cross=False the cross-attention memory is zeroed, which
the cross-attention ablation uses to measure how much the other view contributes.
"""

import torch
from torch import Tensor, nn

from config import GeometryFMConfig
from head import PointmapHead

from nanovision.vit import ViT
from nanovision.transformer import TransformerBlock


class _CrossDecoder(nn.Module):
    """A stack of non-causal cross-attending transformer blocks.

    Each block: bidirectional self-attention over this view's tokens, then cross-attention to
    the other view's memory, then the feed-forward. With memory zeroed, the cross-attention
    sub-layer adds (close to) nothing, which is the ablation handle.
    """

    def __init__(self, dim: int, n_heads: int, depth: int):
        super().__init__()
        self.blocks = nn.ModuleList(
            TransformerBlock(dim, n_heads, causal=False, cross_attn=True)
            for _ in range(depth)
        )

    def forward(self, x: Tensor, memory: Tensor) -> Tensor:
        for blk in self.blocks:
            x = blk(x, kv=memory)
        return x


class GeometryFM(nn.Module):
    """Siamese ViT encoder, two cross-attending decoders, two pointmap heads."""

    def __init__(self, cfg: GeometryFMConfig):
        super().__init__()
        self.cfg = cfg
        self.grid = cfg.img_size // cfg.patch
        self.encoder = ViT(
            img_size=cfg.img_size,
            patch=cfg.patch,
            in_chans=cfg.in_chans,
            dim=cfg.dim,
            depth=cfg.enc_depth,
            n_heads=cfg.enc_heads,
        )
        self.dec1 = _CrossDecoder(cfg.dim, cfg.dec_heads, cfg.dec_depth)
        self.dec2 = _CrossDecoder(cfg.dim, cfg.dec_heads, cfg.dec_depth)
        self.head1 = PointmapHead(cfg.dim, self.grid, cfg.head_hidden)
        self.head2 = PointmapHead(cfg.dim, self.grid, cfg.head_hidden)

    def forward(self, img1: Tensor, img2: Tensor, use_cross: bool = True):
        """Run both views.

        Args:
            img1, img2: (B, 3, H, W) the two images.
            use_cross: if False, zero the cross-attention memory so each decoder ignores the
                other view (the cross-attention ablation).

        Returns:
            pts1, conf1, pts2, conf2 - pointmaps (B, grid, grid, 3) and confidences
            (B, grid, grid), both in cam1 frame.
        """
        f1 = self.encoder.forward_features(img1)     # (B, N, dim)
        f2 = self.encoder.forward_features(img2)     # (B, N, dim)
        mem2 = f2 if use_cross else torch.zeros_like(f2)
        mem1 = f1 if use_cross else torch.zeros_like(f1)
        d1 = self.dec1(f1, mem2)                     # view 1 attends to view 2
        d2 = self.dec2(f2, mem1)                     # view 2 attends to view 1
        pts1, conf1 = self.head1(d1)
        pts2, conf2 = self.head2(d2)
        return pts1, conf1, pts2, conf2
