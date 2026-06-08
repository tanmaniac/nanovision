"""The DETR detection model. Provided wiring; the lesson is the matcher and loss.

A ViT backbone gives per-patch features, a set of learned object-query embeddings cross-
attends to those features through TransformerBlock layers, and two heads read each refined
query into a class distribution and a box. forward(img) -> (logits (B, N, C+1),
boxes (B, N, 4) cxcywh in [0, 1]).

This build deliberately omits two real DETR pieces:
  - Per-decoder-layer re-injection of a 2D image positional encoding into the cross-attention
    keys. Here the only spatial signal is the ViT's single learned positional embedding,
    added once at the patch-embedding layer.
  - Per-layer auxiliary losses (a copy of the matching loss at every decoder layer). Here the
    loss is computed only from the final layer's output.
Box localization is bounded by the 4-pixel patch stride: the backbone sees the image at an
8x8 grid, so a center cannot be pinned finer than that grid allows.
"""

import torch
from torch import Tensor, nn

from nanovision.transformer import TransformerBlock
from nanovision.vit import ViT


class DETR(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_classes = cfg.num_classes
        self.num_queries = cfg.num_queries
        dim = cfg.vit_dim

        # Backbone. img_size and patch are pinned so the ViT's learned pos_embed (grid-locked
        # to (img_size/patch)^2 = 64 tokens) matches the patch grid; forward_features adds it.
        self.backbone = ViT(
            img_size=cfg.img_size,
            patch=cfg.patch,
            in_chans=3,
            dim=dim,
            depth=cfg.vit_depth,
            n_heads=cfg.vit_heads,
            num_classes=cfg.num_classes,  # unused (we call forward_features), but a valid head
            pool="cls",
        )
        expected = (cfg.img_size // cfg.patch) ** 2
        assert self.backbone.n_patches == expected == 64, (
            f"ViT grid mismatch: n_patches={self.backbone.n_patches}, expected {expected}. "
            "Pin img_size=32, patch=4 so the grid-locked pos_embed covers 64 tokens."
        )

        # N learned object-query embeddings.
        self.query_embed = nn.Parameter(torch.zeros(cfg.num_queries, dim))
        nn.init.trunc_normal_(self.query_embed, std=0.02)

        # Decoder: queries self-attend (bidirectional) then cross-attend to the patch tokens.
        self.decoder = nn.ModuleList([
            TransformerBlock(dim, cfg.vit_heads, causal=False, cross_attn=True)
            for _ in range(cfg.dec_depth)
        ])

        # Heads.
        self.class_head = nn.Linear(dim, cfg.num_classes + 1)
        h = cfg.box_mlp_hidden
        self.box_head = nn.Sequential(
            nn.Linear(dim, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, 4),
        )

    def forward(self, img: Tensor) -> tuple[Tensor, Tensor]:
        B = img.shape[0]
        memory = self.backbone.forward_features(img)        # (B, 64, dim)
        q = self.query_embed[None].expand(B, -1, -1)         # (B, N, dim)
        for block in self.decoder:
            q = block(q, kv=memory)
        logits = self.class_head(q)                          # (B, N, C+1)
        boxes = self.box_head(q).sigmoid()                   # (B, N, 4) cxcywh in [0, 1]
        return logits, boxes
