"""Hyperparameters for the DETR detection toy. Provided.

The core of A11 is the matcher and the loss, not the model sizes. The ViT backbone is
pinned to img_size=32, patch=4 so its grid-locked learned positional embedding covers
exactly 8x8 = 64 patch tokens; model.py asserts this. The cost and loss weights are the
DETR values (Carion et al., 2020): class 1, L1 5, GIoU 2, with the no-object (eos) class
downweighted by 0.1 in the classification loss.
"""

from dataclasses import dataclass


@dataclass
class DETRConfig:
    # Image and backbone.
    img_size: int = 32
    patch: int = 4              # 32 / 4 -> 8x8 = 64 patch tokens (pos_embed is grid-locked)
    vit_dim: int = 128
    vit_depth: int = 3
    vit_heads: int = 4

    # Detection head.
    num_queries: int = 10       # N learned object-query slots
    num_classes: int = 3        # number of square colors; no-object class is index num_classes
    dec_depth: int = 2          # decoder TransformerBlock layers
    box_mlp_hidden: int = 128

    # Matching-cost weights (non-differentiable; DETR values).
    cost_class: float = 1.0
    cost_l1: float = 5.0
    cost_giou: float = 2.0

    # Training-loss weights (differentiable; DETR values).
    loss_class: float = 1.0
    loss_l1: float = 5.0
    loss_giou: float = 2.0
    eos_coef: float = 0.1       # no-object class weight in the CE term

    # Toy data and optimization.
    batch: int = 4
    max_objects: int = 2
    lr: float = 1e-3
