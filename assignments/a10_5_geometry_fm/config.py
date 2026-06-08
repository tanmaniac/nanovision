"""Hyperparameters for the DUSt3R-style pointmap toy. Provided.

The toy is two 16x16 images of an off-center sphere viewed from a wide baseline. ViT with
patch 4 gives a 4x4 = 16 patch grid; the model predicts one 3D point per patch (patch
resolution), so each pointmap is (B, 4, 4, 3). Real DUSt3R predicts at pixel resolution
through a DPT head; the toy stays at patch resolution to keep the network tiny and the tests
fast on CPU.
"""

from dataclasses import dataclass


@dataclass
class GeometryFMConfig:
    img_size: int = 16
    patch: int = 4
    in_chans: int = 3

    dim: int = 64               # shared ViT + decoder token dimension
    enc_depth: int = 2          # ViT encoder depth
    enc_heads: int = 4
    dec_depth: int = 2          # cross-attending decoder depth per view
    dec_heads: int = 4
    head_hidden: int = 128      # pointmap-head MLP width

    alpha: float = 0.2          # confidence regularizer weight (DUSt3R)

    # Toy scene geometry: off-center sphere + wide baseline (avoids the centered-sphere
    # degeneracy where the two views are near-identical and cross-attention is idle).
    radius: float = 1.6
    cam_dist: float = 4.0
    sphere_center: tuple = (0.7, 0.0, 0.4)  # off the world origin
    view1: int = 0              # camera index into the ring
    view2: int = 3              # wide baseline from view1 (opposite side of the ring)
    n_ring: int = 8             # cameras on the ring the two views are drawn from

    lr: float = 2e-3
    n_steps: int = 1500         # overfit steps for viz
