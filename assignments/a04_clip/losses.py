"""A4 - the two contrastive losses. Fill the two holes (Tasks 1-2).

clip_loss is CLIP's symmetric InfoNCE; siglip_loss is SigLIP's per-pair sigmoid loss.
Build the InfoNCE cross-entropy by hand (no F.cross_entropy) so the softmax denominator
over the in-batch negatives is visible. The reference is in solution/losses.py.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def clip_loss(image_features: Tensor, text_features: Tensor, logit_scale: Tensor) -> Tensor:
    """Symmetric InfoNCE with a learnable temperature (CLIP, Radford et al., 2021).

    image_features, text_features: L2-normalized (N, D). logit_scale: the temperature in
    log space (logit_scale.exp() is the logit scale 1/tau). The matched pairs are the
    diagonal of the NxN similarity matrix. Do NOT use F.cross_entropy / F.log_softmax -
    build the cross-entropy by hand so the softmax denominator over the in-batch negatives
    is explicit. See the symmetric InfoNCE section of the README.
    """
    raise NotImplementedError("A4 Task 1: implement clip_loss (symmetric InfoNCE)")


def siglip_loss(image_features: Tensor, text_features: Tensor, logit_scale: Tensor,
                bias: Tensor) -> Tensor:
    """Pairwise sigmoid loss (SigLIP, Zhai et al., 2023), Algorithm 1.

    image_features, text_features: L2-normalized (N, D). logit_scale: temperature in log
    space. bias: the learnable SigLIP bias. Each (i, j) pair is an independent binary
    classification, matched (diagonal) or not. See the sigmoid loss section of the README.
    """
    raise NotImplementedError("A4 Task 2: implement siglip_loss (sigmoid loss)")
