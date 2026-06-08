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
    diagonal of the NxN similarity matrix.

    Implement (do NOT use F.cross_entropy / F.log_softmax - build it so the denominator
    is explicit):
        1. logits = logit_scale.exp() * image_features @ text_features.T      # (N, N)
        2. log_softmax each row by hand: log_prob = z - logsumexp(z, dim=1, keepdim=True)
        3. the image->text CE is -mean of the diagonal of log_prob(logits)
        4. the text->image CE is the same on logits.T
        5. return 0.5 * (image->text CE + text->image CE)
    """
    raise NotImplementedError("A4 Task 1: implement clip_loss (symmetric InfoNCE)")


def siglip_loss(image_features: Tensor, text_features: Tensor, logit_scale: Tensor,
                bias: Tensor) -> Tensor:
    """Pairwise sigmoid loss (SigLIP, Zhai et al., 2023), Algorithm 1.

    image_features, text_features: L2-normalized (N, D). logit_scale: temperature in log
    space. bias: the learnable SigLIP bias. Each (i, j) pair is an independent binary
    classification, matched (diagonal) or not.

    Implement:
        1. logits = logit_scale.exp() * image_features @ text_features.T + bias   # (N, N)
        2. labels = 2 * eye(N) - 1                # +1 on the diagonal, -1 off
        3. return -F.logsigmoid(labels * logits).sum() / N   # sum over NxN, divide by N
    """
    raise NotImplementedError("A4 Task 2: implement siglip_loss (sigmoid loss)")
