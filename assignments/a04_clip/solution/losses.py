"""Reference: the two contrastive losses, CLIP's InfoNCE and SigLIP's sigmoid.

Answer key for the top-level losses.py. clip_loss builds the symmetric softmax
cross-entropy by hand (no F.cross_entropy) so the in-batch-negative denominator is
visible; siglip_loss is the SigLIP paper's per-pair sigmoid loss.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def clip_loss(image_features: Tensor, text_features: Tensor, logit_scale: Tensor) -> Tensor:
    """Symmetric InfoNCE with a learnable temperature (CLIP, Radford et al., 2021).

    image_features, text_features: L2-normalized (N, D). logit_scale: the temperature
    parameter in log space (so logit_scale.exp() is the logit scale 1/tau). Build the
    scaled cosine-similarity matrix, then average the image->text and text->image
    cross-entropies with the matched pair on the diagonal.
    """
    logits = logit_scale.exp() * image_features @ text_features.T   # (N, N)

    def ce_diag(z: Tensor) -> Tensor:
        # log_softmax over each row, by hand: the logsumexp is the stable denominator
        # summed over all N negatives, which is the contrastive structure being taught.
        log_prob = z - torch.logsumexp(z, dim=1, keepdim=True)
        return -log_prob.diagonal().mean()

    return 0.5 * (ce_diag(logits) + ce_diag(logits.T))


def siglip_loss(image_features: Tensor, text_features: Tensor, logit_scale: Tensor,
                bias: Tensor) -> Tensor:
    """Pairwise sigmoid loss (SigLIP, Zhai et al., 2023), Algorithm 1.

    logits = logit_scale.exp() * img @ txt.T + bias; label matrix is +1 on the diagonal
    and -1 off it; the loss is the negative log-sigmoid summed over the full NxN matrix
    and divided by N (the batch size), NOT meaned over N*N. Each pair is an independent
    binary classification, so there is no softmax denominator over the batch.
    """
    n = image_features.shape[0]
    logits = logit_scale.exp() * image_features @ text_features.T + bias   # (N, N)
    labels = 2.0 * torch.eye(n, device=logits.device, dtype=logits.dtype) - 1.0
    return -F.logsigmoid(labels * logits).sum() / n
