"""Reference: zero-shot classification by cosine similarity in the shared space.

Answer key for the top-level inference.py. The classifier has no trained head: the
decision boundary is the geometry of the shared embedding space.
"""

import torch
from torch import Tensor


def zero_shot_classify(image_features: Tensor, text_features: Tensor) -> tuple[Tensor, Tensor]:
    """Classify images against class-prompt text embeddings by cosine similarity.

    image_features: L2-normalized (B, D). text_features: L2-normalized (K, D), one row
    per class prompt (or per-class averaged prompt ensemble, already re-normalized).
    Returns (logits (B, K) = cosine similarities, predictions (B,) = argmax class).
    """
    logits = image_features @ text_features.T   # (B, K) cosine sims (features are unit-norm)
    preds = logits.argmax(dim=-1)
    return logits, preds
