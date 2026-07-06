"""A4 - zero-shot classification. Fill the one hole (Task 3).

The classifier has no trained head: the decision boundary is the geometry of the shared
embedding space. The reference is in solution/inference.py.
"""

import torch
from torch import Tensor


def zero_shot_classify(image_features: Tensor, text_features: Tensor) -> tuple[Tensor, Tensor]:
    """Classify images against class-prompt text embeddings by cosine similarity.

    image_features: L2-normalized (B, D). text_features: L2-normalized (K, D), one row
    per class prompt (or per-class averaged, re-normalized, prompt ensemble). Returns
    (logits (B, K), preds (B,)). See the zero-shot classification section of the README.
    """
    raise NotImplementedError("A4 Task 3: implement zero_shot_classify")
