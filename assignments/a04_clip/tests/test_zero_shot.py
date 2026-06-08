"""Zero-shot classification by cosine similarity, and prompt-ensemble averaging."""

import torch
import torch.nn.functional as F

from inference import zero_shot_classify


def test_zero_shot_recovers_class():
    torch.manual_seed(0)
    k, d, b = 4, 8, 12
    protos = F.normalize(torch.randn(k, d), dim=-1)
    cls = torch.randint(0, k, (b,))
    imgs = F.normalize(protos[cls] + 0.05 * torch.randn(b, d), dim=-1)
    _, preds = zero_shot_classify(imgs, protos)
    assert (preds == cls).float().mean() > 0.95


def test_prompt_ensemble_averaging():
    # Average several noisy per-class templates, re-normalize, then classify. The
    # ensemble prototype is closer to the true class direction than any one template.
    torch.manual_seed(0)
    k, d, t = 3, 8, 6
    true = F.normalize(torch.randn(k, d), dim=-1)
    templates = F.normalize(true.unsqueeze(1) + 0.3 * torch.randn(k, t, d), dim=-1)  # (K, T, D)
    ensemble = F.normalize(templates.mean(dim=1), dim=-1)                            # (K, D)
    imgs = F.normalize(true + 0.1 * torch.randn(k, d), dim=-1)
    _, preds = zero_shot_classify(imgs, ensemble)
    assert (preds == torch.arange(k)).all()
