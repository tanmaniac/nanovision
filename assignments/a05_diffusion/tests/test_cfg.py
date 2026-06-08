"""Classifier-free guidance: the combine formula and the training-time label dropout."""

import torch

from diffusion import diffusion_loss
from sampling import classifier_free_guidance
from schedule import cosine_alpha_bar


def test_guidance_combine():
    c = torch.randn(2, 1, 4, 4)
    u = torch.randn(2, 1, 4, 4)
    assert torch.allclose(classifier_free_guidance(c, u, 1.0), c)        # w=1 -> conditional
    assert torch.allclose(classifier_free_guidance(c, u, 0.0), u)        # w=0 -> unconditional
    assert torch.allclose(classifier_free_guidance(c, u, 3.0), u + 3.0 * (c - u))


class RecordLabels:
    null_index = 99

    def __init__(self):
        self.seen = None

    def __call__(self, x, t, labels=None):
        self.seen = None if labels is None else labels.clone()
        return torch.zeros_like(x)


def test_label_dropout_rate():
    _, abar = cosine_alpha_bar(20)
    x0 = torch.randn(8, 1, 8, 8)
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])

    m = RecordLabels()
    diffusion_loss(m, x0, abar, kind="v", num_classes=3, cfg_drop_prob=1.0, labels=labels)
    assert torch.all(m.seen == 3)                  # all dropped to the null index

    m = RecordLabels()
    diffusion_loss(m, x0, abar, kind="v", num_classes=3, cfg_drop_prob=0.0, labels=labels)
    assert torch.all(m.seen == labels)             # none dropped
