"""detr_loss: a differentiable scalar, gradcheckable, with the no-object downweight applied."""

import torch
from torch.autograd import gradcheck

from loss import detr_loss


def _fixed_setup():
    # B=2, N=4 queries, C=3 classes (+ no-object = index 3). Fixed indices (no matcher).
    num_classes = 3
    gt_labels = [torch.tensor([0, 2]), torch.tensor([1])]
    gt_boxes = [
        torch.tensor([[0.30, 0.30, 0.20, 0.20], [0.70, 0.70, 0.16, 0.16]]),
        torch.tensor([[0.50, 0.50, 0.24, 0.24]]),
    ]
    # image 0: query 0 -> gt0, query 2 -> gt1; image 1: query 1 -> gt0.
    indices = [
        (torch.tensor([0, 2]), torch.tensor([0, 1])),
        (torch.tensor([1]), torch.tensor([0])),
    ]
    return num_classes, gt_labels, gt_boxes, indices


def test_scalar_with_grad():
    nc, gl, gb, idx = _fixed_setup()
    logits = torch.randn(2, 4, nc + 1, requires_grad=True)
    boxes = (torch.rand(2, 4, 4) * 0.5 + 0.25).requires_grad_(True)
    total, comps = detr_loss(logits, boxes, gl, gb, idx, nc)
    assert total.ndim == 0 and total.requires_grad
    assert set(comps) == {"class", "l1", "giou"}
    total.backward()
    assert logits.grad is not None and boxes.grad is not None


def test_gradcheck():
    nc, gl, gb, idx = _fixed_setup()
    logits = torch.randn(2, 4, nc + 1, dtype=torch.float64, requires_grad=True)
    boxes = (torch.rand(2, 4, 4, dtype=torch.float64) * 0.4 + 0.3).requires_grad_(True)

    def f(lg, bx):
        return detr_loss(lg, bx, gl, gb, idx, nc)[0]

    assert gradcheck(f, (logits, boxes), eps=1e-6, atol=1e-4)


def test_eos_downweight_scales_unmatched_term():
    # A single image, one matched query, the rest unmatched (no-object). Raising eos_coef
    # must raise the classification loss, since the unmatched queries' no-object CE term is
    # weighted by eos_coef.
    nc = 3
    gl = [torch.tensor([0])]
    gb = [torch.tensor([[0.5, 0.5, 0.2, 0.2]])]
    idx = [(torch.tensor([0]), torch.tensor([0]))]
    torch.manual_seed(0)
    logits = torch.randn(1, 5, nc + 1)
    boxes = torch.rand(1, 5, 4) * 0.4 + 0.3

    _, c_low = detr_loss(logits, boxes, gl, gb, idx, nc, eos_coef=0.1)
    _, c_high = detr_loss(logits, boxes, gl, gb, idx, nc, eos_coef=1.0)
    assert c_high["class"].item() > c_low["class"].item()
    # box terms do not depend on eos_coef.
    assert torch.allclose(c_low["l1"], c_high["l1"])
    assert torch.allclose(c_low["giou"], c_high["giou"])
