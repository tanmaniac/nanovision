"""generalized_iou and the box-format round-trip."""

import torch
from torch.autograd import gradcheck

from boxes import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, generalized_iou


def _xyxy_to_cxcywh_box(x0, y0, x1, y1):
    return torch.tensor([[(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]])


def test_identical_boxes_giou_is_one():
    # Identical boxes are a min/max kink, so this is a forward-value assert only (never
    # inside the gradcheck input).
    b = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    g = generalized_iou(b, b)
    assert g.shape == (1, 1)
    assert torch.allclose(g, torch.ones(1, 1), atol=1e-6)


def test_far_apart_boxes_negative_bounded():
    a = torch.tensor([[0.1, 0.1, 0.1, 0.1]])   # tiny box near top-left
    b = torch.tensor([[0.9, 0.9, 0.1, 0.1]])   # tiny box near bottom-right
    g = generalized_iou(a, b).item()
    assert g < 0.0
    assert g > -1.0


def test_known_overlapping_value():
    # A xyxy [0,0,2,2], B xyxy [1,1,3,3]: IoU = 1/7, GIoU = 1/7 - 2/9.
    a = _xyxy_to_cxcywh_box(0.0, 0.0, 2.0, 2.0)
    b = _xyxy_to_cxcywh_box(1.0, 1.0, 3.0, 3.0)
    g = generalized_iou(a, b).item()
    expected = 1 / 7 - 2 / 9
    assert abs(g - expected) < 1e-5, (g, expected)


def test_shape_pairwise():
    a = torch.rand(3, 4) * 0.5 + 0.25
    b = torch.rand(5, 4) * 0.5 + 0.25
    assert generalized_iou(a, b).shape == (3, 5)


def test_gradcheck_general_position():
    # General-position boxes: distinct coordinates, overlapping but not touching/identical,
    # so no min/max kink is hit at the 1e-6 perturbation.
    boxes1 = torch.tensor([[0.40, 0.45, 0.20, 0.30]], dtype=torch.float64, requires_grad=True)
    boxes2 = torch.tensor([[0.55, 0.50, 0.24, 0.22]], dtype=torch.float64, requires_grad=True)
    assert gradcheck(lambda a, b: generalized_iou(a, b), (boxes1, boxes2), eps=1e-6, atol=1e-4)


def test_format_roundtrip():
    b = torch.rand(7, 4) * 0.5 + 0.25   # positive w, h, inside [0,1]
    rt = box_xyxy_to_cxcywh(box_cxcywh_to_xyxy(b))
    assert torch.allclose(b, rt, atol=1e-6)
