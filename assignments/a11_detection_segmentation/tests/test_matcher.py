"""HungarianMatcher: the matcher must solve the assignment, not pick per-column argmins.

The discriminating case uses N=5 queries and M=2 ground-truth objects placed at the SAME
box but with different classes. One query (q2) sits exactly on that box, so it is the
cheapest column entry for BOTH ground-truths. A naive per-column argmin double-assigns q2 to
both, which is not a valid one-to-one matching; the Hungarian optimum is forced to give the
second ground-truth to the next-cheapest query (q4). M = 1 would make Hungarian a no-op that
a broken greedy implementation would also pass, so the test uses M >= 2.
"""

import torch

from boxes import generalized_iou
from matcher import HungarianMatcher


def _build_case():
    # Two gts at identical geometry, different class.
    gt_boxes = torch.tensor([[0.50, 0.50, 0.16, 0.16],
                             [0.50, 0.50, 0.16, 0.16]])
    gt_labels = torch.tensor([0, 1])
    pred_boxes = torch.tensor([
        [0.05, 0.05, 0.16, 0.16],   # q0 far
        [0.05, 0.95, 0.16, 0.16],   # q1 far
        [0.50, 0.50, 0.16, 0.16],   # q2 on the box: cheapest for both gts
        [0.95, 0.05, 0.16, 0.16],   # q3 far
        [0.46, 0.46, 0.16, 0.16],   # q4 near the box: second cheapest for both
    ])
    pred_logits = torch.zeros(1, 5, 4)   # uniform logits (3 classes + no-object)
    return pred_logits, pred_boxes[None], [gt_labels], [gt_boxes]


def test_shapes_and_lengths():
    logits, boxes, labels, gt = _build_case()
    out = HungarianMatcher().forward(logits, boxes, labels, gt)
    assert len(out) == 1
    row, col = out[0]
    assert row.shape == (2,) and col.shape == (2,)   # length M


def test_hungarian_beats_greedy():
    logits, boxes, labels, gt = _build_case()
    matcher = HungarianMatcher()

    # The cost matrix per-column argmin (greedy) double-assigns q2.
    prob = logits[0].softmax(-1)
    cost_class = -prob[:, labels[0]]
    cost_l1 = torch.cdist(boxes[0], gt[0], p=1)
    cost_giou = -generalized_iou(boxes[0], gt[0])
    C = matcher.cost_class * cost_class + matcher.cost_l1 * cost_l1 + matcher.cost_giou * cost_giou
    greedy = C.argmin(0)
    assert greedy[0].item() == greedy[1].item() == 2, "setup invalid: greedy must double-assign q2"

    row, col = matcher.forward(logits, boxes, labels, gt)[0]
    # Hungarian must use two DISTINCT queries, exactly the optimal set {q2, q4}.
    assert set(row.tolist()) == {2, 4}
    assert sorted(col.tolist()) == [0, 1]
    # And it must beat the (invalid) greedy total cost: a valid assignment over the diagonal.
    total = C[row, col].sum().item()
    assert total <= C[torch.tensor([2, 4]), torch.tensor([0, 1])].sum().item() + 1e-6


def test_cost_matrix_finite_and_no_grad():
    logits, boxes, labels, gt = _build_case()
    boxes = boxes.clone().requires_grad_(True)
    out = HungarianMatcher().forward(logits, boxes, labels, gt)
    row, col = out[0]
    assert not row.requires_grad and not col.requires_grad
    assert row.dtype == torch.long and col.dtype == torch.long


def test_empty_gt():
    logits, boxes, _, _ = _build_case()
    out = HungarianMatcher().forward(logits, boxes, [torch.zeros(0, dtype=torch.long)],
                                     [torch.zeros(0, 4)])
    row, col = out[0]
    assert row.numel() == 0 and col.numel() == 0
