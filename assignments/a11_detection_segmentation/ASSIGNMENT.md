# assignments/a11_detection_segmentation/ASSIGNMENT.md

```yaml
id: a11_detection_segmentation
title: Detection and segmentation as set prediction (DETR)
module: 4
type: Core
estimated_learner_hours: 8
depends_on: [a00_harness, a01_transformer, a02_vit]
builds_into_shared_lib: []
forbidden_imports:
  - import torchvision
  - from torchvision
  - import detectron2
  - from detectron2
  - import ultralytics
  - from ultralytics
  - import mmdet
  - from mmdet
  - import mmcv
  - from mmcv
  - import transformers
  - from transformers
fits_12gb: true
external_data: "none (synthetic colored-squares detection toy)"
```

## motivation
DETR (Carion et al., 2020) reframes object detection as set prediction: emit a fixed-size
unordered set of (class, box) pairs in one forward pass, with no anchor boxes and no
non-maximum suppression (NMS). One-to-one bipartite matching between predictions and ground
truth supplies the training target and, as a side effect, trains the model against producing
duplicates, so NMS is not needed. The mechanism is the content. You build generalized IoU,
the Hungarian matching cost, and the set-prediction loss; the ViT backbone, query decoder,
and heads are provided. See the README for the math and the lineage to Deformable DETR,
DINO, RT-DETR, Mask2Former, Grounding DINO, and SAM.

## background
See the README. Convention: boxes are normalized cxcywh in [0, 1]. The model outputs N=10
query slots, each a class distribution over C+1 classes (the last is the no-object class) and
a box. The matcher builds a per-image cost matrix and solves the one-to-one assignment with
the Hungarian algorithm; the cost is non-differentiable (numpy, detached). The loss is the
differentiable objective computed after the assignment is fixed. The cost uses the raw class
probability -p[c]; the loss uses cross-entropy -log p[c]. L1 and GIoU are the same function
in both.

## what_you_implement
- box_xyxy_to_cxcywh, generalized_iou (boxes.py): the format inverse and pairwise GIoU.
- HungarianMatcher.forward (matcher.py): the cost matrix and the Hungarian assignment.
- detr_loss (loss.py): CE with the no-object downweight, L1 and 1-GIoU on matched pairs.

The ViT backbone, query decoder, heads (model.py), config, viz, and the colored-squares toy
(nanovision.data.toy.detection_batch) are provided.

## tasks
1. `box_xyxy_to_cxcywh` (`boxes.py`): center is the corner midpoint, width/height the corner
   differences. (`box_cxcywh_to_xyxy` is provided.)
2. `generalized_iou` (`boxes.py`): convert cxcywh -> xyxy, areas, clamped intersection,
   union, IoU = inter / (union + eps); enclosing box C; GIoU = IoU - (area_C - union) /
   (area_C + eps). eps = 1e-7 in BOTH denominators. Returns (N1, N2) in (-1, 1].
3. `HungarianMatcher.forward` (`matcher.py`): per image,
   C = cost_class*(-softmax(logits)[:, gt_labels]) + cost_l1*cdist(boxes, gt, p=1) +
   cost_giou*(-GIoU); assert finite; scipy linear_sum_assignment on the detached numpy matrix;
   return (row, col) long tensors. Empty gt -> empty indices. No gradient through the matcher.
4. `detr_loss` (`loss.py`): target class = gt class on matched queries, num_classes (no-object)
   elsewhere; CE with a length-(C+1) weight vector, weight[num_classes] = eos_coef = 0.1; L1 and
   1-GIoU on matched pairs; weighted sum (class 1, L1 5, GIoU 2).

## tests
Run with `/home/tanmay/miniconda3/envs/nanovision/bin/python -m pytest` in this order:
1. `tests/test_giou.py` - identical-box GIoU = 1 (forward only, a kink); far-apart pair in
   (-1, 0); a hand-computed overlapping value (1/7 - 2/9); float64 gradcheck on
   general-position boxes; cxcywh<->xyxy round-trip.
2. `tests/test_matcher.py` - N=5, M=2 case where per-column argmin double-assigns one query
   but Hungarian picks distinct queries {q2, q4}; shapes (N, M) and length M; indices carry no
   grad; empty-gt case.
3. `tests/test_loss.py` - scalar with requires_grad; float64 gradcheck w.r.t. logits and
   boxes at fixed indices; raising eos_coef raises the classification loss (box terms
   unchanged).
4. `tests/test_overfit.py` - 4-image toy, 500 Adam steps: final loss < 0.5 and < 0.1*first;
   matched center error < 0.06 of the image size.
5. `tests/test_forbidden_imports.py` - no torchvision/detectron2/ultralytics/mmdet/mmcv/
   transformers; scipy is allowed. Passes with the holes in place too.

Solution mode (`NANOVISION_IMPL=solution`) is fully green. Default mode fails only at the
holes with NotImplementedError, except `test_forbidden_imports`; the overfit test also surfaces
the ViT backbone's own hole until that assignment is filled.

## provided_boilerplate
`model.py` `DETR` (ViT backbone via `nanovision.vit`, learned query embeddings, a
`TransformerBlock(cross_attn=True)` decoder via `nanovision.transformer`, class and box heads).
`config.py` `DETRConfig`. `viz.py` overfits the toy and draws matched predictions vs ground
truth. `nanovision.data.toy.detection_batch` returns images (B,3,32,32), padded boxes
(B,M,4) cxcywh, labels (B,M), and a valid mask.

## compute_notes
All graded tests run on CPU. The overfit test is 500 Adam steps on the 4-image toy, ~15s on
CPU. The loss falls from ~4.8 to a floor near 0.1-0.25, dominated by the GIoU term: the ViT's
4-pixel patch stride bounds how finely a box can be localized at 32x32, so the GIoU residual
does not vanish. Center error still reaches well under 0.06. viz uses the GPU when present.

## solution_notes
The cost and loss differ in the class term only: cost uses -softmax probability, loss uses
cross-entropy. The eos downweight is a CE weight vector (`weight[num_classes] = 0.1`), not a
post-hoc scaling of unmatched queries; with N=10 queries and 1-2 objects per image, the
no-object class is the overwhelming majority target, so without the downweight the model
collapses to predicting no-object. The GIoU intersection width/height must be clamped to >= 0
(disjoint boxes otherwise get spurious negative area), and eps goes in both denominators so
degenerate boxes do not divide by zero. The matcher must use M >= 2 to discriminate Hungarian
from greedy; M = 1 makes Hungarian a no-op. The model omits two real DETR pieces: per-layer 2D
positional re-injection into the cross-attention keys, and per-layer auxiliary losses.
