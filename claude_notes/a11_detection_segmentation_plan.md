# A11 - detection and segmentation (DETR set prediction): build plan

Status: draft for expert review. Build subagent reads this file plus `agent_build_guide.md`
and mirrors `assignments/a06_0_flow_matching` for layout.

## What this assignment teaches

DETR (Carion et al., ECCV 2020) reframes detection as set prediction: output exactly the right
unordered set of (class, box) pairs in one forward pass, no anchors, no NMS. The duplicate
suppression that NMS used to do emerges from the training objective - one-to-one bipartite
matching forces each ground-truth object onto at most one query slot, so the model is trained
directly against producing duplicates.

The build core is the matching mechanism and the explicit distinction the whole assignment turns
on: the MATCHING COST is a fixed, non-differentiable function (computed in numpy, run through the
Hungarian algorithm) used only to decide which predicted slot is compared to which ground-truth
box; the TRAINING LOSS is the differentiable objective computed AFTER the assignment is fixed. The
Hungarian assignment itself never needs a gradient - its indices are constants in the backward
pass. Conflating the two is the common misconception; the assignment separates them in code.

Survey (README): the convergence fixes and lineage DETR -> Deformable DETR -> DAB-DETR -> DN-DETR
-> DINO-DETR -> RT-DETR; Mask2Former (same queries+matching, mask outputs); GroundingDINO
(open-vocab via language); SAM -> SAM 2 -> SAM 3 (promptable segmentation).

## Reused, already in place (do NOT rebuild)

- `from nanovision.vit import ViT` - the A2 ViT backbone; `forward_features(img)` gives the
  (B, N, dim) patch-token feature map the decoder cross-attends to.
- `from nanovision.transformer import TransformerBlock` - build the query decoder from
  `TransformerBlock(dim, n_heads, causal=False, cross_attn=True)`: object queries self-attend
  (bidirectional) then cross-attend to the ViT patch tokens. (NOT TransformerDecoder - it is
  hardcoded causal, wrong for the query set.)
- `scipy.optimize.linear_sum_assignment` - installed (added for the flow-matching OT coupling).
- `nanovision.determinism.default_device` for the GPU viz; tests stay CPU.

## Toy data: `nanovision.data.toy.detection_batch(...)` (NEW, provided)

Add to `nanovision/data/toy.py`. Tiny detection: images with 1-2 solid colored squares at known
locations. Return (images (B,3,32,32) in [0,1], list/padded boxes (B, M, 4) in normalized cxcywh
in [0,1], labels (B, M) class ids, and a valid/length mask since M varies per image). A square's
class is its color; the box is its exact extent. Deterministic per seed. Keep M small (1-2) and
the image 32x32 so the overfit fits in CPU-seconds. Do not edit other toy.py functions.

## Shapes (fix these numbers)

- Image (B, 3, 32, 32). ViT patch 4 -> 8x8 = 64 patch tokens, dim ~128.
- N = 10 object queries (learned embeddings, dim 128).
- Decoder -> N query features -> two heads: class logits (B, N, C+1) including the no-object
  class at index C, and box (B, N, 4) cxcywh via sigmoid.
- Ground truth per image: M (1-2) boxes (M,4) cxcywh and labels (M,).
- Matcher: per image a cost matrix (N, M); returns (row_idx, col_idx) of length M (= min(N,M)).

## Files (mirror the exemplar layout)

### `boxes.py` (holed; solution copy) - assignment-local

Holes:
- `box_cxcywh_to_xyxy(b)` and `box_xyxy_to_cxcywh(b)`: the format conversions (provide one as the
  hole, or both). Small but used everywhere; state the normalized [0,1] convention.
- `generalized_iou(boxes1, boxes2)`: pairwise GIoU. Convert cxcywh -> xyxy, intersection area
  (CLAMP the intersection width/height to >= 0 or non-overlapping boxes give spurious negative
  area), union area, smallest enclosing box area $C$, $\mathrm{GIoU} = \mathrm{IoU} - |C \setminus
  (A\cup B)| / |C|$. Add an `eps` (1e-7) to BOTH denominators (the union in IoU and $|C|$) so
  zero-area degenerate boxes do not divide by zero. Returns (N1, N2), bounded in $(-1, 1]$.
  Differentiable (reused in the training loss). State that GIoU is defined even for non-overlapping
  boxes (unlike IoU), which is why the matching cost and the box loss use it. The cxcywh->xyxy
  conversion assumes positive w,h (true for sigmoid box outputs and positive-extent GT).

### `matcher.py` (holed; solution copy) - assignment-local

Hole:
- `HungarianMatcher.forward(self, pred_logits, pred_boxes, gt_labels, gt_boxes)`: build the per-
  image cost matrix $C[i,j] = \lambda_{cls}\,(-p_i[c_j]) + \lambda_{L1}\,\lVert b_i - b_j\rVert_1
  + \lambda_{giou}\,(-\mathrm{GIoU}(b_i, b_j))$ where $p = \mathrm{softmax}(\text{logits})$, boxes
  in cxcywh; DETACH it to numpy and call `scipy.optimize.linear_sum_assignment`; return the
  (row, col) index pairs per image. This is the COST - explicitly non-differentiable (state in the
  docstring that the indices are constants in the backward pass; no gradient flows through the
  matcher). Default weights $\lambda_{cls}=1, \lambda_{L1}=5, \lambda_{giou}=2$ (DETR values).

### `loss.py` (holed; solution copy) - assignment-local

Hole:
- `detr_loss(pred_logits, pred_boxes, gt_labels, gt_boxes, indices, num_classes, *, eos_coef=0.1)`:
  the differentiable TRAINING LOSS computed AFTER the matcher fixes `indices`. (a) Classification:
  cross-entropy over all N queries to their target class - matched queries get their gt class,
  unmatched get the no-object class (index = num_classes). Implement the downweight as a length
  $C{+}1$ CE weight vector with `weight[num_classes] = eos_coef = 0.1` and 1.0 for real classes,
  passed to `F.cross_entropy(weight=...)` - NOT a post-hoc scaling of unmatched queries. (b) Box L1
  on matched pairs (cxcywh). (c) GIoU loss $1 - \mathrm{GIoU}$ on matched pairs. Weighted sum with
  the DETR weights (cls 1, L1 5, giou 2). Returns the scalar loss (and optionally the components).
  State the contrast precisely: L1 and GIoU are the SAME function used in both the cost and the
  loss, but the CLASS term differs - the cost uses $-p[c]$ (raw probability) while the loss uses
  cross-entropy $-\log p[c]$. Same information, different function; do not call them identical.

### `model.py` (provided) - assignment-local

`DETR(nn.Module)`: ViT backbone (forward_features), N learned object-query embeddings, a decoder
of `TransformerBlock(causal=False, cross_attn=True)` (queries self-attend then cross-attend to the
patch tokens), a class head `Linear(dim, C+1)` and a box head (MLP -> sigmoid -> cxcywh).
`forward(img)` -> (logits (B,N,C+1), boxes (B,N,4)). Provided wiring; the lesson is the matcher +
loss, not the plumbing. WIRING: the ViT must be built with `img_size=32, patch=4` so n_patches=64
matches its grid-locked learned pos_embed - pin these in config.py and assert n_patches in
`__init__`, or `forward_features` breaks on the pos_embed add. This build deliberately OMITS two
real DETR pieces: per-decoder-layer re-injection of 2D image positional encoding into the
cross-attention keys (here the only spatial signal is the ViT's single learned pos_embed), and the
per-layer auxiliary losses (a loss at every decoder layer). State both as simplifications in the
README; the box localization is bounded by the 4-pixel patch stride.

### `config.py`, `viz.py` (provided)

- config: image 32, patch 4, vit dim/depth, N=10 queries, num_classes (= number of square colors),
  cost/loss weights, eos_coef 0.1, lrs.
- viz (GPU via default_device): overfit the toy detection set, draw predicted boxes over the images
  next to GT, and show the matching (which query -> which GT). Save to out/. Not graded.

### `conftest.py`, `__init__.py`, `solution/`

Mirror the exemplar. `solution/` holds boxes.py, matcher.py, loss.py (plus __init__.py). model.py,
config.py, viz.py top-level only.

## Tests (env python, CPU, seconds each; training-free exact checks preferred)

1. `test_giou.py`: `generalized_iou` - identical boxes give 1.0 (a forward-value assertion ONLY -
   identical boxes are a min/max kink, do not put them inside the gradcheck input); two far-apart
   non-overlapping boxes give a value < 0 and > -1; a known overlapping case matches a hand-computed
   value; gradcheck (float64) on generalized_iou using GENERAL-POSITION boxes (no equal
   coordinates, not exactly touching, not identical - so no kink is hit at the 1e-6 perturbation).
   Also box format round-trip cxcywh->xyxy->cxcywh exact.
2. `test_matcher.py`: a hand-built case (N=5 queries, M>=2 gt) where a NAIVE per-row argmin would
   DOUBLE-ASSIGN one query to both gt boxes, but the Hungarian optimum assigns distinct queries -
   so the test actually discriminates Hungarian from greedy (an M=1 case makes Hungarian a no-op
   and would pass a broken implementation). Assert the returned (row,col) indices are exactly the
   known Hungarian optimum; cost matrix shape (N, M); indices length M. Confirm the matcher output
   carries NO grad (it is detached/numpy). Assert the cost matrix is finite before the scipy call.
3. `test_loss.py`: `detr_loss` is a scalar with requires_grad=True; gradcheck (float64) w.r.t.
   pred_logits and pred_boxes with fixed `indices`; the no-object downweight is applied (an
   unmatched query's class-loss contribution scales with eos_coef).
4. `test_overfit.py`: overfit the 4-image toy detection set (bounded epochs, ~a few hundred). The
   total loss falls near zero and predicted matched boxes overlay the GT squares within a measured
   center-error tolerance. Report the floors; set thresholds from them (no-thrash). Keep it CPU-
   seconds; if 32x32 + ViT is too slow on CPU, shrink (smaller ViT, fewer steps) and measure.
5. `test_forbidden_imports.py`: static scan; forbid torchvision detection ops / ready-made DETR
   (torchvision.ops boxes/giou, detectron2, ultralytics); scipy.optimize.linear_sum_assignment is
   ALLOWED (it is the matcher's intended tool, like the OT coupling in flow matching). Mirror the
   exemplar.

Solution mode all green; default mode fails only at the holes, except test_forbidden_imports. Run
with `/home/tanmay/miniconda3/envs/nanovision/bin/python`.

## README (comprehensive lecture notes, per the skill, real LaTeX)

Fixed section order. Cover:
- Set prediction vs anchors/NMS: why a permutation-invariant loss is needed (gt boxes have no
  canonical order), how one-to-one matching removes NMS.
- Bipartite matching and the Hungarian algorithm: the COST (non-differentiable, scipy) vs the
  LOSS (differentiable, after assignment) distinction as the central teaching point - spell it out,
  it is the most-conflated thing. The three cost terms (class prob, L1, GIoU) and why GIoU (defined
  for non-overlapping boxes).
- Object queries: learned positional slots that specialize spatially/by size; DAB-DETR's
  anchor-box queries as the more explicit version.
- Why NMS disappears (one-to-one matching trains against duplicates).
- DETR's convergence problem (500 epochs) and the fixes: deformable attention (local sampled keys,
  Deformable DETR), query denoising (DN-DETR), contrastive denoising + mixed query selection
  (DINO-DETR). The lineage diagram as text/list (not a flowchart if the style rule prefers prose;
  a small mermaid is acceptable for the lineage). RT-DETR as the real-time answer that beats YOLO,
  removing the "DETR is too slow" objection.
- Segmentation: Mask2Former (same queries + matching, masked cross-attention, mask outputs instead
  of boxes), GroundingDINO (open-vocab via a text backbone + language-guided queries), SAM ->
  SAM 2 (streaming memory for video) -> SAM 3 (concept/noun-phrase + exemplar prompts). Survey
  only, do not implement.
- Where this goes: the query + bipartite-matching paradigm is the dominant class for detection,
  segmentation, tracking, and open-vocab grounding; anchor-based YOLO is the resource-constrained
  competitor. Relevant to the AV stack (RT-DETR/DINO in production, Grounding DINO for rare classes,
  SAM 2/3 in annotation).
- TOY-SCOPE DISCLAIMER (required, course principle): state explicitly that this toy proves the
  mechanism runs and OVERFITS a tiny set - it proves NOTHING about convergence speed (the
  500-epoch DETR-is-slow / Deformable-DETR-fixes-it story is about COCO, not 4 overfit images),
  about NMS-free behavior under real object density (1-2 objects never stress duplicate
  suppression), about the ViT being a good detector backbone, or about any AP/accuracy number.
  Those belong to the survey papers, not the toy. Present the lineage as a pedagogical ordering,
  not a strict ancestry: DAB-DETR and DN-DETR are parallel early-2022 contributions that DINO
  unifies (its name bundles DeNoising + Anchor boxes), so write "DINO combines DAB's anchor-box
  queries and DN's denoising," not "DN descends from DAB". Use the official spelling "Grounding
  DINO" (two words). For SAM 3 (released Nov 2025) and SAM 3.1 (Mar 2026): assign NO arXiv id
  (none is verifiable), link the Meta release page if citing, and hedge their capability claims
  (noun-phrase/exemplar prompts, multi-object tracking) as reported, not established fact.

Verify every arXiv id by fetching `https://arxiv.org/abs/<id>`: DETR 2005.12872, Deformable DETR
2010.04159, DAB-DETR 2201.12329, DN-DETR 2203.01305, DINO-DETR 2203.03605, RT-DETR 2304.08069,
Mask2Former 2112.01527, GroundingDINO 2303.05499, SAM 2304.02643, SAM 2 2408.00714. Re-verify each
title. Run the mandatory context-less style review on the README.

## ASSIGNMENT.md

Concise builder contract in `TEMPLATE.md` format: holes (box conversions, generalized_iou,
HungarianMatcher.forward, detr_loss), what is provided, the verify command, the measured
thresholds. Do not echo the README prose.

## Decisions resolved by the expert review (build to these)

The expert verified all 10 arXiv ids correct and confirmed the cost-vs-loss mechanism and DETR
weights. Resolutions folded in:

1. Cost ($-p[c]$, L1, $-$GIoU, scipy, detached) and loss (CE, L1, $1-$GIoU, weighted) correct;
   weights $\lambda_{cls}=1,\lambda_{L1}=5,\lambda_{giou}=2$, eos_coef 0.1 are paper values. FIX:
   the class term is NOT the same in cost ($-p$) vs loss (CE = $-\log p$); only L1/GIoU are the
   same function twice. eos_coef is a length-$C{+}1$ CE weight vector, not a post-hoc scaling.
2. GIoU formula confirmed, bounded $(-1,1]$. FIX: eps in BOTH denominators, clamp intersection to
   >=0, general-position boxes for gradcheck (identical-box GIoU=1 is a kink - forward assert only).
3. A2 ViT backbone is an acceptable simplification. Pin img_size=32, patch=4 (pos_embed is
   grid-locked to 64 tokens); patch stride bounds localization. Overfit demo, not a ViT-detector
   claim.
4. The toy is sufficient but the MATCHER TEST must use M>=2 where greedy argmin double-assigns (M=1
   makes Hungarian a no-op). Loss floor ~0.05-0.3 (CE residual), center error ~0.06 of image size -
   MEASURE, do not hardcode.
5. Lineage accurate as a teaching arc (DAB/DN parallel into DINO, not linear). All ids verified.
   "Grounding DINO" two words. SAM 3 / 3.1: no arXiv id, hedge capability claims.
6. Build core: pin/assert the ViT grid; note the omitted per-layer pos re-injection and aux losses;
   assert the cost matrix is finite before scipy; explicit toy-scope disclaimer in the README.
