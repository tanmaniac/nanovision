# A11 - detection and segmentation as set prediction

Object detection asks a model to output a set of objects: for each one, a class and a
bounding box. The set has no canonical order (the three cars in an image are not numbered),
and its size changes from image to image. For years detectors sidestepped both problems with
a fixed scaffold of anchors and a hand-built deduplication step. DETR (the detection
transformer) removed that scaffold by training directly against the set: predict a fixed
number of slots, match them one-to-one to the ground-truth objects, and let the matching loss
do the work that anchors and non-maximum suppression used to do.

Build the matching mechanism and the set-prediction loss on a tiny toy of colored squares on a
black background. The pieces to implement are generalized IoU, the Hungarian matching cost,
and the loss. The ViT backbone, the query decoder, and the output heads are provided, because
the lesson is the matcher and the loss, not the plumbing.

Required reading before starting:
- Carion et al. 2020, "End-to-End Object Detection with Transformers" (DETR),
  [arXiv:2005.12872](https://arxiv.org/abs/2005.12872).
- Rezatofighi et al. 2019, "Generalized Intersection over Union",
  [arXiv:1902.09630](https://arxiv.org/abs/1902.09630).

## Lecture notes

### The pre-DETR detection pipeline and its scaffolding

A detector has to turn a dense feature map into a variable-length list of objects. Before
DETR, the dominant answer was anchors: tile the image with a fixed grid of reference boxes
at several scales and aspect ratios (thousands to hundreds of thousands of them), and for
each anchor predict whether an object is centered there, which class, and a small offset that
refines the anchor into the final box. Faster R-CNN, RetinaNet, and the YOLO family are all
variants of this idea.

Anchors create two problems that the pipeline then has to clean up. First, many anchors near
a true object all fire, so the raw output holds several overlapping boxes per object.
Non-maximum suppression (NMS) removes the duplicates: sort boxes by confidence, keep the
highest, delete every remaining box whose IoU with it exceeds a threshold, repeat. NMS is a
greedy, non-differentiable post-process with its own threshold to tune, and it sits outside
the network, so the model is never trained against the duplicates it produces. Second,
training needs a rule that says which anchor is responsible for which object. That rule is
also hand-built: assign each ground-truth box to the anchors whose IoU with it exceeds a
threshold, mark the rest as background, and accept the class imbalance (most anchors are
background) with focal loss or hard-negative mining.

DETR's claim is that both pieces, the anchors and the NMS, are scaffolding that can be removed
by training the model to emit the set directly.

### Set prediction needs a permutation-invariant loss

Suppose the model outputs $N$ slots, each a (class, box) prediction, and the image contains
$M \le N$ objects. The loss must compare an ordered list of $N$ predictions to an unordered
set of $M$ ground-truth objects. Pairing prediction $i$ with ground-truth $i$ by index would
punish the model for emitting the right objects in the wrong slot order, which is a
meaningless target, because the objects have no order.

The fix is to find the best assignment first, then compute the loss against that assignment.
Pad the ground truth to size $N$ with a special no-object class $\varnothing$, so both sides
have $N$ elements, and search over permutations $\sigma$ of the predictions for the one that
minimizes a matching cost:

$$\hat{\sigma} = \arg\min_{\sigma}\ \sum_{i=1}^{N} \mathcal{C}_{\text{match}}\big(y_i,\ \hat{y}_{\sigma(i)}\big)$$

where $y_i$ is the $i$-th ground-truth object (real or $\varnothing$) and $\hat{y}_{\sigma(i)}$
is the prediction assigned to it. This is a bipartite matching between $N$ predictions and $N$
padded targets. Because $\varnothing$ targets contribute a constant to the cost regardless of
which slot they land on, the search effectively assigns the $M$ real objects to $M$ distinct
predictions and leaves the rest as no-object.

### Bipartite matching and the Hungarian algorithm

A bipartite matching pairs each element of one set with a distinct element of another to
minimize total cost. With a square cost matrix $C[i, j]$ (cost of assigning prediction $i$ to
target $j$), the optimal one-to-one assignment is the linear assignment problem, solved
exactly in $O(n^3)$ by the Hungarian algorithm (Kuhn 1955). SciPy's
`scipy.optimize.linear_sum_assignment` is the standard solver, the same one the flow-matching
optimal-transport coupling used.

The greedy alternative, for each target pick its lowest-cost prediction independently, can
assign the same prediction to two targets, which is not a valid one-to-one matching. Suppose
two ground-truth objects sit at the same location and one query is closest to both: per-column
argmin double-assigns that single query, while the Hungarian solution is forced to give the
second object to the next-cheapest query.

#### The cost is not the loss

DETR keeps two separate quantities, and conflating them is the common mistake.

The matching cost $\mathcal{C}_{\text{match}}$ decides which prediction is compared to which
ground-truth object. It is computed once per training step, detached to numpy, and fed to the
Hungarian solver. It produces a set of integer index pairs. No gradient flows through it: the
assignment indices are constants in the backward pass.

The training loss is the differentiable objective computed after the assignment is fixed. It
takes the index pairs as given and computes a loss that the model's parameters are updated
against.

The Hungarian assignment never needs a gradient, because its output is a discrete choice of
indices, and there is no way to backpropagate through $\arg\min$ over permutations. The errors
to avoid are trying to differentiate through the matcher and reusing the cost as the loss.

```mermaid
flowchart LR
    A["model forward: logits, boxes (with grad)"] --> B["matcher: build cost, detach to numpy"]
    B --> C["scipy linear_sum_assignment"]
    C --> D["index pairs (row, col), no grad"]
    A --> E["loss: CE + L1 + GIoU at fixed indices"]
    D --> E
    E --> F["scalar loss, backward to model params"]
```

The cost uses three terms with the DETR weights $\lambda_{\text{cls}}=1$, $\lambda_{L1}=5$,
$\lambda_{\text{giou}}=2$:

$$\mathcal{C}[i, j] = \lambda_{\text{cls}}\,\big(-p_i[c_j]\big)
  + \lambda_{L1}\,\lVert b_i - b_j \rVert_1
  + \lambda_{\text{giou}}\,\big(-\mathrm{GIoU}(b_i, b_j)\big)$$

where $p_i = \mathrm{softmax}(\text{logits}_i)$ and $c_j$ is target $j$'s class. The class
term is the negative predicted probability of the true class, so a query that already assigns
high probability to the right class is cheap. The L1 term is the absolute difference of the
cxcywh box coordinates. The GIoU term rewards box overlap.

### Generalized IoU

Intersection-over-union, $\mathrm{IoU} = |A \cap B| / |A \cup B|$, is the standard box-overlap
score, but it is zero for any two disjoint boxes regardless of how far apart they are. A loss
or cost built only on IoU is flat across all non-overlapping configurations, so it gives no
gradient to pull a far-off prediction toward the target. Generalized IoU (Rezatofighi et al.
2019) fixes this by subtracting the fraction of the smallest enclosing box $C$ that the union
does not cover:

$$\mathrm{GIoU}(A, B) = \mathrm{IoU}(A, B) - \frac{|C \setminus (A \cup B)|}{|C|}
  = \mathrm{IoU} - \frac{|C| - |A \cup B|}{|C|}$$

where $C$ is the smallest axis-aligned box containing both $A$ and $B$. When the boxes
coincide, $|C| = |A \cup B|$ and GIoU = IoU = 1. As the boxes move apart, $|C|$ grows while
the union stays fixed, so the penalty grows and GIoU keeps decreasing toward its lower bound,
approaching $-1$ for boxes that are far apart relative to their size. GIoU is therefore in
$(-1, 1]$ and gives a usable gradient even when the boxes do not overlap, which is why both
the matching cost and the box loss use it.

Two implementation points matter. The intersection width and height must be clamped to be
non-negative before multiplying, or two disjoint boxes produce a negative "intersection area"
and a wrong score. And a small $\varepsilon$ goes in both denominators (the IoU union and
$|C|$) so a zero-area degenerate box does not divide by zero. The function is differentiable,
so the same one is reused in the loss.

### The set-prediction loss

After the matcher fixes the assignment, the loss has three parts, summed with the same DETR
weights (class 1, L1 5, GIoU 2).

Classification is a cross-entropy over all $N$ queries. Each matched query's target is its
ground-truth class; each unmatched query's target is the no-object class at index
`num_classes`. With $N$ slots and a handful of objects per image, most queries are no-object,
so the no-object term would dominate and the model would collapse to predicting no-object
everywhere. DETR downweights it with a class weight vector: a length-$(C{+}1)$ vector that is
$1$ for the real classes and a small `eos_coef` (DETR uses $0.1$) for the no-object class,
passed straight to a weighted cross-entropy. This is the standard class-imbalance
reweighting, not a post-hoc scaling of the unmatched queries' loss after the fact.

The box L1 and the GIoU loss ($1 - \mathrm{GIoU}$) are computed only on the matched (query,
ground-truth) pairs; unmatched queries have no box target. The class term is the one place the
cost and the loss differ: the cost uses the raw probability $-p[c]$, the loss uses
cross-entropy $-\log p[c]$. They carry the same information through a different function. The
L1 and GIoU terms are the identical function in both the cost and the loss.

### Object queries

DETR replaces anchors with object queries: $N$ learned embedding vectors, one per output slot,
that the decoder refines by self-attending among themselves and cross-attending to the image
features. A query is a learned slot, not a position on a grid. Over training, queries
specialize: in the original DETR, different queries learn to prefer different image regions
and object sizes, which is the learned analogue of the anchor grid. The decoder is a stack of
transformer blocks with bidirectional self-attention (the output set has no order) followed by
cross-attention to the patch tokens.

DAB-DETR (Liu et al. 2022, [arXiv:2201.12329](https://arxiv.org/abs/2201.12329)) later made
the query an explicit 4-D anchor box $(x, y, w, h)$ that each decoder layer updates, which both
speeds up convergence and makes the query interpretable as a box prior rather than an opaque
vector.

### Why NMS disappears

Because the matching is one-to-one, every ground-truth object is assigned to exactly one
query, and every other query is trained to predict no-object. The model is therefore trained
directly against producing two boxes for one object: if two queries both fire on the same
object, only one can be matched, and the other takes a no-object penalty. Duplicate
suppression moves from a post-process into the training objective, so no NMS is needed at
inference.

### DETR's convergence problem and the fixes

The original DETR worked but trained slowly: about 500 epochs on COCO, roughly ten times a
comparable Faster R-CNN schedule. The follow-up papers diagnosed why and fixed it, and that
lineage runs through most current detectors.

Deformable DETR (Zhu et al. 2020, [arXiv:2010.04159](https://arxiv.org/abs/2010.04159)) traced
much of the slowness to global dense cross-attention, where every query attends to every pixel
and the attention map has to learn from scratch where to look. It replaced that with
deformable attention: each query attends to a small set of sampled key locations whose offsets
are predicted, so attention is local and sparse from the start. This cut training to about 50
epochs and handled small objects better through multi-scale features.

DN-DETR (Li et al. 2022, [arXiv:2203.01305](https://arxiv.org/abs/2203.01305)) found a second
cause: the bipartite matching is unstable early in training, because which query matches which
object flips between epochs, so the targets a query chases keep changing. It added query
denoising: feed in ground-truth boxes with added noise as extra queries that bypass the
matcher and are trained to reconstruct the clean boxes, giving a stable auxiliary target that
does not depend on the (still noisy) matching.

DINO (Zhang et al. 2022, [arXiv:2203.03605](https://arxiv.org/abs/2203.03605)) combined
DAB-DETR's anchor-box queries and DN-DETR's denoising into one model, added contrastive
denoising (positive and negative noised boxes so the model learns to reject near-duplicates)
and mixed query selection, and became the first DETR-family detector to top the COCO
leaderboard. DAB-DETR and DN-DETR are parallel early-2022 contributions that DINO unifies; the
name DINO bundles denoising and anchor boxes, so read this as a teaching arc, not a strict
ancestry where one descends from the other.

RT-DETR (Zhao et al. 2023, [arXiv:2304.08069](https://arxiv.org/abs/2304.08069)) answered the
last standing objection, that DETR is too slow for real-time use, with an efficient hybrid
encoder and a real-time decoder that beats the YOLO models at comparable latency while keeping
the NMS-free, anchor-free design. Anchor-based YOLO remains the resource-constrained
competitor; RT-DETR removed the "you must use YOLO for speed" argument.

```mermaid
flowchart TD
    DETR["DETR 2020: set prediction, bipartite matching, no NMS"]
    DEF["Deformable DETR 2020: sparse sampled attention, ~50 epochs"]
    DAB["DAB-DETR 2022: anchor-box queries"]
    DN["DN-DETR 2022: query denoising, stable matching"]
    DINO["DINO 2022: anchor boxes + denoising + contrastive DN"]
    RT["RT-DETR 2023: real-time, beats YOLO"]
    DETR --> DEF
    DEF --> DINO
    DAB --> DINO
    DN --> DINO
    DINO --> RT
```

### Segmentation and open-vocabulary detection on the same paradigm

The query-plus-matching idea generalizes past boxes. Mask2Former (Cheng et al. 2021,
[arXiv:2112.01527](https://arxiv.org/abs/2112.01527)) keeps the object queries and the
bipartite matching but has each query predict a binary mask instead of a box, and adds masked
attention where each query attends only inside its current mask prediction. One architecture
then handles instance, semantic, and panoptic segmentation, which previously needed separate
designs.

Grounding DINO (Liu et al. 2023, [arXiv:2303.05499](https://arxiv.org/abs/2303.05499)) adds a
text backbone so the detector is open-vocabulary: a noun phrase produces detections of the
matching objects, instead of a fixed class list. The language features guide the queries, so
the model detects rare or novel categories without retraining a closed classifier.

The Segment Anything line is a separate family. SAM (Kirillov et al. 2023,
[arXiv:2304.02643](https://arxiv.org/abs/2304.02643)) is a promptable segmentation model: a
point, box, or rough mask prompt produces an object mask, trained on a very large mask
dataset. SAM 2 (Ravi et al. 2024, [arXiv:2408.00714](https://arxiv.org/abs/2408.00714))
extends this to video with a streaming memory that tracks an object across frames. SAM 3 (Meta,
released Nov 2025) and SAM 3.1 (Mar 2026) reportedly add concept prompts (a noun phrase or an
exemplar image patch rather than a single click) and multi-object tracking; no arXiv preprint
is available for these, so treat those capability claims as reported by the release, not
established here.

### Where this goes

The query-plus-bipartite-matching paradigm is now the dominant class for detection,
segmentation, tracking, and open-vocabulary grounding; anchor-based YOLO is the
resource-constrained competitor that still wins on the tightest latency budgets. In an
autonomous-driving perception stack, RT-DETR or a DINO variant runs as the production
detector, Grounding DINO covers rare or long-tail classes the closed detector was never
trained on, and SAM 2 or 3 sits in the annotation loop to cut labeling cost. The matcher and
loss built here are the core of all of these.

## The assignment

Implement generalized IoU, the Hungarian matching cost, and the set-prediction loss. The ViT
backbone, the query decoder, and the output heads are provided, since the lesson is the
cost-versus-loss separation and the one-to-one matching, not the network. Each file's
docstrings give the exact signatures, shapes, and the index conventions (boxes are normalized
cxcywh in $[0, 1]$, the model emits $N = 10$ query slots over $C+1$ classes with the last as
no-object). Read those in the files; this section maps each file to the concept above.

### Files to modify

`boxes.py` holds the box geometry. Write `box_xyxy_to_cxcywh` (the format inverse, the center
is the corner midpoint and the width/height the corner differences) and `generalized_iou`
(the GIoU score from the generalized-IoU section, with the clamped intersection and the
$\varepsilon$ in both denominators). `box_cxcywh_to_xyxy` is provided.

`matcher.py` is the cost side. Write `HungarianMatcher.forward`, which builds the per-image
cost matrix from the three terms in the matching-cost section and solves the one-to-one
assignment with `scipy.optimize.linear_sum_assignment` on the detached numpy matrix. The whole
call runs under `torch.no_grad()` and returns index tensors, so no gradient flows through it.

`loss.py` is the loss side. Write `detr_loss`, the differentiable objective from the
set-prediction-loss section: the weighted cross-entropy with the no-object downweight over all
queries, plus L1 and $1 - \mathrm{GIoU}$ on the matched pairs only.

`model.py` (`DETR`, the ViT backbone, query decoder, and heads), `config.py`, `viz.py`, and
the colored-squares toy (`nanovision.data.toy.detection_batch`) are provided.

### Running and validating

Activate the environment (`conda activate nanovision`), then:

```
make test     A=a11_detection_segmentation   # run the tests against the top-level files (the holes)
make verify   A=a11_detection_segmentation   # run the same tests against the reference solution/
make viz      A=a11_detection_segmentation   # render the figures from the reference solution
make viz-mine A=a11_detection_segmentation   # render the figures from your own code (holes filled)
```

`make test` is the command to run while working. It runs the suite in
`assignments/a11_detection_segmentation/tests/` against the top-level files (the ones with the
holes), and goes from red (the holes raise `NotImplementedError`) to green as the holes are
filled. `make verify` runs the identical suite against the reference `solution/` by setting
`NANOVISION_IMPL=solution`, so it is green from the start and shows the target. The goal is to
bring `make test` to the same green as `make verify`. (`test_forbidden_imports` passes with the
holes in place; the overfit test also surfaces the ViT backbone's own hole until the ViT
assignment is filled.)

The suite checks GIoU on an identical pair (1), a far-apart pair (in $(-1, 0)$), a
hand-computed overlapping value, a float64 gradcheck, and the cxcywh-xyxy round-trip; the
matcher on a case where per-column argmin double-assigns one query but Hungarian picks distinct
queries, plus shapes, no-grad indices, and the empty-gt case; the loss for a scalar with grad,
a gradcheck at fixed indices, and that raising `eos_coef` raises the classification loss while
the box terms are unchanged; and an overfit on the 4-image toy.

`make viz` renders the overfit visualization (matched predictions vs ground truth) from the
reference solution and writes PNGs to `out/` with matplotlib's headless Agg backend, so it
works over SSH, in WSL, and in CI with no display. `make viz-mine` renders the same from the
top-level code (needs the holes filled, since it trains a model). Add `SHOW=1` to also open
interactive windows.

What you should see when you run this. The overfit test runs 500 Adam steps on a 4-image toy,
about 15 seconds on CPU; the total loss falls from roughly 4.8 to a floor near 0.1-0.25, and
the matched predicted box centers land on the ground-truth centers to well under 0.06 of the
image side. The remaining floor is dominated by the GIoU term: the ViT's 4-pixel patch stride
bounds how finely a box can be localized at 32x32, so a small GIoU residual does not vanish.
The toy omits two real DETR pieces, per-decoder-layer re-injection of a 2D positional encoding
into the cross-attention keys and per-layer auxiliary losses, which also keeps the floor up.

These are toy artifacts on 32x32 images, and they confirm only that the matcher, the loss, and
the model run end to end and overfit a tiny fixed set. They say nothing about convergence speed
(the 500-epoch DETR-is-slow story is about COCO, not four overfit images), nothing about
NMS-free behavior under real object density (one or two well-separated squares never stress
duplicate suppression the way a crowded scene does), and nothing about average precision. Those
belong to the papers above, not to a 32x32 toy.

## References

- Carion et al. 2020, DETR, [arXiv:2005.12872](https://arxiv.org/abs/2005.12872).
- Rezatofighi et al. 2019, generalized IoU, [arXiv:1902.09630](https://arxiv.org/abs/1902.09630).
- Zhu et al. 2020, Deformable DETR, [arXiv:2010.04159](https://arxiv.org/abs/2010.04159).
- Liu et al. 2022, DAB-DETR, [arXiv:2201.12329](https://arxiv.org/abs/2201.12329).
- Li et al. 2022, DN-DETR, [arXiv:2203.01305](https://arxiv.org/abs/2203.01305).
- Zhang et al. 2022, DINO, [arXiv:2203.03605](https://arxiv.org/abs/2203.03605).
- Zhao et al. 2023, RT-DETR, [arXiv:2304.08069](https://arxiv.org/abs/2304.08069).
- Cheng et al. 2021, Mask2Former, [arXiv:2112.01527](https://arxiv.org/abs/2112.01527).
- Liu et al. 2023, Grounding DINO, [arXiv:2303.05499](https://arxiv.org/abs/2303.05499).
- Kirillov et al. 2023, SAM, [arXiv:2304.02643](https://arxiv.org/abs/2304.02643).
- Ravi et al. 2024, SAM 2, [arXiv:2408.00714](https://arxiv.org/abs/2408.00714).
- Kuhn 1955, "The Hungarian Method for the Assignment Problem".
