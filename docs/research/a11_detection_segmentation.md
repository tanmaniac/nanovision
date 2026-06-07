# A11 — Modern detection and segmentation (Mixed: build + survey)

*Researched June 2026. Sources verified against original papers and current repos.*

---

## 1. Key concepts a student must learn

### Set prediction and why it replaces anchors/NMS

Classical detectors (Faster R-CNN, SSD, YOLO up through v7/v8) produce redundant candidate boxes from a dense grid of anchors, then post-process with NMS to suppress duplicates. The hyperparameters that govern anchor design, IoU thresholds, and NMS cutoffs are dataset-specific and not part of the learned objective.

DETR (Carion et al., ECCV 2020) reframes detection as a set prediction problem: given an image, produce exactly the right unordered set of (class, box) pairs. The model is asked to predict at most N objects in one forward pass; duplicate suppression emerges from the training objective, not from a post-hoc filter.

The student should understand why the set formulation requires a permutation-invariant loss: ground-truth boxes have no canonical order, so a fixed-assignment MSE or cross-entropy would penalize a correct prediction placed in the "wrong" slot.

### Bipartite matching and the Hungarian algorithm

The training loss must first decide which predicted slot to compare against which ground-truth box. DETR solves this with the Hungarian algorithm (Kuhn-Munkres), which finds the minimum-cost perfect matching in a bipartite graph in O(N^3).

Two distinct quantities must not be confused:

**Matching cost** — used only to find the assignment. It does not need to be differentiable with respect to model parameters, because the indices it produces are treated as constants during the backward pass. The cost combines:
- negative class log-probability for the ground-truth class: `-p_hat[c_i]`
- L1 distance between predicted and ground-truth box coordinates (normalized center-x, center-y, width, height)
- negative generalized IoU (GIoU), which is defined even when boxes do not overlap: `GIoU = IoU - |C \ (A ∪ B)| / |C|`, where C is the smallest enclosing box

**Training loss** — computed after the assignment is fixed. It uses the same three terms but now with appropriate scaling weights and, for the class term, standard cross-entropy rather than raw log-probability.

The practical implementation uses `scipy.optimize.linear_sum_assignment` on the CPU, detached from the autograd graph. The assignment indices are then used to permute the prediction tensor, and the actual loss is computed in PyTorch with full gradient tracking.

Unmatched query slots are assigned a "no-object" class (background) and contribute only to the classification loss with a downweighted coefficient (the paper uses 1/10).

### Object queries

The decoder receives N learned embeddings called object queries. These are not spatial queries derived from the image; they are trained, query-specific positional slots that specialize during training. After convergence, different queries tend to specialize spatially (some favor top-left corners, others center regions) and by object size. The queries serve as the "what to look for" prompt for each decoder cross-attention step.

DAB-DETR (Liu et al., ICLR 2022) made this more explicit by parameterizing queries as (x, y, w, h) anchor boxes that are updated layer-by-layer, which improved interpretability and convergence speed. DINO-DETR further builds on this.

### Why NMS disappears

The one-to-one Hungarian matching enforces that each ground-truth object can be assigned to at most one query slot. Any query that produces a box similar to an already-matched box will be assigned "no-object" in training, so the model is directly trained against producing duplicates. No suppression step is needed at inference.

### DETR's convergence problem and how it was fixed

DETR requires 500 epochs on COCO to converge, roughly 10-20x slower than Faster R-CNN. The root cause is not the transformer itself but the global attention in the encoder. Early in training, all queries attend broadly over the full feature map; learning to localize attention to specific regions takes many gradient steps. The effect is compounded by the instability of bipartite matching in early training: the optimal assignment can flip between iterations, giving contradictory gradient signals for the same query.

Three main lines of fix:

**Deformable attention (Deformable DETR, Zhu et al., ICLR 2021).** Instead of attending to all spatial positions, each attention head samples a small number of key points (4 by default) around a learnable reference point. This constrains attention to local neighborhoods, providing an inductive bias similar to convolution, and reduces the complexity of encoder attention from O(HW)^2 to O(HW * K). Multi-scale deformable attention additionally aggregates features across FPN-like levels, fixing the small-object weakness. Result: matches or beats DETR at 1/10th the training epochs.

**Query denoising (DN-DETR, Li et al., CVPR 2022).** A second set of "noisy" queries is constructed by adding random perturbations to ground-truth box coordinates. These denoising queries are fed through the decoder in parallel with the standard object queries (with an attention mask that prevents the two groups from interacting). The decoder must reconstruct the original boxes from the noisy ones, providing a direct and stable supervision signal that bypasses the matching instability. This auxiliary task is removed at inference.

**Contrastive denoising + mixed query selection (DINO-DETR, Zhang et al., ICLR 2023).** DINO-DETR adds contrastive denoising: for each ground-truth box, it creates both positive noisy queries (perturbed but inside the box) and negative noisy queries (perturbed to be far from the box). The positive ones must predict the correct class and box; the negative ones must predict no-object. This gives the decoder a harder, more informative training signal. Mixed query selection initializes part of the object queries from top-scored encoder output regions rather than from fixed learned embeddings. Look-forward-twice uses gradient information from the next layer's prediction to refine the current layer's box estimate. Together these push DINO to 49.4 AP at 12 epochs (vs. DETR's ~45 AP at 500 epochs).

### The DETR family lineage (for the survey)

```
DETR (ECCV 2020)
  └─ Deformable DETR (ICLR 2021)  — deformable attention, multi-scale, 10x faster
       └─ DAB-DETR (ICLR 2022)    — anchor-box queries, per-layer refinement
            └─ DN-DETR (CVPR 2022) — query denoising
                 └─ DINO-DETR (ICLR 2023) — contrastive denoising, mixed init, look-forward-twice
                      └─ Co-DETR (ICCV 2023) — collaborative aux heads for encoder supervision
```

RT-DETR (Zhao et al., CVPR 2024, Baidu) is a parallel branch: it uses an efficient hybrid CNN + transformer encoder (not the full Deformable encoder stack), drops the iterative refinement decoder in favor of a simpler IoU-based query selection, and achieves competitive accuracy at real-time speeds, outperforming YOLOv8 at comparable sizes. RT-DETRv2 (2024) improved it further with a bag of training tricks.

### Promptable segmentation: SAM and SAM 2

Segment Anything Model (SAM, Kirillov et al., ICCV 2023) defines a new task: given a prompt (a point, a box, or a mask), segment the indicated object. The architecture has three components: a heavy ViT image encoder (run once), a lightweight prompt encoder, and a small two-way transformer mask decoder. The SA-1B dataset contains 1 billion masks on 11 million images, generated by a data engine where SAM iteratively annotated its own training data.

SAM 2 (Ravi et al., 2024) extends the model to video: it adds a streaming memory module (a per-frame memory bank and a cross-attention lookup) that allows the model to track a prompted object across frames in real time (~44 FPS). SAM 2 is a unified image and video segmenter, and its technical report was accepted to ICLR 2025.

SAM 3 (Meta, released November 2025, accepted ICLR 2026) extends further to open-vocabulary promptable segmentation: prompts can be short noun phrases or image exemplars, not just spatial indicators. SAM 3.1 (March 2026) introduces shared-memory multi-object tracking. The progression from SAM -> SAM 2 -> SAM 3 is the current frontier for promptable segmentation.

### Mask2Former and universal segmentation

Mask2Former (Cheng et al., CVPR 2022) is a single architecture that handles panoptic, instance, and semantic segmentation by framing all three tasks as masked-attention transformer decoding over learned queries. The key innovation is masked cross-attention: during the decoder forward pass, the cross-attention of each query is restricted to within its current predicted mask region, forcing the decoder to refine a local region rather than attending globally. This replaces three separate specialized heads (one per task) with one query-based decoder. It set state of the art on all three COCO segmentation tasks at the time of publication.

Mask2Former is architecturally adjacent to DETR-lineage detection: it uses the same "N queries -> N predictions" paradigm with bipartite matching, except predictions are (class, binary mask) pairs instead of (class, box) pairs.

---

## 2. Mechanisms to implement from scratch

### Build task 1: Hungarian matcher on toy boxes

**What to implement.** A `HungarianMatcher` class that takes a batch of predicted (logits, boxes) and a batch of ground-truth (labels, boxes) and returns, per image, the row/column index pairs from the optimal bipartite matching.

The cost matrix for one image (N predictions, M ground-truth boxes) is:
```
C[i, j] = alpha * cls_cost[i, j] + beta * l1_cost[i, j] + gamma * giou_cost[i, j]
```
where:
- `cls_cost[i, j] = -softmax(logits)[i, c_j]` (negative probability of ground-truth class)
- `l1_cost[i, j] = ||box_i - box_j||_1` (boxes in normalized cxcywh format)
- `giou_cost[i, j] = -GIoU(box_i, box_j)`

After computing the cost matrix (in numpy, detached), call `scipy.optimize.linear_sum_assignment(C)` to get `(row_indices, col_indices)`. These index the predictions and ground-truths that are paired.

**Verifiable toy problem.** Create a batch of 2 images, each with 2 ground-truth boxes and N=5 queries. Manually set the logits and boxes so that the optimal assignment is deterministically known. Assert that the returned indices are exactly correct. Also run `torch.autograd.gradcheck` on the training loss (not the matcher): perturb the prediction tensors with `requires_grad=True`, compute the post-assignment loss, and verify finite differences match autograd.

**Shape tests to write:**
- `cost_matrix.shape == (N_pred, N_gt)` for each image
- `row_indices` and `col_indices` are both of length `min(N_pred, N_gt)`
- The total loss is a scalar with `.requires_grad == True`

**GIoU implementation check.** Implement GIoU from scratch (convert cxcywh to x1y1x2y2, compute intersection area, union area, enclosing box area). Test against a known case: two identical boxes should give GIoU = 1; two non-overlapping boxes with a large gap should give a value below 0; `giou(A, A) == 1.0`.

### Build task 2: minimal query decoder — overfit one batch

**What to implement.** A minimal DETR-style decoder that overfits a tiny detection dataset. Reuse the cross-attention module from A1.

Architecture (keep it small enough for one forward pass on a laptop CPU):
- Backbone stub: a 3-layer CNN that maps a 128x128 image to a 16x16x256 feature map
- Positional encoding: 2D sinusoidal added to the feature map, flattened to 256 x 256
- Encoder: 1 transformer encoder layer (256-dim, 4 heads)
- Object queries: N=10 learned embeddings of dim 256
- Decoder: 1 transformer decoder layer (self-attention over queries, then cross-attention between queries and encoder output)
- Prediction heads: 2 linear layers, one for class (C+1 outputs) and one for box (4 outputs, sigmoid)
- Hungarian matcher + set prediction loss

**Toy dataset:** 4 images, each 128x128 pixels, each containing 1 or 2 colored squares at known locations. Ground-truth boxes are exact. No data augmentation.

**Verifiable task.** Train for up to 200 epochs on the 4-image batch. The loss should reach near-zero and the predicted boxes should overlay the ground-truth squares within a small tolerance (e.g., center error < 2 pixels). Log the matching cost and the training loss separately to confirm they decrease together.

**Failure-mode to understand.** If all queries collapse to predicting background, the issue is the no-object weight being too large or the learning rate being too small. The student should be able to diagnose this from the class loss component alone.

---

## 3. Assessment of the draft scope

### What is right

The build core — Hungarian matcher + minimal query decoder on a toy detection task — is exactly the right depth. Implementing these from scratch forces understanding of the matching/loss distinction and the query mechanism. Reusing A1 cross-attention is a good explicit dependency.

The survey arc (DETR -> Deformable DETR -> DINO-DETR) covers the right milestone papers.

SAM/SAM 2 as a promptable segmentation survey is appropriate and useful.

### What is missing or should be added

**1. The distinction between matching cost and training loss must be an explicit teaching point, not just implicit in the code.** Many learners conflate the two. The matching cost is a fixed function used to index into predictions; the training loss is the differentiable objective. Separating these clearly prevents a common misconception that the Hungarian assignment itself needs to be differentiable.

**2. DN-DETR's denoising query trick should be given brief mention alongside the DETR -> Deformable -> DINO arc.** DN-DETR (CVPR 2022) introduced the denoising idea that DINO-DETR builds on; presenting DINO without DN is like explaining a design choice without the motivation. The arc is more accurately: DETR -> Deformable DETR -> DAB-DETR -> DN-DETR -> DINO-DETR.

**3. RT-DETR should be added to the survey.** As of 2024-2026, RT-DETR is the answer to "how do I do DETR-quality detection at real-time speeds?" It has been adopted by Ultralytics and is deployed in production. The learner coming from a YOLOv5-era background needs to understand that DETR-lineage can now match YOLO speeds - this removes a key objection to the paradigm shift.

**4. Mask2Former should be added as a brief survey item.** The course already covers segmentation as a topic (per the module title); Mask2Former shows how the same query + bipartite matching idea extends from detection (box outputs) to segmentation (mask outputs) in one unified architecture. It is the dominant approach for panoptic segmentation and is architecturally adjacent to what the student just built. A two-paragraph read-only treatment is sufficient.

**5. GroundingDINO should be mentioned as the next step after DINO-DETR.** GroundingDINO (Liu et al., ECCV 2024) extends DINO-DETR with a text backbone, grounded pre-training, and language-guided query selection, enabling zero-shot and open-vocabulary detection. It is widely used as a practical tool (paired with SAM for text-prompted segmentation) and represents the natural extension of DINO for an AV/robotics engineer who needs to detect novel categories without retraining. A brief description and a "run it" task is sufficient.

**6. SAM 3 should replace the "SAM/SAM 2" framing.** SAM 3 was released November 2025 and accepted to ICLR 2026; the course is taught in 2026. SAM 3 adds concept-level prompting (text noun phrases, image exemplars) that goes beyond SAM 2's spatial prompting. The survey should describe SAM -> SAM 2 -> SAM 3 as a progression, noting what each version added.

**7. OWLv2 can be mentioned briefly but is less critical than GroundingDINO.** OWLv2 (Minderer et al., NeurIPS 2023) scales open-vocabulary detection via self-training on web data, but GroundingDINO is more commonly encountered in practice and more architecturally connected to the DINO lineage being taught.

**8. YOLO-World is worth one sentence.** YOLO-World (Cheng et al., CVPR 2024) extends the YOLO family with open-vocabulary capability via RepVL-PAN. It is an alternative paradigm (anchor-based + language) vs. the query-based paradigm. Mentioning it helps the learner understand the full design space.

### What is correctly omitted or can be kept out

- Anchor-based transformers (Anchor DETR, SAM-DETR) are intermediate research and need not be taught.
- Instance segmentation heads on top of Faster R-CNN (Mask R-CNN) can remain as background knowledge from the 2020 baseline.
- Detailed panoptic/universal segmentation implementation from scratch is too large for one module; a read-and-run of Mask2Former suffices.

### Ordering suggestion

1. Set prediction + bipartite matching theory (with explicit cost vs. loss distinction)
2. Build: HungarianMatcher from scratch (verifiable)
3. Object query mechanism — what queries are, how they specialize
4. Build: minimal query decoder, overfit one batch
5. DETR convergence problem — why it is slow, what the root causes are
6. Survey arc: Deformable DETR -> DAB-DETR -> DN-DETR -> DINO-DETR (each in one paragraph)
7. Survey: RT-DETR as practical real-time answer (brief)
8. Survey: Mask2Former — same query idea, mask outputs instead of box outputs
9. Survey: GroundingDINO — open vocabulary via language grounding
10. Survey: SAM -> SAM 2 -> SAM 3 — promptable segmentation progression

---

## 4. Dependencies and connections

**Depends on A1 (Transformer from scratch).** The cross-attention module built in A1 is directly reused in the query decoder. The student must understand encoder-decoder cross-attention to understand why queries attend to specific image regions.

**Depends on A2 (likely positional encoding / ViT features).** 2D sinusoidal positional encoding added to the CNN feature map before the encoder is a direct extension of the 1D encoding from A1/A2. Deformable DETR's per-level positional encoding is a further extension.

**Bridges classical perception to modern.** The learner arrives knowing anchor-based detectors, NMS, and FPN-style multi-scale features. This module shows how each of those components is replaced: anchors by object queries, NMS by matching, FPN by multi-scale deformable attention. The connection should be made explicit: Deformable DETR's deformable attention is, in spirit, a learned version of the deformable convolution + FPN pipeline the learner already knows.

**Relevant to AV module.** In autonomous driving, detection must handle a large number of object categories at varying scales and distances in real time. DETR-lineage models (especially RT-DETR and DINO-DETR) are now competitive with YOLO in production AV pipelines. GroundingDINO enables novel-category detection without retraining, which is important for rare obstacle classes. SAM 2 / SAM 3 are used in annotation pipelines for video data.

---

## 5. Must-read sources

1. **Carion et al., "End-to-End Object Detection with Transformers" (ECCV 2020).** The original DETR. Read sections 2-3 carefully (set prediction loss, bipartite matching, architecture). This is the primary implementation target.

2. **Zhu et al., "Deformable DETR: Deformable Transformers for End-to-End Object Detection" (ICLR 2021, Oral).** The first serious fix to DETR's convergence. Section 3 (deformable attention module) is essential; understand how the sampling offset network works.

3. **Zhang et al., "DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection" (ICLR 2023).** The current strong baseline for DETR-lineage detection. Read alongside DN-DETR (Li et al., CVPR 2022) to understand where the denoising idea came from.

4. **Zhao et al., "DETRs Beat YOLOs on Real-time Object Detection" (CVPR 2024).** RT-DETR. One-page architecture section covers the hybrid encoder; shows that DETR-lineage is now practical for real-time deployment. Completes the "why DETR matters for production" argument.

5. **Cheng et al., "Masked-attention Mask Transformer for Universal Image Segmentation" (CVPR 2022).** Mask2Former. Shows how the query + matching paradigm extends to all three segmentation tasks. The masked cross-attention insight is clean and worth understanding.

6. **Kirillov et al., "Segment Anything" (ICCV 2023).** SAM 1. Read Section 3 (task formulation, promptable segmentation, data engine). The SA-1B dataset scale and the model-in-the-loop annotation approach are the key contributions beyond the model architecture itself.

7. **Ravi et al., "SAM 2: Segment Anything in Images and Videos" (ICLR 2025).** Extension to video via streaming memory. Approximately half-paper read (sections on memory mechanism and the SA-V dataset).

**Flagged omissions in the original draft scope:**

- DN-DETR (Li et al., CVPR 2022) — not listed but needed as a stepping stone to DINO.
- DAB-DETR (Liu et al., ICLR 2022) — not listed but needed to understand anchor-box queries in DINO.
- Mask2Former (Cheng et al., CVPR 2022) — not listed; add as survey item.
- RT-DETR (Zhao et al., CVPR 2024) — not listed; add as survey item.
- GroundingDINO (Liu et al., ECCV 2024) — not listed; add as brief survey + run-it task.
- SAM 3 (Meta, ICLR 2026) — not listed; update the SAM survey arc to include it.

---

## 6. 2024-2026 developments that change how this should be taught

**RT-DETR changes the practical framing.** Before 2024, a legitimate objection to teaching DETR-lineage was "it's too slow for production." RT-DETR (Zhao et al., CVPR 2024, Baidu) settled this: at comparable parameter counts, RT-DETR-L achieves 53.0 AP at 114 FPS on a T4 GPU, outperforming YOLOv8-L (52.9 AP at 87 FPS). The module should use this as a motivating conclusion rather than leaving the learner with the impression that DETR is a research curiosity.

**GroundingDINO (ECCV 2024) and YOLO-World (CVPR 2024) establish open-vocabulary detection as the new standard expectation.** The question for a practitioner is no longer "can I detect the 80 COCO classes?" but "can I detect arbitrary classes from a text prompt without retraining?" GroundingDINO achieves 52.5 AP zero-shot on COCO (without seeing any COCO training images) by fusing DINO-DETR with a BERT text backbone. YOLO-World does the same from the YOLO side. The survey portion should frame these as the current production state and explain the architectural difference (query-based vs. anchor-based + language conditioning).

**SAM 3 (ICLR 2026) extends promptable segmentation to concepts.** The course as drafted describes SAM and SAM 2 as the endpoint. SAM 3 (November 2025, presented at ICLR 2026) accepts noun-phrase and exemplar prompts, enabling segmentation of any named or shown concept across images and video. SAM 3.1 (March 2026) adds efficient multi-object tracking. The survey arc needs to be extended through SAM 3.

**DINO-DETR remains the right "benchmark" detector for the survey.** As of mid-2026, Co-DETR (ICCV 2023) with ViT-L holds the highest COCO AP (66.0 on test-dev), but the architecture and training recipe are complex. DINO-DETR with R50 at 12 or 24 epochs is still the reference implementation for teaching because the code is clean, the ablations are well-documented, and the IDEA-Research GitHub repo is actively maintained.

**Mask2Former has been surpassed for specialized tasks** (by methods like ODISE for open-vocabulary panoptic segmentation), but for a fixed-vocabulary panoptic/instance task it remains the dominant architecture and the most pedagogically clean example of the "queries for segmentation" idea. It should stay in the survey.

**The teaching framing for 2026.** Rather than "DETR is how we do detection now," the more accurate framing is: "The query + bipartite matching paradigm is now the dominant architecture class for detection and segmentation, with anchor-based YOLO as the main competitor at the resource-constrained end. Understanding Hungarian matching and object queries is a prerequisite for reading the current literature across detection, segmentation, tracking, and open-vocabulary grounding." This motivates both the build task (implement the core mechanism) and the survey (trace how it evolved and spread).
