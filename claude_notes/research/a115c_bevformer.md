# A11.5c — BEVFormer-style attention (camera-to-BEV via query pull): scope validation

*Researched June 2026. Sources verified against original papers.*

---

## 1. Key concepts a student must learn

### BEV queries as the representational substrate

BEVFormer (Li et al., ECCV 2022) maintains a grid of learnable query vectors Q of shape H_bev x W_bev x C, one per BEV cell. On nuScenes the canonical grid is 200x200 at 0.512 m/cell, covering [-51.2, 51.2] m in both X and Y. Each query represents "what is at this ground-plane location." The queries are persistent across the encoder layers - they are the BEV feature map being built, not fixed inputs.

This is the defining choice that separates the query-pull family from the depth-lift family: in LSS (A11.5b), image pixels are pushed outward into 3D via predicted depth; in BEVFormer, BEV cells reach back into image space and pull the features they need.

### Geometry as an attention prior: reference points

For BEV query at grid position p = (x, y), BEVFormer does not ask the query to attend uniformly to all image pixels. Instead, it computes a set of 3D reference points by forming a vertical pillar at (x', y') in world coordinates with N_ref = 4 anchor heights uniformly spaced from -5 m to 3 m. Each 3D point (x', y', z_j) is then projected into every camera using the known intrinsic/extrinsic matrices:

    u, v = K @ [R | t] @ [x', y', z_j, 1]^T

If the projected pixel (u, v) falls within a camera's image bounds, that camera is a "hit view" for this query. The projection is computed offline per frame (geometry changes with ego motion) and serves as the reference location around which deformable attention will sample.

This is the geometry-as-attention-prior idea: the model does not need to learn where to look from scratch. Camera calibration tells it the plausible region; deformable attention adds a small learned offset to refine the sampling location.

### Spatial cross-attention (SCA)

    SCA(Q_p, F_t) = (1 / |V_hit|) * sum_{i in V_hit} sum_{j=1}^{N_ref} DeformAttn(Q_p, P(p, i, j), F_t^i)

where:
- V_hit is the set of cameras that "see" BEV position p
- P(p, i, j) is the 2D projection of the j-th reference point onto camera i
- F_t^i is the feature map of camera i at time t
- DeformAttn samples a small number (typically 4 per head) of points around P(p,i,j) using learned offsets

The result is averaged over hit views. Each query aggregates information only from the parts of image space geometrically consistent with its BEV location.

### Deformable attention

Full deformable attention (Zhu et al., Deformable DETR, ICLR 2021) predicts K offsets delta_p_k and attention weights w_k from the query, then computes:

    DeformAttn(q, p, x) = sum_{k=1}^{K} w_k * x(p + delta_p_k)

where x(.) is bilinear interpolation into the feature map. The offsets are predicted by a small linear layer applied to the query. This has two properties crucial for teaching: (a) it is differentiable end-to-end via the bilinear interpolation gradients, and (b) the network can move the sampling locations beyond the reference point to find the most informative features.

For pedagogy, there is a strong case for teaching a simplified version first: bilinear-sample-at-reference-points with no learned offsets. This isolates the core projection geometry (the hard and important part) from the offset-prediction machinery (a refinement that adds engineering complexity). A student who has implemented clean bilinear sampling at projected 3D pillars understands 90% of BEVFormer's spatial mechanism. Learned offsets can be added as a follow-on exercise.

### Temporal self-attention (TSA)

Between consecutive frames, BEVFormer preserves the BEV feature from the previous step, B_{t-1}. Before use it is warped by ego motion to align with the current BEV grid:

    B'_{t-1}[x, y] = B_{t-1}[R_ego * [x,y,0] + t_ego]

Then TSA runs deformable attention over both the current query Q and the aligned history B'_{t-1}:

    TSA(Q_p, {Q, B'_{t-1}}) = sum_{V in {Q, B'_{t-1}}} DeformAttn(Q_p, p, V)

The offsets for TSA are predicted from the concatenation of Q_p and B'_{t-1}(p), so they encode motion: if an object moved between frames the offsets can track it across the aligned grid. The recurrent structure means that at inference time only one frame of BEV history is carried, making memory cost constant regardless of sequence length.

### The LSS-vs-BEVFormer conceptual contrast

This is the right spine for the topic. The contrast is clean and illuminating:

- LSS (depth-lift, forward projection): estimates a per-pixel depth distribution, voxelizes features into 3D space, collapses to BEV. Needs explicit depth supervision or self-supervised depth signals. Computation scales with the BEV grid density. Error in depth propagates directly into BEV localization.
- BEVFormer (query-pull, backward projection): starts in BEV, uses calibrated camera geometry to find where to sample in image space, learns to pull relevant features there. No depth estimate needed. Temporal state is maintained as a BEV feature buffer.

Neither is strictly superior. LSS-family methods (BEVDepth, BEVDet, etc.) benefit from explicit depth supervision and transfer well to occupancy prediction. BEVFormer-family methods benefit from temporal state accumulation and handle cameras with imperfect depth signals more gracefully. By 2024-2025, the dominant trend is toward sparse query methods that inherit BEVFormer's query-pull geometry but abandon the dense BEV grid entirely.

---

## 2. Mechanisms to implement from scratch

### Core implementation: SCA with bilinear sample at reference points

**Recommended approach: implement the simplified (no learned offsets) version first.**

True deformable attention requires a small offset-prediction MLP and careful weight initialization to prevent all offsets from collapsing to the same point. It works well in the full system but is a source of silent failures in a from-scratch implementation. The simplified version - bilinear sample at exactly the projected reference points, then average - preserves all the geometric reasoning and is straightforwardly differentiable.

**Minimal implementation task:**

Given:
- N_cam = 6 cameras with known K, R, t
- Multi-scale image features F: shape (N_cam, C, H_feat, W_feat)
- BEV queries Q: shape (H_bev, W_bev, C), e.g., 50x50
- BEV reference points R3D: shape (H_bev, W_bev, N_ref, 3) with N_ref = 4 heights

Step 1: project R3D into each camera using batch matrix multiply:
    uvw = K @ (R @ R3D + t)  ->  uv = uvw[:2] / uvw[2]  ->  shape (N_cam, H_bev, W_bev, N_ref, 2)

Step 2: normalize uv to [-1, 1] for `F.grid_sample`.

Step 3: for each camera, mask points that project outside image bounds (the hit/miss logic).

Step 4: call `F.grid_sample(F_i, uv_i, mode='bilinear', align_corners=False)` to get sampled features shape (H_bev, W_bev, N_ref, C).

Step 5: average over N_ref and over hit cameras to get updated BEV features.

Verifiable tasks:
- Shape test: output is (H_bev, W_bev, C) and does not change shape under valid input permutations.
- `torch.autograd.gradcheck` on a single BEV cell: gradients flow through grid_sample back to both the image features and (if queries are used in offset prediction) the queries.
- Overfit one batch: a tiny 4x4 BEV grid with synthetic camera images and known ground-truth BEV labels should reach near-zero loss within 100 steps.

### Follow-on: add learned offsets

After the bilinear version works, add an offset head: a linear layer that takes Q_p as input and outputs K * N_ref * 2 offsets (K = 4 sampling points per reference point). These are added to the reference point projections before grid_sample. Verify with gradcheck that offsets receive gradients.

### Temporal self-attention

**Minimal task:**

Store B_{t-1} from the previous forward pass (detach from computation graph). Warp it using known ego-motion delta (a 2D translation + rotation on the BEV grid, implemented as an affine grid_sample). Concatenate current Q with aligned B'_{t-1} along the channel dimension and run a single-head cross-attention where Q queries against both. Verify:
- On a two-frame synthetic sequence where an object moves 1 cell between frames, the TSA output has lower loss than the no-temporal baseline.
- Gradients flow to the image features of both frames.

### Toy BEV task

BEV semantic segmentation on nuScenes v1.0-mini: 5 classes (road, vehicle, pedestrian, bicycle, free space) at 50x50 resolution, 6 cameras, single-scale FPN features from a frozen MobileNetV3. No depth head. Training target: rasterized map + detection boxes from nuScenes annotations. This fits easily on 12 GB with batch size 2.

---

## 3. Assessment of the draft scope

### What is right

- BEV queries pulling from multi-cam features via cross-attention at projected reference points: correct and central.
- Temporal self-attention across consecutive frames: correct and should be implemented.
- Reuse of A1 attention primitives: appropriate, BEVFormer's deformable attention is a specialization of the cross-attention from A1.
- Reuse of A11.5a projection geometry: appropriate, A11.5a should establish the camera-intrinsic/extrinsic projection pipeline that A11.5c reuses.
- LSS-vs-BEVFormer contrast as the written takeaway: correct, this is the right conceptual spine.

### What is missing or under-emphasized

**PETR deserves explicit mention as an alternative query-pull architecture.** PETR (Liu et al., ECCV 2022) is conceptually simpler than BEVFormer for one reason: it has no BEV grid at all. PETR unprojects image pixel locations into 3D frustum coordinates using camera geometry, encodes those 3D coordinates as position embeddings, and adds them to the image features. Object queries then attend to these 3D-position-aware image features in a single global cross-attention, similar to standard DETR. There is no reference-point pillar computation, no hit/miss mask, no deformable attention - just 3D-PE-decorated image features and global cross-attention. For a student who has already implemented multi-head attention in A1, PETR can be demonstrated in roughly 30 lines of new code. BEVFormer requires substantially more. The draft should at minimum mention PETR in a note contrasting grid-BEV queries (BEVFormer, dense) with sparse object queries with 3D PE (PETR/DETR3D, sparse). Whether to teach PETR instead of or in addition to BEVFormer depends on the course's emphasis on dense BEV maps (needed for A11.5d/e occupancy/map tasks) vs. object-detection-only pipelines.

**The draft should note deformable attention is optional for a first pass.** The draft says "(deformable) spatial cross-attention" but does not explicitly advise whether to implement true deformable attention. Given the course's philosophy of verifiable scratch implementations, I recommend the scope document state clearly: implement bilinear-sample-at-reference-points first, add learned offsets second. This is a real curriculum decision.

**BEVFormer v2 (Yang et al., CVPR 2023) is worth a single paragraph.** It is not a pedagogically distinct mechanism, but it introduced perspective supervision (an auxiliary 2D detection head that forces the backbone to learn 3D-informative features) and hybrid object queries. The practical lesson for students: backbone fine-tuning for BEV tasks requires 3D-aware supervision signals, not just BEV loss.

**DETR3D (Wang et al., CoRL 2021) is the direct predecessor.** DETR3D uses sparse 3D object queries (one per predicted object, not one per BEV cell), projects each query's 3D anchor into cameras, bilinear-samples, and runs cross-attention. The conceptual difference from BEVFormer is dense grid vs. sparse object queries. The draft should mention DETR3D at least in the dependency chain since it clarifies why BEVFormer chose a dense BEV grid rather than sparse queries.

### What is outdated or needs context

**By 2026, sparse query methods have largely superseded dense BEV grid methods for detection.** Sparse4D v2/v3 (Lin et al., 2022-2023), PETR/PETRv2, and SparseDrive (2024) all maintain no BEV grid, use recurrent anchor-based temporal fusion, and achieve better detection efficiency than BEVFormer on nuScenes. The dense BEV grid remains important for map segmentation and occupancy prediction (where you need spatial completeness), but for 3D detection it is now recognized as an unnecessary bottleneck. The draft should acknowledge this 2024-2025 context so students understand they are learning a foundational mechanism, not the current SOTA architecture.

**BEVFormer's temporal self-attention is elegant but fragile at long range.** The recurrent ego-motion warping accumulates error across frames and is sensitive to ego localization noise. By 2023-2024, methods using explicit temporal position encoding (PETRv2's 3D PE for temporal alignment) or anchor-based recurrent fusion (Sparse4D v2) had largely replaced BEVFormer-style temporal self-attention in production systems. Worth a brief note so students can place the mechanism historically.

### Suggested reordering

The draft lists the deformable attention sampling before the reference-point projection geometry. For teaching order it makes more sense to: (1) establish the 3D pillar reference point construction and projection first, (2) show that bilinear sampling at projected points already produces reasonable BEV features, (3) motivate learned offsets as a refinement, (4) then add temporal TSA.

---

## 4. Connections to other topics

**Depends on:**
- A1 (Transformer, multi-head attention): BEVFormer's spatial cross-attention and temporal self-attention are direct instances of the cross-attention and self-attention mechanisms from A1. DeformAttn is cross-attention with learned sparse sampling locations instead of full dot-product over all keys.
- A11.5a (camera projection geometry): the entire reference-point pillar construction and projection to image coordinates is camera calibration applied to transformer attention. Without A11.5a, students cannot implement SCA.
- A11.5b (LSS): the contrast that motivates this topic. Students who have implemented LSS's depth distribution lifting will immediately see what BEVFormer eliminates (explicit depth) and what it adds (query-driven geometry).

**Contrasts:**
- A11.5b (LSS / depth-lift): the pedagogical counterpoint, discussed throughout. LSS pushes features into BEV; BEVFormer pulls them.

**Feeds:**
- A11.5d (occupancy prediction): dense BEV feature grids from BEVFormer are a direct input to voxel/occupancy heads. TPVFormer and OccFormer extend BEVFormer's query mechanism to tri-plane or volumetric representations.
- A11.5e (map segmentation): BEV feature maps from BEVFormer are fed directly to semantic segmentation heads for HD map prediction (MapTR, etc.).
- Any end-to-end AD topic: BEVFormer's BEV feature map is the canonical intermediate representation for downstream planning in methods like UniAD.

---

## 5. Must-read sources

1. **Li et al., "BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers," ECCV 2022.** arXiv:2203.17270. The primary source. Read sections 3 (architecture), 3.2 (SCA formula), 3.3 (TSA), and Table 1 (vs. LSS ablation). The 200x200 grid, 4-height pillar, and hit-view averaging are all here.

2. **Zhu et al., "Deformable DETR: Deformable Transformers for End-to-End Object Detection," ICLR 2021.** arXiv:2010.04159. Defines the deformable attention module that BEVFormer's SCA builds on. Section 3.1 covers the K-point sampling and offset prediction. Read alongside BEVFormer section 3.2 to see exactly what BEVFormer borrows and what it changes (the reference point source changes from learned 2D anchors to camera-projected 3D points).

3. **Wang et al., "DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries," CoRL 2021.** arXiv:2110.06922. The direct predecessor: sparse 3D object queries projected into cameras, bilinear sample, standard cross-attention. Simpler than BEVFormer (no BEV grid, no TSA), but the reference-point projection idea is essentially the same. Reading DETR3D first makes BEVFormer's dense grid extension clear.

4. **Liu et al., "PETR: Position Embedding Transformation for Multi-View 3D Object Detection," ECCV 2022.** arXiv:2203.05625. The cleaner alternative: unproject image coordinates to 3D frustum, encode 3D coordinates as PE added to image features, run global cross-attention from object queries. No BEV grid, no deformable attention, no pillar construction. Pedagogically instructive as a contrast: same geometry, radically simpler attention mechanism, worse at dense prediction tasks, better for detection-only.

5. **Lin et al., "Sparse4D: Multi-view 3D Object Detection with Sparse Spatial-Temporal Fusion," arXiv 2022; v2: arXiv:2305.14018, 2023.** Shows where the query-pull lineage went after BEVFormer: anchor-based sparse queries, multi-scale sampling at multiple 3D instance points (not a BEV grid), O(1) recurrent temporal fusion. Good for students to read after A11.5c to see how BEVFormer's mechanism evolved.

6. **Yang et al., "BEVFormer v2: Adapting Modern Image Backbones to Bird's-Eye-View Recognition via Perspective Supervision," CVPR 2023.** arXiv:2211.10439. Shows that BEVFormer's spatial encoder is reusable but the bottleneck in 2023 was backbone adaptation. Perspective supervision (auxiliary 2D detection head) is the key fix. Read the ablation in Table 5 to understand how much of the gain comes from perspective loss vs. architectural changes.

7. **Liu et al., "PETRv2: A Unified Framework for 3D Perception from Multi-Camera Images," ICCV 2023.** arXiv:2206.01256. Extends PETR with temporal modeling via 3D-PE alignment across frames (rather than BEV warping). Demonstrates that temporal alignment can be done in position-embedding space rather than feature space, with competitive results and O(1) temporal complexity. Useful counterpoint to BEVFormer's TSA.

---

## 6. 2024-2026 developments that change how this should be taught

### Dense BEV grid is no longer the standard for detection

By 2024-2025, the nuScenes leaderboard for camera-only 3D detection is dominated by sparse-query methods (Sparse4D v3, SparseDrive, SparseAD) that maintain no BEV grid. They inherit the same camera-projection geometry as BEVFormer (project 3D anchor points to camera images, sample features) but replace the 200x200 grid with a set of ~900 instance-level anchors that track objects across time. This cuts FLOPs substantially and removes the fixed-range constraint of BEV grids. The pedagogical implication: BEVFormer should be taught as the foundational query-pull mechanism, but students should be told that in 2026, production systems use sparse variants. The dense BEV grid is still useful and standard for map/occupancy tasks.

### Dense BEV grid remains the standard for occupancy and map prediction

OccFormer, TPVFormer, SurroundOcc, and numerous occupancy prediction methods (active through 2025) all use dense BEV or tri-plane query grids derived directly from BEVFormer's architecture. If A11.5d/e cover these tasks, then the dense BEV grid is not just historical - it is the active representation. This means the A11.5c scope is correctly motivated as a prerequisite for those topics even in 2026.

### 3D Gaussian Splatting as an emerging BEV representation

By 2025, several methods (DLWM, GaussianFlowOcc) use Gaussian splatting for 3D scene representation rather than explicit BEV grids. These are upstream of the query-pull mechanism but suggest the longer-term direction: explicit 3D representations rendered from multiple views rather than implicit BEV feature grids. This is too advanced for A11.5c but worth a one-line note in the forward connections.

### Foundation model backbones changed what BEVFormer v2 diagnosed

BEVFormer v2's core finding (CVPR 2023) was that pre-trained 2D vision backbones (ViT-based, trained on ImageNet) transfer poorly to BEV tasks without perspective supervision because they have never been asked to reason about 3D structure. This perspective-supervision insight is now standard practice in the field (applied even in sparse-query methods). Students implementing BEVFormer from scratch on a small dataset should understand this: if they freeze a 2D backbone and observe poor BEV performance, the cause is likely this supervision gap, not a bug in their attention code.

### PETR's 3D-PE approach merits more attention than the draft gives it

By 2024, the line between BEVFormer-style and PETR-style has partially blurred (both project 3D coordinates and sample image features), but the conceptual difference - dense BEV grid queries vs. global cross-attention over 3D-PE-decorated image features - is worth teaching explicitly. PETR requires no deformable attention, no pillar construction, no hit/miss masking, and runs standard multi-head cross-attention. For a course that has just taught A1 attention from scratch, PETR is a 30-line extension that immediately makes the geometry-as-PE idea tangible. BEVFormer adds the BEV grid and temporal state on top. Teaching PETR as a brief warm-up before BEVFormer would significantly reduce the implementation complexity of A11.5c's first exercise.
