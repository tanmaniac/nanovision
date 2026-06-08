# A11.5b — Lift-Splat-Shoot: camera-to-BEV via depth lifting

Research date: 2026-06-06.

---

## 1. Key concepts a student must learn

### Categorical depth distribution (the "implicit" depth)

LSS predicts, for each image pixel, a probability distribution over D discrete depth bins rather than a single depth value. In the original paper this is D=41 bins from 4.0 m to 45.0 m in 1.0 m steps. The softmax over these bins is what Philion & Fidler call "implicit" depth: the network is never directly supervised to produce correct depth; it learns to place features wherever they are most useful for the downstream task (BEV segmentation). This is the central conceptual point of the paper and what makes LSS distinct from pseudo-lidar or depth-completion approaches.

The word "implicit" means the model is not told where objects are in depth - only told whether the final BEV prediction is correct. Depth is a latent variable optimized end-to-end. This is both an advantage (no depth labels needed at train time in the vanilla LSS) and a practical weakness (discussed below under BEVDepth).

### Outer-product lift and frustum construction

For each pixel the backbone produces two outputs: a context vector c of dimension C, and a depth distribution alpha of dimension D (D-way softmax). The feature at depth bin d along that pixel's ray is:

    c_d = alpha_d * c

This is the outer product: a DxC tensor per pixel. Across all H*W pixels this creates a frustum-shaped 3D feature volume of shape [D, H, W, C]. Combined with the known camera intrinsics, each (d, h, w) entry has a definite 3D point location. Stacking all cameras gives a 3D point cloud with attached features.

The outer product is the key differentiable operation. It means features are "soft" - a pixel that is confident at one depth concentrates its feature at one 3D location; a pixel uncertain over depth spreads its feature across many locations proportionally. Backpropagation flows through alpha and c jointly.

### Cumsum trick for voxel pooling (the "splat")

The BEV grid divides the ego-frame into pillars. To aggregate the frustum point cloud into the BEV grid, LSS needs to sum features of all frustum points that fall into each BEV pillar. The naive approach builds a hash map or uses scatter-add. LSS instead:

1. Projects every frustum point into its BEV pillar index.
2. Sorts all frustum points by pillar index (a single argsort).
3. Computes a cumulative sum over all features in sorted order.
4. At each pillar boundary, subtracts the cumsum value at the start from the cumsum value at the end to get the pillar sum.

This avoids variable-length padding and reduces to a sort + cumsum + gather, which are all differentiable and GPU-friendly. The authors report a 2x speedup over naive backprop and derive the analytic gradient explicitly. BEVPoolv2 (2022) later replaces this with a custom CUDA kernel that skips computing the full frustum tensor and achieves a further 15x speedup; for a course implementing from scratch, the cumsum trick is the right target.

### End-to-end differentiability

The full LSS pipeline - image backbone, depth head, lift via outer product, coordinate projection with intrinsics/extrinsics, cumsum splat, BEV encoder, segmentation head - is differentiable throughout. No discrete argmax appears before the loss. This is non-obvious because the cumsum pooling involves a sort (which has zero gradient), but the gradient with respect to the feature values is still well-defined: each feature's contribution to the pillar sum equals its depth probability weight, so gradients flow cleanly.

### Why "implicit" depth matters pedagogically

The contrast with explicit depth methods (BEVDepth, pseudo-lidar) is a core teaching point. LSS learns depth from task supervision alone; BEVDepth adds a lidar-projected depth supervision signal, which turns out to matter a lot for detection accuracy. Students should understand why: without direct depth supervision, the network can learn to produce high-entropy depth distributions (spreading features everywhere) which works for BEV segmentation with soft targets but degrades for 3D object detection where precise localization is needed.

---

## 2. Mechanisms to implement from scratch

### Mechanism A: depth head and outer-product lift

Implementation task: given a (B, C_img, H_feat, W_feat) feature map from a backbone, implement a two-headed 1x1 conv that produces (B, D, H_feat, W_feat) depth logits and (B, C_ctx, H_feat, W_feat) context features. Apply softmax to depth logits. Compute the outer product to produce (B, D, H_feat, W_feat, C_ctx). This should be a single einsum or broadcast multiply.

Verifiable: shape test passes; `torch.autograd.gradcheck` passes on a small input (B=1, D=4, H=2, W=2, C_ctx=8). Confirm that when alpha is one-hot at bin d, the output at other depth bins is zero.

### Mechanism B: frustum coordinate generation

Implementation task: given camera intrinsics K and extrinsics E (ego-to-cam), generate for each (d, h, w) in the frustum the 3D ego-frame coordinate. This requires unprojecting pixel (h, w) to a unit ray, scaling by depth d, then applying E^{-1}. Result shape: (D, H_feat, W_feat, 3).

Verifiable: for a camera looking straight forward, the center pixel's ray should point in the +x or +y direction (depending on convention); the depth-d point should be exactly d meters out. Check with known camera parameters from nuScenes-mini using pyquaternion (substrate library, not the taught mechanism).

### Mechanism C: cumsum splat

Implementation task: given a set of N 3D points with attached C-dimensional features and a 2D BEV grid specification (x_range, y_range, resolution), implement the sort-and-cumsum pooling to produce a (B_x, B_y, C) BEV feature map. No scatter_add; no custom CUDA; pure torch on CPU for the tiny case.

Verifiable: for two points in the same pillar with known features, the output at that pillar equals their sum. For points outside the grid bounds, they should be dropped. Run gradcheck.

### Mechanism D: BEV vehicle segmentation head

Implementation task: a lightweight BEV encoder (two ResNet blocks or a simple conv stack) followed by a 1x1 conv producing a (B, 1, X, Y) binary segmentation logit. Train with binary cross-entropy on the nuScenes-mini vehicle occupancy ground truth.

Verifiable: overfit-one-batch test: single batch of 1-3 nuScenes scenes, loss should reach <0.05 and predicted BEV map should visually align with vehicle positions.

---

## 3. Assessment of the draft scope

### What the draft gets right

The draft correctly identifies all five main components: categorical depth distribution, outer-product lift, frustum construction, voxel/pillar pooling splat, BEV segmentation head. The overfit-one-batch vehicle segmentation on nuScenes-mini is the right evaluation target for a 12 GB GPU. Depth-distribution visualization (show that the predicted alpha values concentrate at plausible depths) is a good diagnostic and worth keeping.

### Corrections and additions

**Add: explicit depth supervision via lidar (BEVDepth) as a required discussion point, not optional.**

This is the most important practical improvement to LSS and is currently under-represented in the draft. BEVDepth (Li et al., arXiv 2206.10092, AAAI 2023) projects lidar point clouds onto the image plane during training and supervises the depth head with a one-hot classification loss at those pixels. This single change, applied to an otherwise-LSS architecture, is the main driver of BEVDet → BEVDepth performance gain. Students should implement or at least replicate the supervision signal: project a lidar sweep (available in nuScenes-mini) onto the image plane, form sparse depth labels, and add a cross-entropy loss on the depth logits at those locations. The pedagogical point: the implicit depth assumption works for BEV segmentation but is insufficient for precise 3D detection; the lidar supervision is cheap to add and large in effect.

**Add: BEVDet framing to situate LSS.**

BEVDet (Huang et al., arXiv 2112.11790, 2021) is the first paper to apply the LSS view transformer directly to 3D object detection and is the direct parent of BEVDepth. Students should know this lineage: LSS (BEV seg, 2020) → BEVDet (BEV detection, 2021) → BEVDet4D (temporal, 2022) → BEVDepth (explicit depth, AAAI 2023).

**Clarify: cumsum trick vs BEVPoolv2.**

The draft says "cumsum-trick / voxel pooling splat" which is ambiguous. The recommendation: implement the cumsum trick from the original paper (sort + cumsum). Note that BEVPoolv2 replaces it with a CUDA kernel that eliminates the large intermediate frustum tensor entirely (15x speedup), but this is a deployment optimization, not a conceptual one. The course should implement the cumsum approach and mention BEVPoolv2 as a pointer for production use.

**Clarify: "implicit" depth is not the same as "no depth supervision".**

The draft uses "implicit depth distribution" correctly but should make explicit that LSS's implicit depth means the depth is learned from downstream task loss only. The contrast with BEVDepth (explicit) and with BEVFormer (no depth prediction at all - attention pull instead) is the key organizing concept for the A11.5 module.

**Add: discuss why the outer product creates a large intermediate tensor and why this is the efficiency bottleneck.**

The frustum tensor is [N_cams, D, H_feat, W_feat, C_ctx]. With N=6, D=41, H=16, W=44, C=64 (a small config) this is 6*41*16*44*64 = ~110M elements at float32 = ~440 MB before the BEV encoder. This is why BEVPoolv2 matters for deployment. Worth one sentence in lecture.

**Nothing to cut.** The BEV segmentation head and visualization are appropriate scope. The depends-on chain (A11.5a geometry, A2 backbone) is correct.

**Temporal LSS (BEVDet4D)** is not required scope for this topic but one sentence noting that temporal BEV fusion by warping a previous BEV frame improves velocity estimation should be included.

---

## 4. Dependencies and connections

### Depends on

**A11.5a (multi-view geometry and BEV basics):** LSS requires working intrinsic/extrinsic camera matrices to unproject pixels to 3D rays and to project 3D points into the BEV grid. Students who have not implemented coordinate transforms will find the frustum construction opaque. The cumsum splat also assumes a student already understands what a BEV pillar is.

**A2 (image backbone):** LSS wraps any image backbone. A student who already has a working EfficientNet or ResNet feature extractor can plug it in as the stem. The depth and context heads are thin 1x1 convolution heads on top of backbone features.

### Contrasts with A11.5c (BEVFormer)

BEVFormer takes the opposite approach: instead of pushing image features out into 3D space ("lift"), it pulls features from image space by projecting 3D BEV query points back to 2D and applying deformable attention. There is no depth distribution and no voxel pooling. The contrast in data flow direction - LSS pushes out, BEVFormer pulls in - is the most important concept in the A11.5 module. Neither approach is universally better; LSS-family methods tend to be faster to implement and easier to understand; BEVFormer-family methods have proven strong on temporal tasks.

### Feeds A11.5d (occupancy prediction)

3D semantic occupancy (e.g., OpenOccupancy, Occ3D, SurroundOcc) is a direct extension of the BEV segmentation in LSS. Instead of collapsing the depth axis to produce a 2D BEV map, occupancy methods retain the full (X, Y, Z) voxel grid and predict semantic labels per voxel. The depth-lifting mechanism from LSS (frustum → 3D feature volume) is used without modification; only the BEV encoder and head are replaced with 3D conv or attention-based decoders. Students who implement LSS have the hardest part done.

---

## 5. Must-read sources

1. **Philion & Fidler (ECCV 2020)** - "Lift, Splat, Shoot: Encoding Images from Arbitrary Camera Rigs by Implicitly Unprojecting to 3D." arXiv 2008.05711. The primary source. Read sections 3 (method) and 4 (experiments) first; section 3.1 defines the outer product, section 3.2 defines the cumsum splat.

2. **Huang et al. (arXiv 2112.11790, 2021)** - "BEVDet: High-performance Multi-camera 3D Object Detection in Bird-Eye-View." The paper that applies LSS to 3D detection and introduces the data augmentation strategy that makes training stable. Required reading for understanding why LSS needed engineering work beyond the ECCV paper.

3. **Li et al. (AAAI 2023, arXiv 2206.10092)** - "BEVDepth: Acquisition of Reliable Depth for Multi-view 3D Object Detection." Introduces explicit lidar depth supervision and the camera-aware depth module. The most important practical improvement to the LSS family. Read to understand why implicit depth is insufficient for 3D detection.

4. **Huang et al. (arXiv 2211.17111, 2022)** - "BEVPoolv2: A Cutting-edge Implementation of BEVDet Toward Deployment." Shows how the frustum tensor bottleneck is eliminated with a custom CUDA kernel. Not required for implementation, but the comparison table (15x speedup) motivates why the cumsum trick matters for production.

5. **Li et al. (ECCV 2022, arXiv 2203.17270)** - "BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers." The main alternative architecture. Reading the BEVFormer ablation table (which includes LSS as a baseline) gives the clearest single-paper comparison between push-out (LSS) and pull-in (attention) approaches.

6. **Lu et al. (CVPR 2025, arXiv 2504.01957)** - "GaussianLSS: Toward Real-world BEV Perception: Depth Uncertainty Estimation via Gaussian Splatting." Replaces discrete depth bins with continuous 3D Gaussians to address bin sparsity and unstable softmax distributions. Represents the 2025 state of the push-out family and shows the direction of active research.

---

## 6. Developments from 2024-2026 that change how this should be taught

### Lidar depth supervision is now table stakes

As of 2023-2024, every competitive LSS-family model on nuScenes uses explicit lidar depth supervision from BEVDepth. Teaching LSS without mentioning that the implicit-depth variant is largely superseded in detection (though still relevant for segmentation) would leave a false impression. The course should frame vanilla LSS as the mechanistic foundation and BEVDepth's depth supervision as the standard production extension, not an optional add-on.

### The push-out vs pull-in framing has clarified

By 2024-2025, the community distinguishes three BEV construction strategies cleanly: (a) depth-based push-out (LSS family), (b) attention-based pull-in (BEVFormer family), and (c) hybrid or sparse methods (Sparse4D, DETR3D). This taxonomy is now standard in survey papers and should be the organizing framework for the A11.5 module. A11.5b teaches (a), A11.5c teaches (b).

### Gaussian splatting has entered the LSS family

GaussianLSS (CVPR 2025) replaces the discrete D-bin softmax with a continuous Gaussian depth representation, projecting each pixel's uncertainty into a 3D Gaussian primitive before splatting. This reduces the sparsity problem of the discrete bins and achieves comparable or better IoU at significantly lower memory. Teaching the categorical bin design should now include one slide noting that the bins are a simplification and that Gaussian/continuous representations are the active research frontier.

### BEV occupancy has become a primary task

OpenOccupancy, Occ3D, SurroundOcc, and the nuScenes occupancy benchmark all arrived in 2023-2024, making 3D semantic occupancy a first-class evaluation. The BEV segmentation head in vanilla LSS is now typically described as a 2D reduction of the 3D occupancy task. Teaching BEV segmentation as the head while noting that dropping the BEV pooling collapse (and instead using a 3D decoder) gives occupancy is the right framing going into A11.5d.

### Temporal fusion is now expected in any serious BEV model

BEVDet4D (2022), BEVDepth temporal extensions, and BEVFormer's temporal attention have made single-frame BEV models look dated for detection. For A11.5b the single-frame model is fine for teaching the mechanism; one sentence noting that BEVDet4D adds temporal by concatenating a warped previous-frame BEV feature map is enough.

### Efficiency is not a research topic anymore - it is a deployment baseline

BEVPoolv2 and TensorRT deployments are described in production documentation as of 2024. For the course this means: do not spend lecture time on BEVPoolv2 kernel internals, but do state the order-of-magnitude cost of the naive frustum tensor so students understand why it was optimized.
