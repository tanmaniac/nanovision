# A11.5d — 3D occupancy prediction: validation report

**Date validated:** 2026-06-06  
**Validator model:** Claude Sonnet 4.6

---

## 1. Key concepts a student must learn

### The voxel occupancy grid

The scene is partitioned into a regular 3D grid of voxels. Each voxel carries a binary occupancy label (free or occupied) and optionally a semantic class. The standard Occ3D-nuScenes grid is 200 x 200 x 16 at 0.4 m resolution, covering [-40 m, -40 m, -1 m] to [40 m, 40 m, 5.4 m]. This is important context: the grid is much wider than tall because AV scenes are flat, which creates the severe class-imbalance problem below.

Students should understand the difference between three related tasks:

- **Semantic scene completion (SSC):** given partial observations (one or a few images), predict the full 3D semantic voxel grid, including unobserved regions. MonoScene (CVPR 2022) is the canonical SSC paper.
- **3D occupancy prediction (vision-centric):** given surround-view camera frames, predict what voxels are occupied and with what class. BEVFormer/TPVFormer-style methods belong here.
- **Self-supervised occupancy:** no 3D labels; supervision comes entirely from rendering losses applied to 2D images. RenderOcc/OccNeRF belong here.

### Occupancy vs semantic occupancy

Binary occupancy asks only: is this voxel filled? Semantic occupancy asks: if filled, what class? In AV perception the semantic version is always desired because the downstream planner needs to distinguish driveable surface from pedestrian from static obstacle. The 17-class Occ3D-nuScenes taxonomy (16 nuScenes-lidarseg classes + free) is the de-facto benchmark set. Teaching binary occupancy alone is insufficient; the head must output C+1 logits per voxel (C semantic classes plus free).

### The dense volumetric loss and class imbalance

The loss over a dense 200x200x16 grid is a per-voxel cross-entropy (or weighted cross-entropy + Lovász-softmax). The critical practical issue is severe class imbalance:

- Free voxels make up roughly 95-98% of the grid.
- Among occupied classes, driveable surface and terrain dominate; bicycle and motorcycle appear ~10,000x less frequently.

Standard cross-entropy will collapse to predicting everything as free. Mitigations used in the literature: inverse-frequency class weights in cross-entropy, focal loss (γ ≥ 2), Lovász loss (differentiable IoU surrogate), and geo/semantic loss decoupling. Students must implement at least inverse-frequency weighting and understand why focal loss or Lovász is often added.

### The link between occupancy prediction and NeRF-style volume rendering

This connection is the most important conceptual bridge in the topic. In NeRF, a ray through the scene accumulates color and opacity as:

```
C(r) = ∫ T(t) * σ(t) * c(t) dt
T(t) = exp(-∫₀ᵗ σ(s) ds)
```

where σ is volume density. In the discretized form, the transmittance is a product of `(1 - αᵢ)` terms, and αᵢ = 1 - exp(-σᵢ δᵢ) is the per-voxel opacity, which is exactly the occupancy probability. The two tasks differ only in direction:

- **NeRF (A9):** given a density/occupancy field, render a pixel color by integrating along rays.
- **Occupancy prediction:** given pixel observations (images), invert that rendering process to estimate the density/occupancy field.

RenderOcc (ICRA 2024) makes this duality the training objective: build a voxel occupancy field from multi-view images, render it back to 2D using the NeRF alpha-compositing formula, and supervise with 2D depth + semantic labels instead of 3D occupancy labels. This eliminates the need for expensive 3D annotations.

The formula a student should memorize: rendered depth `D(r) = Σᵢ Tᵢ αᵢ dᵢ`, rendered semantic `S(r) = Σᵢ Tᵢ αᵢ sᵢ`. Supervision is applied to D and S against 2D ground truth. The A9 module's alpha-compositing code is re-used verbatim; only the supervision target changes.

### BEV-to-voxel lifting

Most methods start from a BEV feature map (a 2D spatial tensor, e.g. H/8 x W/8 after BEVFormer or LSS) and need to lift it to a 3D voxel tensor. Three approaches matter:

1. **Pillar extrusion:** replicate a BEV feature along the Z axis. Fast but loses height information.
2. **Height-aware MLP:** learn a small per-BEV-cell network that predicts how to distribute features vertically. Used in BEVDet4D and OccupancyNet.
3. **3D deformable attention (TPVFormer):** each voxel query attends to the three TPV planes. More expressive but heavier.

For the nanovision scratch implementation, pillar extrusion from BEV features is the correct starting point. It is a single `repeat` or `unsqueeze` + `expand` operation and connects directly to the BEV encoder from A11.5b/c.

---

## 2. Mechanisms to implement from scratch

### Memory warning: dense voxel grids are expensive

A full Occ3D-nuScenes grid at float32 logits (200 x 200 x 16 x 18 classes) = ~46 MB per sample. With a batch of 4, decoder feature maps at multiple scales, and BEV backbone, this exceeds 12 GB before backward pass. The pedagogically correct approach for a 12 GB GPU is a **tiny synthetic grid.**

**Recommended grid for nanovision:** 32 x 32 x 8, 3-5 semantic classes. This produces tensors of ~25 K floats per voxel prediction map, leaves ample room for backward pass and BEV feature backbone.

### Mechanism 1: occupancy head with volumetric loss

**What to implement:**  
- A 3D convolution head (two Conv3d layers) that takes a BEV-derived voxel tensor [B, C, Z, Y, X] and outputs per-voxel class logits [B, num_classes, Z, Y, X].  
- Cross-entropy loss with inverse-frequency class weights.  
- Lovász-softmax loss (standard reference implementation exists; student may use it as substrate, but the occupancy head and weighting must be from scratch).

**Minimal verifiable task:**  
- Shape test: feed random BEV features [2, 64, 8, 32, 32] → logits output must be [2, 5, 8, 32, 32].  
- `torch.autograd.gradcheck` on the weighted cross-entropy.  
- Overfit-one-batch: a single synthesized voxel scene (e.g. a few occupied voxel clusters) should reach IoU > 0.85 on the train sample within 200 steps.

### Mechanism 2: NeRF-style rendering supervision (the RenderOcc/OccNeRF approach)

**What to implement:**  
- From the occupancy field [B, 1, Z, Y, X] (occupancy probability per voxel), implement a ray-marcher that casts rays through the voxel grid, accumulates alpha-composited depth and semantic labels.  
- This re-uses the alpha-compositing code from A9's NeRF module: Tᵢ = ∏ⱼ<ᵢ (1 - αⱼ), rendered depth D = Σ Tᵢ αᵢ dᵢ.  
- Render depth and compare to a sparse LiDAR depth map (from nuScenes-mini, which does have LiDAR). The loss is L1 or Huber on rendered vs observed depth.

**This is the pedagogically correct spine** because it makes the A9 rendering code directly reusable and shows students that occupancy prediction and NeRF rendering are the same kernel in opposite directions.

**Minimal verifiable task:**  
- Gradcheck on the ray-marching accumulation function.  
- Overfit-one-batch: given one nuScenes-mini frame with camera intrinsics/extrinsics and sparse LiDAR depth, the rendered depth error should drop to < 0.5 m within 500 steps (starting from a random initialization).  
- Shape test: ray batch [N_rays, N_samples] alphas → accumulated depth [N_rays].

### Mechanism 3: BEV-to-voxel pillar extrusion (connector to A11.5b/c)

**What to implement:**  
- Take a BEV tensor [B, C, H_bev, W_bev] from the BEVFormer/LSS module, produce a voxel tensor [B, C, Z, H_bev, W_bev] by a learnable height distribution: a small Conv2d that outputs [B, C*Z, H, W], then reshape to [B, C, Z, H, W].

**Minimal verifiable task:**  
- Shape test and gradcheck.

---

## 3. Assessment of the draft scope

### What is right

- The voxel occupancy grid, per-voxel semantics, and volumetric loss are correctly identified as core.
- The NeRF connection is sound and important.
- Depending on A11.5a (camera-to-BEV) and A9 (NeRF rendering) is the correct dependency structure.

### What is missing or under-weighted

**The NeRF rendering supervision angle should be the spine, not a side connection.** The draft says "explicitly connect to A9" but treats it as a conceptual footnote. Given that RenderOcc (ICRA 2024) and OccNeRF (2023) have demonstrated that NeRF-based 2D rendering supervision can match fully 3D-supervised methods while requiring only 2D labels, and given that nuScenes-mini does *not* have Occ3D-style 3D occupancy labels (see below), this approach is also more practical for the course dataset.

**Occ3D-nuScenes mini availability is a real constraint.** Occ3D-nuScenes is derived from the full nuScenes 700-scene split (600 train / 150 val). The nuScenes v1.0-mini split has only 10 scenes and is not part of the Occ3D release. There is no officially released mini subset of Occ3D-nuScenes with precomputed occupancy labels. Options for nanovision:
1. Use NeRF rendering supervision (RenderOcc approach): only needs 2D semantics + LiDAR depth from nuScenes-mini, both of which are available.
2. Generate occupancy labels for the 10 mini scenes manually using the Occ3D label pipeline (the code is open-source at github.com/Tsinghua-MARS-Lab/Occ3D), but this requires running the densification pipeline.
3. Use synthetic voxel ground truth for the overfit-one-batch task and treat nuScenes-mini as evaluation-only.

**Option 1 (rendering supervision) is the recommended path** - it avoids the label availability problem, directly uses A9's rendering code, and is pedagogically superior because it makes the NeRF-occupancy duality concrete rather than verbal.

**TPVFormer should be discussed but not implemented from scratch.** Its tri-perspective view is a meaningful extension of BEV representation and is worth ~one page of explanation. The full deformable attention mechanism is too heavy for a 12 GB scratch exercise. The student should understand TPV as "BEV plus two perpendicular planes" and why it resolves the height ambiguity of flat BEV.

**SurroundOcc label generation pipeline** (Poisson reconstruction + voxelization from multi-frame LiDAR) is worth one paragraph as context for where occupancy labels come from in practice. Students should understand that "occupancy ground truth" is itself a derived, imperfect product, not a clean annotation.

**Sparse occupancy methods (SparseOcc, CVPR 2024; SparseOcc ECCV 2024) should be noted** as the direction the field has moved post-2023. Dense 200x200x16 grids are being replaced by sparse representations that discard >90% of empty voxels. This explains why the tiny-grid exercise remains relevant: it isolates the non-sparse core without requiring sparse voxel convolution libraries (MinkowskiEngine, etc.).

**Gaussian-based occupancy (GaussianOcc, ICCV 2025; GaussianFlowOcc, ICCV 2025)** represents the current frontier. These methods represent the scene as 3D Gaussians rather than voxels and use Gaussian splatting for rendering supervision. This is the 2025-2026 replacement for both NeRF-density and dense voxel grids. It should be mentioned as a "where this leads" note but not implemented - Gaussian splatting is A9-territory extension.

### What to cut or reorder

- The draft does not mention the label availability problem. This must be addressed first.
- "Overfit voxel occupancy on a few scenes" should specify: synthetic voxels for the 3D supervised path, or nuScenes-mini camera+LiDAR for the rendering-supervised path.
- The BEV-to-voxel lifting mechanism deserves its own implementation step, not just a mention.

### Suggested reordering

1. BEV-to-voxel lift (pillar extrusion from A11.5b/c BEV features) - 30 min exercise.
2. Occupancy head + weighted CE loss on synthetic voxels - 1 hr exercise.
3. NeRF-rendering supervision on nuScenes-mini LiDAR - spine exercise, ~2 hr.
4. TPVFormer as conceptual extension (read + describe, no implementation).

---

## 4. Dependencies and conceptual bridge

### Depends on

- **A11.5a (image backbone + multi-camera feature extraction):** provides the per-camera feature maps that feed BEV/voxel lifting.
- **A11.5b/c (BEV encoding):** provides the BEV feature tensor [B, C, H, W] that is lifted to 3D.
- **A9 (NeRF volume rendering):** provides the alpha-compositing ray-marching function. In the rendering-supervised path, this is the actual loss computation kernel.

### Bridge to AV perception

Occupancy prediction is the module where the geometric machinery from the 3D vision modules (multi-view geometry, depth estimation, BEV encoding) becomes a directly plannable scene representation. Classical object detection outputs bounding boxes, which miss free-space geometry and fail for irregular objects like scaffolding or debris. Occupancy prediction outputs a metric voxel grid that a motion planner can query directly as a signed-distance field or free-space mask.

The NeRF connection (A9 → A11.5d) is not merely conceptual: it explains why self-supervised occupancy is possible at all. NeRF showed that a density field can be optimized purely from 2D image observations via volume rendering. Occupancy prediction inherits this and extends it to surround-view rigs and semantic labels.

---

## 5. Must-read sources

1. **MonoScene (Cao et al., CVPR 2022)** - arxiv:2112.00726. First paper to lift a single 2D image to a 3D semantic occupancy grid; defines the SSC formulation and FLoSP feature lifting that all subsequent camera-based methods build on.

2. **TPVFormer (Huang et al., CVPR 2023)** - arxiv:2302.07817. Introduces the tri-perspective view representation: BEV + two perpendicular planes. Shows that three 2D feature planes can approximate a full 3D volume much more efficiently than a dense voxel tensor, with concrete mIoU results on nuScenes.

3. **Occ3D (Tian et al., NeurIPS 2023 Datasets & Benchmarks)** - arxiv:2304.14365. Establishes the Occ3D-nuScenes benchmark (200x200x16 grid, 18 classes, 600 training scenes). The current standard evaluation protocol and label-generation pipeline. Read to understand what "occupancy ground truth" is and how it is generated from LiDAR.

4. **SurroundOcc (Wei et al., ICCV 2023)** - arxiv:2303.09551. Surround-view multi-camera occupancy prediction with a Poisson-reconstruction-based dense label generation method. Paired reading with Occ3D to see two different annotation strategies.

5. **RenderOcc (Pan et al., ICRA 2024)** - arxiv:2309.09502. The direct NeRF-rendering-supervised occupancy paper. Trains a 3D occupancy model using only 2D depth + semantic labels via NeRF-style volume rendering. The implementation blueprint for the rendering-supervision exercise and the primary justification for treating A9 and A11.5d as the same rendering kernel.

6. **OccNeRF (Zhang et al., 2023)** - arxiv:2312.09243. Fully self-supervised surround-view occupancy via temporal photometric consistency (no depth labels needed). Shows the logical endpoint of the rendering-supervision idea when even 2D depth is unavailable.

7. **SparseOcc (Tang et al., CVPR 2024 / Liu et al., ECCV 2024)** - arxiv:2404.09502 and arxiv:2312.17118. Represents the post-2023 move from dense voxel grids to sparse latent representations. Read to understand why production systems are abandoning the dense grid that the course exercise implements.

---

## 6. 2024-2026 developments that change how this should be taught

### Sparse representations have replaced dense grids

By 2024-2025, the standard approach in published work is sparse voxel prediction (SparseOcc CVPR 2024, SparseOcc ECCV 2024, MinkOcc 2025). Dense 200x200x16 grids are now the "baseline to beat," not the production approach. The course should teach the dense grid as the conceptual foundation but flag that 90% of voxels are empty and that sparse methods exploit this directly.

### Gaussian splatting has superseded NeRF-density in rendering-supervised methods

GaussianOcc (ICCV 2025) and GaussianFlowOcc (ICCV 2025) replace NeRF-style density integration with 3D Gaussian splatting for the rendering step. Gaussians render faster, are differentiable, and represent surfaces more compactly than density volumes. The mathematical connection to the A9 module weakens slightly (A9 covers NeRF, not 3DGS), but the conceptual bridge - "invert a renderer to get geometry" - remains valid. Instructors should note this evolution in a sidebar.

### RayIoU replaces mIoU as the primary metric

SparseOcc (ECCV 2024) introduced RayIoU, which evaluates occupancy along camera rays rather than over all voxels. This addresses a major flaw in mIoU: the metric is dominated by the large number of free voxels and penalizes depth errors twice (once for the occupied voxel predicted in the wrong place, once for the free voxel predicted as occupied). RayIoU is now standard in competitive submissions. The course should implement a simple mIoU for the overfit exercise but mention RayIoU and why it matters.

### Self-supervised / annotation-free methods are now competitive

RenderOcc (2024), OccNeRF (2023), GaussianOcc (2025) all demonstrate that methods trained with zero 3D occupancy labels can match or approach fully supervised baselines. For the course, this makes the rendering-supervision path the more forward-looking one to emphasize.

### Occupancy as a world model representation

The 2024-2025 literature increasingly frames 4D occupancy (3D + time) as the scene representation for neural world models in AV. Drive-OccWorld (AAAI 2025) and PreWorld (ICLR 2025) predict future occupancy states for planning. This extends directly from the static occupancy head taught in this module. The course can note this as the destination without implementing it.

### nuScenes-mini label availability remains a gap

As of 2026, no official 3D occupancy label release covers the 10-scene nuScenes-mini split. The Occ3D-nuScenes release targets the full 750-scene split. A course using nuScenes-mini must either (a) run the label generation pipeline on the 10 mini scenes, (b) use the rendering-supervised path (no 3D labels required), or (c) use synthetic voxel ground truth for the main exercise. This is a concrete constraint the draft scope does not address.
