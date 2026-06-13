# BUILD_ORDER.md - dependency-ordered build plan

This file tells the builder the order to construct assignments and exactly what
each one depends on. Build strictly top-to-bottom; never build an assignment
before its dependencies exist, because later assignments import earlier
primitives from the shared `nanovision/` package.

This plan reflects the curriculum validation in `curriculum_review.md` and
the decisions recorded in `revised_curriculum.md`: each assignment builds the
2026-consensus variant as its core with the original taught as historical
contrast, four content items were added (geometry foundation models, a VQ
tokenizer, a video/temporal half-assignment, and a ConvNeXt block folded into
A2), the autonomous-driving sub-module stays at five assignments, and A5/A6 stay
separate. The course is 22 built assignments.

Legend: **[Core]** = learner implements the mechanism from scratch.
**[Survey]** = guided notebook, mostly pretrained weights + a few fill-ins.
**[Mixed]** = some from-scratch implementation + some survey.

---

## Phase 0 - Foundation (build first, everything depends on it)

### A0 - Harness & primitives  **[Core]**
- **Builds:** `nanovision.primitives` (LayerNorm, gelu, MLP),
  `nanovision.trainer.Trainer`, `nanovision.gradcheck`, `nanovision.determinism`,
  `nanovision.data.toy` + `nanovision.data.images`, `nanovision.viz` (stubs).
- **Depends on:** nothing.
- **Verifiable result:** gradcheck passes on LayerNorm/MLP; Trainer overfits a
  toy linear-regression batch to ~0.
- **Why first:** every later assignment imports the harness and primitives.

### A1 - The Transformer, from scratch  **[Core]**
- **Builds:** `nanovision.attention` (SDPA, MultiHeadAttention),
  `nanovision.transformer` (blocks, encoder/decoder, positional encodings,
  RoPE, RMSNorm, SwiGLU).
- **Implements:** scaled dot-product attention, multi-head, causal masking;
  the LLaMA-style block as the core - RMSNorm, RoPE, SwiGLU - with sinusoidal/
  learned positional encodings and LayerNorm taught as historical contrast;
  GQA/MQA conceptually plus a toy implementation; AdamW+warmup recipe; trains a
  decoder-only tiny char-LM and a copy/sort toy task.
- **Depends on:** A0.
- **Forbidden imports:** `nn.MultiheadAttention`, `nn.Transformer*`,
  `F.scaled_dot_product_attention`.
- **Verifiable result:** gradcheck on attention; overfit copy task to ~0;
  tiny char-LM produces coherent toy text.
- **Reading notes:** FlashAttention (IO-aware attention); Vision Mamba / SSM
  backbones.

---

## Phase 1 - Visual representations

### A2 - Vision Transformers  **[Core]**
- **Implements:** patch embedding, class token, positional embeddings; assembles
  ViT from `nanovision.transformer`; register tokens (core, not stretch); the
  DeiT augmentation recipe (RandAugment, MixUp, CutMix, stochastic depth) as part
  of the mechanism; a CLS-vs-mean-pool measurement; a ConvNeXt block build (the
  conv-vs-transformer backbone story); trains a tiny ViT on CIFAR-10.
- **Depends on:** A0, A1.
- **Builds into shared lib:** a ConvNeXt block in `nanovision.primitives`.
- **Probe notebook:** load pretrained DINOv2 and DINOv2-with-registers (via
  `timm`), show the attention-map difference. (probe only; not mechanism code)
- **Notes:** 2D sincos / axial-RoPE positional schemes mentioned; Swin demoted
  to context for the isotropic-vs-hierarchical tradeoff.

### A3 - Self-supervised learning (MAE + DINO)  **[Core]**
- **Implements:** (a) MAE - random masking, encoder-on-visible-only, lightweight
  decoder, pixel-reconstruction loss; (b) DINO (the build core) - EMA teacher,
  multi-crop, centering + sharpening, the collapse-avoidance mechanics; (c) iBOT
  as a ~20-line extension on DINO (the DINOv2 patch-level objective).
- **Depends on:** A1, A2.
- **Verifiable result:** MAE reconstructs held-out patches; the linear probe is a
  DINO-vs-MAE-vs-random comparison (the gap is the lesson); explicit collapse
  test (removing centering collapses the representation).
- **Conceptual page:** DINOv2's three-loss structure (DINO + iBOT + KoLeo) and
  the I-JEPA latent-prediction paradigm.

### A3.5 - Video / temporal modeling  **[Core, compact]**
- **Implements:** spatiotemporal tube masking extending MAE; tubelet embedding; a
  tiny video ViT.
- **Depends on:** A2, A3.
- **Verifiable result:** the video MAE reconstructs masked tubelets on a toy
  video set; overfits a single batch.
- **Why:** bridges A3 → BEVFormer temporal attention (A11.5c) → world models
  (A12); temporal modeling was a gap in the original list.

### A4 - CLIP & open-vocabulary  **[Core]**
- **Implements:** dual encoder (reuse the A2 ViT image tower and A1 transformer
  text tower); the SigLIP sigmoid loss as the primary objective (it trains at
  small batch on 12GB), with softmax InfoNCE taught as the historical
  predecessor; the SigLIP bias initialization; zero-shot classification via
  prompt embeddings. A4 builds the loss and zero-shot inference, not the encoder.
- **Depends on:** A1, A2.
- **Probe notebook:** load real CLIP/SigLIP weights, reproduce zero-shot CIFAR.
- **Required reading:** the OWL-ViT → OWLv2 → GroundingDINO open-vocab thread.
- **Why here:** the multimodal bridge; A8 (VLM) and A13 (VLA) depend on it.

---

## Phase 2 - Generative models

### A5 - Diffusion (DDPM / DDIM)  **[Core]**
- **Implements:** forward noising schedule (linear + cosine), time-embedded U-Net,
  v-prediction as the required objective (with an epsilon-vs-v comparison), the
  score-epsilon (Tweedie) derivation as an explicit exercise, ancestral (DDPM) +
  DDIM sampling, classifier-free guidance.
- **Depends on:** A0 (and the U-Net uses primitives; time embedding uses A1
  sinusoidal).
- **Forbidden imports:** `diffusers`, any prebuilt scheduler/UNet.
- **Verifiable result:** unconditional MNIST/shapes generation that visibly
  improves over training; this FITS and converges on a 4080.
- **Reading:** Min-SNR loss weighting; EDM preconditioning; the score-SDE paper.

### A6 - Flow matching & rectified flow  **[Core]**
- **Implements:** the conditional flow matching objective (velocity field along
  straight-ish probability paths); optimal-transport coupling as a first-class
  build task (straightens trajectories without reflow; the production default);
  logit-normal timestep sampling as an ablation; the rectified-flow reflow
  procedure; few-step ODE sampling; reuses A5's network backbone.
- **Depends on:** A5.
- **Verifiable result:** matches A5 sample quality in far fewer steps; OT coupling
  and reflow visibly straighten trajectories (plot them on 2D toy distributions).
- **Notes:** teach the exact diffusion↔flow-matching equivalence for Gaussian
  paths as the spine; flag MeanFlow / Rectified Diffusion as the methods that
  supersede reflow for few-step generation.

### A6.5 - VQ tokenizer  **[Core, compact]**
- **Implements:** a VQ-VAE/VQ-GAN codebook with the straight-through estimator; a
  tiny autoregressive prior over the discrete tokens (reuse A1).
- **Depends on:** A1, A2.
- **Builds into shared lib:** `nanovision.quantize` (vector-quantization codebook
  + straight-through).
- **Verifiable result:** the tokenizer reconstructs a toy image set; the AR prior
  overfits a single batch and samples tokens that decode to plausible images.
- **Why:** unlocks the unified multimodal generation literature (Chameleon,
  Janus, LlamaGen); contrasts with the continuous KL-VAE of A7.

### A7 - Latent diffusion & a tiny DiT  **[Core]**
- **Implements:** a small KL-VAE (encoder/decoder + KL), then a flow-matching DiT
  (not DDPM) in the VAE latent space - patchify latents, condition on timestep +
  class via adaLN-Zero - instead of the U-Net.
- **Depends on:** A1, A5, A6, A6.5.
- **Verifiable result:** latent-space generation on a toy set; the DiT overfits a
  single batch; sampling reconstructs through the VAE decoder.
- **Reading:** MM-DiT (joint text+image tokens, SD3/FLUX) and REPA representation
  alignment; the KL-VAE (continuous, diffusion) vs VQ (discrete, autoregressive)
  distinction. Conceptual basis for the video-generation reading note.

---

## Phase 3 - Multimodal understanding & 3D

### A8 - Vision-Language Models  **[Core]**
- **Implements:** LLaVA-style bridge - frozen vision encoder (A2/A4) + an MLP
  projector into the LM embedding space (core) + a tiny decoder-only LM (A1); an
  optional 1-layer cross-attention resampler so both connector families are
  concrete; the explicit two-stage train (projector-align then instruction-tune);
  an AnyRes/tiling shape exercise; train on a toy VQA/caption set.
- **Depends on:** A1, A2, A4.
- **Verifiable result:** model answers toy questions grounded in the image;
  ablation showing the projector carries grounding.
- **Note:** "frozen encoder" is a pedagogical simplification (2024+ models
  fine-tune the ViT).

### A9 - NeRF (the prequel)  **[Core, compact]**
- **Implements:** positional (Fourier) encoding motivated by showing the
  spectral-bias failure first; the MLP radiance field; volumetric rendering (ray
  sampling, alpha compositing along rays); fit ONE tiny synthetic scene.
- **Depends on:** A0. Camera geometry is prerequisite knowledge for this learner;
  the only new work is getting the coordinate conventions right (OpenCV vs OpenGL
  vs NeRF c2w). It is reused by A10, A10.5, and A11.5a.
- **Verifiable result:** renders a held-out view of the toy scene.
- **Notes:** coarse/fine sampling optional and ungraded; Instant-NGP/Zip-NeRF as
  one-paragraph context (the end of the NeRF arc before 3DGS).

### A10 - 3D Gaussian Splatting  **[Core]**
- **Implements:** the differentiable rasterizer FORWARD pass - 3D Gaussians
  (position, covariance via scale+rotation, opacity, SH/color), the EWA Jacobian
  projection to a 2D covariance, tile/alpha compositing; fit a small scene from
  posed images by gradient descent (autograd through the forward, no custom CUDA).
- **Depends on:** A9 (shares camera geometry & volumetric intuition).
- **Verifiable result:** fits a handful of posed images of a toy scene; renders a
  novel view; visibly faster inference than the A9 NeRF (5-20x in pure PyTorch).
- **Notes:** gradcheck the scale+rotation→covariance factorization; densification
  is read-and-understand only (a simple prune suffices); EWA/Zwicker required
  reading; 2DGS, Mip-Splatting, gsplat as context. Frame it as differentiable
  optimization over a 3D representation, close to bundle adjustment.

### A10.5 - Geometry foundation models  **[Mixed]**
- **Implements (build part):** a pointmap-regression head (DUSt3R-style) on a toy
  stereo pair - regress per-pixel 3D points in a shared frame from an image pair.
- **Survey part:** DepthAnything v2, Marigold (diffusion depth), MASt3R, VGGT
  (multi-view 3D in under a second).
- **Depends on:** A1, A2, A9.
- **Builds into shared lib:** pointmap/depth utilities in `nanovision.geometry`.
- **Verifiable result:** the pointmap head overfits a toy stereo pair; predicted
  points reproject consistently across the two views.
- **Why:** the learned replacement for the keypoint→COLMAP pipeline - the biggest
  gap relative to this learner's SfM/SLAM background.

### A11 - Modern detection & segmentation  **[Mixed]**
- **Implements (Core part):** DETR-style set prediction - the Hungarian bipartite
  matching loss and the query-based decoder head (reuse A1 cross-attention).
  Teach the matching cost (non-differentiable, for assignment) vs the training
  loss (differentiable, after assignment) explicitly.
- **Survey part:** the lineage DETR → DAB-DETR / DN-DETR → Deformable DETR →
  DINO-DETR → RT-DETR (read + run); Mask2Former (unified segmentation);
  GroundingDINO (open-vocab); promptable segmentation SAM → SAM 2 → SAM 3.
- **Depends on:** A1, A2.
- **Verifiable result:** the matching loss correctly assigns predictions to a toy
  set of boxes; the query decoder overfits a tiny detection task.

---

## Phase 3.5 - Autonomous-driving perception (nuScenes-mini substrate)

> **Substrate:** nuScenes `v1.0-mini` (~4GB, account + license click-through).
> A11.5a owns dataset setup ("step zero") and the loader/calibration utilities.
> All training here targets correctness + few-scene overfitting; real BEV training
> needs a cluster and will NOT fit a 4080. Each README sets that expectation.

### A11.5a - Camera geometry & the BEV transform  **[Core]**
- **Builds:** `nanovision.geometry` (pinhole project/unproject, the four SE(3)
  primitives `make_transform`/`apply_transform`/`invert_transform`/
  `compose_transforms`, `CameraRig`, `ipm_to_bev`) and
  `nanovision.data.nuscenes_mini` (loader + calib utils).
- **Implements:** the full intrinsic/extrinsic chain for the 6-camera nuScenes
  rig; the four-step lidar→camera projection chain; a temporal-offset exercise
  (naive single-ego-pose vs timestamp-correct projection - ~50ms offset is ~1.5m
  at highway speed); the ego-centric BEV grid definition (the shared contract for
  the whole module); ground-plane IPM to a first naive BEV. Pinhole projection is
  one-paragraph review for this audience.
- **Depends on:** A0 (and geometry intuition from A9/A10).
- **Verifiable result:** lidar points project into all 6 cameras and land on the
  right objects; the temporal-correct overlay is visibly tighter than the naive
  one; multi-cam images warp into a shared BEV grid; flat-ground breakage is
  visible and explained.
- **Notes:** nuScenes images are pre-undistorted (K is exact pinhole, no
  distortion terms); pyquaternion is scalar-first (w, x, y, z). State both.
- **Hard prerequisite for:** A11.5b, c, d, e.

### A11.5b - Lift-Splat-Shoot  **[Core]**
- **Implements:** per-pixel categorical depth distribution; the outer-product
  "lift" into a frustum point cloud; "splat" via the cumsum-trick voxel/pillar
  pooling into a BEV feature map (the designed pooling, implemented from scratch);
  a small lidar-depth-supervision task (project mini lidar to the image plane for
  sparse depth labels + an auxiliary depth loss, BEVDepth-style); a tiny BEV
  segmentation head.
- **Depends on:** A11.5a, A2.
- **Verifiable result:** overfits BEV vehicle-segmentation on a few nuScenes-mini
  scenes; the depth distribution visualization is sensible.
- **Notes:** lineage LSS → BEVDet → BEVDepth; BEVPoolv2 is a deployment pointer;
  GaussianLSS (continuous depth) noted.

### A11.5c - BEVFormer-style attention  **[Core]**
- **Implements:** BEV queries that pull from multi-cam image features via spatial
  cross-attention at projected 3D reference points - build the
  bilinear-sample-at-reference-points version first, add learned deformable
  offsets as a follow-on; temporal self-attention across consecutive frames.
  Reuse A1 attention + A11.5a projection + A3.5 temporal intuition.
- **Depends on:** A11.5a, A1, A11.5b (for comparison), A3.5.
- **Verifiable result:** overfits the same toy BEV task; the LSS-vs-BEVFormer
  contrast (depth-push vs query-pull) is the written takeaway.
- **Notes:** PETR (3D position embeddings + global cross-attention) is a cleaner
  ~30-line warm-up worth showing; DETR3D is the predecessor; sparse-query methods
  (Sparse4D/SparseDrive) now dominate detection while dense BEV persists for
  occupancy/mapping.

### A11.5d - 3D occupancy  **[Core, compact]**
- **Implements:** an occupancy head over a voxel grid (per-voxel occupancy +
  semantics) supervised by NeRF-style 2D rendering (RenderOcc/OccNeRF) - because
  nuScenes-mini has no official Occ3D voxel labels, supervision comes from 2D
  lidar depth + camera semantics, reusing the A9 alpha-compositing integral
  directly; the volumetric loss with mandatory class-imbalance weighting (free
  voxels are ~97% of the grid).
- **Depends on:** A11.5a, A9, A11.5b/c (BEV features feed the voxel head).
- **Verifiable result:** overfits voxel occupancy on a few scenes via rendering
  supervision; use a small synthetic grid (~32³ × 8) to fit 12GB.
- **Notes:** the NeRF↔occupancy duality (same geometry from opposite ends) is the
  spine; sparse/Gaussian occupancy and the RayIoU metric as context.

### A11.5e - Unified perception → prediction → planning  **[Mixed]**
- **Implements (build part):** a multimodal motion-prediction head mapping BEV
  features to future agent trajectories with a winner-take-all min-of-N loss
  (select the winner by minFDE at the endpoint, regress in agent-centric
  coordinates, extract agent features via RoI-align rather than a single cell).
- **Survey part:** the end-to-end differentiable stack framing (UniAD → VAD →
  DriveTransformer, the last running perception/prediction/planning queries in
  parallel); how a planning head bolts on. Full E2E planning is NOT built.
- **Depends on:** A11.5b/c, A11.5d.
- **Verifiable result:** the trajectory head overfits a few scenes.
- **Notes:** make the open-loop-metric critique a primary teaching point - the
  AD-MLP "ego status is all you need" result shows a velocity-only MLP matches
  full stacks because ~74% of nuScenes is straight driving; name NAVSIM and
  Bench2Drive as the real (closed-loop) evaluation standard.

---

## Phase 4 - Action & dynamics

### A12 - World models (RSSM / Dreamer)  **[Core]**
- **Implements:** a DreamerV3-style recurrent state-space model - deterministic
  (GRU) + stochastic (categorical) latent, the reconstruction objective, KL
  balancing and free bits as distinct mechanisms, symlog + two-hot as a
  first-class exercise, unimix and actor-critic return normalization, the
  straight-through estimator, and "imagine" rollouts in latent space; on a toy
  gridworld/control env.
- **Depends on:** A0, A1, A3.5; conceptual link to A5/A7 (generative dynamics).
- **Verifiable result:** the model imagines plausible latent rollouts;
  reconstructs observations; a simple policy learned in imagination beats random.
- **Reading:** video world models (Genie 2, DIAMOND, DreamerV4) - the pixel-space
  vs latent-control distinction; DreamerV4's flow-matching transformer is where
  the field is heading but needs far more compute than a 4080.

### A13 - VLA / embodied (capstone)  **[Core]**
- **Implements:** a VLM-style policy (A8) with a flow-matching action head (A6,
  with DDPM as the contrast) doing action chunking on a toy 2D manipulation/
  navigation task; the chunk-size H=1 vs H>1 ablation is the central quantitative
  deliverable; the discretized-action-token vs continuous-generation debate made
  explicit. Ties the course together: perception (A2/A4) + language interface
  (A8) + generative action (A5/A6) + (optionally) latent dynamics (A12).
- **Depends on:** A5/A6, A8, (A12 optional, strictly).
- **Verifiable result:** the policy completes the toy task; action chunking vs
  single-step ablation shows the benefit.
- **Anchors:** OpenVLA / OpenVLA-OFT and Octo as the studyable open references
  (RT-2 is proprietary and 55B).

---

## Phase 5 - Classical SLAM / localization (C++)

A later addition that fills the classical geometric SLAM canon the deep-learning course
skips. All mechanism code is C++17 with Eigen, exposed to the pytest green-red bar via
pybind11; visualization is Rerun (interactive plus headless `.rrd`); production libraries
(GTSAM, Open3D, g2o) appear as labeled comparison oracles, never as the graded
implementation. Full design, holes, tests, and conventions are in
`claude_notes/a14_classical_slam_plan.md` (expert-reviewed). Visualization-first by request:
each assignment ships an animated, steppable view, and every README ends with an "In
interviews" depth section.

### A14.0 - Lie groups for state estimation  **[Core]**
- **Implements:** SO(3)/SE(3) exp/log, hat/vee, left/right Jacobians and inverses, the
  adjoint, and box-plus/box-minus (right-perturbation convention, used module-wide).
- **Depends on:** A11.5a (matches its frame/naming conventions; no code shared across the
  language boundary - reimplementing SE(3) in C++ is intended practice).
- **Verifiable result:** exp/log round-trips and the adjoint identity to ~1e-9; the
  numerical-vs-analytic right Jacobian to ~1e-6. Viz: manifold vs naive pose interpolation
  (the naive blend leaves SO(3); det drops from 1, the manifold path stays rigid).

### A14.1 - KF / EKF / UKF  **[Core]**
- **Implements:** linear KF (Joseph form); EKF with a nonlinear process model + Jacobian and
  a range-bearing measurement model; UKF unscented transform; information form.
- **Depends on:** A14.0.

### A14.2 - EKF-SLAM  **[Core]**
- **Implements:** state augmentation on landmark init, the joint O(n^2) covariance update,
  NN + Mahalanobis-gate data association, on a 2D range-bearing world with loop closures.
- **Depends on:** A14.1. Framed as the superseded filter that factor graphs replaced.

### A14.3 - Multi-view geometry estimators  **[Core]**
- **Implements:** triangulation (DLT + nonlinear), normalized eight-point E/F, PnP, a RANSAC
  robust wrapper, and a composed two-view relative-pose front-end.
- **Depends on:** A14.0, A11.5a.

### A14.4 - Point-cloud registration (ICP)  **[Core]**
- **Implements:** point-to-point (Umeyama SVD) and point-to-plane ICP, correspondence +
  outlier rejection (the kd-tree is provided); Open3D/GICP as the comparison oracle.
- **Depends on:** A14.0.

### A14.5 - Pose-graph / bundle adjustment  **[Core]**
- **Implements:** SE(2)/SE(3) pose-graph residuals + analytic Jacobians, Gauss-Newton/LM,
  the Schur complement for landmark marginalization, a loop-closure edge; g2o/GTSAM oracle.
- **Depends on:** A14.0, A14.3.

### Reading-only note (a14_classical_slam/notes/)
IMU pre-integration and VIO, time-sync and extrinsic calibration, place recognition /
loop-closure detection, and a bridge from classical to the learned geometry models in the
repo (A10.5 DUSt3R/VGGT, A11.5 BEV).

## Reading-only notes (NOT built; bundled as Markdown notes in the repo)

- **Video generation:** flow matching (A6) in a spatiotemporal VAE latent (A7);
  why it won't fit a 4080; what to read.
- **VLA data engines / scaling:** Open X-Embodiment, DROID, the FAST tokenizer;
  how the data side now dominates model design.
- **Efficient attention:** FlashAttention / IO-aware attention (attached to A1).
- **State-space backbones:** Vision Mamba / SSMs (attached to A1).
- **MM-DiT and REPA** training techniques (attached to A7).
- **Video world models:** Genie 2, DIAMOND, DreamerV4 (attached to A12).

---

## Build cadence recommendation

Build and fully verify in this order, pausing after the first three for the
learner to calibrate house style and difficulty:

1. **A0, A1, A11.5a** (foundation from both ends + the unfamiliar AV plumbing).
2. Then the rest of Module 1 (A2, A3, A3.5, A4).
3. Then Module 2 (A5, A6, A6.5, A7).
4. Then Module 3 (A8, A9, A10, A10.5, A11).
5. Then the rest of Module 3.5 (A11.5b–e).
6. Then Module 4 (A12, A13).
