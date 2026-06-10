# Revised curriculum (decision-applied)

This is the fleshed-out coursework after the 19-agent validation
(`curriculum_review.md`) and your decisions:

- Adopt the 2026-consensus variant as the build core for each assignment, with the
  original as historical contrast.
- Add all four content items: geometry foundation models (new), VQ tokenizer
  (new), a video/temporal half-assignment (new), ConvNeXt folded into A2, plus
  efficient-attention/Mamba reading notes.
- Keep the autonomous-driving sub-module at five assignments.
- Keep A5 (diffusion) and A6 (flow matching) as two assignments.

Result: 22 built assignments across the same six modules, plus reading notes.
New or changed entries are marked. This plan has been propagated into the
canonical spec: `BUILD_ORDER.md`, `ARCHITECTURE.md` (§2, §3, §6), and
`BUILD_CHECKLIST.md` now describe this 22-assignment curriculum. This file remains
as the rationale and the per-assignment scope-change record.

## Dependency-ordered assignment list

Legend: [Core] build the mechanism; [Survey] guided/pretrained; [Mixed] both.
(NEW) = added by this revision. (CHANGED) = scope altered from the draft.

### Module 0 - Foundations
- A0 harness & primitives [Core] - deps: none. Unchanged.
- A1 transformer [Core] (CHANGED) - deps: A0. Build the LLaMA-style stack as core:
  RMSNorm, RoPE, SwiGLU; sinusoidal/learned PE and LayerNorm taught as historical
  contrast. GQA/MQA conceptually + a toy implementation. Attach reading notes:
  FlashAttention (IO-aware attention) and Vision Mamba / SSM backbones.

### Module 1 - Visual representations
- A2 ViT [Core] (CHANGED) - deps: A0, A1. Add register tokens to core; name the
  DeiT augmentation recipe as part of the mechanism; CLS-vs-mean-pool measurement;
  fold in a ConvNeXt block build (conv-vs-transformer story); 2D sincos / axial
  RoPE noted. DINOv2 + DINOv2-with-registers probe.
- A3 SSL, MAE + DINO [Core] (CHANGED) - deps: A1, A2. Original DINO stays the build
  core; add iBOT as a ~20-line extension; DINOv2 three-loss structure and I-JEPA as
  a conceptual page; linear probe is a DINO-vs-MAE-vs-random comparison.
- A3.5 video / temporal modeling [Core, compact] (NEW) - deps: A2, A3. Extend MAE
  to spatiotemporal tube masking; tubelet embedding; a tiny video ViT. Bridges to
  BEVFormer temporal attention (A11.5c) and world models (A12).
- A4 CLIP & open-vocab [Core] (CHANGED) - deps: A1, A2. Build SigLIP sigmoid loss
  as primary (trains at small batch on 12GB), softmax InfoNCE as historical
  predecessor; explain the SigLIP bias init; A4 builds the loss + zero-shot
  inference, not the encoder. OWL-ViT -> GroundingDINO open-vocab thread as required
  reading. Real-CLIP probe.

### Module 2 - Generative
- A5 diffusion, DDPM/DDIM [Core] (CHANGED) - deps: A0, A1. v-prediction required
  (eps-vs-v comparison); the score-epsilon (Tweedie) derivation as an explicit
  exercise; Min-SNR weighting and EDM preconditioning as reading; score-SDE on the
  must-read list. Trains for real on MNIST/shapes on a 4080.
- A6 flow matching & rectified flow [Core] (CHANGED) - deps: A5. Add OT coupling as
  a first-class build task; logit-normal timestep sampling as an ablation; reflow
  taught as foundational with MeanFlow / Rectified Diffusion flagged as the
  superseding few-step methods. The exact diffusion-flow equivalence is the spine.
- A6.5 VQ tokenizer [Core, compact] (NEW) - deps: A1, A2. VQ-VAE/VQ-GAN codebook +
  straight-through estimator (~100 lines); a tiny autoregressive prior over the
  discrete tokens (reuse A1). Unlocks the unified multimodal generation literature
  (Chameleon, Janus, LlamaGen). Contrast with the continuous KL-VAE of A7.
- A7 latent diffusion + DiT [Core] (CHANGED) - deps: A1, A5, A6, A6.5. Small KL-VAE,
  then a flow-matching DiT (not DDPM) in latent space with adaLN-Zero conditioning;
  MM-DiT and REPA in the reading note; KL-VAE-vs-VQ distinction explicit.

### Module 3 - Multimodal & 3D
- A8 VLM [Core] (CHANGED) - deps: A1, A2, A4. MLP projector as core + optional
  1-layer cross-attention resampler; explicit two-stage train (align then
  instruction-tune); AnyRes/tiling shape exercise; note frozen-encoder is a
  simplification.
- A9 NeRF [Core, compact] (CHANGED) - deps: A0. Volume-rendering integral derived
  cold; positional encoding motivated by showing the spectral-bias failure first;
  camera geometry stated as prerequisite (only conventions are new work);
  coarse/fine optional-ungraded; Instant-NGP/Zip-NeRF as context.
- A10 Gaussian splatting [Core] (CHANGED) - deps: A9. EWA Jacobian projection and
  scale+rotation->covariance (gradcheck it); alpha compositing ties to A9;
  densification read-and-understand only; EWA/Zwicker required reading,
  2DGS/Mip-Splatting/gsplat context; realistic 5-20x speed expectation.
- A10.5 geometry foundation models [Mixed] (NEW) - deps: A1, A2, A9. Build a
  pointmap-regression head (DUSt3R-style) on a toy stereo pair; survey
  DepthAnything v2, Marigold, MASt3R/VGGT. The biggest gap relative to your
  SfM/SLAM background: the learned replacement for the keypoint -> COLMAP pipeline.
- A11 detection & segmentation [Mixed] (CHANGED) - deps: A1, A2. Build core
  (Hungarian matcher + minimal query decoder on toy boxes) unchanged; teach
  matching-cost vs training-loss explicitly; survey lineage DAB/DN-DETR, RT-DETR,
  Mask2Former, GroundingDINO; SAM -> SAM2 -> SAM3.

### Module 3.5 - Autonomous-driving perception (nuScenes-mini)
- A11.5a camera geometry & BEV [Core] (CHANGED) - deps: A0. Add the temporal-offset
  exercise (naive vs timestamp-correct lidar->camera projection); name the four
  SE(3) primitives as deliverables; define the ego-centric BEV grid here; pinhole
  projection is one-paragraph review for this learner; nuScenes gotchas
  (pre-undistorted images, pyquaternion scalar-first) stated in text. Owns dataset
  step-zero.
- A11.5b Lift-Splat-Shoot [Core] (CHANGED) - deps: A11.5a, A2. Outer-product lift +
  cumsum-trick splat from scratch; add a small lidar-depth-supervision task
  (BEVDepth); lineage LSS -> BEVDet -> BEVDepth; GaussianLSS noted.
- A11.5c BEVFormer [Core] (CHANGED) - deps: A11.5a, A1, A11.5b, A3.5. Build the
  bilinear-sample-at-reference-points version first, deformable offsets as a
  follow-on; consider PETR as a cleaner warm-up; add DETR3D as predecessor; note
  sparse-query (Sparse4D/SparseDrive) dominance for detection. LSS-vs-BEVFormer is
  the spine.
- A11.5d 3D occupancy [Core, compact] (CHANGED) - deps: A11.5a, A9, A11.5b/c. Switch
  to NeRF rendering supervision (RenderOcc/OccNeRF) since mini has no Occ3D labels;
  this makes the A9 alpha-compositing reuse the spine; mandatory class-imbalance
  weighting; small synthetic grid (~32^3 x 8); sparse/Gaussian occupancy + RayIoU
  as context.
- A11.5e prediction -> planning [Mixed] (CHANGED) - deps: A11.5b/c, A11.5d. Build a
  multimodal trajectory head on frozen BEV features with a winner-take-all min-of-N
  loss (select by minFDE, agent-centric, RoI-align agent features); make the
  open-loop-metric critique (AD-MLP "ego status is all you need") a primary teaching
  point; name NAVSIM/Bench2Drive and DriveTransformer.

### Module 4 - Action & dynamics
- A12 world models, Dreamer/RSSM [Core] (CHANGED) - deps: A0, A1, A3.5. RSSM build
  core; elevate symlog/two-hot to a first-class exercise; separate KL balancing
  from free bits; add unimix, return normalization, straight-through estimator;
  video world models (Genie 2, DIAMOND, DreamerV4) as reading notes.
- A13 VLA capstone [Core] (CHANGED) - deps: A5/A6, A8, A12 (optional). VLM backbone
  + flow-matching action head (pi0-style); action chunking H=1 vs H>1 as the
  central deliverable; discretized-token vs continuous-generation debate explicit;
  anchor on OpenVLA/OpenVLA-OFT and Octo; flow matching default, DDPM contrast.

## Reading-only notes (not built)
- Video generation: flow matching in a spatiotemporal VAE latent (won't fit 4080).
- VLA data engines / scaling (Open X-Embodiment, DROID, FAST tokenizer).
- Efficient attention: FlashAttention / IO-aware attention (attached to A1).
- State-space backbones: Vision Mamba / SSMs (attached to A1).
- Video world models: Genie 2, DIAMOND, DreamerV4 (attached to A12).
- MM-DiT and REPA training tricks (attached to A7).

## New shared-library symbols to reserve

So later additions do not churn import paths, the shared-lib contract should
anticipate:

- `nanovision.transformer`: RoPE, RMSNorm, SwiGLU as first-class (A1); tubelet /
  spatiotemporal patch embedding (A3.5).
- `nanovision.quantize`: vector-quantization codebook + straight-through (A6.5).
- `nanovision.geometry`: pointmap / depth utilities for A10.5 alongside the existing
  pinhole + CameraRig + ipm_to_bev.
- `nanovision.primitives`: a ConvNeXt block (A2).

## Build cadence (unchanged first deliverable)

The first deliverable is still A0, A1, A11.5a, then a calibration pause. The only
change is that A1 and A11.5a now build to the revised scope above. After the pause,
the build order follows this file top to bottom.
