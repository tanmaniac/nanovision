# Curriculum review: validating the nanovision subject list

The original 19-assignment list was drafted by a single model. To validate it, I
ran 19 research agents in parallel (one per concept, plus one whole-curriculum
audit), each grounding its findings in 2026 sources via web search. Each agent
identified the key concepts a student must learn, the from-scratch mechanism worth
building on a tiny problem, and an opinionated critique of the draft scope. The
per-topic reports are in `docs/research/`; this file is the synthesis.

A caveat on sources: the per-topic reports were written by a smaller model and
some cite specific arXiv IDs or dates that should be spot-checked before they go
into a learner-facing handout. The method names and the conceptual corrections
below are reliable; treat the exact citation strings in `docs/research/*.md` as
leads to verify, not as verified references.

## Verdict

The list is sound in shape. No assignment is fundamentally misconceived, and the
build-the-mechanism-on-a-tiny-problem framing holds up topic by topic. The
corrections cluster into four recurring themes plus a set of structural questions
about coverage. Nothing here blocks starting the build; most changes are scope
refinements to individual assignments, and the structural questions only affect
assignments well past the first deliverable (A0, A1, A11.5a).

## Four cross-cutting themes

1. Teach the 2026 consensus, not the 2020-2022 original. The single most common
   finding. Several assignments default to the first version of an idea when a
   later, often simpler, variant is now standard and frequently fits the 12GB
   budget better: RMSNorm/RoPE/SwiGLU over LayerNorm/sinusoidal/GELU (A1), SigLIP
   sigmoid loss over softmax InfoNCE (A4), v-prediction over epsilon-prediction
   (A5), flow-matching DiT over DDPM DiT (A7), register tokens in ViT (A2).

2. The diffusion-to-flow-matching unification is a spine, not three separate
   topics. A5, A6, A7, and A13 all touch it. The score-epsilon equivalence
   (Tweedie) in A5, the exact Gaussian-path equivalence in A6, the flow-matching
   denoiser in A7, and the flow-matching action head in A13 should be taught as
   one connected thread so the learner sees diffusion and flow matching as two
   views of one object.

3. The 12GB ceiling should drive method choice, not just hyperparameters. SigLIP
   trains at small batch where InfoNCE fails; a synthetic 32^3 occupancy grid
   fits where a full Occ3D grid does not; a pure-PyTorch splatter is 5-20x faster
   than the NeRF, not the 100-1000x of CUDA. Where the budget changes which method
   is the right thing to teach, the assignment should say so.

4. Some draft "build on a few nuScenes-mini scenes" targets need a feasibility
   pass. The sharpest case: nuScenes-mini's 10 scenes have no official Occ3D
   occupancy labels (A11.5d), so that assignment must switch to NeRF-style 2D
   rendering supervision (lidar depth + camera semantics, both present in mini)
   rather than supervised voxel labels.

## Per-assignment findings

For each: whether the build core stands, the key concepts to learn, and the top
changes. Detail and sources are in the matching `docs/research/` file.

### A1 transformer (`a01_transformer.md`)
Core stands. Key concepts: attention as content-addressable lookup with the
sqrt(d) scaling, the pre-norm block, relative position via rotation.
Changes: promote RMSNorm and RoPE to core (every later module assumes the
LLaMA-style stack), SwiGLU at least a warm stretch, add GQA/MQA conceptually. The
"upgrade GPT-2 to the 2026 consensus stack" framing is the cleanest spine.

### A2 ViT (`a02_vit.md`)
Core stands. Key concepts: ViT is just A1 over patch tokens (only the patchifier,
class token, and positional embedding are vision-specific); no inductive bias means
small-data training needs the DeiT augmentation recipe; register tokens clean up
attention maps. Changes: register tokens to core, name the DeiT recipe as part of
the mechanism, add a CLS-vs-mean-pool measurement, demote Swin to context, mention
2D sincos/axial-RoPE positional schemes.

### A3 SSL, MAE + DINO (`a03_ssl.md`)
Core stands; original DINO remains the right build target. Key concepts: the
collapse problem and why centering + sharpening + EMA teacher together avoid it;
MAE's asymmetric encoder-decoder and the load-bearing 75% mask ratio; the family
map (contrastive / distillation / masked-pixel / masked-latent). Changes: add iBOT
as a ~20-line extension (explains the DINOv2 patch objective), give DINOv2's
three-loss structure and I-JEPA a conceptual page, make the linear probe a
DINO-vs-MAE-vs-random comparison (the gap is the lesson).

### A4 CLIP (`a04_clip.md`)
Core stands but the loss should change. Key concepts: symmetric InfoNCE on the NxN
matrix with all-but-diagonal as negatives, hence batch-size dependence and a
learnable temperature; SigLIP's sigmoid loss removes the global softmax and works
at small batch; zero-shot via prompt embeddings. Changes: teach SigLIP as the
primary loss (it actually trains on a 12GB card), InfoNCE as the historical
predecessor; A4 builds the loss and zero-shot inference, not the encoder (that is
A2); explain the SigLIP bias init; make the OWL-ViT to GroundingDINO open-vocab
thread required reading.

### A5 diffusion (`a05_diffusion.md`)
Core stands; algorithm sequence and MNIST target are right. Key concepts: the
score-epsilon equivalence via Tweedie (unifies the ELBO and score views), the
probability-flow ODE behind DDIM, v-prediction's better conditioning. Changes:
promote v-prediction to required with an eps-vs-v comparison; make the
score-epsilon derivation an explicit exercise so A6 follows naturally; add Min-SNR
loss weighting (one line, faster convergence) and EDM preconditioning as reading;
put the score-SDE paper on the must-read list.

### A6 flow matching (`a06_flow_matching.md`)
Core stands. Key concepts: the conditional flow matching objective (analytic
constant-velocity target, simpler than score matching), OT coupling as the
straightening mechanism, the exact diffusion-flow equivalence for Gaussian paths.
Changes: add OT coupling as a first-class build task (straightens without reflow;
it is the production default), add logit-normal timestep sampling as an ablation,
teach reflow as foundational but flag that MeanFlow / Rectified Diffusion have
superseded it for few-step generation.

### A7 latent diffusion + DiT (`a07_latent_dit.md`)
Core stands. Key concepts: the autoencoder + generative-prior split as a design
principle, adaLN-Zero conditioning (zero-init so each block starts as identity),
patchify/unpatchify as the spatial-to-sequence interface. Changes: train the DiT
with flow matching, not DDPM (matches SD3/FLUX, beats DDPM at equal FLOPs per SiT);
add MM-DiT and REPA to the reading note; state the KL-VAE (continuous, for
diffusion) vs VQ (discrete, for autoregressive) distinction explicitly.

### A8 VLM (`a08_vlm.md`)
Core stands. Key concepts: the visual-token interface (encoder -> connector ->
prepend to LM context), the connector families (MLP projector vs cross-attention
resampler vs early fusion), two-stage training (projector-align then
instruction-tune). Changes: build the MLP projector as core, add an optional
1-layer cross-attention resampler so both families are concrete; make the two-stage
split explicit; add an AnyRes/tiling shape exercise; flag that "frozen encoder" is
a pedagogical simplification (2024+ models fine-tune the ViT).

### A9 NeRF (`a09_nerf.md`)
Core stands; "prequel to splatting" framing is right for 2026. Key concepts: the
volume-rendering integral and its discretization into alpha compositing (derive it
cold), positional encoding as a spectral-bias fix, ray generation from camera
calibration (prerequisite knowledge for this learner, only the coordinate
conventions are new work). Changes: motivate positional encoding by showing the
failure first; state camera geometry is prerequisite; make coarse/fine sampling
optional-ungraded; add Instant-NGP/Zip-NeRF as one-paragraph context.

### A10 Gaussian splatting (`a10_gaussian_splatting.md`)
Core stands. Key concepts: the EWA Jacobian projection (Sigma_2D = J W Sigma_3D
W^T J^T, drop the third row/col) which reuses the perspective Jacobian from bundle
adjustment; the scale+rotation -> covariance factorization (gradcheck this);
front-to-back alpha compositing identical to NeRF transmittance. Changes: make
densification read-and-understand (no gradients through it; a simple prune
suffices); add EWA/Zwicker as required reading plus 2DGS/Mip-Splatting/gsplat
context; set realistic speed expectations (5-20x over the NeRF in pure PyTorch).

### A11 detection & segmentation (`a11_detection_segmentation.md`)
Build core (Hungarian matcher + minimal query decoder on toy boxes) is the right
size, no change. Key concepts: matching cost (non-differentiable, for assignment)
vs training loss (differentiable, after assignment); the DETR convergence problem
and the deformable-attention and denoising-query fixes; object queries as learned
positional slots. Changes (survey): fill the lineage with DAB-DETR/DN-DETR, add
RT-DETR (removes the "too slow" objection), Mask2Former for the segmentation half,
GroundingDINO for open-vocab, update the SAM arc to SAM -> SAM2 -> SAM3.

### A11.5a camera geometry & BEV (`a115a_camera_geometry_bev.md`)
Core stands; primitives are exactly what the later AV assignments need. Key
concepts: the four-step lidar-to-camera chain and the ~50ms lidar/camera temporal
offset (~1.5m error at speed); the ego-centric BEV grid as the shared contract for
the whole module; the nuScenes gotchas (pre-undistorted images, pyquaternion
scalar-first). Changes: add the temporal-offset exercise (naive vs
timestamp-correct projection); name the four SE(3) primitives as deliverables;
define the BEV grid here, not in A11.5b; treat pinhole projection as one-paragraph
review for this audience.

### A11.5b Lift-Splat-Shoot (`a115b_lift_splat_shoot.md`)
Core stands. Key concepts: the outer-product lift (per-pixel depth softmax times
context vector), the cumsum-trick splat (the designed pooling, worth implementing),
implicit vs explicit depth. Changes: add lidar depth supervision (BEVDepth) as a
required discussion plus a small task (project mini lidar to the image plane for
sparse depth labels + an auxiliary depth loss); spell out the LSS -> BEVDet ->
BEVDepth lineage; note GaussianLSS (2025) on continuous depth.

### A11.5c BEVFormer (`a115c_bevformer.md`)
Core stands; LSS-vs-BEVFormer (depth-push vs query-pull) is the right spine. Key
concepts: geometry-as-attention-prior (project 3D reference points into cameras,
sample there), spatial cross-attention as depth-free multi-view fusion, temporal
self-attention as O(1) recurrent history. Changes: build the
bilinear-sample-at-reference-points version first, add learned deformable offsets
only as a follow-on (full deformable attention fails silently from scratch);
consider teaching PETR first as a cleaner ~30-line warm-up; add DETR3D as
predecessor; tell the learner sparse-query methods (Sparse4D/SparseDrive) now
dominate detection while dense BEV persists for occupancy/mapping.

### A11.5d 3D occupancy (`a115d_occupancy.md`)
Core stands only with the supervision change. Key concepts: the NeRF-occupancy
duality (the A9 alpha-compositing integral is the occupancy rendering formula),
severe class imbalance (free voxels ~97%), voxel resolution as the memory
constraint. Changes: make NeRF rendering supervision (RenderOcc/OccNeRF) the
primary target because mini has no Occ3D labels; this promotes the A9 connection
from a footnote to the spine; mandate class-imbalance weighting; use a small
synthetic grid (~32^3 x 8); note sparse/Gaussian occupancy and RayIoU as context.

### A11.5e prediction -> planning (`a115e_pred_planning.md`)
Core stands. Key concepts: the multi-future / mode-averaging problem and the
winner-take-all min-of-N loss; query-based agent representation enabling end-to-end
gradient flow; the open-loop nuScenes planning metric is not a valid measure of
perception-conditioned planning. Changes: build the multimodal trajectory head on
frozen BEV features (select winner by minFDE, regress agent-centric, RoI-align the
agent features); make the open-loop critique a primary teaching point (AD-MLP "ego
status is all you need" - a velocity-only MLP matches full stacks because 74% of
nuScenes is straight driving); name NAVSIM/Bench2Drive as the real eval standard
and DriveTransformer as the parallel-query successor to UniAD.

### A12 world models (`a12_world_models.md`)
Core stands; DreamerV3 RSSM is the right buildable target. Key concepts: the RSSM
split (deterministic GRU state + stochastic categorical latent), KL balancing and
free bits as distinct mechanisms, symlog + two-hot as the core robustness
contribution. Changes: elevate symlog/two-hot to a first-class exercise; separate
KL balancing from free bits; add unimix, return normalization, and the
straight-through estimator (omitting them breaks training); put video world models
(Genie 2, DIAMOND, DreamerV4) in reading notes as the pixel-vs-latent distinction.

### A13 VLA (`a13_vla.md`)
Core stands. Key concepts: the VLM-backbone + flow-matching action head split
(pi0), action chunking and temporal ensembling as the fix for compounding error in
behavior cloning, the discretized-action-token vs continuous-generation debate.
Changes: make that debate explicit; set flow matching as default with DDPM as
contrast; anchor on OpenVLA/OpenVLA-OFT and Octo as studyable open references
(RT-2 is proprietary/55B); the chunk-size H=1 vs H>1 ablation is the central
quantitative deliverable; mark the A12 dependency strictly optional.

## Curriculum-level structural findings (`_curriculum_gaps.md`)

The audit's main point: the list covers what it includes well, but has gaps
relative to your "be able to read current papers" goal, with the sharpest one
sitting right next to your own background.

Proposed additions, in the audit's priority order:

1. Geometry foundation models (DUSt3R / MASt3R / VGGT / DepthAnything / Marigold).
   The audit calls this the most glaring gap given your SfM background: DUSt3R
   replaces the keypoint -> COLMAP pipeline with pointmap regression, VGGT does
   multi-view 3D in under a second. A buildable version is a pointmap-regression
   head on a toy stereo pair plus a DepthAnything/Marigold survey.
2. A VQ-VAE / VQ-GAN discrete tokenizer (the codebook + straight-through estimator,
   ~100 lines). Load-bearing for the unified multimodal generation literature
   (Chameleon, Janus, LlamaGen) the current list does not prepare you to read.
3. ConvNeXt / ConvNeXt V2 folded into A2 (the conv-vs-ViT backbone story; ~30
   minutes of code, recurs in every detection/segmentation paper).
4. Video / temporal modeling promoted from a reading note to a half-assignment
   (spatiotemporal tube masking on MAE), which bridges A3 -> BEVFormer temporal
   attention -> world models.
5. Efficient attention (FlashAttention) and state-space backbones (Vision Mamba),
   at least as reading notes, since they are assumed infrastructure in recent
   papers.

Proposed cuts/merges (more debatable, and they touch your stated AV interest):
merge A5+A6 into one diffusion-and-flow assignment; merge A11.5d+e into one "AV
outputs" assignment to trim the AV sub-module from 5 to 4. I would not act on the
AV trim without your sign-off, since deep AV/BEV is a stated reason you are taking
this course; the merge logic is "occupancy is incremental over BEVFormer for this
learner," which is an opinion, not a fact about your goals.

## What this changes for the build

Nothing blocks the first deliverable. A0 is unaffected. A1 and A11.5a get scope
refinements already captured above (the LLaMA-stack upgrades for A1; the
temporal-offset exercise, named SE(3) primitives, and BEV-grid definition for
A11.5a). The structural decisions (new assignments, merges) only matter for the
later phases, so they do not need to be settled before building starts - but the
shared-library contract should anticipate the likely additions (for example, a
discrete-tokenizer module path and a geometry-foundation module path) so we do not
churn import paths later.

Decisions taken: adopt the consensus variants wholesale, add all four content
items (geometry foundation models, VQ tokenizer, video/temporal, ConvNeXt into
A2), keep the AV sub-module at five, keep A5/A6 separate. These are propagated into
`BUILD_ORDER.md`, `ARCHITECTURE.md`, and `BUILD_CHECKLIST.md`; the resulting
22-assignment plan is `docs/revised_curriculum.md`.
