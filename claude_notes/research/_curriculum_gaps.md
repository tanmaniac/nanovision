# Curriculum gaps audit — nanovision 19-assignment list

Audited against the 2020-2026 CV/perception literature. The learner profile: senior perception/robotics engineer (stereo VO, SLAM, multi-sensor EKF fusion, multi-camera SfM, classical BEV), last built vision systems circa 2020 (EfficientDet/DETR era), goal is to be able to read current (2026) CV/perception papers. Hardware: single 12 GB RTX 4080; overfit-only or survey where full training is infeasible.

Current list for reference:
A0 harness & primitives · A1 transformer · A2 ViT · A3 SSL (MAE+DINO) · A4 CLIP · A5 diffusion (DDPM/DDIM) · A6 flow matching/rectified flow · A7 latent diffusion + DiT · A8 VLM (LLaVA-style) · A9 NeRF · A10 Gaussian splatting · A11 detection/segmentation (DETR + SAM survey) · A11.5a camera geometry & BEV · A11.5b Lift-Splat-Shoot · A11.5c BEVFormer · A11.5d 3D occupancy · A11.5e prediction→planning · A12 world models (Dreamer/RSSM) · A13 VLA capstone.
Reading-only: video generation, VLA data engines/scaling.

---

## 1. Major missing topics

### 1a. Modern ConvNets and the conv-vs-transformer story — MISSING (no treatment at all)

ConvNeXt (Liu et al., Jan 2022, CVPR 2022) and ConvNeXt V2 (Woo et al., CVPR 2023) are not covered anywhere. This is a significant gap for a learner coming from a 2020 baseline, because the canonical 2022-2025 narrative in backbone design is: "we modernized ResNet step-by-step toward ViT's training recipe, and it nearly matches ViT at equal compute — so why exactly does attention help, and where does it actually matter?" ConvNeXt V2 adds Global Response Normalization and shows that masked autoencoders work on fully-convolutional architectures too, which sheds light on why MAE (A3) works at all. Without this the learner cannot participate in conversations about backbone selection in detection/segmentation papers, where ConvNeXt-L/H remains a competitive baseline through 2025.

**Recommendation:** fold into A2 (ViT) as a ~0.5-assignment add-on, or add a short "architecture story" section to A11 (detection). Build a tiny ConvNeXt block (depthwise conv, inverted bottleneck, LayerNorm) and compare parameter efficiency vs a ViT patch. Fits 12 GB trivially.

---

### 1b. Image tokenizers / VQ-VAE / VQ-GAN — MISSING (structurally important, not covered)

VQ-VAE (van den Oord 2017) is foundational but not the concern here. The critical gap is VQ-GAN (Esser et al., CVPR 2021) and its 2024-era successors (LlamaGen, VAR, DALL-E 3 tokenizer improvements). These are load-bearing for two reasons:

1. Every autoregressive image-generation system (including LlamaGen 2024, which shows LLaMA-architecture transformers beat diffusion at class-conditional generation) uses a discrete image tokenizer. The learner will encounter this in any multimodal LLM paper that does unified text+image generation.
2. The unified modeling paradigm — treating images as discrete token sequences alongside text — is central to the 2024-2026 generation of VLMs (Chameleon, LlamaGen, Janus, Show-o). Without understanding VQ-GAN / codebook quantization the learner cannot read these papers.

A5 (DDPM) covers continuous latent diffusion, and A7 (latent diffusion + DiT) covers the VAE encoder, but neither explains discrete quantization or the codebook loss. The current VLM assignment (A8) is LLaVA-style (frozen image encoder + LLM), which deliberately avoids unified generation.

**Recommendation:** add a half-assignment (A4.5 or inside A7) that builds a minimal VQ-VAE/VQ-GAN: codebook, straight-through estimator, perceptual + GAN loss sketch. Codebook size 512 on a tiny 32×32 dataset is fine. Fits easily in 12 GB.

---

### 1c. Geometry foundation models (DUSt3R / MASt3R / VGGT / DepthAnything / Marigold) — MISSING, highly relevant to this learner

This is the most glaring gap relative to the learner's background. Between 2023-2025, the classical SfM pipeline the learner knows — keypoints → matching → COLMAP → MVS — has been substantially supplanted or augmented by:

- **Depth Anything v1 (CVPR 2024)** and **v2 (NeurIPS 2024)**: DINOv2-DPT trained on 62M+ unlabeled images; the de-facto monocular depth foundation model; cited in almost every AV and robotics paper.
- **Marigold (CVPR 2024, oral, best paper candidate)**: repurposes Stable Diffusion for monocular depth; shows that generative priors transfer to geometric tasks.
- **DUSt3R (CVPR 2024)**: replaces the full classical SfM pipeline with a pointmap-regression transformer; no feature matching, no camera priors, outputs dense 3D from any image pair.
- **MASt3R (ECCV 2024)**: extends DUSt3R with local feature matching head, making it a drop-in replacement for SuperGlue+COLMAP.
- **VGGT (CVPR 2025 Best Paper)**: feed-forward transformer that takes 1-to-hundreds of images and returns camera parameters, dense depth, point map, and 3D tracks in under 1 second. Trained on large 3D-annotated corpora.

For a learner who implemented multi-camera SfM in 2020, DUSt3R and VGGT represent a paradigm shift that is impossible to read without understanding the pointmap formulation and the cross-attention over multi-view tokens. DepthAnything v2 is now cited as a baseline in essentially every monocular depth paper. None of these appear in the current curriculum.

**Recommendation:** add a dedicated assignment (A9.5 or between A9/A10 in Module 3) — "geometry foundation models." Build task: implement the pointmap-regression idea on a tiny synthetic scene (toy camera pair, 2D → pointmap head over a ViT encoder, global alignment loss from DUSt3R). Survey DepthAnything and Marigold architecture. This is a BUILD assignment and fits in 12 GB.

---

### 1d. Efficient/long-context attention and state-space backbones (FlashAttention, linear attention, Mamba) — MISSING

FlashAttention (Dao et al., 2022) and FlashAttention-2 (2023) are now infrastructure: every serious transformer implementation uses them, and papers routinely cite sequence-length capability enabled by FA2. Vision Mamba / VMamba (2024, ICML) apply bidirectional SSMs as O(N) vision backbones, competitive with ViT-S/B at equal params. VideoMamba (ECCV 2024) applies the same idea to video. The learner will encounter these in every paper that cares about compute efficiency.

The curriculum teaches ViT (A1/A2) and writes naive attention. A reader encountering "FA2-accelerated attention" or "linear SSM backbone" in a paper will be lost about why the complexity matters and what IO-aware tiling does.

**Recommendation:** this does not need its own assignment. Add to A1 (transformer): one section on memory/compute complexity of naive vs tiled attention, with a from-scratch tiling sketch (no CUDA, just the algorithm). Add to A2 or A3: a survey note on Vision Mamba — what changes from ViT (selective scan replacing attention), and why O(N) vs O(N²) matters for video/high-res. No build needed for the hardware-level parts; conceptual understanding suffices.

---

### 1e. Video understanding and temporal modeling — MISSING (reading note insufficient)

Video is only mentioned as a "reading-only note" on video generation (diffusion video). But the learner will encounter temporal modeling in:
- VideoMAE / VideoMAE V2 (NeurIPS 2022 / CVPR 2023): masked autoencoders on spatiotemporal tubes — the dominant video pretraining recipe 2022-2025
- Video ViTs (ViViT, TimeSformer 2021): how to extend image ViT to handle time — factorized attention, tubelet embeddings
- VideoMamba (ECCV 2024): O(N) scanning over space-time
- The AV module (A11.5b-e) implicitly assumes temporal feature aggregation, and BEVFormer uses temporal self-attention explicitly

**Recommendation:** add a half-assignment or survey note (not reading-only, at minimum a survey with equations) after A3. Task: extend a tiny MAE to spatiotemporal tube masking on a 3-frame toy video. Conceptually bridges A3 (MAE) → A11.5c (BEVFormer temporal attention) → A12 (world models). Can be a light build (tube masking + 3D patch embedding) that fits in 12 GB.

---

### 1f. Open-vocabulary detection and grounding — MISSING

GLIP (2022), OWL-ViT (2022), Grounding DINO (ECCV 2024) are the standard zero-shot detection paradigm. Every robotics perception paper that does "detect anything the user asks for" uses one of these. The learner's A11 covers closed-vocabulary DETR and SAM (which is promptable but not language-conditioned). Without understanding how text embeddings are aligned with region proposals, a paper like "detect X from a text description" is opaque.

**Recommendation:** fold into A11 as a survey section (not a separate build). The learner already knows DETR architecture; adding "replace the fixed class head with a text encoder similarity" is conceptually small. One section explaining CLIP-backbone + box-head fusion, RPN-free vs two-stage grounding, and the GLIP contrastive grounding loss. No build needed.

---

### 1g. Point clouds / LiDAR networks and LiDAR-camera fusion — MISSING, relevant to AV module

The AV sub-module (A11.5a-e) is entirely camera-based. Real-world autonomous driving stacks — and the nuScenes benchmark that BEVFormer/LSS use — include LiDAR. BEVFusion (MIT, 2022) is the canonical LiDAR-camera fusion paper, combining LSS-style camera BEV with pillar-based LiDAR BEV. Without understanding PointPillars / VoxelNet (2018-2019, now background) and BEVFusion, the learner cannot read 2023-2025 AV papers that compare against or build on LiDAR-camera fusion.

Note: this is less critical than items 1a-1f because the learner's stated background is camera-centric, and the course explicitly covers BEV. But the gap will appear immediately when reading papers.

**Recommendation:** add a survey note to A11.5b (LSS) or A11.5d (occupancy): one section on PointPillars BEV representation and one on BEVFusion's dual-stream architecture. No build needed. The key mental model (LiDAR gives free metric depth; camera gives semantics; fuse in BEV) takes two pages to explain.

---

### 1h. Consistency models and diffusion distillation — UNDER-COVERED

A5 (DDPM/DDIM) and A6 (flow matching) cover the generation side. Consistency models (Song et al., 2023) and score distillation (CVPR 2023), along with LCM (latent consistency models, 2023), are how practitioners actually run diffusion at inference: 1-4 steps instead of 50-1000. These appear constantly in 2024-2025 papers (video generation, 3D generation, image editing). Marigold (mentioned above) also uses DDIM steps at inference.

**Recommendation:** add a survey section to A5 or A6 covering the consistency trajectory idea (self-consistency condition, why it enables few-step sampling), progressive distillation, and LCM. No build needed; the conceptual equations fit in one page.

---

### 1i. Mixture of Experts (MoE) in vision — LOW PRIORITY, survey sufficient

V-MoE (2021, NeurIPS), and more recently MoE layers in DiT-MoE (2024) and Janus-Pro (2025). Relevant primarily because the learner will see "MoE" in large model papers. Not worth a build assignment.

**Recommendation:** one paragraph in A7 (DiT) noting what MoE layers replace (FFN → set of expert FFNs + router) and why it matters for scaling. No build, no dedicated note needed.

---

### 1j. Retrieval / RAG for vision — LOW PRIORITY for this learner

Visual RAG (VisRAG 2024, ColPali 2024) is useful for document-understanding applications but not core to perception/robotics. The learner's goal is perception papers, not retrieval systems.

**Recommendation:** skip. Can be mentioned as a footnote in A8 (VLM).

---

### 1k. Quantization and deployment efficiency — SURVEY NEEDED

INT8/INT4 post-training quantization (GPTQ, AWQ for LLMs; QAT for vision), model pruning, and knowledge distillation are now assumed background in any paper that discusses deployment. The learner will see "4-bit quantized" or "distilled to ViT-S" and need a mental model.

**Recommendation:** one survey note (not a build) either in A0 or as a standalone note between A7 and A8, covering: floating-point formats (FP32/BF16/FP8/INT8/INT4), quantization-aware training vs post-training, scale-zero-point formulation, and knowledge distillation loss. No build; conceptual background only.

---

### 1l. Evaluation and benchmarking practices — IMPLICIT GAP

The curriculum builds mechanisms but does not address how current papers measure them. COCO AP, nuScenes NDS/mAP, FVD for video, FID for images, DPT for depth, ADE20k mIoU for segmentation — these appear on every paper's table. The learner knows some of these but not 2024-era ones.

**Recommendation:** fold into each module's assignment spec: one paragraph per assignment explaining the standard benchmark and metric. No separate assignment needed.

---

## 2. Over-coverage and redundancy

### A5 + A6 + A7: three consecutive generative assignments, partially redundant

A5 (DDPM/DDIM), A6 (flow matching/rectified flow), and A7 (latent diffusion + DiT) form three full assignments on generative models. For a learner whose goal is perception paper literacy, this is over-weighted. Specifically:

- **A6 (flow matching/rectified flow)** is the most redundant. Flow matching is a cleaner mathematical framing of diffusion but for reading perception papers the relevant outputs are "fast sampling" and "straighter trajectories." The learner who understands DDPM already grasps the core probabilistic framework; flow matching adds the ODE/SDE viewpoint. This could be a survey extension of A5 rather than a standalone assignment.
- **A7** is justified because latent diffusion + DiT is genuinely load-bearing for understanding how Stable Diffusion and Sora-style architectures work.

**Recommendation:** merge A6 into A5 as a Part 2 (Part 1: DDPM/DDIM score matching; Part 2: flow matching / rectified flow comparison). Save one assignment slot for a geometry foundation model (gap 1c) or video understanding (gap 1e).

### A9 (NeRF) and A10 (Gaussian splatting): sequential, justified, but NeRF build may be oversized

Both are justified for this learner given the 3D background. However, classic NeRF (original Mildenhall et al. 2020 with positional encoding + volumetric rendering) is now a stepping-stone to Gaussian splatting; the learner should spend more time on 3DGS and understand its connection to NeRF rather than spending a full assignment on NeRF. Instant-NGP (hash encoding, 2022) is the practical production version, not the original MLP NeRF.

**Recommendation:** condense A9 to a "NeRF core" (ray marching, volume rendering integral, positional encoding — one week, all buildable in 12 GB) and move the freed time to Instant-NGP as a reading survey. A10 (3DGS) stays full.

### A11.5a through A11.5e: five AV assignments, some granularity concerns

A11.5d (3D occupancy) and A11.5e (prediction→planning) are both narrow sub-topics of the AV stack. For a learner who already knows BEV geometry, the occupancy prediction idea (voxelizing BEV into height bins + semantic classes) is incremental over BEVFormer. Similarly, prediction and planning (A11.5e) is more of a systems integration topic than a new architectural mechanism.

**Recommendation:** merge A11.5d and A11.5e into a single "AV outputs" assignment. The freed slot could be used for the LiDAR survey (gap 1g) or for folding in UniAD (CVPR 2023 best paper) as the end-to-end unification example. This gives the module 4 assignments instead of 5, which is already substantial.

---

## 3. Module structure and ordering

### Overall 6-module structure

The 6-module split (foundations / visual representations / generative / multimodal+3D / AV perception / action+dynamics) is coherent. The dependency graph is sound: A0→A1→A2 is the correct transformer foundation; A3 (SSL) depends on A2 (ViT); A4 (CLIP) depends on A2+A3; diffusion (A5-A7) can run in parallel with A8 but is correctly placed before it because A8 (LLaVA) uses a frozen CLIP/ViT encoder trained by methods from A3/A4.

### Dependency issue: A8 (VLM) before A9/A10 (NeRF/3DGS)

The 3D module (A9-A10) comes after the VLM module (A8). This is acceptable but there is a slightly better ordering: geometry foundation models (gap 1c) naturally sit between A10 (3DGS) and A11 (detection), because DUSt3R/VGGT build on ViT encoders and volumetric/pointmap concepts from NeRF/3DGS. If gap 1c is added as A10.5, the sequence A9→A10→A10.5 gives a clean narrative: implicit NeRF → explicit Gaussian → feed-forward pointmap → detection.

### Video understanding placement

If a video understanding note (gap 1e) is added, it belongs after A3 (SSL/MAE) and before A5 (diffusion), because VideoMAE extends MAE to spatiotemporal tubes. This also benefits the AV module: when BEVFormer's temporal self-attention is introduced (A11.5c), the learner already has the mental model from VideoMAE.

### Suggested revised module ordering (changes only)

- After A3: add A3.5 "video and temporal modeling" (survey + light build, spatiotemporal tube masking)
- Merge A5+A6 into a single "DDPM+flow matching" assignment
- Add A4.5 or inside A7: VQ-VAE/VQ-GAN discrete tokenizer
- After A10: add A10.5 "geometry foundation models" (DUSt3R/VGGT build + DepthAnything/Marigold survey)
- Merge A11.5d+A11.5e into one "AV outputs" assignment
- Move ConvNeXt coverage into A2 or A11
- Add survey sections to A1 (FA2 complexity), A11 (open-vocab detection), A11.5b (LiDAR/BEVFusion)

---

## 4. Balance check: build vs survey vs reading note

### Things marked build that cannot fit 12 GB

None of the existing BUILD assignments appear to exceed 12 GB if implemented at the intended toy/overfit scale. The risk areas are:

- **A7 (latent diffusion + DiT)**: using a pre-trained VAE from Stability AI (430M SD-1.5) plus training a tiny DiT on MNIST latents is fine. Risk: if the assignment asks the learner to train a DiT on ImageNet latents at 256×256 this will not converge meaningfully in a single-GPU setting. Recommendation: specify training on class-conditional CIFAR-10 latents or a 64×64 toy dataset; the architecture mechanism is the goal, not quality.
- **A13 (VLA capstone)**: depends entirely on what VLA backbone is used. If it inherits a frozen LLaMA-7B + frozen ViT-L, VRAM consumption will be at or over 12 GB in bf16. Recommendation: specify a quantized (4-bit) or smaller backbone (LLaMA-3-1B or SmolVLM) for the build component; the action head is what the learner implements.

### Things currently survey/reading that should be built

- **Geometry foundation models** (gap 1c): the pointmap regression idea (A10.5 above) is a genuine build candidate — implementing the cross-view attention + pointmap head from scratch is the kind of mechanism that this course is designed for, and it directly exercises the learner's SfM intuition. Currently completely absent.
- **VQ-GAN tokenizer** (gap 1b): the codebook + straight-through estimator is a ~100-line build that illuminates how image tokens work, which is reading-load-bearing for unified VLMs. Should be a build, not a survey.

### Things marked build that could be surveys

- **A6 (flow matching)** as a standalone assignment: the mathematical distinction between flow matching and DDPM is small enough that a build-from-scratch on a 2D toy (Gaussian→banana-shaped distribution) can be incorporated into A5 rather than requiring a separate full assignment. The current structure treats it as a full build, which may be generous given the information density.

---

## 5. Prioritized recommendations

Ranked by impact on the stated goal (read 2026 CV/perception papers) relative to effort.

**1. Add A10.5: geometry foundation models (DUSt3R / VGGT / DepthAnything / Marigold)**
This is the highest-priority gap. The learner's SfM background makes this the most natural extension, and the DUSt3R/VGGT paradigm shift from classical pipelines to feed-forward transformers is one of the two or three biggest structural changes in 3D CV since 2020. Currently completely absent. Build: pointmap regression head on a toy stereo pair. Survey: DepthAnything v2 + Marigold.

**2. Merge A5+A6 and use the freed slot for VQ-GAN tokenizer (A4.5)**
Discrete image tokenization is load-bearing for the entire 2024-2025 unified multimodal generation literature (Chameleon, LlamaGen, Janus, Show-o). Currently absent. The VQ-GAN build is small (~100 lines, trivially fits 12 GB). The A5+A6 merge loses no conceptual coverage because the flow-matching math can be a Part 2 of the DDPM assignment.

**3. Add ConvNeXt to A2 (or A11)**
The conv-vs-transformer story is background for essentially every backbone comparison in 2022-2025 papers. Building a ConvNeXt block takes 30 minutes; the conceptual payoff (why depthwise conv + inverted bottleneck = approximate attention in a local window) is high. Currently absent.

**4. Add a video/temporal survey + light build after A3 (A3.5)**
VideoMAE, Video ViTs, tubelet embeddings. BEVFormer temporal attention (A11.5c) and world models (A12) both require temporal sequence modeling intuition. A half-assignment extending MAE to spatiotemporal tubes is sufficient.

**5. Merge A11.5d + A11.5e into one "AV outputs" assignment**
Occupancy and prediction/planning are incremental over BEVFormer for this learner. Merging them saves a slot and reduces the sense that the AV sub-module is disproportionately large (5 assignments covering roughly a 2-year niche vs. 2 assignments for all of generative vision pre-DDPM). Add a LiDAR/BEVFusion survey section to A11.5b instead of a dedicated assignment.

**6. Fold FlashAttention complexity and Mamba survey into A1/A2**
Not a new assignment — just two sections. FlashAttention tiling is infrastructure-level knowledge; Vision Mamba is a competing paradigm to ViT that appears in ~20% of 2024 backbone papers. Takes half a day to add; avoids the learner being puzzled by "O(N) SSM backbone" in every efficiency paper.

**7. Add open-vocabulary detection survey to A11**
Grounding DINO / GLIP / OWL-ViT. Two paragraphs in the A11 assignment spec explaining how a CLIP backbone + box head replaces fixed-class detection heads. No build needed; the learner already knows DETR from A11.

**8. Clarify VRAM budget for A7 (DiT) and A13 (VLA capstone)**
A7 should specify training on CIFAR-10 or toy 32×32 latents, not ImageNet-scale. A13 should specify a quantized or small-scale VLA backbone (e.g., SmolVLM + 4-bit quantization) to stay within 12 GB. These are not curriculum content changes but specification tightening to prevent wasted effort.
