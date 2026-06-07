# A2 — Vision Transformers: validation report

**Date:** 2026-06-06  
**Topic:** A2 — Vision Transformers  
**Draft scope under review:** patch embedding; class token; positional embeddings; assemble ViT from A1 transformer block; train tiny ViT on CIFAR-10; probe notebook loading pretrained DINOv2 (timm), visualize emergent attention/segmentation; stretch — Swin windowed/shifted attention, patch-dropout.

---

## 1. Key concepts a student must learn

### Architecture core

**Patch embedding as a linear projection.** An image of shape `(H, W, C)` is split into non-overlapping patches of size `p×p`, flattening each into a vector of length `p²C`. A learned linear layer (equivalently, a Conv2d with kernel size `p` and stride `p`) projects each patch to the model dimension `d`. The sequence length is `N = HW/p²`. Students must be able to derive this, implement it in both forms, and understand why the two forms produce identical outputs.

**Class token prepended to the patch sequence.** A single learnable vector `[CLS] ∈ R^d` is prepended to the `N` patch tokens, giving a sequence of length `N+1`. After all transformer blocks, the `[CLS]` output is the image representation fed to the classifier head. The original ViT paper (Dosovitskiy et al., 2020) motivates this by analogy with BERT; however, mean-pooling over all patch tokens consistently matches or beats `[CLS]` at small scales and is simpler to reason about. Both must be understood.

**Positional embeddings and their interpolation.** ViT adds a learned 1D positional embedding table of shape `(N+1, d)` to the token sequence. The key insight is that these are learned per-position, not 2D-sinusoidal, and must be bicubically interpolated to handle higher-resolution inputs at inference time. Students should implement learned 1D PE, understand why 2D structure is implicit (row-major flattening), and be aware that 2D sincos PE (used in MAE, BEiT, DINOv2) removes the need for interpolation heuristics and generalizes better across resolutions. As of 2024, axial 2D RoPE (Heo et al., ECCV 2024) is the strongest current choice for resolution-agnostic ViTs and appears in production systems like Gemini's visual encoder (SigLIP 2).

**Transformer block reuse.** ViT stacks standard pre-norm transformer blocks (LayerNorm → MSA → residual; LayerNorm → MLP → residual). The MSA and MLP are identical to NLP transformers; there is no 2D-specific induction. This is the central pedagogical point: vision and language share one mechanism.

**Quadratic attention cost and its implication for resolution.** For an input of 224×224 with patch size 16, `N = 196` tokens, making global attention cheap. At 1024×224 (driving camera) with p=16, `N ≈ 4096`, making naive ViT expensive. Students should compute this, understand that Swin's O(N) windowed attention is a direct response, and recognize why MAE's masking (discarding 75% of tokens before the encoder) makes large-scale ViT training tractable.

**Inductive bias gap vs. CNNs.** ViT has no built-in locality or translation equivariance; every spatial relationship is learned from data. This is why ViT needs large pretraining data (JFT-300M in the original paper) or heavy augmentation (DeiT's RandAugment, MixUp, CutMix, stochastic depth) to generalize. On CIFAR-10 from scratch with standard training, even a tiny ViT-Tiny overfits badly relative to a small ResNet. This is a feature for the course — it makes the inductive bias tradeoff concrete and measurable.

### Mathematical core

- Self-attention: `Attn(Q, K, V) = softmax(QK^T / sqrt(d_k)) V`, complexity `O(N^2 d_k)`.
- Multi-head attention: split `d` into `h` heads of dimension `d/h`; concatenate outputs and project.
- MLP block: two linear layers with a GELU activation, expansion ratio typically 4×.
- Patch embedding as a strided convolution: `nn.Conv2d(C, d, kernel_size=p, stride=p)` — the weight matrix is equivalent to the patch projection matrix when reshaped.
- Positional embedding interpolation: bicubic resize of the embedding table when input resolution at inference differs from training.
- Register tokens: extra learnable tokens appended to the sequence (not corresponding to image patches) that absorb the high-norm "outlier" tokens otherwise observed in DINOv2 and similar self-supervised ViTs. Introduced by Darcet et al. (ICLR 2024).

---

## 2. Mechanisms worth implementing from scratch

Each mechanism below is self-contained enough to be verified by shape tests and `torch.autograd.gradcheck` on a tiny synthetic input, and by overfitting a batch of 4–8 images on CIFAR-10 or a random dataset.

### 2a. Patch embedding

**Implementation:** `nn.Conv2d(in_channels=3, out_channels=d, kernel_size=p, stride=p)` plus a learned `[CLS]` token and a learned positional embedding table. Verify:

- Output shape `(B, N+1, d)` for a batch of CIFAR-10 images (32×32, p=4 → N=64).
- Gradient flows through all parameters (`gradcheck` with `float64`, sequence dimension collapsed to scalar via sum).
- Mean-pool vs. CLS-pool: replace the classifier head and measure loss descent on a single batch of 8 images; both should overfit to zero training loss.

**Minimal problem:** 32×32 CIFAR-10 images, patch size 4, `d=64`, 2 transformer blocks, 4 heads.

### 2b. Learned 1D positional embedding with interpolation

**Implementation:** `nn.Embedding(N+1, d)` initialized from a range. Then implement the bicubic resize needed to transfer weights to a longer sequence.

**Verifiable task:** train tiny ViT at 32×32 (N=64), save the PE table, resize it to serve a 48×48 input (N=144), run a forward pass and confirm shapes are correct without re-initializing the patch projector.

### 2c. ViT assembled from A1 transformer blocks

**Implementation:** chain patch embedding → positional embedding addition → stack of pre-norm transformer blocks (imported from A1) → LayerNorm → CLS (or mean) pool → linear head.

**Verifiable task:** overfit 8 examples from CIFAR-10 to zero cross-entropy loss within 500 steps; plot training loss to confirm monotonic decrease. This test verifies correct data flow through the full model.

### 2d. Windowed attention (Swin)

**Implementation:** partition the token grid into non-overlapping windows of size `w×w`, apply self-attention within each window independently, shift windows by `(w/2, w/2)` in alternating blocks and apply a cyclic-shift mask.

**Verifiable task:** 2×2 window on a 4×4 token grid (16 tokens, 4 windows); check that attention is block-diagonal with no cross-window leakage; then shift and verify attention pattern changes. `gradcheck` on the attention weights. This is harder to implement correctly than global attention — the cyclic shift + masking is where bugs typically live.

### 2e. Patch dropout (optional stretch)

**Implementation:** randomly drop a fraction `r` of patch tokens (not CLS) before the transformer blocks, analogous to MAE's masking strategy. Restore original positions at the decoder side for reconstruction, or simply skip restoration and train a classification head on the surviving tokens.

**Verifiable task:** overfit 8 images with r=0.5 (half the patches dropped); compare wall-clock time per step vs. r=0; verify that gradients still flow to the patch projector for surviving tokens only.

---

## 3. Assessment of draft scope

### What is right

- Patch embedding, class token, positional embeddings are the correct starting trio.
- Assembling ViT from A1 transformer blocks is the right pedagogical move: it forces students to see that ViT is NLP transformer + patch tokenizer, nothing more.
- Loading DINOv2 (or DINO) from timm and visualizing attention maps is high-value: it concretely shows emergent segmentation properties and motivates the SSL modules later.
- Swin windowed/shifted attention is worth covering, though its status has changed (see below).
- Patch dropout as a stretch is sound.

### What is missing

**Register tokens (Darcet et al., ICLR 2024) — should be in the main scope, not a stretch.**  
Register tokens are now standard in all new DINOv2-family models (DINOv2-reg, released 2024; also used in SigLIP 2 and other production ViTs). The concept is simple enough to implement in one line (append `k` learnable tokens, discard after encoding) and directly observable in the visualization notebook: DINOv2 without registers shows blotchy high-norm artifacts in background patches; DINOv2-reg does not. This is a two-paragraph concept with a one-line code change and a striking visual payoff. It belongs in the main notebook, not as a footnote.

**CLS vs. mean-pool comparison — should be an explicit experiment.**  
The draft names the class token but does not tell students to compare it against mean pooling. The literature is clear: mean pooling equals or beats CLS at small scale, and many production ViTs (MAE, BEiT, SimMIM) use mean pooling. A two-line change and re-run of the overfit experiment makes this concrete in minutes.

**Strong augmentation / regularization as a prerequisite for ViT on small data — needs explicit treatment.**  
The draft says "train a tiny ViT on CIFAR-10" without specifying the training recipe. If students use vanilla SGD + random crop/flip, the model will not converge meaningfully and they will wrongly conclude ViT is simply bad on small data. The correct lesson, established by DeiT (Touvron et al., ICML 2021), is that ViT without DeiT's augmentation/regularization (RandAugment, MixUp, CutMix, label smoothing, stochastic depth) trains 6+ points worse than with it on ImageNet-1k. Even for the overfit-one-batch exercise this does not matter, but for any "train on CIFAR-10" goal the recipe is the mechanism being tested, not just the architecture. The scope should name the required recipe explicitly.

**Positional embedding interpolation and resolution generalization — undersold.**  
Students are building on this model in later modules (CLIP, VLMs, detection). Those all use ViT backbones at higher resolution than they were trained. The bicubic PE interpolation trick should be a named verifiable task, not left implicit.

**2D sincos positional embedding (used in MAE, DINOv2, BEiT) — should be mentioned.**  
The original ViT uses learned 1D PE. All major ViTs from 2022 onward (MAE, BEiT, DINOv2) use 2D sinusoidal PE because it interpolates better across resolutions without the bicubic heuristic. Students should at minimum know this variant exists; implementing it as an alternative to learned 1D PE is a 20-line exercise.

### What is outdated or mis-emphasized

**"Train a tiny ViT on CIFAR-10" is a reasonable exercise but the framing is problematic.**  
CIFAR-10 at 32×32 with patch size 4 is a legitimate tiny-scale sanity check, and a ViT-Tiny (d=192, 12 heads, 12 layers) can reach ~72–75% with a DeiT recipe after 300 epochs, which is 15 points below a small ResNet and 10 below a ConvNeXt-Tiny. The exercise is useful if the takeaway is the inductive-bias deficit and the need for augmentation, not if it is presented as "ViT achieves X% on CIFAR-10." The scope should frame this explicitly.

**Swin is important context, but its centrality has declined.**  
Swin V1 (Liu et al., 2021) was the dominant backbone for detection and segmentation from 2021–2023. As of 2025–2026, isotropic ViT backbones have largely replaced Swin-style hierarchical designs in the highest-performing systems (DINOv2 + neck, SAM2, ViT-Det, EVA-02). ConvNeXt-V2 (CVPR 2023) also matches or beats Swin-V2 at the same parameter count with simpler implementation. Swin is still worth understanding as the canonical hierarchical ViT and as background for dense prediction, but it should not be the main stretch — it should be taught alongside ConvNeXt as an illustration of the isotropic vs. hierarchical tradeoff. The draft currently presents Swin as the natural extension; a more honest framing is "Swin won 2021–2023; isotropic ViT + hierarchical neck wins 2024–2026."

**DINOv2 probe notebook should include DINOv2-with-registers (2024 variant).**  
timm ships `vit_base_patch14_reg4_dinov2.lvd142m` (the register variant). The notebook should load both, visualize attention maps from both, and show the difference. This requires zero extra implementation and is one of the most visually compelling demonstrations in modern computer vision.

**Stretch "patch dropout" is better framed as "token masking."**  
Patch dropout in training (a feature of `timm`'s ViT) and masked autoencoding (MAE, He et al. 2022) share the same mechanism. The stretch exercise should be named "token masking" or "random patch dropping" and linked explicitly to MAE, which the course presumably covers later.

### Recommended additions

1. **Register tokens:** add to main scope; implement as 4 learnable tokens appended after CLS, discarded after transformer blocks.
2. **CLS vs. mean-pool experiment:** explicit two-line comparison during the overfit exercise.
3. **DeiT training recipe:** name it explicitly in the CIFAR-10 training exercise; the augmentations are the lesson, not just the architecture.
4. **2D sincos PE:** name it and optionally implement as an alternative; link to MAE.
5. **DINOv2-with-registers vs. plain DINOv2:** show both in the probe notebook.
6. **PE interpolation:** make it an explicit verifiable mini-exercise.

### Recommended cuts / de-emphasis

- Swin as a stretch: keep as context for hierarchical vs. isotropic discussion, but do not position it as the natural next step.
- "Train to convergence on CIFAR-10" framing: scope to overfit-one-batch + short training run with DeiT recipe; full convergence is not the goal per course philosophy.

---

## 4. Connections to other course topics

**Feeds A1 (transformer block).** ViT is the first architecture that uses the A1 block on a structured 2D input. Any bug in A1 shows up immediately in A2.

**Feeds SSL (DINO, MAE, DINOv2).** The visualization notebook in A2 is a preview of the SSL module. Token masking (MAE) reuses the patch embedding with a mask; DINO's self-distillation operates on the CLS token output. Register tokens are a concept that originates in SSL training dynamics.

**Feeds CLIP / VLMs.** CLIP uses a ViT vision encoder (ViT-B/32, ViT-L/14) directly. The CLS output is the image embedding. Students who understand ViT output structure understand what CLIP's image encoder returns. PE interpolation becomes critical when VLMs process images at multiple resolutions.

**Feeds dense prediction / detection.** Hierarchical backbones (Swin) and ViT + FPN necks (ViT-Det) both start from ViT concepts. Students must understand patch token spatial arrangement — that the `(i, j)` patch corresponds to a specific grid position — to understand how detection heads read from ViT feature maps.

**Feeds LSS / BEVFusion.** In camera-based BEV perception, the image backbone is typically a ViT (DINOv2 frozen or EVA-02 fine-tuned) or a Swin. The neck that lifts image features into 3D (LSS, BEVDepth) consumes per-patch features, not just CLS. Students need to know that ViT patch tokens are spatially arranged and can be reshaped to `(H/p, W/p, d)` feature maps.

**Feeds SAM / SAM2.** SAM's image encoder is a ViT-H with windowed attention in early blocks. SAM2 uses a hierarchical image encoder (Hiera). Both are direct descendants of the Swin/ViT tradeoff covered in A2.

---

## 5. Must-read sources

1. **Dosovitskiy et al., "An Image is Worth 16x16 Words" (2020/2021, ICLR 2021).** arXiv:2010.11929. The original ViT paper. Read sections 3 (model description) and 4 (experiments) carefully; the appendix has the full hyperparameter table and the JFT vs. ImageNet ablation that establishes the data-hunger problem.

2. **Touvron et al., "Training data-efficient image transformers & distillation through attention" — DeiT (ICML 2021).** arXiv:2012.12877. Essential companion to the original ViT: establishes that strong augmentation (RandAugment, MixUp, CutMix) and stochastic depth close most of the gap vs. CNNs without large-scale pretraining. Any CIFAR-10 training exercise in the course should use this recipe.

3. **Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows" (ICCV 2021).** arXiv:2103.14030. The canonical hierarchical ViT; best background for understanding dense prediction backbones and the O(N) complexity argument. Read sections 3.1–3.3 on window partitioning and cyclic shifting.

4. **He et al., "Masked Autoencoders Are Scalable Vision Learners" — MAE (CVPR 2022).** arXiv:2111.06377. Shows that ViT + 75% patch masking trains faster and more effectively than supervised ViT. The masking trick is the same as patch dropout in the draft scope. Read section 2 for the architecture; figure 1 for the intuition.

5. **Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision" (2023).** arXiv:2304.07193. The reference for emergent ViT features. Trains ViT-g on LVD-142M with a combination of DINO + iBOT + SwAV losses. Key results: zero-shot depth, segmentation, and classification from frozen features. The model used in the probe notebook.

6. **Darcet et al., "Vision Transformers Need Registers" (ICLR 2024).** arXiv:2309.16588. Explains why DINOv2 attention maps have background artifacts, introduces register tokens as a fix, and reports +20 points on object discovery tasks. One-page concept; critical for the visualization notebook and any serious use of ViT in dense prediction.

7. **Heo et al., "Rotary Position Embedding for Vision Transformer" (ECCV 2024).** arXiv:2403.13298. Applies 2D axial RoPE to ViT and demonstrates consistent improvements over learned absolute PE at resolutions above 224. Worth assigning as background for students who will use ViTs in multi-resolution pipelines (VLMs, detection).

**Omissions in the draft:**  
The draft implies using DINOv2 without naming the DINO (v1) paper (Caron et al., 2021, arXiv:2104.14294), which is the cleaner illustration of attention-based segmentation. DINO v1 should be referenced in the visualization notebook section because its attention heads produce cleaner segmentation maps at small model scale and the mechanism (CLS token self-attention) is simpler to explain than DINOv2's combined training objective.

---

## 6. 2024–2026 developments that change how this should be taught

### Register tokens are now standard

From 2024 onward, all new ViT checkpoints intended for dense prediction tasks (DINOv2-reg, SigLIP 2, EVA-02 variants) include register tokens. Students loading a pretrained ViT from timm should be taught to check for register tokens and understand their effect on the attention map. The visual difference is dramatic and motivates the concept in about two minutes. This was not a consideration in 2021 and should now be in the main syllabus.

### Isotropic ViT has largely displaced hierarchical backbones for high-quality vision

Between 2021 and 2023, Swin-style hierarchical ViTs dominated leaderboards for detection and segmentation. From 2024, the pattern has reversed: ViT-Det (2022), EVA-02 (2023), and SAM2 (2024) all use isotropic ViT backbones with separate detection necks (FPN, etc.), and DINOv2 frozen features transferred to detection with a simple MLP neck outperform Swin-based systems in many zero-shot settings. The course should present Swin as a historically important architectural idea (windowed attention + patch merging = O(N) complexity) rather than as the recommended path forward.

### 2D RoPE is replacing learned absolute positional embeddings

Axial 2D RoPE (Heo et al., ECCV 2024) improves resolution generalization without the bicubic interpolation heuristic. SigLIP 2 (2024) and several VLM image encoders use it. Teaching learned 1D PE as "the" positional embedding is now slightly misleading; the topic should end with "and here is why DINOv2 / SigLIP / modern systems use 2D sincos or 2D RoPE instead."

### FlexiViT / NaViT: patch size is no longer fixed

FlexiViT (Beyer et al., CVPR 2023, arXiv:2212.08013) trains a single ViT with randomized patch size, enabling a compute-budget trade-off at inference. NaViT (2023, arXiv:2307.06304) packs patches from multiple images of different native resolutions into one sequence. These ideas are not essential for A2 but are important when students reach VLM modules (LLaVA, Gemini), which routinely process images at variable resolution. A brief mention with a pointer is appropriate.

### The "ViT needs huge data" narrative is partially obsolete

DeiT (2021) challenged it with augmentation. MAE (2022) showed that ViT-H could be pretrained on ImageNet-1k alone. By 2024, ViT-S trained with DINOv2-style objectives or with MAE pretraining reaches competitive accuracy on CIFAR-100 without JFT. The course should not repeat the 2021 received wisdom that ViT requires Google-scale data; it should say "ViT without inductive bias requires either large data or a strong self-supervised pretraining objective, both of which are now accessible via standard recipes."

### "Do all ViTs need registers?" — an active question

A March 2026 paper (arXiv:2603.25803) performs a cross-architectural reassessment and finds that the register benefit does not universally apply to all models and training objectives. A separate May 2026 paper (arXiv:2506.08010) proposes test-time registers that do not require retraining. The course does not need to resolve this debate, but students should be told that the design space is open.
