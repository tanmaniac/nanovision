# A3 — Self-supervised learning (MAE + DINO): research validation

*Validated: 2026-06-06. Sources consulted via web search and paper fetches.*

---

## 1. Key concepts a student must learn

### The core problem: learning without labels

Self-supervised learning (SSL) defines a pretext task from the unlabeled data itself to produce transferable representations. For images, three families of pretext tasks dominate:

- **Contrastive** (SimCLR, MoCo): maximize agreement between differently-augmented views of the same image, push apart views of different images. Requires negative pairs or large queues.
- **Distillation / self-distillation** (BYOL, DINO): a student network tries to match the output of a teacher network that is an exponentially-moving-average (EMA) copy of the student. No negative pairs required.
- **Masked prediction** (MAE, iBOT): reconstruct or predict the content of masked-out regions. For images, the signal can be raw pixels (MAE) or patch tokens predicted in a feature space (iBOT, I-JEPA).

These families differ in *what they optimize*, *what information they require at training time*, and *what their frozen features are good at*.

---

### MAE: masked autoencoders

**Patchification and masking.** An image is divided into a grid of non-overlapping patches (e.g., 16×16 pixels for ViT-B/16 on a 224×224 image → 196 patches). A random 75% are masked out; only the remaining 25% visible patches are passed to the encoder.

**The 75% masking ratio is not arbitrary.** Natural images are highly spatially redundant — adjacent patches correlate strongly. At low masking ratios (e.g., 15%, the BERT figure) a model can interpolate from neighbors without learning semantics. At 75% the task requires genuine scene understanding. He et al. ablate this and show linear-probe accuracy rises monotonically with masking ratio up to ~75%, then plateaus.

**Asymmetric encoder-decoder.** The encoder (large ViT) processes only the ~49 visible tokens, with positional embeddings to encode which patches they came from. A lightweight decoder (typically 8 transformer blocks vs. 24 in the encoder) receives the full set of 196 tokens — the encoded visible tokens plus learned mask tokens (one shared learned embedding per masked position, with positional embeddings). The decoder reconstructs raw pixel values for each masked patch. Loss is MSE on masked patches only, normalized per patch.

**Why this is efficient.** Encoding only visible patches reduces encoder FLOP cost by ~4×, making large-model pretraining feasible on a single-node cluster.

**What the representations are.** MAE features are strong when fine-tuned end-to-end (ViT-H achieves 87.8% top-1 on ImageNet-1K with fine-tuning, using only ImageNet-1K for training). However, under linear probing MAE ViT-B achieves only ~67.5%, well below DINO's ~78%. MAE features require a nonlinear head to express their full capacity; they encode low-to-mid-level structure rather than the globally consistent prototypes that linear classifiers prefer.

---

### DINO: self-distillation with no labels

**Architecture.** Two networks with the same architecture: a student and a teacher. The teacher is never trained by backprop — it is an EMA of the student weights: `θ_t ← λ θ_t + (1-λ) θ_s`, with λ starting near 0.996 and increasing toward 1 over training.

**Multi-crop augmentation.** Each image produces two "global" crops (≥50% of image area) and several "local" crops (<50%). The teacher only sees global crops; the student sees all crops. The student is trained to predict the teacher's distribution on each global view given any view (global or local). This "local-to-global" correspondence forces the model to recognize the scene from a small context.

**The loss.** For each pair of student view s and teacher view t:

```
L = -sum_k  p_t(k) * log p_s(k)
```

where p_t and p_s are softmax distributions over K prototype dimensions (default 65,536). The student softmax uses a higher temperature (sharper → lower entropy) than the teacher.

**Collapse and the centering + sharpening fix.**

Two types of trivial collapse exist:
1. *Uniform collapse*: the model outputs equal probabilities for all prototypes regardless of input. No information is transmitted.
2. *Single-mode collapse*: the model assigns probability 1 to one prototype for every input.

DINO avoids both with two complementary operations applied to the *teacher output only*:

- **Centering**: subtract a running mean `c ← m*c + (1-m)*mean_batch(g_t)` from teacher logits before softmax. This kills single-mode collapse (if one dimension dominates, subtracting its mean brings it back toward zero) but by itself encourages uniform outputs.
- **Sharpening**: use a low temperature τ_t ≈ 0.04 for the teacher softmax, making its distribution peaked. This counters the uniform-distribution tendency that centering introduces.

The EMA teacher is the third pillar: because the student is not directly copied, the teacher is a smoothed, more stable target. This breaks the symmetry that would allow the student to trivially match a "noisy" copy of itself. The asymmetry (predictor on student side in BYOL; EMA + centering + sharpening in DINO) is what prevents collapse in the absence of negative pairs.

**Experimental collapse test.** Removing centering causes rapid convergence to single-prototype collapse within a few hundred iterations. Removing sharpening causes uniform-distribution collapse (all outputs equal). Both are verifiable in a minimal DINO run on CIFAR-10.

**Emerging properties.** DINO ViT features spontaneously produce semantically-meaningful attention maps — the [CLS] token attends to foreground objects without any supervision. This motivated the original paper's focus on ViTs and is a direct consequence of the local-to-global multi-crop objective.

---

### Why EMA teachers work at all (the deeper reason)

The EMA teacher is a smoothed historical average of the student. It acts as a *slowly-changing target*, preventing the student from "chasing its own tail." If the teacher were an instantaneous copy (stop-gradient in SimSiam), collapse avoidance depends entirely on the predictor bottleneck. With EMA the target is anchored to a temporally-smoothed representation, making degenerate solutions dynamically unstable.

---

### DINOv2: combining masked prediction and distillation

DINOv2 (Oquab et al., 2023) is the production version of this family. It trains with three simultaneous objectives:

1. **DINO loss** on [CLS] tokens — image-level self-distillation, as above.
2. **iBOT loss** on patch tokens — student sees masked patches, teacher sees full image; the student must predict the teacher's patch-level distributions for masked positions. This adds local discriminability and boosts dense tasks (segmentation, depth).
3. **KoLeo regularizer** — entropy regularizer on the batch-level feature distribution: `L_koleo = -(1/n) sum_i log(d_nn(i))` where `d_nn(i)` is the nearest-neighbor distance to sample i in the batch. This spreads features across the hypersphere, preventing feature clustering that hurts retrieval (+8% on retrieval benchmarks per the paper).

DINOv2 also uses a short high-resolution fine-tuning phase at the end. Its curated training set (LVD-142M, ~142 million images assembled from web data using similarity-based retrieval against curated sources) is critical — data quality matters as much as architecture.

**DINOv2 frozen ViT-g/14 is the de facto strong backbone** as of 2025 for probing and dense downstream tasks. It achieves performance comparable with weakly-supervised models on tasks including depth estimation, semantic segmentation, and classification — without any fine-tuning.

---

### I-JEPA: latent-space prediction without pixel reconstruction

I-JEPA (Assran et al., CVPR 2023) uses a ViT context encoder to predict the latent representations of target patches in the same image, given non-overlapping context patches. Unlike MAE, the target is a *feature vector* (from a separate EMA target encoder), not raw pixels. This biases the pretext task toward semantic, not textural, content. I-JEPA is more compute-efficient than MAE and avoids the texture-bias pitfall of pixel reconstruction, but its frozen features are somewhat weaker than DINOv2 on classification. V-JEPA extends this to video (Bardes et al., 2024).

---

### Family map

```
SSL families
├── Contrastive (negative pairs required)
│   ├── SimCLR (2020, Chen et al.) — large batch, no queue
│   └── MoCo (2020, He et al.) — momentum queue
├── Distillation / self-distillation (no negatives)
│   ├── BYOL (2020, Grill et al.) — predictor + EMA, no centering
│   ├── DINO (2021, Caron et al.) — EMA + centering + sharpening + multi-crop
│   └── DINOv2 (2023, Oquab et al.) — DINO + iBOT + KoLeo + curated data
├── Masked prediction (pixel-space target)
│   ├── MAE (2021, He et al.) — pixel reconstruction, asymmetric enc-dec
│   └── BEiT (2021, Bao et al.) — discrete token targets (dVAE)
├── Masked prediction (latent-space target)
│   ├── iBOT (2021, Zhou et al.) — DINO + patch-level distillation
│   └── I-JEPA (2023, Assran et al.) — latent prediction, no augmentation
└── Hybrid
    └── DINOv2 = DINO + iBOT + KoLeo
```

---

## 2. Mechanisms to implement from scratch

All implementations use only PyTorch autograd (no Hugging Face `transformers`, no `timm` MAE/DINO modules). CIFAR-10 or TinyImageNet is the recommended dataset; both fit in GPU memory and overfit in minutes.

---

### Mechanism 1 — MAE on CIFAR-10 patches

**Problem setup.** Use 32×32 CIFAR-10 images. Patch size 4×4 → 64 patches per image. Mask 75% (48 patches). Encoder: 4-block ViT. Decoder: 2-block ViT.

**Tasks to verify:**

| Task | What to check |
|---|---|
| Shape test | After patchify: `(B, 64, 48)`. Encoder input (visible): `(B, 16, 48)`. Decoder input: `(B, 64, D_dec)`. Output: `(B, 48, 48)` (48 masked patches × 16 pixels). |
| `torch.autograd.gradcheck` | Wrap the patchify → mask → encoder → decoder → MSE pipeline; verify gradients w.r.t. encoder parameters. |
| Overfit one batch | Train on a batch of 16 images for 500 steps; reconstruction MSE on masked patches should fall below 0.01 (normalized). |
| Visual check | Plot original vs reconstructed image with masked regions shown; visible patches should be perfect, masked should be plausible. |

**Key implementation details to enforce:**
- The encoder receives only visible-patch tokens, not mask tokens.
- Positional embeddings must be added before masking (encoder) and before decoding (decoder), so both operate in full-grid position space.
- The mask token is a single learned parameter `nn.Parameter(torch.zeros(1, 1, D_dec))` broadcast to all masked positions.
- Loss is computed on masked patches only, averaged over masked positions; do not include visible patches in the loss.

---

### Mechanism 2 — DINO on CIFAR-10 with collapse test

**Problem setup.** CIFAR-10, ViT-Tiny (4 blocks, 192 dim, 3 heads). Two global crops (random resized crop 0.4–1.0) and two local crops (0.05–0.4). 65,536 prototype dims (or 4,096 for the minimal version). Teacher EMA τ = 0.996.

**Tasks to verify:**

| Task | What to check |
|---|---|
| Shape test | Student and teacher outputs: `(B, K)` after softmax. Centering buffer: `(1, K)` EMA. |
| `gradcheck` | Gradient flows through student cross-entropy loss; no gradients through teacher (verify `teacher.requires_grad_(False)` and EMA update is outside the graph). |
| Overfit one batch | On 16 CIFAR-10 images (one batch, 4 crops per image), loss should decrease to near zero in ~200 epochs (student perfectly predicts teacher). |
| Collapse test | Run three variants: (a) full DINO — loss decreases, prototypes remain diverse; (b) no centering — monitor entropy of teacher output distribution; it should collapse to near-zero entropy (single prototype) within 200 steps; (c) no sharpening (high teacher temperature τ_t = 0.5) — monitor entropy; it should converge to near-maximum entropy (uniform). Measure prototype entropy `H = -sum p log p` every 10 steps and plot all three curves. |
| Linear probe | After 50 epochs on full CIFAR-10 (train set), train a linear classifier on frozen teacher features; accuracy should exceed 60% (vs. ~10% for random ViT init). |

**What "collapse" looks like in code.** Track `teacher_entropy = -torch.sum(p_t * torch.log(p_t + 1e-8), dim=-1).mean()` over the batch at each step. Healthy training: entropy stays in mid-range and decreases slowly. No-centering collapse: entropy falls to ~0 within 100 steps. No-sharpening collapse: entropy rises to `log(K)` and stays there.

---

### Mechanism 3 (extension) — iBOT patch-level distillation

Add a patch-token distillation head on top of the DINO implementation above. Mask 30% of patches in the student input; the teacher sees the full sequence. Add a second cross-entropy loss on masked patch positions. This directly demonstrates why DINOv2 outperforms DINO on dense tasks (segmentation) — the patch-level objective forces position-specific discriminative features.

Verifiable task: with patch distillation head, k-nearest-neighbor retrieval at the patch level (crop-level matching) should outperform the DINO-only baseline.

---

## 3. Assessment of the draft scope

### What is right

- MAE mechanism description is accurate: random masking, encoder on visible patches only, lightweight decoder, pixel-reconstruction MSE loss. This is the pedagogically essential core.
- DINO mechanism description is accurate: EMA teacher, multi-crop, centering + sharpening, collapse-avoidance mechanics.
- The collapse test (removing centering collapses representations) is the most important single experiment in the module — it converts an abstract regularization trick into a falsifiable claim. Keep this.
- Linear-probe-beats-random-init is a valid assessment target.

### What is missing or underweighted

**1. The SSL family map is absent.** Without it, students have no context for where MAE and DINO sit relative to SimCLR, MoCo, BYOL, BEiT, and I-JEPA. They cannot evaluate tradeoffs or understand why different tasks call for different methods. The family map (contrastive / distillation / masked-pixel / masked-latent) should be taught explicitly before the implementation exercises.

**2. DINOv2 is the production baseline, not DINO.** As of 2025, DINOv2 is what practitioners actually use as a backbone. The module should explain DINOv2's three-loss combination (DINO + iBOT + KoLeo) and the role of data curation, even if the implementation exercise stays with vanilla DINO. Skipping DINOv2 leaves students unable to understand why frozen ViT-g features appear throughout downstream modules (pose estimation, depth estimation, probing).

**3. iBOT is a direct bridge between DINO and DINOv2** and takes ~20 lines of code to add on top of a DINO implementation. It should be the extension task rather than being absent. Without iBOT, the jump from DINO to DINOv2 feels magical.

**4. KoLeo regularizer deserves one paragraph.** The nearest-neighbor distance maximization in feature space is a clean, implementable idea that improves retrieval by +8% in DINOv2. It belongs in the conceptual material.

**5. I-JEPA deserves a conceptual paragraph.** It represents the "latent prediction" paradigm (not pixel prediction) and is the basis for V-JEPA. The key pedagogical point: predicting in feature space rather than pixel space biases the task toward semantics over texture. This is a different failure mode than either MAE or DINO and is worth one page of text.

**6. The linear-probe test needs a baseline comparison.** The current scope says "linear-probe eval beats random init," which is trivially true for any trained network. The probe should be compared against: (a) supervised ViT of the same size trained for the same wall-clock time, and (b) MAE features from the same backbone. DINO linear probe reliably beats MAE linear probe on the same architecture (78% vs 67% on ViT-B/ImageNet), which is the non-obvious result worth explaining.

**7. The multi-crop augmentation details are not mentioned** in the draft. The student should understand why local crops must be small (< 50%) and why the teacher only receives global crops. The asymmetry is what forces the network to match local context to global semantics. This should be explicit.

### What is outdated or mis-emphasized

**Teaching original DINO (2021) as the pedagogical core is still correct**, because DINO is the minimal model that exhibits all of the important phenomena: EMA teacher, collapse, centering/sharpening, emergent attention maps. DINOv2 adds complexity (three losses, curated data, large models) that obscures these fundamentals. The right ordering is: DINO implementation → understand collapse → add iBOT patch objective → understand DINOv2 as the scaled-up combination.

**I-JEPA should not be primary** for this module but should not be absent. It belongs as a one-page conceptual treatment after the DINO/DINOv2 section, with a pointer to V-JEPA for the video SSL module. The JEPA paradigm is important for understanding the 2024-2025 direction of SSL, but it is not the right implementation target in a 12GB module because I-JEPA's advantages (over MAE) only become clear at scale on ImageNet or larger.

**DINOv3 (August 2025)** extends DINOv2 to 6.7B parameters trained on 1.7B images with Gram anchoring for dense feature stability. It is too recent and too large to implement, but a one-paragraph note in "2024-2026 developments" is appropriate.

### Recommended scope additions/cuts

| Action | Item |
|---|---|
| Add | SSL family map as explicit teaching material |
| Add | DINOv2 conceptual overview (3 losses + data curation) |
| Add | iBOT as extension implementation task |
| Add | KoLeo regularizer paragraph |
| Add | Multi-crop augmentation details (why teacher gets only global views) |
| Add | I-JEPA conceptual paragraph with V-JEPA pointer |
| Revise | Linear probe test: compare DINO vs MAE vs random, not just vs random |
| Keep | MAE implementation (pixel reconstruction, asymmetric encoder-decoder) |
| Keep | DINO collapse test — this is the pedagogical centerpiece |
| Keep | Original DINO (2021) as implementation target |
| No change needed | EMA teacher mechanics |

---

## 4. Connections to other topics

**Produces the backbones used everywhere.** DINOv2 ViT-g/14 features appear as frozen inputs in depth estimation (A5), semantic segmentation (A6), and few-shot recognition modules. Students cannot interpret "we freeze the DINOv2 backbone" without having implemented DINO and understood what the frozen [CLS] and patch tokens represent.

**Connects to ViT (A2).** MAE and DINO both require ViT as the backbone. The attention map visualization in DINO is the clearest demonstration of what ViT attention heads learn, making A3 the payoff for A2.

**Connects to augmentation and data pipelines (A1).** DINO's multi-crop strategy is the most aggressive augmentation pipeline in the course. Understanding why the augmentations must be strong (to prevent learning trivial shortcut solutions) connects directly to data augmentation principles.

**Connects to contrastive learning (A4, if present).** SimCLR and MoCo are the historical predecessors; the family map makes this lineage clear. The move from contrastive to distillation (DINO) to hybrid (DINOv2) is a narrative arc.

**Connects to dense prediction (A5/A6).** MAE features fine-tune well for detection/segmentation; DINO/DINOv2 frozen features probe well for dense tasks. This motivates the different evaluation protocols.

**Connects to foundation models.** DINOv2 is used as the visual encoder in many vision-language models. Understanding its training makes VLM module (if present) concrete.

---

## 5. Must-read sources

1. **He et al. (2021/2022) — "Masked Autoencoders Are Scalable Vision Learners."** arXiv:2111.06377. CVPR 2022. The primary MAE paper; Sections 2-3 cover the full architecture, masking strategy, and ablation of masking ratio. Read before implementing.

2. **Caron et al. (2021) — "Emerging Properties in Self-Supervised Vision Transformers."** arXiv:2104.14294. ICCV 2021. The original DINO paper; Section 2 covers the full algorithm, centering/sharpening, and multi-crop. Algorithm 1 is the implementation spec. The attention map visualizations in Section 4 are the pedagogical payoff.

3. **Oquab et al. (2023) — "DINOv2: Learning Robust Visual Features without Supervision."** arXiv:2304.07193. TMLR 2024. The production backbone; Section 3 covers the three-loss objective and Section 4 covers data curation. Essential for understanding why DINOv2 features appear in downstream modules.

4. **Zhou et al. (2021) — "iBOT: Image BERT Pre-Training with Online Tokenizer."** arXiv:2111.07832. ICLR 2022. Directly extends DINO with patch-level distillation; the bridge from DINO to DINOv2. Reading this makes the DINOv2 iBOT loss term immediately clear.

5. **Grill et al. (2020) — "Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning."** arXiv:2006.07733. NeurIPS 2020. BYOL is the predecessor to DINO; understanding why BYOL's predictor + EMA prevents collapse (without centering) builds intuition for why DINO adds centering for the prototype-distribution setting.

6. **Assran et al. (2023) — "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture."** arXiv:2301.08243. CVPR 2023. The I-JEPA paper; the key conceptual contribution is prediction in feature space rather than pixel space. Read Section 3 for the masking strategy and Section 5 for the comparison with MAE.

7. **Siméoni et al. (2025) — "DINOv3."** arXiv:2508.10104. The current state of the DINO lineage; 6.7B parameters, 1.7B images, Gram anchoring for dense feature stability. A horizon paper — read the abstract and Section 3 (Gram anchoring) to understand the open problem (dense feature degradation at scale) that DINOv2 did not fully solve.

**Omission to flag.** The draft scope does not cite Grill et al. (BYOL 2020), which is important for understanding the ancestry of EMA teachers and the collapse-without-negatives problem. It also omits Chen and He (2021) "Exploring Simple Siamese Representation Learning" (SimSiam, arXiv:2011.10566), which analytically clarifies the role of stop-gradient and predictor asymmetry.

---

## 6. Developments from 2024-2026 that change how this should be taught

**DINOv3 (August 2025).** Trains at 6.7B parameters on 1.7B curated images. The key new idea is *Gram anchoring*: instead of regularizing individual patch features, it anchors the Gram matrix (matrix of all pairwise dot products of patch features) of the student to match an earlier teacher checkpoint. This prevents the dense feature degradation that occurs during extended DINOv2 training schedules, where global metrics keep improving while local (patch-level) feature quality degrades. DINOv3 sets new benchmarks on frozen-backbone dense tasks (COCO detection: 66.1 mAP, ADE20k: 63.0 mIoU). *Teaching implication*: add "dense feature degradation at scale" as a known limitation of DINOv2.

**V-JEPA and V-JEPA 2 (2024-2025).** V-JEPA (Bardes et al., 2024) extends I-JEPA to video by predicting masked spatio-temporal patches in latent space. V-JEPA 2 achieves strong video understanding without labels. This makes the JEPA paradigm practically relevant, not just theoretical. *Teaching implication*: I-JEPA is no longer just a curiosity — it is the basis of a production video SSL system.

**The distillation family converged to hybrid objectives.** By 2025 the field consensus is that pure masked-pixel prediction (MAE) produces features that fine-tune well but probe poorly, while pure distillation (DINO) probes well but fine-tunes comparably. The hybrid (DINOv2 = DINO + iBOT) outperforms both on both axes. *Teaching implication*: teach the family map as a progression ending at the hybrid, not as competing alternatives.

**MAE linear-probe weakness is documented and understood.** Multiple 2023-2024 papers (e.g., "Stare at What You See," arXiv:2211.08887; "Understanding contrastive versus reconstructive") explain that pixel reconstruction biases MAE toward texture and position encoding rather than semantic content. This is why MAE ViT-B linear probe (67.5%) lags DINO ViT-B linear probe (78.2%). *Teaching implication*: frame MAE's pretext task explicitly as a "textural completion" problem and contrast it with DINO's "semantic consistency" objective.

**Singular defects in DINOv2 (2024).** "SINDER: Repairing the Singular Defects of DINOv2" (arXiv:2407.16826) documents that DINOv2 features contain a small number of "register" tokens that act as outlier patch features with very high norm, disrupting local feature quality. This is a known artifact of the training procedure, not a fundamental limitation. *Teaching implication*: when students visualize DINOv2 attention maps they may observe these artifacts; flagging this prevents misdiagnosis.

**Data quality matters as much as loss design.** The DINOv2 ablation (Section 5 of the paper) shows that training DINOv2 on uncurated data (raw web crawl) substantially degrades performance. The LVD-142M pipeline (retrieval-based curation using similarity to seed datasets) is a non-trivial contribution. *Teaching implication*: add a brief discussion of data curation in the SSL context; "scale" without quality does not work.
