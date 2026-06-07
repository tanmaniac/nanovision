# A4 — CLIP & open-vocabulary: research validation

## 1. Key concepts a student must learn

### Contrastive image-text learning

CLIP trains two independent encoders — one for images, one for text — to produce embeddings in a shared L2-normalized space. The only learning signal is alignment: matched pairs should be close, unmatched pairs should be far apart. There is no per-class label, no bounding box, no segmentation mask. This makes the representation inherently open-vocabulary: the text encoder is queried at inference time with arbitrary natural language.

The training set consists of N image-caption pairs per batch. All N×N combinations are evaluated; the diagonal entries are positives, the N(N-1) off-diagonal entries are in-batch negatives. This is the defining structural choice, and it has direct consequences:

- **Quality of negatives scales with N.** A small batch means few negatives, which means weak gradients. The loss is trivially satisfied if images are distinct enough to separate any small set. CLIP trained on WIT-400M used batch size 32,768 specifically to make the negatives hard and diverse.
- **False negatives exist.** Two images in the same batch may share a semantically equivalent caption; treating them as negatives is wrong. This is a known problem that becomes more severe as N grows and class diversity shrinks (e.g., fine-grained datasets).

### Symmetric InfoNCE loss and temperature

The similarity matrix S ∈ R^{N×N} is computed as:

```
S = (1/τ) · image_features @ text_features.T
```

where both feature tensors are L2-normalized (so S_{ij} is cosine similarity scaled by 1/τ), and τ is a scalar learned parameter (the temperature).

CLIP optimizes the average of two cross-entropy losses — one treating rows as distributions (image-to-text), one treating columns (text-to-image):

```
L = 0.5 * (CE(S, labels) + CE(S.T, labels))
labels = torch.arange(N)
```

This is the symmetric InfoNCE. It is equivalent to minimizing the negative log-likelihood that each image ranks its true caption first in a softmax over all N captions, and simultaneously that each caption ranks its true image first.

**Temperature τ** controls the sharpness of the softmax. Low τ concentrates probability on the top-ranked item and produces very hard gradients; high τ flattens the distribution and yields soft gradients that can be informative across many negatives. CLIP initializes τ = 0.07 (stored as `nn.Parameter(torch.tensor(log(1/0.07)))`) and clamps it so the logit scale 1/τ never exceeds 100, which was found empirically necessary to prevent training instability. After training, τ converges to approximately 0.01 for most CLIP checkpoints.

**Batch-size dependence** is not merely practical. The InfoNCE objective is a lower bound on the mutual information between image and text representations, and the bound tightens as N increases. With N = 8 you get 7 negatives per sample; with N = 32,768 you get 32,767. The gradient signal is qualitatively different — small batches collapse the loss too easily, producing mediocre representations.

### Zero-shot transfer via prompt embeddings

At inference time, for a K-class classification problem:
1. Construct K text prompts, one per class (e.g., "a photo of a {classname}").
2. Encode all K prompts with the text encoder to get K template embeddings.
3. Encode the query image with the image encoder.
4. Compute cosine similarities between the image embedding and all K class embeddings.
5. Take the argmax (or softmax for a probability distribution).

There is no fine-tuning, no additional learned head. The classification boundary is defined entirely by the geometry of the shared embedding space. This is why prompt wording matters: CLIP was trained on captions, not bare class names, so "a photo of a golden retriever" out-performs "golden retriever" by several percentage points. Ensembling 80 hand-crafted templates for ImageNet improved accuracy by ~3.5% over a single template. On CIFAR-10, standard CLIP (ViT-B/32) achieves ~88-90% zero-shot; SigLIP pushes this to ~92%, and SigLIP2 to ~94%.

### SigLIP sigmoid loss

Zhai et al. (ICCV 2023) replaced the row/column softmax normalization with an independent binary cross-entropy on every pair:

```
L_sig = -(1/N²) * sum_{i,j} [ y_ij * log σ(s_ij + b) + (1 - y_ij) * log(1 - σ(s_ij + b)) ]
```

where y_ij = 1 if (i,j) is a matched pair and -1 otherwise (using the signed convention in the paper; effectively log-sigmoid for positives and log(1-sigmoid) for negatives), and b is a learnable bias initialized to a large negative value so most pairs start near zero loss.

The crucial difference from InfoNCE: **each pair is treated independently**. There is no global normalization denominator that sums over all N items. This means:
- The loss is well-defined and stable at batch size 1.
- Distributed training does not require an all-reduce over the full similarity matrix.
- Performance degrades much more gracefully at small batch sizes: SigLIP outperforms softmax InfoNCE significantly when N < 16,384.
- At N = 32k+ both methods perform comparably.

For a 12 GB GPU, the sigmoid loss is the practical default — it works acceptably with batches of 256-1024, where InfoNCE would produce poor representations.

### Open-vocabulary detection and segmentation

CLIP produces a single image-level embedding; detection requires per-region embeddings. OWL-ViT (Minderer et al., ECCV 2022) showed the simplest possible transfer: remove the final pooling from the ViT image encoder, add per-token classification and regression heads, and fine-tune on detection data with image-level CLIP features as initialization. Text queries are encoded with the CLIP text encoder at inference time.

OWLv2 (Minderer et al., 2023) scaled this with self-training: use the OWL-ViT detector to pseudo-label a large image-text corpus, then train a stronger detector on those pseudo-labels.

GroundingDINO (Liu et al., ECCV 2024) took a different path: early fusion between text tokens and image tokens inside the transformer (cross-modal attention before the final encoder), producing a stronger detector that can handle referring expressions ("the red cup on the left"). It achieves 52.5 AP zero-shot on COCO — higher than any pure-CLIP-finetuned approach.

These three form the main lineage from CLIP to open-vocabulary perception.

---

## 2. Mechanisms to implement from scratch (12 GB-friendly)

### 2.1 Symmetric InfoNCE loss with learnable temperature

**Implementation target:** a function `clip_loss(image_features, text_features, logit_scale)` that:
- Takes L2-normalized embeddings of shape (N, D)
- Scales the similarity matrix by `logit_scale.exp()`
- Returns the symmetric cross-entropy loss

**Minimal verifiable task:** Overfit a batch of N=8 (image, caption) pairs where images are 32×32 noise blobs with a one-token "label" as caption. After ~200 gradient steps the diagonal of the similarity matrix should be the maximum entry in each row and column, and the loss should be near zero.

**Shape tests:** S = (N, N), labels = (N,), loss is scalar.

**gradcheck:** Wrap `clip_loss` as a function of `image_features` and `text_features` (both as float64 leaf tensors), check with `torch.autograd.gradcheck`.

**Small-batch behavior note:** With N=8 and only 7 negatives, the model overfits trivially. Test that training with N=4 produces a model that fails to generalize (representations not separated), while N=32 produces better separation. This makes the batch-size dependence concrete and verifiable within a 12 GB budget.

### 2.2 SigLIP sigmoid loss

**Implementation target:** a function `siglip_loss(image_features, text_features, logit_scale, bias)` implementing the independent pairwise binary cross-entropy.

**Minimal verifiable task:** Same overfit-one-batch setup as above, but with N=4. The sigmoid loss should converge where InfoNCE with N=4 fails or is much slower, demonstrating the small-batch advantage concretely.

**gradcheck:** Treat logit_scale and bias as learnable parameters; verify gradients for both.

### 2.3 Zero-shot CIFAR-10 with real CLIP weights

**Implementation target:** A zero-shot classifier that:
1. Loads `openai/clip-vit-base-patch32` (or `google/siglip-base-patch16-224`) via `transformers` or `open_clip`.
2. Encodes 10 class name prompts ("a photo of a {c}").
3. Encodes CIFAR-10 test images in batches.
4. Classifies by cosine argmax.
5. Reports accuracy (expected: ~88-90% for CLIP ViT-B/32, ~92% for SigLIP-B/16).

**Why this belongs in the course:** It is the canonical demonstration that contrastive training generalizes without any task-specific labels. It uses real weights to show that the loss the student implemented at scale produces something useful. CIFAR-10's 10 classes are simple enough that the student can see failures and understand why a dog photo is misclassified (e.g., as cat because both are in the same embedding region).

**Memory:** CLIP ViT-B/32 is ~340 MB. CIFAR-10 test is 10k images. This fits on any 12 GB GPU with room to spare.

---

## 3. Assessment of the draft scope

### What is right

- **Dual encoder architecture** is the correct abstraction. Reusing the ViT image tower from A1/A2 and the transformer text tower from A1 is pedagogically tight — the student sees that CLIP's novelty is not in the encoders but in the training objective and the shared projection.
- **Symmetric InfoNCE with learnable temperature** is the right primary mechanism to implement. It is simple (~10 lines of PyTorch), mathematically clean, and directly connects to mutual information estimation.
- **Zero-shot CIFAR probe** with real weights is the right capstone. It is concrete, fast, and gives immediate feedback.

### What is missing or under-emphasized

**SigLIP sigmoid loss should be implemented alongside InfoNCE, not treated as optional context.** Given a 12 GB ceiling, the student cannot meaningfully experience InfoNCE at the batch sizes (16k+) where it works well. A student who only implements InfoNCE and tests it at batch size 64 will observe weak training and not know whether that is expected or a bug. Implementing both losses at N=16 and comparing them is the best way to teach the batch-size dependence problem and its solution simultaneously. SigLIP is also the encoder used in PaliGemma, SigLIP2, and π0, making it more 2026-current than vanilla CLIP.

**The bias parameter in SigLIP is non-trivial.** The initialization of `b = -log(N² - N)` (so the expected loss contribution from negative pairs starts near zero) matters for stable training and should be explained. The draft does not mention it.

**Prompt engineering and its effect on accuracy** is under-specified. The student should implement the ensemble strategy (average over K templates) as part of the zero-shot probe, not just single-template classification.

**The modality gap** — the empirical finding that image and text embeddings live in geometrically separated cones even after training — is a known failure mode of contrastive training that is not mentioned. It is detectable with the student's own probe code (PCA on the joint embedding matrix) and is pedagogically interesting.

### What is outdated or mis-emphasized

**"Reuse ViT image tower + A1 transformer text tower"** is fine if done carefully, but the student must ensure the text encoder produces a pooled [EOS] token representation (as CLIP does), not a sequence of token embeddings. This detail is glossed over in the draft and frequently causes bugs.

**The open-vocabulary detection lineage deserves a section, not just a footnote.** OWL-ViT → OWLv2 → GroundingDINO is a clear progression that shows how a CLIP encoder becomes a detector. The student should read the OWL-ViT paper and understand that removing the global pooling is sufficient for region-level tasks — this is a one-paragraph insight with large downstream importance (it explains how the same encoder serves both classification and detection).

### What to add

- **SigLIP loss implementation** as a required, not optional, mechanism (see section 2.2).
- **Comparison cell:** train the same tiny dual encoder with InfoNCE at N=16 and SigLIP at N=16, measure alignment quality (e.g., fraction of diagonal entries that are the row maximum). This is the clearest demonstration of batch-size insensitivity.
- **Brief mention of EVA-CLIP and OpenCLIP** as the open-weight ecosystem the student will actually use — both are built on the same losses the student implements.

### What to cut or reduce

The draft implies implementing the full dual encoder from scratch, including the ViT image tower. Given that ViT is already A1/A2, the image encoder can be a frozen stub (a random ViT or a tiny custom CNN) for the loss implementation exercise. The pedagogical focus for A4 is the contrastive loss and the zero-shot inference procedure, not re-implementing the encoder.

### Recommended order

1. Implement InfoNCE loss (symmetric, with learnable temperature, with gradcheck).
2. Observe batch-size dependence by training at N=8 vs N=64 on a toy dataset.
3. Implement SigLIP sigmoid loss; repeat the comparison — SigLIP works at N=8.
4. Zero-shot CIFAR-10 with real CLIP weights (one-template and ensemble).
5. Optional: swap CLIP encoder for SigLIP encoder, note the accuracy difference.

---

## 4. Connections to other topics

**Backward (prerequisite):** A4 directly consumes the ViT (A2) and the causal transformer (A1). The student already understands self-attention, positional encoding, and patch embedding. The text encoder is a transformer with a [CLS] or [EOS] pooling trick.

**Forward to VLMs (A5+):** LLaVA-style VLMs are a CLIP/SigLIP image encoder + a linear projection + a frozen or fine-tuned LLM decoder. PaliGemma uses SigLIP-So400M as its vision tower. The projection layer maps image tokens to the LLM's token dimension. A student who understands CLIP's shared embedding space immediately understands why this projection is needed and what it does.

**Forward to VLAs:** π0 (Physical Intelligence, 2024) uses SigLIP+Gemma as its VLM backbone, then adds action tokens. OpenVLA (Stanford, 2024) uses LLaVA with a CLIP ViT-L encoder. The contrastive encoder is the perceptual front-end for every leading open-source VLA.

**Forward to open-vocabulary detection:** OWL-ViT removes the global pooling from the CLIP ViT, adds per-patch classification and box regression heads, and fine-tunes on detection data. The student who implemented the CLIP image encoder in A4 can read OWL-ViT and immediately understand the architectural diff.

**Forward to grounded generation:** GroundingDINO and SAM2 use text-conditioned features to produce spatially grounded outputs. These are A6/A7 material, but their connection to CLIP is direct.

**Lateral (same module):** The symmetric InfoNCE loss is a special case of the more general noise-contrastive estimation (NCE) and NT-Xent (normalized temperature-scaled cross-entropy used in SimCLR). If the course covers self-supervised learning, CLIP's loss is the natural multimodal extension.

---

## 5. Must-read sources

1. **Radford et al., "Learning Transferable Visual Models From Natural Language Supervision," 2021 (arXiv 2103.00020).** The original CLIP paper. Read sections 2 (natural language supervision), 3.1 (contrastive pre-training), and Appendix B (implementation details including temperature initialization, batch size 32,768, and prompt engineering).

2. **Zhai et al., "Sigmoid Loss for Language Image Pre-Training," ICCV 2023 (arXiv 2303.15343).** Introduces SigLIP. Read sections 2 (loss formulation) and 3 (batch-size ablations). Essential for understanding why the softmax-based loss is impractical on consumer hardware.

3. **Tschannen, Gritsenko, et al., "SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features," 2025 (arXiv 2502.14786).** SigLIP2 adds captioning, self-distillation, and masked prediction to the SigLIP recipe. Relevant because its encoder is the current production default in PaliGemma 2 and π0. Read the training objective section and the localization ablation.

4. **Minderer et al., "Simple Open-Vocabulary Object Detection," ECCV 2022.** OWL-ViT. Shows that removing global pooling from a CLIP ViT and adding detection heads is sufficient for open-vocabulary detection. Read section 3 (model architecture). The key insight — patch tokens carry spatial information that global pooling discards — is one paragraph and has large downstream importance.

5. **Liu et al., "Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection," ECCV 2024 (arXiv 2303.05499).** Shows how early cross-modal fusion (text and image tokens interacting inside the transformer) outperforms late fusion (CLIP-style independent encoders with a detection head). Read section 3 (feature enhancer and language-guided query selection).

6. **Ilharco et al., "OpenCLIP: Reproducible Scaling Laws for Contrastive Language-Image Learning," 2022 (arXiv 2212.07143).** Documents training dynamics, scaling laws, and failure modes across data and model size. Includes the open-weight checkpoints the student will use for the CIFAR probe. Read section 4 (scaling laws) and appendix A (training details).

7. **Weng, "Contrastive Representation Learning," Lil'Log, 2021.** Not a paper but an unusually clear survey of InfoNCE, NT-Xent, and their theoretical connections to mutual information estimation. Useful as a self-contained primer before reading the CLIP paper.

---

## 6. 2024-2026 developments that change how this should be taught

### SigLIP is now the practical default, not an extension

PaliGemma (Google, 2024), PaliGemma 2 (2024), and π0 (Physical Intelligence, 2024) all use SigLIP-So400M as their vision encoder. LLaVA 1.6 and its successors use CLIP ViT-L. A course teaching CLIP in 2026 that only covers softmax InfoNCE is teaching the architecture that is one generation behind the field's current practice. The sigmoid loss should be taught as the primary mechanism or at minimum co-primary with InfoNCE.

### SigLIP2 adds multi-task pretraining as a new dimension

SigLIP2 (2025) demonstrates that adding captioning (LocCa decoder), self-distillation, and masked patch prediction to the contrastive recipe produces substantially better dense features and localization (+18 points on RefCOCO for the B/16 model). This is relevant to the course because it shows that contrastive loss alone is insufficient for spatial/dense tasks, motivating the open-vocabulary detection topic. The student should at least read the SigLIP2 abstract to understand why the field moved beyond pure contrastive training.

### GroundingDINO 1.5 (May 2024) and open-vocabulary segmentation matured

GroundingDINO 1.5 (IDEA Research, 2024) pushed zero-shot COCO AP to new levels. Combined with SAM2 (Meta, 2024), it forms the backbone of most current open-vocabulary instance segmentation pipelines. The CLIP → OWL-ViT → GroundingDINO lineage is now standard and should be presented as a progression rather than separate papers.

### EVA-CLIP-18B (February 2024) sets the scaling ceiling

BAAI's EVA-CLIP-18B achieved 80.7% average zero-shot accuracy across 27 benchmarks using 18B parameters. It demonstrates that contrastive CLIP-style training continues to scale, but the gap between open and closed models has narrowed significantly.

### Modality gap is a known measurable artifact

Several 2023-2024 papers (Schrodi et al., 2024; Lee et al., 2024) formalize the modality gap as a geometric phenomenon arising from initialization and contrastive training dynamics. A student can measure it with their own probe code (compute mean image embedding and mean text embedding in the CIFAR-10 experiment; note they are not collocated). Including this as a measurement task in the assessment would deepen intuition without adding implementation complexity.

### Temperature-free losses are an emerging alternative (2025)

Work from 2025 (arXiv 2501.17683) proposes temperature-free contrastive objectives that remove the clipping/initialization sensitivity. These are not yet mainstream but may be worth one sentence in lecture as evidence that the field continues to refine the loss function.

### The compute ceiling argument strongly favors teaching SigLIP first

A practical note that is not in the literature but is true of 2026 pedagogical context: a student on a 12 GB GPU who implements InfoNCE and trains it at N=64 will observe poor representations and may believe they have a bug. InfoNCE only starts working well above N=4,096 for non-trivial datasets. SigLIP works at N=256-512, which is the natural batch size for a 12 GB training run on 224×224 images with a ViT-B backbone. Teaching SigLIP first, then showing InfoNCE as the historical predecessor and explaining *why* it needed 32,768 batch size, is more coherent given the hardware ceiling.
