# A4 build plan - CLIP and SigLIP (contrastive image-text)

Status: planned, pending expert review. Core. Deps: A1 (transformer/attention),
A2 (ViT). Bridges to A8 (VLM), A13 (VLA), and the open-vocab detection thread.

## Scope

A4's deliverable is the contrastive training objective and zero-shot inference, NOT
the encoders. The image and text towers reuse the shared library (nanovision
TransformerEncoder + primitives), the same way A3 and A3.5 provide a small ViT built
on the shared transformer rather than importing a02's local vit.py. The taught
mechanisms are the two contrastive losses and the zero-shot classification procedure.

Primary objective is the SigLIP sigmoid loss (it is what trains on a 12GB budget and
is the 2024-2026 production default in PaliGemma / pi0); softmax InfoNCE is built too,
as the historical predecessor and to make the batch-size problem concrete. Per the
research note (research/a04_clip.md), implementing both and comparing them at
small N is the clearest way to teach why InfoNCE needed batch size ~32k and SigLIP
does not.

## Holes the student writes (3)

1. `clip_loss(image_features, text_features, logit_scale)` (`losses.py`, local).
   Symmetric InfoNCE with a learnable temperature. image_features, text_features are
   L2-normalized (N, D). Build the scaled similarity `logits = logit_scale.exp() *
   image_features @ text_features.T` (N, N); the positives are the diagonal
   (`labels = arange(N)`); return `0.5 * (CE(logits, labels) + CE(logits.T,
   labels))`. The student BUILDS the CE by hand: `log_softmax` (with max-subtraction)
   then gather the diagonal and negate-mean. F.cross_entropy is FORBIDDEN here because
   it hides the softmax-over-negatives denominator that is the entire lesson (the
   reviewer's call; same principle as A3 building dino_loss from log_softmax).
   logit_scale is a learnable scalar stored in log space; the model clamps the log
   parameter at log(100) before exp() (CLIP's stability trick; init log(1/0.07)).
   Teaches the NxN in-batch-negative structure, the symmetric image<->text CE, and
   the temperature.
2. `siglip_loss(image_features, text_features, logit_scale, bias)` (`losses.py`,
   local). Independent pairwise binary classification. Form `logits = logit_scale.exp()
   * image_features @ text_features.T + bias` (N, N); the label matrix is
   `2*eye(N) - 1` (+1 diagonal, -1 off). The loss is the SigLIP paper's Algorithm 1
   exactly: `-sum( logsigmoid(labels * logits) ) / N`, i.e. sum the per-pair
   log-sigmoid over the full NxN matrix and divide by N (the batch size), NOT a mean
   over N*N. (Equivalently: sum each row, then mean over the N rows.) Dividing by N^2
   is wrong by a factor of N and rescales the gradient. F.logsigmoid is ALLOWED (a
   stable elementwise primitive, like gelu). bias is a learnable scalar init -10 (the
   paper's value; keeps the many negative pairs near zero loss at the start);
   logit_scale init log(10) for the SigLIP path (the paper's temp init, not CLIP's
   log(1/0.07)). Teaches that dropping the softmax denominator makes the loss per-pair
   and stable at any batch size.
3. `zero_shot_classify(image_features, text_features)` (`inference.py` or
   `model.py`, local). image_features (B, D) and class-prompt text_features (K, D),
   both L2-normalized. Return cosine-similarity logits (B, K) = image @ text.T and the
   argmax class. Teaches that the classifier boundary is the geometry of the shared
   space, no trained head.

## Provided (no holes)

- `model.py` / `backbone.py`: a tiny dual encoder built on the shared library.
  - Image tower: a small ViT (patch embed via Conv2d, nanovision TransformerEncoder,
    pooled to one (B, dim) vector). Reuses the shared transformer; does not import
    a02's local vit.py.
  - Text tower: token embedding + a small causal nanovision TransformerEncoder, pooled
    at the EOS position the CLIP way: `hidden[arange(N), token_ids.argmax(dim=-1)]`,
    i.e. the hidden state at the position of the highest token id (the EOS/EOT token is
    assigned the largest id, and sequences are padded, so this is NOT `[:, -1]`). Then
    project. Provided + explained in the README, not a hole; it is a known bug source.
    Note CLIP uses this causal-EOS pool; SigLIP's text tower is bidirectional with a
    different (MAP/last-token) pool, so the README must not claim EOS pooling is
    universal.
  - Both towers end in a linear projection to a shared dim D and an L2 normalize.
    Learnable `logit_scale` (init log(1/0.07)) and `bias` (init -10, the SigLIP value).
  - `encode_image`, `encode_text`, and a `forward` that returns the normalized
    features + logit_scale (+ bias) are provided.
- Toy data in `nanovision/data/toy.py`: `image_text_batch(...)` (provided) - N synthetic
  (image, caption) pairs. Use a SMALL number of latent classes C < N (e.g. C=4, N=8 so
  each class appears ~twice) rather than one class per pair: each pair draws a class plus
  a nuisance attribute (color, shape, size) that drives both a colored-shape image and a
  short token caption, with only some attributes shared between the paired image and
  caption. This gives genuine in-batch false negatives (two same-class pairs are
  "negatives" but semantically close, the InfoNCE false-negative pathology) and a graded
  off-diagonal similarity matrix the viz can show, instead of a trivial one-hot-per-pair
  block diagonal that overfits in ~10 steps. Returns images (N, C, H, W) and token ids
  (N, L). The overfit test still uses N=8.
- `config.py`: tiny sizes (image 16 or 32, small dims, D shared ~64, vocab small,
  seq short). Overfit batch N=8; the small-batch comparison uses N in {4, 8, 16}.

## Tests (verify-before-train order)

1. `test_shapes.py`: encode_image/encode_text give (N, D) L2-normalized (norm ~1);
   clip_loss and siglip_loss return scalars; zero_shot_classify gives (B, K) logits
   and (B,) predictions.
2. `test_gradcheck.py`: float64 gradcheck of clip_loss w.r.t. image and text features
   and w.r.t. logit_scale; of siglip_loss w.r.t. features, logit_scale, and bias.
3. `test_losses_reference.py` (reference-value): on a constructed feature set where the
   diagonal is the obvious match, clip_loss is minimized when features align; check the
   symmetric loss equals the hand-computed value on a tiny fixed (2x2 or 3x3) case;
   siglip_loss positive/negative decomposition matches the logsigmoid formula.
4. `test_overfit.py` (overfit-one-batch): the tiny dual encoder trained with siglip_loss
   on N=8 fixed pairs reaches near-zero loss and the similarity-matrix diagonal is the
   max of its row and column for all N (alignment achieved).
5. `test_losses_structural.py` (replaces the small-batch comparison; reviewer's call).
   The InfoNCE-vs-SigLIP REPRESENTATION-QUALITY gap is NOT observable by overfitting one
   tiny batch (overfitting a fixed N=4..16 batch is exactly where InfoNCE is strong;
   the real gap is about held-out transfer during real training at moderate N, which a
   12GB toy cannot reproduce). So DO NOT assert an alignment-quality gap at tiny N (it
   is a coin flip and would be a flaky test). Assert only what is deterministic: (a)
   both losses drive a tiny fixed batch to full alignment (diagonal is the row and
   column max for all N) - a correctness sanity check on both; (b) the structural
   difference that IS robust: siglip_loss is finite and well-defined at N=1 (one pair,
   one sigmoid BCE), while symmetric InfoNCE over a 1x1 similarity is degenerate
   (log_softmax of a single logit is 0, no negatives). The InfoNCE-vs-SigLIP
   alignment-vs-N comparison goes in the VIZ, not a test, with the honest caption that
   the toy shows the losses agree at small N and the real difference needs scale.
6. `test_zero_shot.py` (reference-value): with class-prototype text features and image
   features placed near their class prototype, zero_shot_classify recovers the right
   class; prompt-ensemble averaging of several templates is exercised.
7. `test_forbidden_imports.py`: top-level files + solution use no open_clip, clip,
   transformers, timm, or a prebuilt CLIP/loss. FORBID `F.cross_entropy`, `F.nll_loss`,
   and `F.log_softmax` (the InfoNCE hole must build log-softmax + diagonal gather by
   hand, the reviewer's call - cross_entropy hides the negative-sum denominator that is
   the lesson). ALLOW `F.logsigmoid` for SigLIP (a stable elementwise primitive like
   gelu) and `F.normalize` for the provided L2-norm.

## Real-weights probe (notebook, optional, not a test)

A `notebooks/` probe: load a real CLIP or SigLIP checkpoint (open_clip /
transformers, probe-only deps) and reproduce zero-shot CIFAR-10 (~88-90% CLIP B/32,
~92% SigLIP B/16), single-template vs 80-template ensemble. Also measure the modality
gap (mean image embedding vs mean text embedding are not collocated). This is where
the pretrained libs are allowed; never in mechanism code.

## Layout

Local assignment, no new shared symbol (`builds_into_shared_lib: []`). Top-level:
config.py, model.py (dual encoder, provided), losses.py (clip_loss, siglip_loss holes),
inference.py (zero_shot_classify hole), backbone.py if the towers are split out, viz.py,
README.md, ASSIGNMENT.md, tests/, solution/ with the filled copies. conftest + __init__
per the standard layout.

RESOLVED (review): keep A4 fully local, no shared symbol. A8 (VLM) needs an UN-pooled
tower that emits per-patch tokens for the projector, whereas A4's tower pools to one
(B, D) vector for the contrastive loss; different output contracts, so A8 provides its
own tower regardless (built on the same shared TransformerEncoder). Keep A4's pooling a
thin final step over patch tokens so A8 can mirror the construction.

## Citations to verify before writing the README (per the lecture-notes skill)

- CLIP: Radford et al., 2021, arXiv:2103.00020.
- SigLIP: Zhai et al., 2023, arXiv:2303.15343.
- SigLIP 2: Tschannen et al., 2025, arXiv:2502.14786.
- OWL-ViT: Minderer et al., 2022, arXiv:2205.06230 (CONFIRM id).
- Grounding DINO: Liu et al., 2023, arXiv:2303.05499.
- OpenCLIP: Ilharco et al., 2022, arXiv:2212.07143.

## Disclosures the README must make

- Batch-size dependence: InfoNCE's negatives scale with N, the loss is an MI lower
  bound that tightens with N; CLIP used N=32,768. On 12GB you cannot reach that, which
  is exactly why SigLIP (stable per-pair, works at N=256-1024) is the practical default.
  The tiny overfit here is a mechanism check, not a representation-quality result.
- SigLIP bias init (b ~ -10) keeps the many negative pairs near zero loss at the start;
  state the reasoning (most pairs are negatives, sigmoid(s+b) ~ 0).
- Temperature: learnable, log-space, exp clamped <= 100 for stability; converges ~0.01.
- EOS pooling for the text tower (take the EOS hidden state, not mean), the common bug.
- Modality gap: image and text embeddings sit in separated cones even after training.
- The towers are tiny stand-ins; CLIP's result needs scale. A4 teaches the objective
  and inference, not a competitive encoder.

## Build order

1. config.py + toy.image_text_batch + model.py (dual encoder) provided pieces.
2. solution/losses.py, solution/inference.py; verify the encoder overfits N=8 with
   siglip_loss at solution level before holing.
3. Hole the top-level losses.py / inference.py. No shared shim (no new nanovision symbol).
4. Tests; confirm solution green and holed fails cleanly.
5. viz (similarity matrix heatmap before/after; InfoNCE-vs-SigLIP alignment-vs-N curve),
   README lecture notes, ASSIGNMENT.md.
6. make verify A=a04_clip green; update BUILD_CHECKLIST; commit.
