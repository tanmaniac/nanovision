# assignments/a04_clip/ASSIGNMENT.md

```yaml
id: a04_clip
title: CLIP and SigLIP - contrastive image-text learning
module: 1
type: Core
estimated_learner_hours: 6
depends_on: [a00_harness, a01_transformer, a02_vit]
builds_into_shared_lib: []
forbidden_imports:
  - import open_clip
  - import clip
  - from clip
  - import transformers
  - from transformers
  - import timm
  - F.cross_entropy
  - F.log_softmax
  - F.nll_loss
  - nn.CrossEntropyLoss
fits_12gb: true
external_data: "none for tests (optional real CLIP/SigLIP weights + CIFAR-10 for the probe notebook)"
```

## motivation
CLIP adds language as a third supervision source after labels (A2) and self-supervision
(A3): train an image and a text encoder so matched (image, caption) pairs are close in a
shared L2-normalized space and mismatched pairs are far. The objective is the content. A4
builds the two contrastive losses (CLIP's symmetric InfoNCE and SigLIP's per-pair sigmoid)
and zero-shot inference; the towers are tiny stand-ins built on the shared transformer.
SigLIP is primary (it trains at small batch on 12GB and is the 2024-2026 default in
PaliGemma/pi0); InfoNCE is the predecessor that explains the large-batch requirement. The
README covers the batch-size dependence, the EOS-pooling trick, the bias init, the
modality gap, and the OWL-ViT -> Grounding DINO open-vocab line.

## background
See the README. Features are L2-normalized (N, D); the similarity matrix is (N, N) with
matched pairs on the diagonal. logit_scale is a learnable scalar in log space, clamped at
log(100); bias is a learnable scalar init -10 (SigLIP). Text pooling takes the EOS hidden
state at tokens.argmax(dim=-1) (EOS is the largest token id), not the last position.

InfoNCE: logits = logit_scale.exp() * img @ txt.T; L = 0.5*(CE(logits, arange N) +
CE(logits.T, arange N)), the CE built by hand (log_softmax = z - logsumexp(z, dim=1)).
SigLIP (Algorithm 1): logits = ... + bias; labels = 2*eye(N) - 1; L =
-logsigmoid(labels*logits).sum() / N (sum over NxN, divide by N, NOT mean over N^2).
Zero-shot: logits = img @ class_txt.T; pred = argmax.

## what_you_implement
- clip_loss: symmetric InfoNCE with a learnable temperature, CE built by hand.
- siglip_loss: the per-pair sigmoid loss with the learnable bias.
- zero_shot_classify: cosine-similarity argmax against class-prompt embeddings.

The dual encoder, L2 normalization, temperature/bias parameters, EOS pooling, and the toy
(image, caption) data are provided.

## tasks
- **Task 1 - clip_loss** (`losses.py`): logits = logit_scale.exp() * img @ txt.T;
  log_softmax each row by hand (z - logsumexp(z, dim=1, keepdim=True)); CE = -mean of the
  diagonal; return 0.5*(CE(logits) + CE(logits.T)). No F.cross_entropy/F.log_softmax.
  Teaches the in-batch-negative softmax denominator and the symmetric objective.
- **Task 2 - siglip_loss** (`losses.py`): logits = logit_scale.exp() * img @ txt.T + bias;
  labels = 2*eye(N) - 1; return -F.logsigmoid(labels*logits).sum() / N. Teaches the
  per-pair sigmoid loss with no batch denominator.
- **Task 3 - zero_shot_classify** (`inference.py`): logits = img @ txt.T; preds =
  logits.argmax(-1); return (logits, preds). Teaches inference as geometry of the shared
  space.

## tests
Run in this order (also in the README):
1. `tests/test_shapes.py` - encoder gives L2-normalized (N, D); losses scalar; zero_shot
   gives (B, K) and (B,) (shape).
2. `tests/test_gradcheck.py` - float64 gradcheck of clip_loss (w.r.t. features +
   logit_scale) and siglip_loss (w.r.t. features + logit_scale + bias) (gradcheck).
3. `tests/test_losses_reference.py` - clip_loss == symmetric F.cross_entropy reference;
   siglip_loss == log-sigmoid BCE reference; aligned features give low clip_loss
   (reference-value).
4. `tests/test_overfit.py` - the dual encoder trained with siglip_loss (and clip_loss) on
   N=8 fixed pairs aligns every pair (diagonal is row and column max) (overfit-one-batch).
5. `tests/test_losses_structural.py` - the deterministic small-batch difference: at N=1
   siglip_loss is finite with nonzero gradient while clip_loss is 0 with none
   (reference-value). The InfoNCE-vs-SigLIP quality gap is NOT asserted (it needs scale;
   it lives in the viz with an honest caption).
6. `tests/test_zero_shot.py` - cosine argmax recovers the class; prompt-ensemble averaging
   works (reference-value).
7. `tests/test_forbidden_imports.py` - mechanism code uses no open_clip/clip/transformers/
   timm and no F.cross_entropy/F.log_softmax/F.nll_loss; F.logsigmoid and F.normalize
   allowed. Passes with the holes in place too.

## provided_boilerplate
`model.py` (identical at the top level and in solution/): ImageTower (ViT, mean-pool,
project), TextTower (token embed + causal TransformerDecoder + EOS-pool, project), and
CLIPModel (both towers + L2 normalize + learnable logit_scale init log(1/0.07), clamp
log(100) + bias init -10). `config.py` holds all hyperparameters.
`nanovision.data.toy.image_text_batch` provides the toy pairs (latent class + attribute
-> colored-shape image + token caption; fewer classes than pairs to create in-batch false
negatives). The learner writes only the three mechanism bodies.

## compute_notes
CPU, synthetic seeded pairs, no download. Tiny towers (16x16/p4, dim 64 depth 3; vocab 32,
seq 8), N=8, Adam lr 1e-3, 400 steps, full alignment with either loss. Fits 12GB
trivially; the gating signal is correctness, not representation quality.

## solution_notes
clip_loss builds the CE by hand (log_softmax via z - logsumexp) so the softmax denominator
over in-batch negatives is explicit; F.cross_entropy/F.log_softmax are forbidden because
they hide exactly that. siglip_loss normalization is sum-over-NxN / N (batch size), the
paper's Algorithm 1, NOT mean over N^2 (which is wrong by a factor of N). bias init -10 is
the paper's value; logit_scale init log(1/0.07) is CLIP's (SigLIP uses log(10), immaterial
for the toy). EOS pooling uses tokens.argmax(-1) because EOS is the largest id and
sequences are padded; [:, -1] would pool a pad token. The toy uses n_classes < batch so
two same-class pairs are in-batch negatives but semantically close (graded off-diagonal in
the viz). The small-batch InfoNCE-vs-SigLIP quality gap is NOT testable by overfitting one
tiny batch (overfitting is where InfoNCE is strong); only the N=1 structural difference is
deterministic, so that is what the test asserts and the gap goes in the viz with an honest
caption. A4 adds no shared symbol: A8's VLM needs an un-pooled patch-token tower, a
different output contract, so it builds its own.
