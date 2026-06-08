# assignments/a03_ssl/ASSIGNMENT.md

```yaml
id: a03_ssl
title: Self-supervised learning - MAE and DINO
module: 1
type: Core
estimated_learner_hours: 8
depends_on: [a00_harness, a01_transformer, a02_vit]
builds_into_shared_lib: []
forbidden_imports:
  - import timm
  - import transformers
  - from transformers
  - nn.MultiheadAttention
  - nn.TransformerEncoder
  - nn.TransformerEncoderLayer
  - nn.Transformer
  - torch.nn.functional.scaled_dot_product_attention
  - nn.LayerNorm
fits_12gb: true
external_data: "none for tests (optional CIFAR-10 for the linear-probe run)"
```

## motivation
Self-supervised learning trains a representation from unlabeled images by defining
a pretext task from the data itself. This assignment builds the two methods that
anchor the modern picture: MAE (mask 75% of patches, reconstruct the missing pixels
with an asymmetric encoder-decoder) and DINO (a student matches an EMA teacher's
prototype distribution across crops, with centering and sharpening to avoid
collapse). The README covers the SSL family map, why the 75% ratio matters, the
centering/sharpening collapse mechanics, and the line to DINOv2, iBOT, and I-JEPA.

## background
See the README for the full treatment. Shapes: images are (B, C, H, W); patch
tokens are (B, N, D) with N = (img/patch)^2 = 64 for 32x32 patch-4 inputs.

MAE. random_masking keeps n_keep = round((1 - r) N) tokens per sample via a random
permutation, returning x_kept (B, n_keep, D), a binary mask (B, N) in original
order (1 = masked), and ids_restore (B, N) the inverse permutation. The encoder
runs on visible tokens only; the decoder receives the full grid after a shared
learned mask token is appended and the set is unshuffled by ids_restore. The loss
is per-patch MSE on masked patches only, against per-patch-normalized pixels:

    L_mae = sum_i mask_i * mean_pix( (pred_i - target_i)^2 ) / sum_i mask_i

DINO. Each crop produces (B, K) prototype logits. The teacher distribution is
p_t = softmax((g_t - c) / tau_t) with stop-gradient (centering by c, sharpening by
a small tau_t); the student uses log p_s = log_softmax(g_s / tau_s). The loss sums
H(p_t, p_s) = -sum_k p_t(k) log p_s(k) over (teacher global crop, student crop)
pairs excluding the matched same-crop pair, averaged. The teacher is an EMA of the
student: theta_t <- m theta_t + (1 - m) theta_s. The center is an EMA of the batch
mean: c <- m_c c + (1 - m_c) mean(g_t). The collapse instrument is the mean teacher
entropy H = -sum_k p_t(k) log p_t(k).

## what_you_implement
- MAE random masking (the shuffle-keep-unshuffle trick).
- MAE mask-token assembly (append shared mask tokens, unshuffle to grid order).
- MAE masked-patch reconstruction loss.
- The DINO cross-view distillation loss with centering and sharpening.
- The EMA teacher update and the centering-buffer update.
- The teacher-entropy collapse instrument.

The ViT backbone, the MAE module wiring, the DINO student/teacher construction,
the multi-crop augmentation, the projection head, and the training-step wiring are
provided.

## tasks
- **Task 1 - random_masking** (`mae.py`, `random_masking`): x is (B, N, D).
  Compute n_keep = round((1 - mask_ratio) N), draw noise (B, N), argsort it to a
  random permutation ids_shuffle, take ids_restore = argsort(ids_shuffle). Gather
  the first n_keep shuffled indices into x_kept (B, n_keep, D). Build mask as
  n_keep zeros then ones in shuffled order and gather by ids_restore into original
  patch order (1 = masked). Return (x_kept, mask, ids_restore). Teaches how the
  encoder is made to see only visible tokens while keeping a way to put everything
  back in grid order.
- **Task 2 - append_mask_tokens** (`mae.py`, `append_mask_tokens`): x_enc
  is (B, n_keep, D_dec) encoded visible tokens already projected to decoder dim;
  ids_restore is (B, N); mask_token is (1, 1, D_dec). Broadcast mask_token to the
  N - n_keep masked slots, concatenate [x_enc; masks] in shuffled order, and gather
  by ids_restore so position i holds its encoded visible token or a mask token, in
  original patch order. Return (B, N, D_dec). Teaches the decoder-side assembly that
  inverts the masking shuffle.
- **Task 3 - mae_loss** (`mae.py`, `mae_loss`): pred and target are
  (B, N, p*p*C); target is per-patch-normalized pixels; mask is (B, N) with 1 on
  masked patches. Compute MSE per patch (mean over the pixel dim), then average over
  masked patches only using mask as the weight. Teaches "loss on masked patches
  only, per-patch normalized."
- **Task 4 - dino_loss** (`dino.py`, `dino_loss`): student_out is a list of
  (B, K) logits over all crops; teacher_out is a list of (B, K) logits over the
  global crops. Teacher p_t = softmax((teacher_out - center) / teacher_temp) with
  stop-gradient; student log p_s = log_softmax(student_out / student_temp). Sum
  H(p_t, p_s) over (teacher global crop, student crop) pairs excluding the matched
  same-index pair, averaged over the counted pairs. Teaches centering + sharpening
  and the cross-view distillation objective.
- **Task 5a - ema_update** (`dino.py`, `ema_update`): under no_grad, for
  every parameter and buffer set teacher <- momentum*teacher + (1-momentum)*student
  (copy integer buffers). Teaches the EMA teacher that gives a stable, slowly moving
  target.
- **Task 5b - update_center** (`dino.py`, `update_center`): teacher_out is
  (M, K); return center <- center_momentum*center + (1-center_momentum)*batch_mean,
  outside the autograd graph. Teaches the centering buffer that prevents single-mode
  collapse.
- **Task 6 - teacher_entropy** (`dino.py`, `teacher_entropy`): form
  p_t = softmax((teacher_out - center) / teacher_temp) and return the mean over the
  batch of H = -sum_k p_t(k) log p_t(k). Teaches the instrument the collapse test
  reads.

## tests
Run in this order:
1. `tests/test_shapes.py` - random_masking on (2, 64, D) with r=0.75 gives x_kept
   (2, 16, D), mask (2, 64) with 48 ones per row, ids_restore (2, 64); the MAE
   forward gives pred (2, 64, 48); DINO student/teacher heads give (B, K); the
   center buffer is (1, K) (shape).
2. `tests/test_gradcheck.py` - float64 gradcheck of the MAE encode->decode->loss
   pipeline w.r.t. the encoder patch-embed weight, and of dino_loss w.r.t. the
   student logits; the teacher params have requires_grad False and a student-loss
   backward leaves the teacher grads None (gradcheck + reference-value).
3. `tests/test_mae_masking.py` - random_masking keeps exactly round((1-r)N) tokens,
   mask has the complementary ones, the [kept; placeholder] set unshuffled by
   ids_restore returns visible tokens to their original positions, and the op is
   deterministic under a fixed seed (reference-value).
4. `tests/test_ema.py` - after ema_update the teacher params equal m*old + (1-m)*
   student; update_center moves the center toward the batch mean (reference-value).
5. `tests/test_mae_overfit.py` - the MAE memorizes one fixed batch (fixed mask)
   to masked-patch MSE < 0.05 (overfit-one-batch).
6. `tests/test_dino_overfit.py` - the DINO student loss falls below 0.85 * initial
   when trained to match a frozen teacher target (overfit-one-batch).
7. `tests/test_dino_collapse.py` - the centerpiece: three short variants on one
   synthetic batch, reading teacher_entropy. End-state entropies satisfy
   collapse (no centering) < full < uniform (no sharpening), with margins
   (reference-value).
8. `tests/test_forbidden_imports.py` - the solution uses no prebuilt
   attention/transformer module, fused SDPA, nn.LayerNorm, timm, or transformers in
   actual code (prose mentions allowed). Passes with the holes in place too.

## provided_boilerplate
`backbone.py` (identical at the top level and in solution): the ViT encoder (Conv2d patch
embed, learned PE with bicubic resize for smaller crops, the A1 transformer stack),
patchify / unpatchify / per_patch_normalize helpers, the DINO projection head
(MLP -> L2 normalize -> weight-normalized prototype linear), the DINOModel
(backbone + head), and the multi-crop augmentation. The MAE module wiring, the DINO
student/teacher construction (`build_student_teacher`), and the training step
(`dino_step`) are provided. `config.py` holds all hyperparameters. The learner
writes only the six mechanism bodies.

## compute_notes
Gating is overfit-one-batch and the collapse test on CPU, all on synthetic seeded
tensors with no download. MAE: encoder dim 64 depth 4, decoder dim 48 depth 2, patch
4 (N=64), 75% masking, 16-image batch, Adam lr 2e-3, 800 steps with a fixed mask,
reaching masked-patch MSE under 0.05. DINO: backbone dim 64 depth 4, K=128
prototypes, batch 8, Adam lr 1e-3, 150 steps. The collapse test runs three short
variants; a healthy full-DINO teacher entropy sits between 0 and log K = ln(128) =
4.85, no-centering falls to ~0, no-sharpening rises to ~log K. The tiny ViT fits
12GB trivially. The optional CIFAR-10 linear-probe run (frozen DINO features vs MAE
features vs random init) is described in the README; it is not part of the tests.

## stretch_goals
1. iBOT patch-level distillation on top of DINO: mask ~30% of the student's patch
   tokens, have the teacher see the full sequence, and add a per-patch cross-entropy
   on the masked positions. This is the bridge from DINO to DINOv2.
2. KoLeo regularizer: add -mean_i log(nearest-neighbor distance of feature i in the
   batch) to spread features across the hypersphere.
3. Linear probe on frozen features for DINO, MAE, and random init on CIFAR-10, and
   reproduce DINO probing above MAE at the same backbone size.
4. Swap MAE's pixel target for a latent target from an EMA encoder (the I-JEPA idea)
   and compare what the features encode.

## further_reading
- He et al., "Masked Autoencoders Are Scalable Vision Learners" (2022,
  arXiv:2111.06377) - MAE.
- Caron et al., "Emerging Properties in Self-Supervised Vision Transformers" (2021,
  arXiv:2104.14294) - DINO; Algorithm 1 is the implementation spec.
- Oquab et al., "DINOv2" (2023, arXiv:2304.07193) - the production backbone
  (DINO + iBOT + KoLeo + data curation).
- Zhou et al., "iBOT" (2021, arXiv:2111.07832) - patch-level distillation, the
  DINO-to-DINOv2 bridge.
- Grill et al., "Bootstrap Your Own Latent (BYOL)" (2020, arXiv:2006.07733) - the
  EMA-teacher predecessor that avoids collapse without negatives or centering.
- Assran et al., "I-JEPA" (2023, arXiv:2301.08243) - latent-space prediction
  instead of pixel reconstruction.

## solution_notes
All gating tests use synthetic seeded tensors; none download data. Seeds use
`set_seed(0)` and `torch.manual_seed`.

MAE overfit: the mask pattern is held fixed across steps (the loop re-seeds before
each forward) so the run is a clean memorization signal. With encoder dim 64 depth 4
and decoder dim 48 depth 2, 800 Adam steps at lr 2e-3 reach masked-patch MSE around
8e-3 on the test batch, well under the 0.05 threshold; the viz single-image run
reaches ~1e-4. The threshold is 0.05 (an order of magnitude of headroom). Per-patch
normalization makes the target zero-mean unit-variance per patch, so an untrained
decoder predicting the mean sits near MSE 1.0, which is the initial value.

DINO overfit: the test freezes the teacher and captures its outputs once, so the
target does not move, and trains the student with dino_loss only (no EMA, no
centering update). Two smooth low-frequency views share signal so the cross-view
target is learnable; the loss falls to about 0.74 * initial against the 0.85
threshold. The loss does not reach zero because the head's unit-norm features give
bounded cosine logits that cannot match the sharp teacher_temp target exactly. The
EMA and centering updates are exercised separately (test_ema, test_dino_collapse).

DINO collapse: K = 128 so log K = 4.85. The teacher uses the fast-tracking
collapse_momentum (0.9) so the degenerate state appears within ~150 steps. End-state
mean entropies (last 10 steps): no-centering ~0.0, full DINO ~1.3, no-sharpening
~4.85. The test asserts e_collapse < 0.5, e_uniform > 0.9 log K, and
e_collapse + 0.3 < e_full < e_uniform - 0.3; the observed values clear every margin.
The overfit and collapse tests need opposite teacher regimes (a stable target to
converge to vs a fast-tracking teacher that follows the student into collapse),
which is why config carries both overfit_momentum and collapse_momentum.

The forbidden-imports scan strips comments and docstrings via tokenize, so the
modules may name the forbidden symbols in prose. backbone.py uses nn.GELU and
nn.utils.parametrizations.weight_norm, neither of which is forbidden.
