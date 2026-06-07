# assignments/a02_vit/ASSIGNMENT.md

```yaml
id: a02_vit
title: Vision transformers, from scratch
module: 1
type: Core
estimated_learner_hours: 6
depends_on: [a00_harness, a01_transformer]
builds_into_shared_lib:
  - nanovision.primitives.ConvNeXtBlock
forbidden_imports:
  - nn.MultiheadAttention
  - nn.TransformerEncoder
  - nn.TransformerEncoderLayer
  - nn.Transformer
  - nn.LayerNorm
  - torch.nn.functional.scaled_dot_product_attention
  - timm
  - torchvision.models
fits_12gb: true
external_data: "none (optional CIFAR-10 via torchvision, ~170MB)"
```

## motivation
A ViT is the A1 transformer applied to image patches instead of text tokens: cut
the image into non-overlapping patches, project each to a token, add a class token
and positional embeddings, run the same encoder, and classify. There is no
2D-specific layer. The README covers the patch tokenizer, register tokens, the
CLS-vs-mean-pool choice, PE interpolation, and the ConvNeXt counterpoint with paper
links and the forward connections to CLIP, MAE/DINO, and BEV backbones.

## background
See the README for the worked shapes. The holes implement: the ConvNeXt block
(depthwise 7x7 conv -> channels-last LayerNorm -> Linear dim->4dim -> gelu ->
Linear 4dim->dim -> layer-scale -> residual); patch embedding as a strided Conv2d
(kernel = stride = patch) flattened to (B, N, dim) with N = (img/patch)^2; the token
sequence (prepend [CLS], add the learned absolute PE over [CLS]+patches, append
n_registers register tokens that get no PE); pooling (the [CLS] token at index 0, or
the mean over the N patch tokens, excluding CLS and registers); and bicubic PE
interpolation from an old_grid x old_grid layout to new_grid x new_grid with the CLS
row kept. Shapes: images are (B, C, H, W); patch tokens are (B, N, dim); the encoder
sees (B, 1 + N + n_registers, dim); logits are (B, num_classes).

## what_you_implement
- The ConvNeXt block (canonical in `nanovision/primitives.py`; the only new
  shared-library symbol).
- Patch embedding as a non-overlapping strided convolution.
- The ViT token sequence: CLS + learned PE + register tokens.
- The pooling head: CLS token vs mean over patch tokens.
- Bicubic positional-embedding interpolation for a new input resolution.
- The provided ViT assembly and `train_cifar.py` then overfit a synthetic batch as
  the integration check; the encoder stack itself is imported from A1.

## tasks
- **Task 1 - ConvNeXtBlock.forward** (`nanovision/primitives.py`, holed in
  `starter/primitives.py`): depthwise 7x7 conv, channels-last LayerNorm, inverted
  bottleneck (Linear dim->4dim, gelu, Linear 4dim->dim), optional layer-scale gamma,
  residual add. Input/output (B, dim, H, W). Teaches the modernized-ResNet design
  and that spatial mixing (depthwise conv) and channel mixing (the MLP) separate the
  same way attention and the FFN do.
- **Task 2 - PatchEmbed.forward** (`starter/vit.py`): Conv2d(in_chans, dim,
  kernel_size=patch, stride=patch), flatten the spatial grid, transpose to
  (B, N, dim). Teaches that patch embedding is one shared linear map per patch,
  exactly a strided convolution.
- **Task 3 - ViT token assembly** (`_assemble_tokens` in `starter/vit.py`): prepend
  the learnable [CLS] token, add the learned absolute PE over [CLS]+patches, then
  append n_registers learnable register tokens with no PE. Result
  (B, 1 + N + n_registers, dim). Teaches sequence construction and register tokens.
- **Task 4 - ViT pooling** (`_pool` in `starter/vit.py`): return the [CLS] token
  (index 0) or the mean over the N patch tokens (indices 1 .. 1+N), selected by
  `self.pool`. Teaches the CLS-vs-mean-pool choice and that the mean must exclude
  the register tokens.
- **Task 5 - interpolate_pos_embed** (`starter/vit.py`): bicubically resize the
  patch part of the PE table from old_grid^2 to new_grid^2 tokens (CLS row kept),
  returning (1, 1 + new_grid^2, dim). Teaches resolution generalization.

## tests
Run in this order (also in the README):
1. `tests/test_shapes.py` - ConvNeXt preserves (2,16,8,8); PatchEmbed
   (2,3,32,32)->(2,64,dim); ViT forward (2,3,32,32)->(2,num_classes); the encoder
   sequence length is 1+64+n_registers (shape).
2. `tests/test_gradcheck.py` - `check_gradients` at float64 on ConvNeXtBlock and
   PatchEmbed (gradcheck).
3. `tests/test_patch_equivalence.py` - the Conv2d patch embed equals unfold +
   linear with the conv weight reshaped to (C*p*p, dim) (reference-value).
4. `tests/test_pos_interp.py` - 8x8 -> 12x12 returns (1,1+144,d); same-grid is
   near-identity; a 32x32 ViT runs a 48x48 forward after swapping in the
   interpolated PE (reference-value + shape).
5. `tests/test_registers.py` - registers enter the sequence, receive gradients
   after a backward, and mean-pool excludes them and CLS (reference-value).
6. `tests/test_overfit.py` - the assembled ViT overfits 8 synthetic seeded images
   to cross-entropy < 0.02 in 500 steps for both pool="cls" and pool="mean"
   (overfit-one-batch).
7. `tests/test_forbidden_imports.py` - the solution and `nanovision/primitives.py`
   use no prebuilt attention/transformer module, fused SDPA, `nn.LayerNorm`, timm,
   or `torchvision.models` in actual code (mentions in prose are allowed). This one
   passes on starter too.

## provided_boilerplate
The A1 transformer encoder (imported, classic ViT config: LayerNorm + GELU MLP +
absolute PE, non-causal), the ViT module construction and forward plumbing, the
register/CLS/PE parameters and their init, `config.py`, the `train_cifar.py` wiring
to `nanovision.Trainer`, the CIFAR-10 loader, `Trainer` and `set_seed` from A0, and
the loss-curve and attention-rollout plotting. The learner writes only Tasks 1-5.

## compute_notes
Gating is overfit-one-batch on CPU: dim 64, patch 4 (N=64), depth 2, 4 heads, 4
register tokens, batch 8 synthetic images, Adam lr 3e-3, 500 steps, reaching
cross-entropy well under 0.02 for both pool modes in a few seconds. The tiny ViT
fits 12GB trivially. The optional real CIFAR-10 run (`solution/train_cifar.py`) is a
wiring sanity run, not a convergence run: a tiny ViT from scratch underfits CIFAR-10
without the DeiT recipe (RandAugment, MixUp, CutMix, label smoothing, stochastic
depth), which is the lesson - the inductive-bias deficit, not the architecture.

## stretch_goals
1. Swap the learned 1D PE for 2D sincos PE (MAE/DINOv2 style) and drop the bicubic
   interpolation; compare resolution transfer.
2. Token masking / patch dropout (the MAE mechanism): drop a fraction of patch
   tokens before the encoder; measure step time vs the drop rate.
3. Windowed (Swin) attention on the token grid: block-diagonal attention within
   w x w windows, then the shifted-window variant.
4. Train ConvNeXt and ViT at matched parameter count on CIFAR-10 with the DeiT
   recipe; compare the inductive-bias gap.

## further_reading
- Dosovitskiy et al., "An Image is Worth 16x16 Words" (2020, arXiv:2010.11929) -
  the original ViT.
- Touvron et al., "DeiT" (2021, arXiv:2012.12877) - the augmentation recipe that
  makes ViT train on small data.
- Liu et al., "A ConvNet for the 2020s" - ConvNeXt (2022, arXiv:2201.03545).
- Darcet et al., "Vision Transformers Need Registers" (2024, arXiv:2309.16588).
- He et al., "Masked Autoencoders Are Scalable Vision Learners" (2022,
  arXiv:2111.06377).
- Oquab et al., "DINOv2" (2023, arXiv:2304.07193).

## solution_notes
The overfit batch uses `set_seed(0)` and torch.randn images with random labels, so
the test is deterministic; final cross-entropy is well under the 0.02 threshold for
both pool modes. The patch-equivalence test holds to atol 1e-5: the only error is
float32 rounding because `F.unfold` orders rows (C, p_row, p_col), which matches the
conv weight's (C, p, p) flatten exactly, so the two computations are the same up to
arithmetic order. The PE-interpolation same-grid check uses atol 1e-4: bicubic
resampling onto a matching grid is near-identity but not bit-exact (the cubic kernel
touches neighbors), so a loose tolerance is correct. ConvNeXt uses layer scale
initialized at 1e-6, so an untrained block is close to the identity. The
forbidden-imports scan strips comments and docstrings via tokenize so the modules
can name the forbidden symbols in prose.
