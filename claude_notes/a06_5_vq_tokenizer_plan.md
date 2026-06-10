# A6.5 - VQ tokenizer: build plan

Status: plan for expert review before any code. Module 2 (Generative), compact. Depends on
A1 (the transformer, reused as the autoregressive prior) and A2 (ConvNeXt blocks for the
encoder/decoder). No external research note exists for this assignment, so the expert review
is the sole correctness gate. Forbidden imports: `vector_quantize_pytorch`,
`taming` / `taming-transformers`, `diffusers` VQ models, any prebuilt codebook/quantizer.

Directory: `assignments/a06_5_vq_tokenizer/` (the `_5` suffix; A6 flow matching is the
integer sibling, per the naming convention).

## Goal and scope

The learner builds a VQ-VAE: a convolutional encoder maps an image to a grid of continuous
vectors, a learned codebook snaps each to its nearest discrete code (the vector
quantization), and a decoder reconstructs the image from the quantized grid. The
non-differentiable argmin is trained through with the straight-through estimator (STE). Then
a tiny autoregressive transformer (reused from A1) is trained over the discrete token grid as
a prior, and sampling from it decodes through the codebook + decoder to new images. This is
the discrete-token route to generation (Chameleon, Janus, LlamaGen), the contrast to A7's
continuous KL-VAE.

## What the student implements (holes) vs provided

Three holes, kept compact:

### Files (top-level holes; `solution/<file>.py` is the answer key)

`quantize.py` (the shared symbol, exposed as `nanovision.quantize.VectorQuantizer`)
- HOLE `VectorQuantizer.forward(z_e)`: z_e is the encoder output `(B, D, H', W')`.
  1. Move D to the last axis and flatten to `(B*H'*W', D)`.
  2. Nearest code per vector: distances `||z_e - e_k||^2` to the `K` codebook embeddings
     (use `||z_e||^2 - 2 z_e @ E^T + ||E||^2`), `indices = argmin_k`.
  3. `z_q = E[indices]`, reshaped back to `(B, D, H', W')`.
  4. Straight-through: return `z_q_ste = z_e + (z_q - z_e).detach()` so the decoder sees the
     quantized values but the gradient flows to the encoder as identity.
  5. VQ loss: `codebook = ||sg[z_e] - z_q||^2` (pulls codes toward the encoder),
     `commitment = ||z_e - sg[z_q]||^2` (pulls the encoder toward its code),
     `vq_loss = codebook + beta * commitment` with `beta = 0.25` (van den Oord et al. 2017).
  Returns `(z_q_ste, indices (B, H', W'), vq_loss)`.

`vqvae.py` (local)
- HOLE `vq_vae_loss(x, x_hat, vq_loss)`: total objective
  `mse(x_hat, x) + vq_loss`. (The reconstruction term plus the quantizer's vq term; trivial
  to assemble but it names the full objective and is what the overfit test drives.)
- PROVIDED: `Encoder` (a couple of strided convs / ConvNeXt blocks down to a small grid),
  `Decoder` (transpose-conv upsample back), `VQVAE` module wiring (encode -> quantize ->
  decode), all using `nanovision.quantize.VectorQuantizer` and ConvNeXt blocks via
  `nanovision.primitives`.

`prior.py` (local)
- HOLE `ar_sample(prior, n, grid_hw, num_codes, *, generator=None)`: autoregressively sample
  a token grid. Flatten order row-major; start from an empty/BOS context, at each step run
  the transformer prior, take the next-token logits, sample (multinomial), append, until
  `H'*W'` tokens are produced; reshape to `(n, H', W')`. Returns the integer token grid.
- PROVIDED: `TokenPrior` (a decoder-only transformer from `nanovision.transformer` with a
  token embedding over `K` codes + a small positional scheme, causal), and `ar_nll`, the
  next-token cross-entropy training loss (teacher forcing over the flattened grid), since the
  learner already built that exact loss in A1's char-LM. Sampling is the new mechanism.

`config.py` PROVIDED: `VQConfig` (img 16x16x1 or x3, encoder downsample to 4x4, code_dim 16,
num_codes 32, beta 0.25; prior dim/depth/heads small).

`viz.py` PROVIDED: original vs reconstruction grid; the codebook-usage histogram (how many of
the K codes are used, the collapse diagnostic); AR-prior samples decoded to images.

### Toy data

Reuse `nanovision.data.toy.diffusion_image_batch` (the three shape classes), which gives a
multimodal image set the VQ-VAE can reconstruct and the AR prior can model. No new toy
needed. (Question for the expert: are 32 codes over a 4x4 grid enough to reconstruct three
shape classes at 16x16, and to make AR-prior overfit-one-batch reach low CE, without
codebook collapse swamping the toy?)

### Shared symbol / shim

`nanovision/quantize.py` NEW shim: `VectorQuantizer = load("a06_5_vq_tokenizer",
"quantize").VectorQuantizer` (owned by a06_5). `vqvae.py` and the tests import the quantizer
ONLY via `nanovision.quantize`; A6.5's own `quantize.py` is the owning file. This is the
first new shared symbol since the layout restructure; it follows the same pattern as
primitives/attention/transformer.

## Tests (each fast; per-assignment pytest process)

- `test_shapes.py`: encoder `(B,1,16,16) -> (B,D,4,4)`; `VectorQuantizer` returns z_q_ste
  `(B,D,4,4)`, indices `(B,4,4)` in `[0,K)`, scalar vq_loss; decoder back to `(B,1,16,16)`;
  prior logits `(B, L, K)`.
- `test_quantize.py`: z_q entries equal the nearest codebook vectors (compare against an
  independent brute-force argmin); indices match; vq_loss equals the codebook + 0.25*commit
  reference on fixed inputs.
- `test_straight_through.py`: the centerpiece. With `z_e` requiring grad,
  `VectorQuantizer(z_e)` then `z_q_ste.sum().backward()` gives `z_e.grad` all ones (the STE
  passes the gradient through the argmin as identity); and the forward `z_q_ste` value equals
  the hard quantized `z_q` (the detach does not change the forward value).
- `test_gradcheck.py` (float64): `vq_vae_loss` through the encoder->quantize->decode path is
  differentiable w.r.t. the encoder output (the STE makes this well-defined).
- `test_recon_overfit.py`: the VQVAE reconstructs one fixed shape-image batch to MSE below a
  small threshold (overfit-one-batch); codebook usage > 1 (it did not collapse to a single
  code).
- `test_prior_overfit.py`: the AR prior's cross-entropy on one fixed token batch drops below
  a small threshold (it memorizes the token sequences).
- `test_ar_sample.py`: `ar_sample` returns `(n, H', W')` integer indices all in `[0, K)`,
  deterministic under a fixed generator.
- `test_forbidden_imports.py`: no `vector_quantize_pytorch`, `taming`, prebuilt VQ/codebook;
  scan top-level + solution + the `nanovision.quantize` shim.

## Open questions for the expert

1. STE formulation: is `z_q_ste = z_e + (z_q - z_e).detach()` the correct and standard
   straight-through, and is `sg[z_e]` vs `sg[z_q]` assigned to the codebook vs commitment
   term the right way round (codebook loss updates the codes, commitment updates the
   encoder)? Confirm beta=0.25.
2. Codebook collapse: at K=32 over a 4x4 grid, is plain codebook-loss training enough to keep
   several codes alive on the toy, or should the assignment use EMA codebook updates (van den
   Oord appendix) and/or a dead-code reinit? Keep EMA as a stretch, or make it core?
3. Should the reconstruction loss be MSE in pixel space (simple) for the toy, with the
   perceptual + GAN losses of VQ-GAN (Esser et al. 2021) named only in the README, or is
   pixel MSE too weak to teach anything real? (Plan: pixel MSE core, VQ-GAN README-only.)
4. AR prior over the token grid: row-major raster order with a causal transformer and a
   learned positional embedding - correct and sufficient, or does the toy need 2D-aware
   positions? Is teacher-forced next-token CE the right training loss?
5. Is reusing `diffusion_image_batch` (3 shape classes) a good toy, or does VQ need more
   texture to be a meaningful reconstruction target?

## Compute notes

16x16 images, a 4x4 latent grid, K=32 codes, a small prior. All tests run on CPU in seconds
to under a minute. The overfit tests train one fixed batch. Fits 12GB trivially.

## README (lecture notes) outline

The discrete-representation idea (why quantize at all: a discrete bottleneck gives a finite
token vocabulary that an autoregressive model can model exactly like text), VQ-VAE (van den
Oord et al. 2017) and the codebook nearest-neighbor lookup, the straight-through estimator
(the argmin has zero gradient almost everywhere, so STE copies the decoder gradient straight
to the encoder), the codebook and commitment losses and codebook collapse, VQ-GAN (Esser et
al. 2021) adding perceptual + adversarial losses for sharp reconstructions (named, not
built), the autoregressive prior over tokens and sampling, and the forward pointers: the
unified discrete multimodal models (Chameleon, Janus, LlamaGen) tokenize images this way and
model text + image tokens with one transformer; A7 contrasts this with a continuous KL-VAE +
diffusion. All math as real LaTeX. Mermaid for the encode -> quantize -> decode and the
prior-sampling flow. Verify every arXiv id before citing (van den Oord 1711.00937, VQ-GAN
2012.09841, Chameleon 2405.09818, LlamaGen 2406.06525 - the expert should confirm these).

## Expert review: corrections folded in

The reviewer verified the core math (STE forward/backward, the two loss terms and their
stop-gradients, beta=0.25, the distance expansion, the shape flow, the grad-cleanliness of
the STE test). Folded corrections:
- BOS token (MUST-FIX): the causal prior cannot sample position 0 from an empty context.
  Vocab is K+1 with BOS = index K. Training input is `[BOS, t_0..t_14]`, targets
  `[t_0..t_15]` (the head predicts only the K real codes). `ar_sample` starts from `[BOS]`.
  `TokenPrior`, `ar_nll`, and `ar_sample` share this convention.
- Collapse check: use perplexity `exp(-sum_k p_k log p_k)` of the code-usage distribution
  (1 = total collapse, K = uniform), not "usage > 1". `codebook_perplexity(indices, K)` is a
  provided helper, wired into the test and viz. Threshold set from a measured run.
- All test thresholds pinned to measured runs: recon MSE, prior CE (also assert it dropped
  from the `ln K ~= 3.466` start toward ~0), and the vq_loss reference computed with
  identical stop-gradient placement and beta.
- Image range is [-1, 1] (the toy's convention); the decoder ends in `tanh`, and MSE is in
  that space.
- `test_straight_through` asserts `z_e.grad is None` before backward, then checks all-ones
  after, so a stale grad cannot mask a bug.
- The permute/reshape around quantization is inverse-consistent with `.contiguous()` so
  location <-> code alignment cannot break.
- Codebook init: uniform `U(-1/K, 1/K)` (van den Oord's choice).
- EMA codebook updates, dead-code reinit, cosine-distance (L2-normalized) quantization, and
  VQ-GAN's perceptual+adversarial losses are named in the README as the production tricks,
  not built (keeping the assignment compact); plain codebook-loss VQ is the core path. If a
  measured overfit run shows the toy collapsing, revisit adding EMA.

## Build order

1. Expert review of this plan; fold corrections.
2. `quantize.py` (+ `nanovision.quantize` shim), `vqvae.py`, `prior.py` solutions + holes;
   encoder/decoder/config/viz provided.
3. Tests; verify both modes.
4. `viz.py`.
5. Lecture-notes README via the skill, then the context-less style-review pass.
6. Commit and push.
