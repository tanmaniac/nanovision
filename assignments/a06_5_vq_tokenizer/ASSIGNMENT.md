# assignments/a06_5_vq_tokenizer/ASSIGNMENT.md

```yaml
id: a06_5_vq_tokenizer
title: VQ tokenizer - vector quantization and a discrete autoregressive prior
module: 2
type: Core
estimated_learner_hours: 6
depends_on: [a00_harness, a01_transformer, a02_vit]
builds_into_shared_lib: [nanovision.quantize]
forbidden_imports:
  - import vector_quantize_pytorch
  - from vector_quantize_pytorch
  - import taming
  - from taming
  - import diffusers
  - from diffusers
fits_12gb: true
external_data: "none (synthetic shape images)"
```

## motivation
A VQ-VAE turns an image into a short grid of discrete tokens from a fixed vocabulary, so an
autoregressive transformer can generate images the same way it generates text. The mechanism
is the content: a learned codebook with a nearest-neighbor lookup, the straight-through
estimator that trains through the non-differentiable argmin, the codebook and commitment
losses, and an autoregressive prior over the token grid. This is the discrete-token route
(Chameleon, LlamaGen), the contrast to A7's continuous KL-VAE. See the README for the math.

## background
See the README. Encoder z_e (B,D,4,4); codebook E (K,D); nearest code per vector by
argmin ||z_e - E_k||^2; z_q = E[idx]. Straight-through: z_q_ste = z_e + (z_q - z_e).detach()
so the forward is the hard code and the gradient to the encoder is identity. Losses:
reconstruction MSE + codebook ||sg[z_e]-z_q||^2 + beta||z_e-sg[z_q]||^2, beta=0.25. Images are
in [-1,1] (decoder ends in tanh). The prior is a causal transformer over the 16 flattened
tokens with a learned BOS at index K (vocab K+1); training is teacher-forced next-token CE.

## what_you_implement
- VectorQuantizer.forward (the shared nanovision.quantize symbol): nearest-code lookup, the
  straight-through estimator, the codebook + commitment losses.
- vq_vae_loss: reconstruction MSE + the quantizer's vq loss.
- ar_sample: autoregressive sampling of the token grid from the prior.

The encoder/decoder, VQVAE wiring, TokenPrior + ar_nll, codebook_perplexity, config, toy
data, and viz are provided.

## tasks
1. `VectorQuantizer.forward` (`quantize.py`, shared as `nanovision.quantize.VectorQuantizer`):
   flatten z_e (B,D,H,W) to (B*H*W,D) channel-last (contiguous); squared distance to the K
   codes; argmin -> indices; z_q = codebook[idx] reshaped back (contiguous); codebook =
   ||sg[z_e]-z_q||^2, commit = ||z_e-sg[z_q]||^2, vq_loss = codebook + beta*commit;
   z_q_ste = z_e + (z_q-z_e).detach(); return (z_q_ste, indices (B,H,W), vq_loss).
2. `vq_vae_loss` (`vqvae.py`): (x_hat-x)^2 mean + vq_loss.
3. `ar_sample` (`prior.py`): start from [BOS]; L=H*W steps of run-prior, softmax last-position
   logits, multinomial sample, append; drop BOS, reshape to (n,H,W).

## tests
Run in this order:
1. `tests/test_shapes.py` - VQVAE x_hat/indices/vq_loss; quantizer z_q/idx/loss; prior logits
   (B,L,K) (shape).
2. `tests/test_quantize.py` - indices match a brute-force nearest neighbor; z_q_ste equals the
   hard codebook lookup; vq_loss equals the codebook + 0.25*commitment reference
   (reference-value).
3. `tests/test_straight_through.py` - the centerpiece: z_e.grad is None pre-backward then all
   ones after backprop through z_q_ste (the STE identity gradient); the forward value is the
   hard quantize (autograd-gradient check, not a finite-difference gradcheck, since the STE
   makes those disagree by design).
4. `tests/test_recon_overfit.py` - the VQVAE overfits one shape-image batch to recon MSE <
   0.05 and codebook perplexity > 3 (no collapse) (overfit-one-batch).
5. `tests/test_prior_overfit.py` - the AR prior CE drops from ~ln(K)=3.47 toward its
   position-0 floor ln(B)/L (~0.13 here), under 0.2 and under 0.1*start (overfit-one-batch).
6. `tests/test_ar_sample.py` - sampling returns (n,H,W) indices in [0,K), deterministic under
   a fixed generator (reference-value).
7. `tests/test_forbidden_imports.py` - no vector_quantize_pytorch/taming/diffusers VQ; scans
   top-level + solution + the nanovision.quantize shim. Passes with the holes in place too.

There is no gradcheck test: the straight-through estimator deliberately makes the autograd
gradient differ from finite differences, so test_straight_through asserts the autograd
gradient (identity) directly instead.

## provided_boilerplate
`vqvae.py` `Encoder` (strided convs + a ConvNeXt block from `nanovision.primitives`),
`Decoder` (transpose convs + tanh to [-1,1]), `VQVAE` (encode -> quantize -> decode +
`decode_indices`). `prior.py` `TokenPrior` (token embedding over K+1 codes + a causal
`nanovision.transformer.TransformerDecoder` + a head over K codes) and `ar_nll` (teacher-forced
next-token CE with the BOS shift). `quantize.py` `codebook_perplexity`. `config.py`,
`nanovision.data.toy.diffusion_image_batch`, and `viz.py` (reconstructions, the code-usage
histogram + perplexity, decoded AR samples).

## compute_notes
16x16 images, a 4x4 latent grid, K=32 codes, all tests on CPU in seconds to under a minute.
The VQ-VAE overfit is 1500 Adam steps on 8 images (recon MSE well under 0.05, perplexity ~4).
The prior overfit is 1000 steps on 8 token grids. Fits 12GB trivially.

## solution_notes
The straight-through estimator z_q_ste = z_e + (z_q-z_e).detach() has forward value z_q (the
detach does not change the value) and gradient d z_q_ste/d z_e = identity, so the decoder
gradient reaches the encoder unchanged; the codebook never updates through this path (the code
is detached), only through the codebook loss. Stop-gradient placement: codebook = ||sg[z_e] -
z_q||^2 trains the codes, commitment = ||z_e - sg[z_q]||^2 trains the encoder; beta=0.25 is
van den Oord's value. The permute/reshape around quantization uses .contiguous() so each
location maps back to its own code. The prior needs a learned BOS (index K, vocab K+1) to have
an input at position 0; ar_nll and ar_sample share that convention. The prior CE floors near
ln(B)/L because position 0 sees the identical [BOS] context for every grid in the batch, so
its cross-entropy cannot beat the entropy of the B distinct first tokens - the overfit
assertion accounts for this. No gradcheck: the STE makes autograd disagree with finite
differences on purpose. A6.5 owns the new shared symbol nanovision.quantize (VectorQuantizer +
codebook_perplexity); vqvae.py and the tests import it only via nanovision.quantize, never
bare, so its module identity stays single.
