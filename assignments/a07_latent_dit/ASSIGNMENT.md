# assignments/a07_latent_dit/ASSIGNMENT.md

```yaml
id: a07_latent_dit
title: Latent diffusion and a tiny DiT
module: 2
type: Core
estimated_learner_hours: 9
depends_on: [a00_harness, a01_transformer, a05_diffusion, a06_0_flow_matching]
builds_into_shared_lib: []
forbidden_imports:
  - import diffusers
  - from diffusers
  - import timm
  - from timm
  - import torchvision
  - from torchvision
  - nn.Transformer
  - TransformerEncoder
  - TransformerDecoder
fits_12gb: true
external_data: "none (synthetic shape images from nanovision.data.toy.diffusion_image_batch)"
```

## motivation
See the README. Two stages: a KL-regularized VAE compresses a 16x16x1 image into a 4x4x4
continuous latent, and a diffusion transformer (DiT) predicts a flow-matching velocity in
that latent space, conditioned on timestep and class label. The flow-matching objective is
reused from the flow-matching assignment unchanged; the denoiser now operates on a latent and
uses a transformer with adaLN-Zero conditioning instead of a U-Net.

## background
See the README. Convention (identical to the flow-matching assignment): t=0 is noise
x0~N(0,I), t=1 is data x1 (the VAE latent). Path x_t=(1-t)x0+t*x1, velocity u=x1-x0. Latent
downsample factor f=4 (16->4 per axis), C=4 latent channels, patch size p=1 -> N=16 tokens,
d_model=64, n_heads=2, n_blocks=4. beta=1e-4 (small KL weight). Do NOT use the research note's
opposite t-convention.

## what_you_implement
- reparameterize, kl_divergence, vae_loss (vae.py): the VAE's stochastic sample, its KL
  regularizer, and the recon + beta*KL loss.
- modulate, patchify, unpatchify (dit.py): the adaLN affine and the latent<->token reshape.
- DiTBlock.forward (dit.py): the adaLN-Zero transformer block.

The VAE Encoder/Decoder/KLVAE, the DiT wiring + timestep_embedding, the flow loss + Euler
sampler (flow.py), config, and viz are provided.

## tasks
1. `reparameterize(mu, logvar)` (`vae.py`): z = mu + exp(0.5*logvar)*eps, eps=torch.randn_like(mu).
   Returns (B,C,4,4).
2. `kl_divergence(mu, logvar)` (`vae.py`): 0.5*(exp(logvar)+mu^2-1-logvar), sum over (C,H,W),
   mean over B. Scalar.
3. `vae_loss(x, x_hat, mu, logvar, beta)` (`vae.py`): recon = ((x_hat-x)^2) summed per image,
   mean over batch; total = recon + beta*kl_divergence(mu,logvar). Return (total, recon, kl).
4. `modulate(x, shift, scale)` (`dit.py`): x*(1+scale.unsqueeze(1)) + shift.unsqueeze(1).
   x is (B,N,d); shift/scale are (B,d).
5. `patchify(z, p)` (`dit.py`): (B,C,H,W)->(B,N,p*p*C), N=(H/p)(W/p), row-major. Reshape/permute.
6. `unpatchify(tokens, p, C, H, W)` (`dit.py`): exact inverse of patchify.
7. `DiTBlock.forward(x, c)` (`dit.py`): chunk adaLN_modulation(c) into 6; gated attn and mlp
   residual branches with modulate(norm(x), shift, scale). See the docstring in dit.py.

## tests
Run in this order:
1. `tests/test_shapes.py` - encoder mu/logvar (B,4,4,4); reparameterize (B,4,4,4); decoder
   (B,1,16,16); DiT(z,t,y) (B,4,4,4).
2. `tests/test_patchify_roundtrip.py` - unpatchify(patchify(z,p))==z exactly for p in {1,2};
   token count (H/p)(W/p).
3. `tests/test_kl.py` - kl_divergence matches a hand-computed closed form (constant mu/logvar)
   and is 0 at mu=0,logvar=0.
4. `tests/test_gradcheck.py` - float64 gradcheck of kl_divergence and a single DiTBlock.forward
   (adaLN Linear perturbed off zero). reparameterize is NOT gradchecked (random eps).
5. `tests/test_adaln_identity.py` - a fresh DiTBlock returns x within 1e-6; the full DiT
   predicts all-zeros at init within 1e-6.
6. `tests/test_vae_overfit.py` - 800 Adam steps on 8 images; recon < 5.0 (floors ~0.008).
7. `tests/test_dit_overfit.py` - 2000 Adam steps on a fixed deterministic target; final < 0.05
   and final < 0.01*first (floors ~8e-4, ratio ~6e-6).
8. `tests/test_forbidden_imports.py` - no diffusers/timm/torchvision/nn.Transformer; passes
   with the holes in place too.

## provided_boilerplate
`vae.py` `Encoder`/`Decoder`/`KLVAE` (conv VAE; GroupNorm groups 8 if c%8==0 else 1). `dit.py`
`timestep_embedding` (sinusoidal, copied from the diffusion U-Net), `DiTBlock.__init__`
(norm1/norm2 elementwise_affine=False, MultiHeadAttention from nanovision.attention, GELU MLP,
adaLN_modulation Sequential with zero-init final Linear), `DiT` (patch_embed, learned pos
embed, class embed, time MLP, block stack, zero-init final adaLN + output head). `flow.py`
`cfm_loss` (takes x0 as arg) and `euler_sample` (conditioned on y). `config.py` `DiTConfig`.
`nanovision.data.toy.diffusion_image_batch` (16x16 shape images, 3 classes). `viz.py` trains
the VAE then the DiT and decodes one sample per class.

## compute_notes
All tests CPU, seconds. VAE overfit: per-image-sum recon drops from ~248 to ~0.008 in 800
steps (threshold 5.0 is comfortable). DiT overfit: flow loss drops from ~144 to ~8e-4 (ratio
~6e-6) in 2000 steps on a FIXED target; resampling x0 each step would inject irreducible
variance and break the relative threshold. KL stays ~200 at beta=1e-4 by design (recon
dominant). viz fits 12GB trivially.

## solution_notes
modulate MUST unsqueeze shift/scale from (B,d) to (B,1,d) before the affine, or the (B,N,d)
broadcast is wrong. adaLN_modulation final Linear, the final-layer adaLN Linear, and the
output-head Linear are ALL zero-init (weight and bias) so the DiT predicts zeros at init; the
adaln-identity test depends on it. patchify uses reshape (B,C,nh,p,nw,p) -> permute
(0,2,4,1,3,5) -> reshape; unpatchify reverses it. beta=1e-4 (NOT 1e-2): 1e-2 is beta-VAE
territory and fights recon under the sum reductions. Flow convention is the flow-matching
assignment's (t=0 noise, t=1 data, u=x1-x0); cfm_loss takes x0 as an argument so the overfit
target is deterministic. DiTBlock gradcheck perturbs the zero-init adaLN Linear off zero so it
tests a non-trivial map. reparameterize is excluded from gradcheck (fresh eps per call makes
the finite-difference reference ill-posed). Measured seeds: torch.manual_seed(0) +
Generator(0) for both overfit tests.
