# A7 - latent diffusion and a tiny DiT: build plan

Status: draft for expert review. Build subagent must read this whole file plus
`docs/agent_build_guide.md` and mirror the exemplar `assignments/a06_0_flow_matching`.

## What this assignment teaches

Two stages that sit under every diffusion image generator since 2022 (LDM, SD3, FLUX):

1. A small KL-regularized VAE compresses a 16x16 image into a continuous 4x4 latent with C=4
   channels. The latent is roughly Gaussian (a light KL penalty) but keeps spatial structure,
   so the decoder reconstructs cleanly. This is the continuous-latent route, the contrast to
   the discrete VQ codebook from the VQ tokenizer assignment.
2. A DiT (diffusion transformer, Peebles and Xie 2023) replaces the U-Net denoiser. It
   patchifies the latent into tokens, runs transformer blocks with adaLN-Zero conditioning on
   timestep + class label, and predicts a flow-matching velocity. Sampling integrates the
   velocity ODE in latent space, then the frozen VAE decoder turns the sampled latent into an
   image.

The forward process is the linear-interpolant conditional flow matching from the flow-matching
assignment, reused unchanged except that the denoiser now sees a latent instead of a pixel
image, and is conditioned on the class label. Flow matching (not DDPM) is the deliberate
choice: SD3, FLUX, and SiT all train DiTs with rectified flow, and it is simpler to code than
the DDPM posterior.

The new mechanisms the student implements are the VAE's reparameterization + KL + loss, and
the DiT's adaLN-Zero block + patchify/unpatchify. Everything else (encoder/decoder conv
stacks, the DiT wiring, the flow-matching loss and Euler sampler) is provided, because those
were taught earlier or are scaffolding.

## Convention (read this - it is the most dangerous trap)

A7 uses the SAME flow-matching convention as the flow-matching assignment, NOT the convention
in the research note `docs/research/a07_latent_dit.md`. They are opposite and silently mixing
them flips a sign.

- t = 0 is noise $x_0 \sim \mathcal{N}(0, I)$; t = 1 is data $x_1$ (here, the VAE latent).
- Path: $x_t = (1-t)\,x_0 + t\,x_1$. Conditional velocity: $u = x_1 - x_0$ (constant in t).
- Loss target is $u = x_1 - x_0$; the network predicts velocity $v_\theta(x_t, t, y)$.
- Sampling integrates $\mathrm{d}x/\mathrm{d}t = v_\theta$ forward from t=0 to t=1 with Euler.

The research note writes $x_t = (1-t)x_0 + t\,\varepsilon$, $v = \varepsilon - x_0$, which has
t=0 as data and t=1 as noise. Do NOT use that. Match the flow-matching assignment's
`solution/path.py` and `solution/flow.py` exactly. This mirrors the score-velocity sign issue
already resolved in that assignment.

## Toy data

`nanovision.data.toy.diffusion_image_batch(n, num_classes=3, size=16, channels=1, seed)` already
exists (built for the diffusion assignment): 16x16 grayscale shapes in [-1, 1], 3 classes
(disk, square, cross), class label does not pin position/size. A7 reuses it directly. No new
toy data needed. The class label is the conditioning signal for the DiT.

## Shapes (fix these numbers)

- Image: (B, 1, 16, 16), values in [-1, 1].
- VAE encoder output: (B, 2C, 4, 4) = (B, 8, 4, 4), split into mu, logvar each (B, 4, 4, 4).
  Downsample factor f = 4 (16 -> 8 -> 4 via two stride-2 convs).
- Latent z: (B, C, 4, 4) = (B, 4, 4, 4).
- DiT patchify: patch_size p = 1, so N = (4/1)(4/1) = 16 tokens, each of dim p*p*C = 4,
  linearly projected to d_model = 64.
- DiT: d_model = 64, n_heads = 2, n_blocks = 4, mlp_ratio = 4. Bidirectional self-attention
  (no causal mask, like the ViT encoder).
- DiT output: velocity, same shape as z, (B, C, 4, 4).

p = 1 is intentional: with a 4x4 latent, a patch size of 2 leaves only 4 tokens, too few for
the attention to be interesting. The patchify/unpatchify code must still be written generally
(parameterized by p) and tested for the round trip; p = 1 is just the configured value.

## Files

Mirror the flow-matching assignment's layout exactly. Holed files have a top-level copy (with
`raise NotImplementedError` holes) and an identical `solution/` copy with the holes filled.
Provided files live only at the top level.

### `vae.py` (holed; solution copy required)

Holes the student fills:
- `reparameterize(mu, logvar) -> z`: $z = \mu + \exp(0.5\,\log\sigma^2)\,\varepsilon$,
  $\varepsilon \sim \mathcal{N}(0, I)$ same shape as mu. Use `torch.randn_like`.
- `kl_divergence(mu, logvar) -> Tensor`: the closed-form KL from $\mathcal{N}(\mu, \sigma^2)$
  to $\mathcal{N}(0, I)$, summed over latent dims and averaged over the batch:
  $\mathrm{KL} = \tfrac12 \sum (\exp(\log\sigma^2) + \mu^2 - 1 - \log\sigma^2)$, sum over
  (C, H, W) then mean over B. Must match the analytic test exactly.
- `vae_loss(x, x_hat, mu, logvar, beta) -> (loss, recon, kl)`: recon = MSE summed over pixels
  per image then mean over batch (match the per-image-sum, batch-mean reduction used by the
  flow-matching loss); total = recon + beta * kl. Return the three scalars so the test and viz
  can see the split.

Provided (identical in both copies):
- `Encoder(nn.Module)`: two stride-2 conv blocks (Conv2d + GroupNorm + SiLU), 1 -> 32 -> 64
  channels with spatial 16 -> 8 -> 4, then a 1x1 conv to 2C = 8 channels. `forward(x)` returns
  mu, logvar by chunking the 8 channels into two (B, 4, 4, 4) halves.
- `Decoder(nn.Module)`: 1x1 conv C -> 64, then two nearest-upsample + conv blocks (GroupNorm +
  SiLU) 4 -> 8 -> 16 spatial, 64 -> 32 -> 1 channels, final `nn.Tanh` (images live in [-1, 1],
  same as the decoder in the VQ tokenizer assignment).
- `KLVAE(nn.Module)`: wraps Encoder + Decoder. `encode(x)` -> mu, logvar; `decode(z)` -> x_hat;
  `forward(x)` -> x_hat, mu, logvar using `reparameterize`.

Reuse `nanovision.primitives` for any building block that already exists there (check what is
exported); otherwise plain `nn` modules. GroupNorm groups: use 8 if channels % 8 == 0 else 1
(match the `_groups` helper pattern in the diffusion assignment's U-Net).

### `dit.py` (holed; solution copy required)

Holes the student fills:
- `modulate(x, shift, scale) -> Tensor`: $x \cdot (1 + \text{scale}) + \text{shift}$, where x is
  (B, N, d) and shift/scale are (B, d). They must be unsqueezed to (B, 1, d) INSIDE `modulate`
  before the multiply/add so they broadcast over the N token axis - x*(1+scale)+shift with
  scale as (B, d) does NOT broadcast against (B, N, d) and will error or broadcast wrong. This
  is the adaLN affine; mirror the reference `modulate` (DiT, facebookresearch/DiT models.py)
  which does `scale.unsqueeze(1)` / `shift.unsqueeze(1)`.
- `patchify(z, p) -> Tensor`: (B, C, H, W) -> (B, N, p*p*C) with N = (H/p)(W/p), row-major
  patch order. Pure reshape/permute, no projection (the linear patch-embed projection is a
  separate provided `nn.Linear`).
- `unpatchify(tokens, p, C, H, W) -> Tensor`: exact inverse of patchify, (B, N, p*p*C) ->
  (B, C, H, W). The round-trip `unpatchify(patchify(z)) == z` must hold exactly.
- `DiTBlock.forward(self, x, c)`: the adaLN-Zero block. Given conditioning c (B, d):
  - regress 6 modulation params from c: `shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp,
    gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)`, each (B, d).
  - `x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))`
  - `x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))`
  - norm1, norm2 are `nn.LayerNorm(d, elementwise_affine=False)` (the affine comes from adaLN).
  - attn is bidirectional self-attention via `nanovision.attention.MultiHeadAttention` (no
    mask), matching the ViT encoder's use. mlp is the provided FFN.

Provided (identical in both copies):
- `DiTBlock.__init__`: builds norm1, norm2, attn (`MultiHeadAttention` from
  `nanovision.attention`), mlp (a 2-layer GELU MLP, hidden = mlp_ratio * d), and
  `self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))` with the final Linear
  weight AND bias zero-initialized. The zero-init is what makes the block an identity at init
  (gate = 0 so both residual branches contribute 0). State this in a comment.
- `timestep_embedding(t, dim)`: the same sinusoidal embedding used in the diffusion assignment's
  U-Net (copy it; cos-then-sin, log-spaced freqs). Provided.
- `DiT(nn.Module)`: patch-embed `nn.Linear(p*p*C, d)`, a learned positional embedding
  `nn.Parameter` of shape (1, N, d), a class-embedding table `nn.Embedding(num_classes, d)`, a
  timestep MLP `nn.Sequential(Linear(time_dim, d), SiLU, Linear(d, d))`, the `DiTBlock` stack,
  a final adaLN over the last layer (`nn.LayerNorm(d, elementwise_affine=False)` plus a
  zero-init `nn.Linear(d, 2*d)` producing a final shift/scale), and a zero-init output head
  `nn.Linear(d, p*p*C)`. `forward(self, z, t, y)`:
  - tokens = patch_embed(patchify(z, p)) + pos_embed
  - c = time_mlp(timestep_embedding(t, time_dim)) + class_embed(y)   # (B, d)
  - run blocks with c
  - final-layer modulate + output head, then `unpatchify` back to (B, C, H, W).
  The two zero-init layers (final-layer adaLN and output head) make the whole DiT predict 0 at
  init, the standard DiT initialization.

### `flow.py` (provided only; no hole, no solution copy)

The conditional flow-matching objective and Euler sampler, reused from the flow-matching
assignment and adapted to pass the class label through to the model. Provide:
- `cfm_loss(model, x0, x1, y, t) -> Tensor`: $x_t = (1-t)x_0 + t x_1$, target $u = x_1 - x_0$,
  `pred = model(x_t, t, y)`, return the per-image-sum batch-mean squared error. (x1 is the
  latent batch; x0 the noise; y the class labels.) Takes x0 as an argument (like the
  flow-matching assignment's `cfm_loss`) rather than sampling internally, so the overfit test
  can fix x0 and get a deterministic regression target. The viz/train loop samples a fresh
  `x0 = torch.randn_like(x1)` each step.
- `euler_sample(model, x0, y, n_steps) -> Tensor`: integrate $\mathrm{d}x/\mathrm{d}t = v$ from
  t=0 to t=1 with n_steps forward-Euler steps, conditioned on y. Returns the final latent.

A comment must point out this is the flow-matching assignment's objective with conditioning
added, and must restate the t=0 noise / t=1 data convention.

### `config.py` (provided only)

`@dataclass DiTConfig` with: image size 16, channels 1, num_classes 3, latent_dim C = 4,
f = 4, patch_size = 1, d_model = 64, n_heads = 2, n_blocks = 4, mlp_ratio = 4, time_dim = 64,
beta (KL weight) = 1e-4, n_steps = 50 for sampling. beta = 1e-4 is the top of the production
1e-6..1e-4 range (the expert confirmed 1e-2 is beta-VAE territory and would fight the recon
under this loss's sum reductions, contradicting "beta is small"). With recon as per-image-sum
over 256 pixels and KL as per-image-sum over 64 latent dims, 1e-4 keeps recon dominant while
the latent stays loosely Gaussian.

### `viz.py` (provided only)

Train the VAE briefly on a batch, show originals vs reconstructions; then train the DiT on the
encoded latents and sample one image per class, decode through the VAE decoder, save a figure
to `out/`. Not graded. Mirror the flow-matching assignment's viz structure.

### `conftest.py`, `__init__.py`

Copy the flow-matching assignment's conftest verbatim (adjust the docstring file list). Local
files imported bare; `nanovision.*` for shared.

### `solution/`

Contains only the two holed files filled in: `vae.py`, `dit.py`, plus `__init__.py`.

## Tests (mirror the flow-matching assignment's test set; env python, CPU, seconds each)

1. `test_shapes.py`: encoder -> mu, logvar each (B, 4, 4, 4); reparameterize -> (B, 4, 4, 4);
   decoder -> (B, 1, 16, 16); DiT(z, t, y) -> (B, 4, 4, 4).
2. `test_patchify_roundtrip.py`: for p in {1, 2}, `unpatchify(patchify(z, p), p, C, H, W)`
   equals z exactly (`torch.allclose`, atol 0). Confirms the reshape is a true inverse.
3. `test_adaln_identity.py`: a freshly constructed `DiTBlock` (zero-init adaLN) with arbitrary
   x and arbitrary conditioning c returns output equal to x within atol 1e-6. The core
   adaLN-Zero property: at init every block is identity regardless of c. Also check the full
   `DiT` predicts all-zeros at init (both zero-init final layers).
4. `test_kl.py`: `kl_divergence` matches a hand-computed closed form for a small fixed
   (mu, logvar) - e.g. construct mu, logvar where the analytic KL is known, assert allclose.
   Include the mu=0, logvar=0 case (KL = 0).
5. `test_gradcheck.py`: `torch.autograd.gradcheck` (float64) on `kl_divergence` (w.r.t.
   mu, logvar) and on a single `DiTBlock.forward` (small d, double precision). Do NOT gradcheck
   `reparameterize`: its random eps is resampled per call, so the finite-difference reference
   forward uses a different eps than the analytic pass and the check is ill-posed (the expert
   confirmed this). reparameterize is covered by the shape and overfit tests instead. Follow
   the flow-matching assignment's gradcheck setup (float64, small tensors, requires_grad).
6. `test_vae_overfit.py`: overfit 8 images, Adam, recon MSE falls below a threshold after a
   bounded number of steps (~800). Report the measured floor; set the threshold from it per the
   no-thrash rule. Recon is per-image-sum batch-mean, so the absolute number depends on the sum
   reduction - measure first.
7. `test_dit_overfit.py`: overfit the DiT on FIXED random latents (decoupled from VAE
   convergence, the expert-confirmed honest choice), exactly like the flow-matching overfit
   test. The loss target must be DETERMINISTIC: fix x1 (random latents), fix x0 (seeded
   `torch.randn`), fix per-row t (seeded, e.g. `0.05 + 0.9*rand`), and a fixed class label per
   row; then `cfm_loss(model, x0, x1, y, t)` is a single regression and the relative-drop
   threshold is well-posed. Assert final < 0.05 and final < 0.01 * first; adjust from the
   measured floor per the no-thrash rule. Do NOT resample x0 each step - that injects
   irreducible variance and breaks the relative threshold.
8. `test_forbidden_imports.py`: one static scan over the top-level holed files + solution + any
   shim, mirroring the flow-matching assignment's version. Forbid the obvious shortcuts: no
   importing a ready-made DiT/VAE from torchvision/diffusers/timm; forbid `torch.nn.Transformer`
   if that would bypass the block. Mirror the exact structure of the exemplar's test.

All tests must pass under `NANOVISION_IMPL=solution`; default mode must fail only at the holes
(NotImplementedError), except `test_forbidden_imports` which passes in both modes.

## README (comprehensive lecture notes, per the lecture-notes skill)

Cover, in the skill's fixed section order, with real LaTeX:
- Why latent space: the cost argument with the compression-factor numbers. f is the per-axis
  linear downsample, so the spatial-position (token) count drops by $f^2$: f=4 gives $4^2 = 16$x
  fewer positions per denoising step, production f=8 gives $8^2 = 64$x. State $f^2$ explicitly
  (two spatial axes) so a reader does not read the factor as f. Separately note this counts
  spatial positions, not tensor elements: the latent adds C channels, so for the 16x16x1 ->
  4x4x4 toy the element count goes 256 -> 64 (4x), not 16x - do not present 16x as a
  total-tensor reduction. The two-stage split: the autoencoder is trained once with a
  reconstruction objective and a light KL, the generative model then only learns the structure
  of the compressed distribution.
- The KL-VAE: encoder to (mu, logvar), reparameterization, the closed-form KL to a unit
  Gaussian, why beta is small (keep spatial structure, do not collapse to pure noise), and the
  contrast with the VQ codebook (continuous-for-diffusion vs discrete-for-autoregressive).
- The DiT: patchify latents to tokens, the transformer over tokens, unpatchify. Why a
  transformer replaces the U-Net (scaling, the DiT FID-vs-FLOPs result).
- adaLN-Zero, carefully (the central new mechanism): standard LayerNorm vs adaptive LayerNorm
  that regresses (shift, scale, gate) per sub-layer from the conditioning embedding; the Zero
  init that makes every block identity at start, why that stabilizes training (analogous to
  zero-init residual branches in ResNets); the modulate formula; why this beat cross-attention
  and in-context conditioning in the DiT paper's class-conditional ablation. Scope that claim:
  it is about class conditioning at equal Gflops, NOT a general verdict against cross-attention -
  PixArt-alpha uses cross-attention precisely because adaLN does not scale to long text token
  sequences. Do not let the reader over-generalize.
- The forward process: the linear-interpolant CFM reused from flow matching, now in latent
  space and class-conditioned, with the t=0 noise / t=1 data convention stated explicitly.
- Sampling end to end: Euler ODE on the velocity in latent space, decode through the frozen VAE.
- Where this goes: SD3 / FLUX are flow-matching DiTs in KL-VAE latent space; the only pieces
  not built here are the MM-DiT double stream (image and text tokens with separate weight
  matrices, joined only at attention - explain with a diagram, do not implement) and the text
  encoder. REPA (aligning intermediate DiT activations with frozen DINOv2 features, 17.5x
  training speedup, Yu et al. ICLR 2025) as a training trick worth a paragraph. Video DiTs
  (Sora and the open-weights line) extend this by spacetime patchification - the VAE compresses
  in time too and patchify adds a temporal axis; this is the bridge to the video-generation
  reading note.

Verify every arXiv link by fetching `https://arxiv.org/abs/<id>` before citing. The relevant
IDs (already collected in the research note, but re-verify): LDM 2112.10752, DiT 2212.09748,
SD3 2403.03206, REPA 2410.06940, SiT 2401.08740, PixArt-alpha 2310.00426. FLUX has no paper -
cite the model card / GitHub, not an arXiv id.

Run the mandatory context-less style review on the README before finishing.

## ASSIGNMENT.md

Concise builder contract in `TEMPLATE.md` format: the holes (reparameterize, kl_divergence,
vae_loss, modulate, patchify, unpatchify, DiTBlock.forward), what is provided, the verify
command, the measured pass thresholds. Do not echo the README prose.

## Decisions resolved by the expert review (build to these)

These were open questions; the expert review answered them and the answers are now folded into
the spec above. Recorded here so the build subagent does not re-open them.

1. beta = 1e-4 (not 1e-2). 1e-2 is beta-VAE territory and would fight recon under this loss's
   sum reductions. 1e-4 keeps recon dominant while the latent stays loosely Gaussian. Measure
   the VAE-overfit recon floor at this beta and set the test threshold from it. Do not train to
   KL=0 and do not use beta=0.
2. The DiT overfit test runs on FIXED random latents, decoupled from VAE convergence (the honest
   choice). The true encode -> train -> decode pipeline lives in viz only. The loss target must
   be deterministic (fixed x0, t, y) - see test 7.
3. Keep the final-layer adaLN (paper-faithful) AND the zero-init output head. The zero-init
   output head alone already guarantees the DiT predicts zeros at init, but include the final
   adaLN for fidelity. The identity-at-init test covers the FULL DiT (assert all-zeros output),
   not only one block.
4. Confirmed: match the flow-matching assignment's t=0-noise / t=1-data convention. `cfm_loss` /
   `euler_sample` as specified carry no sign error. Discard the research note's opposite
   convention.
5. f = 4, p = 1 (16 tokens). Stronger latent-compression demo; 16 tokens is enough for attention
   to matter. Keep patchify written generally and tested at p = 2 (round-trip test), even though
   the configured model uses p = 1.
