# A5 - Diffusion (DDPM / DDIM): build plan

Status: plan for expert review before any code. Module 2 (Generative). Depends on A0
(primitives, trainer) and A1 (sinusoidal embedding). Forbidden imports: `diffusers`,
any prebuilt scheduler/UNet, `torchsde`, `k_diffusion`.

Directory: `assignments/a05_diffusion/` (no half-sibling, so no `_0` suffix per the
naming convention).

## Goal and scope

The learner implements the diffusion mechanism end to end on a tiny toy image set that
fits and converges on a 4080: forward noising (linear + cosine schedules), the
closed-form `q(x_t | x_0)`, the three prediction parameterizations (eps, x0, v) and the
algebra that converts between them, v-prediction as the required training objective with
an eps-vs-v comparison, the Tweedie score-eps identity as an explicit exercise, DDPM
(ancestral) and DDIM (deterministic, sub-sampled) samplers, and classifier-free
guidance. Min-SNR loss weighting is a one-line option in the loss.

The research note `research/a05_diffusion.md` is the authority on scope. Its main
corrections to the original draft, already folded in here:
- v-prediction is the required objective, not a stretch (Salimans & Ho 2022; the
  production default since SD 2.x).
- the score-eps connection `s = -eps / sqrt(1 - abar_t)` (Tweedie) is a required
  derivation, not an aside.
- DDIM is the probability-flow ODE integrator, not just "a faster sampler"; name the
  ODE link because it sets up A6 (flow matching).
- class-conditional CFG moves from stretch to required.

## What the student implements (holes) vs what is provided

Two design calls I want the expert to confirm:

1. The time-embedded U-Net is PROVIDED, not a hole. Reason: the diffusion-specific
   content is the schedule, the objective, the parameterization algebra, and the
   samplers. The U-Net is standard CNN plumbing (ResNet blocks, GroupNorm, bottleneck
   self-attention, a sinusoidal-timestep MLP, a class+null embedding). For an
   experienced engineer it adds bulk, not insight, and a buggy student U-Net would make
   the diffusion tests flaky for reasons unrelated to diffusion. The README still
   explains the architecture and the time-injection (AdaGN-style additive shift). The
   two diffusion-specific pieces of the U-Net are holes: `timestep_embedding(t, dim)`
   (the sinusoidal embedding from the transformer, now indexed by the scalar diffusion
   timestep instead of sequence position) and the AdaGN time-injection line in the ResNet
   block. The expert confirmed this split: hole the embedding and the injection (the
   learner should implement how time enters, which A7's adaLN-Zero generalizes), provide
   the rest.

2. Sampler correctness is tested with an analytic oracle model, not a trained network.
   The oracle returns the construction noise `eps = (x_t - sqrt(abar_t)*x0)/
   sqrt(1-abar_t)` for the known `x0` (equivalently, the fixed eps used to build `x_T`).
   With this definition a correct deterministic (eta=0) sampler reconstructs `x0` EXACTLY
   (to float precision, not just approximately), because the oracle keeps the trajectory
   on the original construction line at every step, so discretization error never enters.
   The test asserts 1e-5 (float32) with `clip_x0=False`, no training. This gives each
   sampler test a known-reachable exact pass condition and keeps it fast. (Same move as
   A3, where the DINO overfit test trains against a frozen captured teacher instead of a
   moving one. It avoids the failure mode where a test depends on a training run
   converging.)

### Conventions stated once (README + code docstrings)

- t indexing: schedule arrays have length T, indices `0..T-1`. `abar[0]` is the least
  noised (one forward step from x0); `abar[T-1] ~ 0` is pure noise. The cosine formula's
  continuous arg `t/T` is evaluated to fill these T array entries; pin the off-by-one so
  `beta_1` is well defined.
- t=0 boundary: define `abar_{-1} = 1` (i.e. `x_{-1} = x0`, no noise), so the final
  sampler step maps to clean x0 and adds no noise (DDPM) / has `sigma=0` (DDIM).
- x0 clamping: samplers clamp `x0_hat` to [-1, 1] each step by default (`clip_x0=True`)
  because the x0 estimate is unreliable at high t; the oracle test runs with clamp off.
- v-prediction is well-conditioned at both endpoints; eps and x0 inversions divide by
  `sqrt(abar)` (singular at t=T) or `sqrt(1-abar)` (singular at t=0). A viz row shows
  this blow-up to make the "v is best-conditioned" claim a measured demonstration.
- CFG `w` convention: this plan's `w` is the affine-extrapolation weight where `w=1` is
  the plain conditional (no guidance boost). The diffusers `guidance_scale` equals
  `w+1`; the README states this so cross-reading does not introduce an off-by-one.

### Files (top-level holes; `solution/<file>.py` is the answer key)

`schedule.py`
- HOLE `cosine_alpha_bar(T, s=0.008) -> (betas, alphas_bar)`: Nichol & Dhariwal 2021,
  `abar_t = f(t)/f(0)`, `f(t) = cos^2((t/T + s)/(1+s) * pi/2)`, then
  `beta_t = 1 - abar_t/abar_{t-1}` clipped to <= 0.999. Returns both tensors shape `(T,)`.
- PROVIDED `linear_alpha_bar(T, beta_start=1e-4, beta_end=2e-2)`: historical contrast
  (Ho et al. 2020). Returns `(betas, alphas_bar)`. These constants are calibrated for
  T=1000; at small T the chain barely noises (`abar_T` stays well above 0), so tests at
  T=50 must NOT assert `abar[-1]~0` for the linear schedule (the cosine schedule is
  self-normalizing in T via `t/T` and is fine at any T). The README states that linear
  betas do not transfer across T.
- PROVIDED `gather(a, t)`: index a `(T,)` schedule tensor by an integer time batch `(B,)`
  and reshape to `(B, 1, 1, 1)` for broadcasting against images.

`diffusion.py`
- HOLE `q_sample(x0, t, eps, alphas_bar)`: closed form
  `x_t = sqrt(abar_t) * x0 + sqrt(1 - abar_t) * eps`. Shapes `(B,C,H,W)`.
- HOLE `v_target(x0, eps, abar_t)`: `v = sqrt(abar_t)*eps - sqrt(1-abar_t)*x0`
  (Salimans & Ho 2022, App. D).
- HOLE `to_x0_eps(pred, x_t, abar_t, kind)`: given a model output of kind in
  {"eps","x0","v"}, return `(x0_hat, eps_hat)`. This is the parameterization-agnostic
  bridge the samplers call. Algebra:
  - eps: `eps_hat = pred`; `x0_hat = (x_t - sqrt(1-abar)*eps_hat)/sqrt(abar)`.
  - x0: `x0_hat = pred`; `eps_hat = (x_t - sqrt(abar)*x0_hat)/sqrt(1-abar)`.
  - v: `x0_hat = sqrt(abar)*x_t - sqrt(1-abar)*pred`;
       `eps_hat = sqrt(abar)*pred + sqrt(1-abar)*x_t`.
- HOLE `score_from_eps(eps, abar_t)`: `-eps / sqrt(1 - abar_t)` (Tweedie). One line; the
  derivation lives in the README. Used only in a test and viz, to show eps is a scaled
  score.
- HOLE `diffusion_loss(model, x0, alphas_bar, *, kind="v", num_classes=None,
  cfg_drop_prob=0.1, min_snr_gamma=None, labels=None, generator=None)`: sample
  `t ~ U[0,T)`, `eps ~ N(0,I)`, build `x_t` via `q_sample`, optionally drop `labels` to
  the null index with prob `cfg_drop_prob`, forward the model, form the target by `kind`
  (`eps`, `x0`, or `v_target`), MSE. If `min_snr_gamma` is set, weight each sample by a
  PARAMETERIZATION-AWARE Min-SNR weight (Hang et al. 2023), with `SNR_t = abar_t/(1-abar_t)`:
  - kind="eps": `w_t = min(SNR_t, gamma)/SNR_t`
  - kind="x0":  `w_t = min(SNR_t, gamma)`
  - kind="v":   `w_t = min(SNR_t, gamma)/(SNR_t + 1)`
  These are all "truncate the effective x0-space weight at gamma" expressed in each
  loss's native space; a single `min(SNR,gamma)/SNR` formula is WRONG for the default
  v-loss (it over-weights low-t). Apply the weight per-sample BEFORE the batch mean:
  `(w_t * mse_per_sample).mean()`. MSE is allowed (not a forbidden import); the loss math
  is the point.

`sampling.py`
- HOLE `ddpm_sample(model, shape, alphas_bar, *, kind="v", variance="beta_tilde",
  clip_x0=True, labels=None, guidance=1.0, generator=None) -> x0`: ancestral sampler, t
  from T-1 down to 0. Posterior mean
  `mu = sqrt(abar_{t-1})*beta_t/(1-abar_t) * x0_hat + sqrt(alpha_t)*(1-abar_{t-1})/
  (1-abar_t) * x_t`. Variance is a choice: the TRUE posterior variance is
  `beta_tilde_t = (1-abar_{t-1})/(1-abar_t) * beta_t` (`variance="beta_tilde"`, default),
  and `beta_t` is the other fixed option Ho et al. report as comparable
  (`variance="beta"`). Add noise at all steps except t=0. Boundary: define
  `abar_{-1} = 1` so the final step (t=0) maps to clean x0 with no added noise. If
  `clip_x0`, clamp `x0_hat` to [-1, 1] each step (the x0 estimate is unreliable at high t
  and can blow the trajectory up; real samplers threshold). Uses `to_x0_eps` so it is
  parameterization-agnostic.
- HOLE `ddim_sample(model, shape, alphas_bar, timesteps, *, kind="v", eta=0.0,
  clip_x0=True, labels=None, guidance=1.0, generator=None) -> x0`: `timesteps` is a
  decreasing subset of indices (e.g. 50 of 1000). Step:
  `x_{prev} = sqrt(abar_prev)*x0_hat + sqrt(1 - abar_prev - sigma^2)*eps_hat + sigma*z`,
  `sigma = eta * sqrt((1-abar_prev)/(1-abar_t)) * sqrt(1 - abar_t/abar_prev)`. `eta=0`
  is deterministic (probability-flow ODE); `eta=1` produces variance `beta_tilde_t`, so it
  matches the `variance="beta_tilde"` DDPM sampler on the full consecutive timestep grid
  (not the `beta` one). Same `abar_{-1}=1` boundary and `clip_x0` clamp as DDPM (clamp
  off for the oracle test).
- HOLE `classifier_free_guidance(eps_cond, eps_uncond, w)`:
  `eps_uncond + w*(eps_cond - eps_uncond)`. The samplers run two forwards (cond, null)
  per step when `guidance != 1.0` and combine in eps space.

`unet.py` PROVIDED: `TimeEmbeddedUNet` (small: base width ~32-64, 2-3 resolution levels,
GroupNorm ResNet blocks, one bottleneck self-attention, sinusoidal time MLP, class
embedding with an extra null row for CFG). TWO small holes inside it: (1)
`timestep_embedding(t, dim)`, the sinusoidal embedding from the transformer now indexed
by the scalar diffusion timestep instead of sequence position; (2) the AdaGN
time-injection line in the ResNet block, where the projected time+class embedding is
applied as a scale/shift (or additive shift) into the block's features. Holing the
injection makes the learner implement HOW the timestep enters the network, not just the
embedding; A7's adaLN-Zero generalizes exactly this. Everything else (blocks, norm,
attention, down/up sampling) is provided so a buggy student U-Net cannot make the
diffusion tests flaky.

`config.py` PROVIDED: `DiffusionConfig` (img 1x16x16 toy; `T=1000` for schedule math,
but tests pass a small `T` like 50; base_width 32; num_classes 3; cfg_drop_prob 0.1).

`viz.py` PROVIDED: (a) `abar` vs t for linear vs cosine; (b) a forward-noising strip of
one image at t in {0, T/4, T/2, 3T/4, T-1}; (c) a 64-sample grid from a trained model,
DDPM-1000 vs DDIM-50 side by side; (d) CFG sweep w in {1, 3, 7.5} for each class; (e) an
eps-vs-v training-curve overlay; (f) the eps/x0 inversion magnitude vs t blowing up near
the schedule endpoints while v stays bounded. Real-training viz samples from an EMA copy
of the weights (decay ~0.999); raw-weight samples look markedly worse, and the README
notes this so the learner does not conclude the model is broken.

### Toy data

Add `nanovision/data/toy.py::diffusion_image_batch(n, num_classes=3, size=16,
channels=1, generator=None) -> (images, labels)`: each class is a distinct simple shape
(filled disk, axis-aligned square, plus/cross) on a zero background, values in [-1, 1].
Three classes, kept at 16x16 1-channel (no color - it adds cost without helping CFG).
Give each sample WIDE intra-class variation that the label does not fully determine
(position and radius/size jittered over a broad range, optionally two sizes per shape),
so high-w CFG visibly trades position/size diversity for canonical-shape fidelity (the
diversity-collapse half of the trade-off, which is subtle if intra-class variation is
small). Deterministic per seed so overfit-one-batch is exact.

## Tests (each is fast; per-assignment pytest process)

- `test_shapes.py`: schedule `(T,)`; `q_sample` `(B,C,H,W)`; `unet(x,t,labels)` ->
  `(B,C,H,W)`; `ddpm_sample`/`ddim_sample` -> `(B,C,H,W)`.
- `test_schedule.py`: both schedules `abar[0] ~ 1`, strictly decreasing, `betas in (0, 1)`.
  The `abar[-1] < 0.1` assertion is gated to the COSINE schedule only (linear betas are
  T=1000-specific and do not reach ~0 at the test's small T). Cosine matches a reference
  value array at a few t (precomputed in the test, not from the student code).
- `test_q_sample_moments.py`: Monte Carlo over many eps at fixed `(x0, t)`: empirical
  mean ~ `sqrt(abar_t)*x0`, empirical var ~ `(1-abar_t)`. Size N and tolerances together
  (MC error ~ 1/sqrt(N)): mean tol a few `sqrt((1-abar)/N)`, variance tol looser, so it
  is not flaky.
- `test_parameterization.py`: for random `(x0, eps, t)` build `x_t`; check
  `to_x0_eps(eps_target, ...)`, `to_x0_eps(x0, ...)`, `to_x0_eps(v_target, ...)` all
  recover the same `(x0, eps)` to 1e-5 (the three parameterizations are algebraically
  equivalent). `score_from_eps` equals `-eps/sqrt(1-abar)`.
- `test_gradcheck.py` (float64): `q_sample`, `v_target`, `to_x0_eps`, and
  `diffusion_loss` with a tiny linear stand-in model are differentiable.
- `test_sampler_oracle.py`: build `x_T` from a known `x0`; the oracle returns the
  construction eps (and, in a second case, the construction v) at each t; assert
  deterministic DDIM (eta=0, `clip_x0=False`) reconstructs `x0` EXACTLY to 1e-5; assert
  this holds for both kind="eps" and kind="v" oracles (the samplers are
  parameterization-agnostic via `to_x0_eps`). This is the exact, training-free sampler
  correctness test.
- `test_ddim_ddpm_consistency.py`: with `eta=1`, the full consecutive timestep list, and
  the `variance="beta_tilde"` DDPM sampler, a seeded DDIM trajectory equals the DDPM
  ancestral trajectory (DDIM eta=1 produces exactly `beta_tilde_t`). The test must use
  the `beta_tilde` DDPM variant, NOT `beta`; against `beta` the noise scales differ and
  the invariant is false.
- `test_cfg.py`: `classifier_free_guidance(c, u, w=1)` returns `c`; the formula is the
  affine extrapolation; in `diffusion_loss`, with `cfg_drop_prob=1.0` every label is the
  null index, with `0.0` none are (seeded).
- `test_overfit.py`: train the provided U-Net on one B=4 toy batch with v-prediction for
  a few hundred steps; assert loss drops below a small threshold; then DDIM-sample from
  the overfit model and assert the samples are close to the training batch (the overfit
  network memorized them). Keep T small (e.g. 50) and steps bounded so the test is well
  under a minute.
- `test_forbidden_imports.py`: scan top-level + solution + any shim for `diffusers`,
  `k_diffusion`, `torchsde`, prebuilt schedulers/UNets.

## Shared symbols / shims

A5 owns no `nanovision.*` shared symbol yet. A6 (flow matching) "reuses A5's network
backbone" and A7 (latent DiT) reuses the schedule, eps/v objective, CFG, and DDIM. When
those are built they will promote `unet.py` and/or `schedule.py`/`sampling.py` to a
`nanovision.diffusion` shim owned by a05 (the loader pattern, keyed on NANOVISION_IMPL),
following the import rule (shared owned file imported only via `nanovision.*`; A5's own
files stay bare). Not building those shims now avoids the dual-module-identity trap; A5
imports all its files bare.

## Compute notes

Everything is 1x16x16 with a base-width-32 U-Net. The overfit test and all unit tests
run on CPU in seconds to a minute. A real unconditional + class-conditional training run
to "visibly improves" quality is a `viz`/notebook activity on the 4080, not a test.

## README (lecture notes) outline

Historical landscape (score matching and NCSN before DDPM; why exact-likelihood models
and GANs each fell short), what DDPM changed (Ho et al. 2020: the closed-form forward,
the eps-prediction objective, the ELBO that reduces to weighted MSE), the score-eps
identity via Tweedie and why eps-prediction is denoising score matching (Vincent 2011),
the three parameterizations and why v is best-conditioned, DDIM and the probability-flow
ODE, CFG and the implicit `p(x|c)^w/p(x)^{w-1}` distribution, a one-paragraph EDM /
Min-SNR note, and named forward pointers: A6 replaces the SDE with a directly
parameterized velocity field (v-prediction is the bridge), A7 swaps the U-Net for a DiT
with adaLN-Zero, A13 applies diffusion to action trajectories (Diffusion Policy). Verify
every arXiv id before citing. Mermaid diagrams for the forward/reverse chain and the
U-Net data flow.

## Build order

1. Expert review of this plan; fold corrections.
2. `schedule.py`, `diffusion.py`, `sampling.py` solutions + holes; `unet.py`,
   `config.py` provided; toy generator.
3. Tests; verify both modes (`NANOVISION_IMPL=solution` green, top-level fails at holes).
4. `viz.py`.
5. Lecture-notes README via the skill, then the context-less style-review pass.
6. Commit.
