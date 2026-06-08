# assignments/a05_diffusion/ASSIGNMENT.md

```yaml
id: a05_diffusion
title: Diffusion - DDPM, DDIM, and classifier-free guidance
module: 2
type: Core
estimated_learner_hours: 10
depends_on: [a00_harness, a01_transformer]
builds_into_shared_lib: []
forbidden_imports:
  - import diffusers
  - from diffusers
  - import k_diffusion
  - from k_diffusion
  - import torchsde
  - DDPMScheduler
  - DDIMScheduler
  - UNet2DModel
fits_12gb: true
external_data: "none (synthetic shape images)"
```

## motivation
Diffusion learns to generate by reversing a fixed Gaussian noising process: add noise to an
image over many steps until it is pure noise, train a network to undo one step, then sample
from noise. The training loss is a plain MSE on the added noise, which is stable where GANs
are not. The mechanism is the content. You build the forward schedules, the closed-form
q_sample, the three prediction parameterizations and the algebra between them, the score
connection, the DDPM and DDIM samplers, and classifier-free guidance. v-prediction is the
required objective (the production default since Stable Diffusion 2); epsilon and x0 are
the comparison points. See the README for the historical landscape and the math.

## background
See the README. Index convention: schedule arrays have length T, indices 0..T-1;
alphas_bar[0] is least noised, alphas_bar[T-1] ~ 0 is pure noise, and alphas_bar[-1] := 1
is the clean-image boundary at t=0. With a = sqrt(abar_t), b = sqrt(1-abar_t):
forward x_t = a*x0 + b*eps; v = a*eps - b*x0; score = -eps/b. The samplers convert any
prediction to (x0_hat, eps_hat) and clamp x0_hat to [-1,1] by default.

## what_you_implement
- cosine_alpha_bar: the Nichol & Dhariwal cosine schedule.
- q_sample: the closed-form forward process.
- v_target, to_x0_eps, score_from_eps: the parameterization algebra and the Tweedie score.
- diffusion_loss: sample t and eps, build x_t, optional CFG label drop, target by kind,
  parameterization-aware Min-SNR weighting.
- classifier_free_guidance, ddpm_sample, ddim_sample: the guidance combine and the two
  samplers.
- timestep_embedding and the AdaGN injection line in the U-Net.

The U-Net body, config, toy data, and viz are provided.

## tasks
- **Task 1 - cosine_alpha_bar** (`schedule.py`): build f on the t=0..T grid, normalize by
  f(0), return alphas_bar = abar[1:] (length T) and betas = 1 - abar_t/abar_{t-1} clipped
  to <= 0.999. (linear_alpha_bar and gather are provided.)
- **Task 2 - q_sample** (`diffusion.py`): x_t = sqrt(abar_t)*x0 + sqrt(1-abar_t)*eps, with
  abar_t pulled by gather.
- **Task 3 - parameterizations** (`diffusion.py`): v_target = a*eps - b*x0; to_x0_eps for
  kind in {eps, x0, v} returning (x0_hat, eps_hat) (ValueError otherwise);
  score_from_eps = -eps/b.
- **Task 4 - diffusion_loss** (`diffusion.py`): sample t ~ U[0,T), eps ~ N(0,I); build x_t;
  drop labels to the null index `num_classes` with prob cfg_drop_prob; pred = model(x_t, t,
  labels); per-sample MSE against the kind's target; if min_snr_gamma set, weight per
  sample by min(SNR,g)/SNR (eps), min(SNR,g) (x0), or min(SNR,g)/(SNR+1) (v) before the
  mean.
- **Task 5 - classifier_free_guidance** (`sampling.py`): eps_uncond + w*(eps_cond -
  eps_uncond).
- **Task 6 - ddpm_sample** (`sampling.py`): ancestral loop t=T-1..0 with the posterior mean
  and beta_tilde (default) or beta variance; no noise at t=0; uses _predict (provided).
- **Task 7 - ddim_sample** (`sampling.py`): the DDIM step with sigma(eta); abar_prev := 1
  past the end; noise only when eta>0 and a real prev exists.
- **Task 8 - timestep_embedding + AdaGN** (`unet.py`): the sinusoidal embedding of the
  scalar timestep, and the per-channel time+class shift into each ResNet block.

## tests
Run in this order (also in the README):
1. `tests/test_shapes.py` - schedule (T,); q_sample (B,C,H,W); U-Net forward; sampler
   outputs (shape).
2. `tests/test_schedule.py` - endpoints, monotonicity, betas in (0,1]; cosine matches a
   reference array; the abar[-1] ~ 0 endpoint asserted on cosine only (reference-value).
3. `tests/test_q_sample_moments.py` - Monte Carlo E[x_t] = sqrt(abar)*x0, Var = 1-abar
   (reference-value, tolerance sized to 1/sqrt(N)).
4. `tests/test_parameterization.py` - the three predictions all recover (x0, eps) to 1e-4;
   score = -eps/sqrt(1-abar) (reference-value).
5. `tests/test_gradcheck.py` - float64 gradcheck of q_sample, v_target, to_x0_eps
   (gradcheck).
6. `tests/test_sampler_oracle.py` - a construction-prediction oracle makes DDIM (eta=0) and
   DDPM reconstruct x0 exactly to 1e-5, for eps/x0/v (oracle, training-free).
7. `tests/test_ddim_ddpm_consistency.py` - DDIM eta=1 on the full grid equals the
   beta_tilde DDPM trajectory (reference-value).
8. `tests/test_cfg.py` - the guidance combine; label dropout hits all/none at
   cfg_drop_prob 1.0/0.0 (structural).
9. `tests/test_overfit.py` - the v-prediction loss on a fixed 4-image batch drops below 0.1
   and to under half the start (overfit-one-batch).
10. `tests/test_forbidden_imports.py` - no diffusers/k_diffusion/torchsde/prebuilt
    scheduler-UNet. Passes with the holes in place too.

## provided_boilerplate
`unet.py` `TimeEmbeddedUNet` (GroupNorm ResNet blocks, 4x4 bottleneck self-attention, skip
connections, sinusoidal-time MLP, class embedding with a null row) with only
timestep_embedding and the AdaGN injection line holed. `sampling.py` `_predict` (the CFG
two-pass and x0 clamping) is provided so the sampler holes are the update equations.
`config.py` `DiffusionConfig`. `nanovision.data.toy.diffusion_image_batch` (three shape
classes, wide intra-class position/size variation, values in [-1,1]). `viz.py` renders the
schedules, the forward-noising strip, the parameterization conditioning, and a trained
sample grid with a CFG sweep.

## compute_notes
CPU for all tests, synthetic seeded images, no download. Tiny U-Net (base width 32, 16x16,
1 channel), small T in tests (8-100). Overfit-one-batch trains 400 steps on 4 images and is
well under a minute. The viz trains a few thousand steps and fits 12GB trivially; real
sample quality would use an EMA copy of the weights.

## solution_notes
The cosine schedule is built on the T+1 grid so abar normalizes to f(0) and betas[0] is
well defined; betas are clipped to 0.999. v-prediction is the default because the eps and
x0 inversions divide by sqrt(1-abar) (singular at t=T) or sqrt(abar) (singular at t=0),
while v stays bounded. Min-SNR weighting is parameterization-specific: min(SNR,g)/SNR for
eps, min(SNR,g) for x0, min(SNR,g)/(SNR+1) for v; the single eps-form on a v loss
over-weights low-t and is wrong. The DDPM default variance is the true posterior beta_tilde
= (1-abar_prev)/(1-abar_t)*beta_t; beta_t is the other fixed choice Ho et al. report as
comparable. DDIM eta=1 produces beta_tilde, so the consistency test compares against the
beta_tilde DDPM sampler, not beta. The sampler-correctness tests use a construction oracle
(returns the eps/x0/v consistent with a known x0 at the current x_t) so x0_hat = x0 every
step and the final step (abar_prev := 1) returns x0 exactly, with no training, which avoids
a test that depends on a training run converging. A5 adds no shared nanovision symbol; A6
and A7 will promote the schedule and the network when they reuse them.
