# A6 - Flow matching and rectified flow: build plan

Status: plan for expert review before any code. Module 2 (Generative). Depends on A5
(reuses the time-embedded U-Net for the optional image demo) and A0/A1. Forbidden imports:
`torchcfm`, `torchdyn`, `torchdiffeq`, `diffusers`, `k_diffusion`. `scipy.optimize`
(linear_sum_assignment) is ALLOWED, it is the OT-coupling utility, not a flow library.

Directory: `assignments/a06_0_flow_matching/` (the `_0` suffix because A6.5, the VQ
tokenizer, is its half-sibling, per the naming convention).

## Convention pinned once

t = 0 is noise, t = 1 is data. The prior is `x0 ~ N(0, I)`; the data sample is `x1`. The
linear (rectified-flow) conditional path and its velocity:

$$x_t = (1-t)\,x_0 + t\,x_1, \qquad u_t = \frac{dx_t}{dt} = x_1 - x_0.$$

Sampling integrates `dx/dt = v_theta(x, t)` forward from t=0 (noise) to t=1 (data). I will
keep this convention everywhere; the research note swaps it in one place (its score formula
line), which I have re-derived below for THIS convention.

## Goal and scope

The learner builds conditional flow matching (CFM) end to end on a 2D toy distribution
where everything is visible (plot the velocity field on a grid, plot trajectories over the
data), then the additions the field actually uses: minibatch optimal-transport (OT)
coupling, logit-normal timestep sampling (SD3), Euler ODE sampling, a straightness metric,
and the rectified-flow reflow procedure. The diffusion-flow equivalence is taught with the
exact score-velocity relation. The image-scale CFM (reusing A5's U-Net with the objective
swapped) is a viz demonstration, not a graded test.

The research note `docs/research/a06_flow_matching.md` is the authority. Its corrections to
the original draft, folded in here:
- OT coupling is first-class, not optional (Tong et al. 2023, OT-CFM). Implement and
  compare against independent coupling.
- logit-normal timestep sampling is a standard component (Esser et al. 2024, SD3), not a
  footnote.
- the score-velocity relation is stated exactly, making the diffusion-flow equivalence
  precise rather than a "two views" metaphor.
- reflow is the canonical first few-step method, but framed as one of several; MeanFlow
  (Geng et al. 2025), Rectified Diffusion (ICLR 2025), and consistency flow matching are
  named as the 2025 frontier (README only, not implemented).

## Why the graded tests use a local MLP, not A5's U-Net

A6's holes (path, loss, OT, Euler, straightness) are the flow-matching mechanism, which is
independent of the network. Testing them on a small 2D `VelocityMLP` keeps each test fast,
exact, and fully visible, and avoids a cross-assignment trap: A5's U-Net carries holes
(timestep_embedding, AdaGN), so importing it into A6's default-mode tests would fail at
A5's holes, not A6's. The image-scale CFM (swap A5's diffusion objective for CFM, same
backbone) goes in `viz.py`, which runs in solution mode where A5's U-Net is filled. A5's
U-Net is promoted to a `nanovision.unet` shim (owned by a05) for that reuse, following the
import rule (shared owned file imported only via `nanovision.*`).

## What the student implements (holes) vs provided

### Files (top-level holes; `solution/<file>.py` is the answer key)

`path.py`
- HOLE `linear_path(x0, x1, t)`: `x_t = (1-t)*x0 + t*x1`. `t` is `(B,)` broadcast to the
  feature dims. Shapes `(B, D)` for the 2D toy.
- HOLE `linear_velocity(x0, x1)`: `x1 - x0` (the constant conditional velocity).

`timesteps.py`
- HOLE `sample_timesteps(n, dist="uniform", loc=0.0, scale=1.0, generator=None)`: returns
  `(n,)` in (0,1). `"uniform"` is `U[0,1]`; `"logit_normal"` is `sigmoid(loc + scale*z)`,
  `z ~ N(0,1)` (Esser et al. 2024). The README shows logit-normal concentrating mass near
  t=0.5.

`flow.py`
- HOLE `cfm_loss(model, x0, x1, t, *, generator=None)`: build `x_t = linear_path(x0,x1,t)`,
  target `u = linear_velocity(x0, x1)`, predict `v = model(x_t, t)`, MSE
  `mean(||v - u||^2)`. `t` is passed in (sampled by the caller via `sample_timesteps`) so
  the loss is deterministic given its inputs and gradchecks cleanly.
- HOLE `score_from_velocity(v, x_t, t)`: the exact score-velocity relation for the linear
  path. Derivation (this convention): `x_t | x1 ~ N(t*x1, (1-t)^2 I)`, so the conditional
  score is `-(x_t - t*x1)/(1-t)^2`; with `u = (x1 - x_t)/(1-t)` substituted, this reduces
  to `score = (t*v - x_t)/(1-t)`. So `score_from_velocity` returns `(t*v - x_t)/(1-t)`.
  (Question for the expert: confirm `(t*v - x_t)/(1-t)`, not `/(1-t)^2`; the research note
  uses `/(1-t)^2` under its swapped t convention.)

`coupling.py`
- HOLE `ot_coupling(x0, x1)`: minibatch OT. Build the pairwise squared-distance matrix
  `C[i,j] = ||x0[i] - x1[j]||^2`, solve the assignment with
  `scipy.optimize.linear_sum_assignment(C)`, and return `x1` reordered so row `i` of `x0`
  pairs with its optimal `x1`. Returns the permuted `x1` (and the permutation). Straightens
  trajectories without reflow (Tong et al. 2023).

`sampling.py`
- HOLE `euler_sample(model, x0, n_steps, *, return_traj=False)`: integrate
  `x <- x + v(x, t)*dt` for `n_steps` from t=0 to t=1, `dt = 1/n_steps`. Returns the final
  `x1_hat` (or the full trajectory if `return_traj`).
- HOLE `straightness(model, x0, n_steps)`: the rectified-flow straightness metric
  `E[ || (x1_hat - x0) - v(x_t, t) ||^2 ]` averaged over the Euler trajectory points (how
  much the instantaneous velocity deviates from the net displacement; 0 for a perfectly
  straight constant-velocity flow).

`model.py` PROVIDED: `VelocityMLP` for the 2D toy: input `x (B,D)` and `t (B,)`, a
sinusoidal time embedding (reused from the transformer / A5 construction) concatenated or
added, a few-layer MLP with SiLU, output `(B,D)`. Small (width ~128).

`reflow.py` PROVIDED: `reflow_pairs(model, n, n_steps, generator)` draws `x0 ~ N(0,I)`,
runs `euler_sample` to get `x1_hat`, returns `(x0, x1_hat)` pairs to retrain on (the 2-
rectified-flow data). It is provided because it is just `euler_sample` plus pairing; the
reflow lesson is that retraining CFM on these pairs straightens the flow, shown in viz.

`config.py` PROVIDED: `FlowConfig` (data_dim 2, mlp_width 128, toy "two_moons" / "8gauss",
batch 256, logit-normal loc 0 scale 1, n_steps for sampling).

`viz.py` PROVIDED: (a) the learned velocity field on a grid at t in {0, 0.25, 0.5, 0.75, 1}
over the data; (b) sample trajectories, independent coupling vs OT coupling vs 2-rectified,
showing OT and reflow straighten them; (c) the straightness metric for the three; (d)
samples at 1, 2, 4, 10, 100 Euler steps; (e) uniform vs logit-normal timestep histograms;
(f) the image-scale demo: A5's U-Net (via `nanovision.unet`) trained with `cfm_loss` on the
toy shape images, sampled with Euler, next to A5's DDIM samples.

### Toy data

Add to `nanovision/data/toy.py`: `two_moons(n, noise, generator)` and `eight_gaussians(n,
generator)` returning `(n, 2)` data samples in a bounded 2D region, deterministic per seed.
Multimodal so the learned flow is non-trivial and the OT/reflow straightening is visible.
(Question for the expert: two-moons vs 8-Gaussians vs checkerboard as the default toy for
showing OT straightening most clearly at batch 256?)

### Shared symbols / shims

- `nanovision/unet.py` NEW shim: `TimeEmbeddedUNet = load("a05_diffusion","unet").TimeEmbeddedUNet`
  (owned by a05), for the viz image demo only. A5 keeps importing its `unet` bare; the shim
  is a separate module identity used only by A6 viz, so no dual-identity collision in A5's
  own process.
- A6 owns no new `nanovision.*` symbol yet. A7 (DiT) will reuse `cfm_loss` and
  `sample_timesteps`; when built it will promote `flow.py` to a `nanovision.flow` shim owned
  by a06.

## Tests (each fast; per-assignment pytest process)

- `test_shapes.py`: `linear_path`/`linear_velocity` `(B,D)`; `sample_timesteps` `(n,)` in
  (0,1); `cfm_loss` scalar; `VelocityMLP(x,t)` `(B,D)`; `euler_sample` `(B,D)`;
  `ot_coupling` returns `(B,D)` and a length-B permutation.
- `test_path.py`: `linear_path(x0,x1,0)=x0`, `linear_path(x0,x1,1)=x1`, midpoint is the
  average at t=0.5; `linear_velocity = x1 - x0`.
- `test_timesteps.py`: uniform stays in (0,1) with mean ~0.5; logit-normal stays in (0,1),
  median ~ `sigmoid(loc)`, and (loc=0,scale=1) puts more mass in [0.25,0.75] than uniform
  does (the mid-concentration property). Size N large, tolerances to 1/sqrt(N).
- `test_euler_oracle.py`: an oracle velocity field returning the per-sample constant
  `x1 - x0` integrates from `x0` to `x1` EXACTLY (the linear path is straight, so Euler is
  exact at any `n_steps`, even 1). Assert 1e-6 for n_steps in {1, 4, 50}. This is the
  training-free exact sampler test (the A5-oracle analog).
- `test_score_velocity.py`: for random `(x0, x1, t)`, `score_from_velocity(u, x_t, t)`
  equals the independently computed conditional score `-(x_t - t*x1)/(1-t)^2` to 1e-5
  (self-checking: if the `(1-t)` vs `(1-t)^2` algebra is wrong the test fails).
- `test_ot_coupling.py`: on a constructed swap case (`x0 = [[0,0],[1,1]]`,
  `x1 = [[1,1],[0,0]]`) OT returns the swapped pairing; in general the OT total squared
  distance is `<=` the identity pairing's, and the returned permutation is a valid bijection.
- `test_straightness.py`: the constant-velocity oracle flow has straightness ~0; a
  deliberately curved field (e.g. `v = rot90(x)`) has straightness `> 0`.
- `test_gradcheck.py` (float64): `linear_path`, `cfm_loss` with a tiny linear stand-in
  model, and `euler_sample` (a few steps) are differentiable.
- `test_overfit.py`: fix one batch `(x0, x1, t)` from the toy; train `VelocityMLP` with
  `cfm_loss` on that fixed batch; loss drops below a small threshold (the field memorizes
  the constant target at those points). Bounded steps, well under a minute.
- `test_forbidden_imports.py`: no `torchcfm`/`torchdyn`/`torchdiffeq`/`diffusers`/
  `k_diffusion`; `scipy.optimize` is allowed and present in `coupling.py`.

## Diffusion-flow equivalence (README, with one check)

Teach the algebra: under the linear path, `v = x1 - x0` and the score is
`(t*v - x_t)/(1-t)`; a VP/cosine path with v-prediction (A5) is the same object with a
different schedule and a time-dependent loss weight. State the three genuine differences
(linear schedule is not variance-preserving so the effective weighting differs; OT coupling
is flow-matching-only; the velocity parameterization gives exact log-likelihood via the
continuous change-of-variables). `test_score_velocity` is the concrete check that the
relation is right.

## Compute notes

2D toy: everything is `(B, 2)` with a width-128 MLP; all tests run on CPU in seconds. The
overfit test is a few hundred steps on one batch. The viz image demo trains a few thousand
steps on A5's U-Net (solution mode) and fits 12GB trivially; it is a demonstration, not a
graded test.

## README (lecture notes) outline

CNF background (why simulation-free regression beats maximum-likelihood ODE simulation,
Chen et al. 2018 -> Lipman et al. 2022), the CFM objective and why the conditional target
has the same gradient as the intractable marginal, the linear path and its constant
velocity, OT coupling and why crossing paths curve the marginal field, the diffusion-flow
equivalence with the exact score-velocity relation, logit-normal timestep sampling,
Euler sampling and straightness, reflow, and a named survey of the 2025 few-step frontier
(MeanFlow, Rectified Diffusion, consistency flow matching). Forward pointers: A7 (DiT)
reuses CFM + logit-normal with a transformer backbone (SD3/FLUX), A13 (VLA) uses a flow-
matching action head (pi0, 10 Euler steps at 50 Hz). All math as real LaTeX. Mermaid for
the path/coupling/reflow diagrams. Verify every arXiv id before citing.

## Expert review: corrections folded in

The reviewer verified the core algebra (path, conditional velocity, the score-velocity
relation, CFM loss, Euler exactness, the straightness metric). Folded corrections:
- score_from_velocity is `(t*v - x_t)/(1-t)` with denominator `(1-t)` (confirmed by
  derivation); the research note's `/(1-t)^2` is for its swapped t-convention and is NOT
  used here. Add a documented domain guard: the score is singular at t=1 ((1-t)->0), so the
  function asserts/expects `t < 1` and the README states the singularity.
- `test_score_velocity` samples t in [0.05, 0.9] (away from the t=1 singularity) and builds
  one `x_t` passed to both sides, so the 1e-5 tolerance holds for correct code.
- straightness uses the per-point form `E||(x1_hat - x0) - v(x_t,t)||^2` (Liu et al. 2022);
  the README must NOT present the research note's `E||x1 - x0 - integral v dt||^2` as
  straightness (it reduces to endpoint error). x1_hat is the realized Euler endpoint.
- `cfm_loss` is intentionally unweighted; the README states that logit-normal t-sampling is
  the weighting mechanism (tying back to A5's v-prediction weighting), so no explicit loss
  weight is added.
- The README teaches the marginal-vs-conditional velocity explicitly: the MSE minimizer is
  `v(x,t) = E[x1 - x0 | x_t = x]`, the conditional average; where paths cross, that average
  curves the marginal field, which is what OT coupling reduces.
- `ot_coupling`: `linear_sum_assignment` returns `row_ind` already sorted, so apply
  `col_ind` to reorder x1 (no double-permute); squared-L2 cost is the OT-CFM convention.
- `reflow_pairs` integrates with ~100 Euler steps so x1_hat approximates the true ODE
  endpoint, not a 1-step discretization.
- overfit test: sample a fresh t per row (so the net regresses the field, not one point),
  assert relative loss (final < 1e-3 * Var(u)) rather than an absolute 1e-6.
- logit-normal test asserts the median (~ sigmoid(loc), exact), not the mean, and uses a
  comfortable margin on the mid-mass check.
- viz default toy is 8-Gaussians for the OT/reflow straightening figures (clearest mode
  structure at batch 256); two-moons stays as the secondary non-Gaussian-manifold example.
- viz image demo samples from an EMA copy (decay ~0.999) of the U-Net weights.
- Citations: arXiv:2405.20320 is Lee, Lin & Fanti (not "Dao et al."); stochastic
  interpolants (arXiv:2303.08797) is Albergo, Boffi & Vanden-Eijnden (three authors).

## Build order

1. Expert review of this plan; fold corrections (especially the score-velocity formula).
2. `path.py`, `timesteps.py`, `flow.py`, `coupling.py`, `sampling.py` solutions + holes;
   `model.py`, `reflow.py`, `config.py` provided; the `nanovision.unet` shim; 2D toy data.
3. Tests; verify both modes.
4. `viz.py`.
5. Lecture-notes README via the skill, then the context-less style-review pass.
6. Commit.
