# assignments/a06_0_flow_matching/ASSIGNMENT.md

```yaml
id: a06_0_flow_matching
title: Flow matching and rectified flow
module: 2
type: Core
estimated_learner_hours: 9
depends_on: [a00_harness, a01_transformer, a05_diffusion]
builds_into_shared_lib: []
forbidden_imports:
  - import torchcfm
  - from torchcfm
  - import torchdyn
  - from torchdyn
  - import torchdiffeq
  - from torchdiffeq
  - import diffusers
  - from diffusers
  - import k_diffusion
fits_12gb: true
external_data: "none (synthetic 2D distributions and shape images)"
```

## motivation
Flow matching learns a velocity field that transports noise to data along a probability
path, trained by plain MSE regression onto a known conditional target. It is simpler than
score matching (no score function, no constraint on the forward process) and is the
objective behind production text-to-image systems since mid-2024 (SD3, FLUX). The mechanism
is the content. The holes to fill are the linear path and its constant velocity, the CFM loss, minibatch
optimal-transport coupling, logit-normal timestep sampling, Euler ODE sampling, the
straightness metric, and (provided) reflow, all on a 2D toy where the field and trajectories
are fully visible. The diffusion-flow equivalence ties this back to A5 through the
score-velocity relation. See the README for the math.

## background
See the README. Convention: t=0 is noise x0~N(0,I), t=1 is data x1. Linear path
x_t=(1-t)x0+t*x1, conditional velocity u=x1-x0 (constant in t). CFM loss is the MSE between
v_theta(x_t,t) and u, unweighted; logit-normal t-sampling does the weighting. Score-velocity
relation: score=(t*v-x_t)/(1-t), singular at t=1. OT coupling pairs x0 to x1 by the exact
Hungarian solution under squared-L2 cost. Euler integrates dx/dt=v from t=0 to t=1.

## what_you_implement
- linear_path, linear_velocity: the straight conditional path and its displacement velocity.
- sample_timesteps: uniform and logit-normal (sigmoid(loc+scale*z)) timestep sampling.
- cfm_loss: MSE of the predicted velocity against x1-x0 on the path.
- score_from_velocity: the (t*v-x_t)/(1-t) score relation.
- ot_coupling: minibatch OT via scipy linear_sum_assignment on squared-L2 cost.
- euler_sample, straightness: forward-Euler ODE sampling and the rectified-flow straightness.

The 2D velocity MLP, config, reflow-pair generation, toy data, and viz are provided.

## tasks
1. `linear_path` / `linear_velocity` (`path.py`): x_t=(1-t)*x0+t*x1 with t (B,) broadcast;
   velocity x1-x0.
2. `sample_timesteps` (`timesteps.py`): "uniform" -> U[0,1]; "logit_normal" ->
   sigmoid(loc+scale*z), z~N(0,1); ValueError otherwise.
3. `cfm_loss` (`flow.py`): build x_t, target u=x1-x0, pred=model(x_t,t), MSE (sum over
   feature dims, mean over batch). Unweighted.
4. `score_from_velocity` (`flow.py`): (t*v-x_t)/(1-t), with t broadcast; defined for t<1.
5. `ot_coupling` (`coupling.py`): C[i,j]=||x0[i]-x1[j]||^2 (torch.cdist squared),
   linear_sum_assignment, return (x1[col], col). Row indices come back sorted, so col is the
   permutation.
6. `euler_sample` (`sampling.py`): x <- x + v(x,t)*dt over n_steps from t=0; return final or
   the (n_steps+1, B, D) trajectory.
7. `straightness` (`sampling.py`): mean over the Euler trajectory of
   ||(x1_hat-x0) - v(x_t,t)||^2.

## tests
Run in this order:
1. `tests/test_shapes.py` - path/velocity (B,D); timesteps (n,) in (0,1); cfm_loss scalar;
   MLP forward; euler output and trajectory; ot_coupling output + permutation (shape).
2. `tests/test_path.py` - endpoints x0 at t=0, x1 at t=1, average at t=0.5; velocity=x1-x0
   (reference-value).
3. `tests/test_timesteps.py` - uniform mean ~0.5; logit-normal median ~0.5 and ~0.73 mass in
   [0.25,0.75]; unknown dist raises (reference-value).
4. `tests/test_euler_oracle.py` - the constant-velocity oracle integrates x0->x1 EXACTLY at
   n_steps in {1,4,50} to 1e-6 (training-free exactness).
5. `tests/test_score_velocity.py` - score_from_velocity equals the conditional score
   -(x_t-t*x1)/(1-t)^2 to 1e-5 for t in [0.05,0.9] (reference-value, self-checking).
6. `tests/test_ot_coupling.py` - the swap case returns the swap; a valid bijection; OT total
   cost <= identity cost (reference-value).
7. `tests/test_straightness.py` - constant field ~0, curved (rotation) field > 0.1
   (reference-value).
8. `tests/test_gradcheck.py` - float64 gradcheck of linear_path, score_from_velocity,
   cfm_loss (linear stand-in model), euler_sample (gradcheck).
9. `tests/test_overfit.py` - the velocity MLP on a fixed (x0,x1,t) batch drops ~1000x to
   under 0.05 and under 0.01*initial (overfit-one-batch).
10. `tests/test_forbidden_imports.py` - no torchcfm/torchdyn/torchdiffeq/diffusers/
    k_diffusion; scipy is allowed. Passes with the holes in place too.

## provided_boilerplate
`model.py` `VelocityMLP` (sinusoidal time embedding + MLP over (x, t)). `reflow.py`
`reflow_pairs` (euler_sample + pairing). `config.py` `FlowConfig`. `nanovision.data.toy`
`eight_gaussians` and `two_moons` (2D targets). `viz.py` renders the velocity field,
trajectories under independent/OT/reflow coupling, the straightness bar chart, few-step
samples, the timestep histograms, and an image-scale CFM demo reusing A5's U-Net via
`nanovision.unet`.

## compute_notes
2D toy, all tests on CPU in seconds. The overfit test is 3000 Adam steps on one 64-point
batch; the loss falls ~1000x but floors near 0.015 (finite MLP capacity where distinct pairs
land near the same (x_t,t) with different targets), so the assertion is relative plus a
comfortable absolute. The viz image demo trains A5's U-Net with the CFM objective for a few
thousand steps and fits 12GB trivially; it runs in solution mode where A5's U-Net is filled.

## solution_notes
The score-velocity denominator is (1-t), not (1-t)^2: under this convention (t=0 noise,
x_t=(1-t)x0+t*x1) the conditional is N(t*x1,(1-t)^2 I), and substituting u=(x1-x_t)/(1-t)
into the conditional score -(x_t-t*x1)/(1-t)^2 collapses to (t*v-x_t)/(1-t). The research
note's /(1-t)^2 is for the swapped t-convention and is not used. cfm_loss is intentionally
unweighted; logit-normal t-sampling is the weighting (the link to A5's v-prediction
weighting). OT coupling uses squared-L2 (the OT-CFM / 2-Wasserstein convention), and
linear_sum_assignment returns sorted row indices so the column indices are the permutation
to apply to x1. euler_sample is exact for a constant velocity, which is why the oracle test
reconstructs x1 to 1e-6 with even one step. straightness uses the per-point Liu-2022 form
||(x1_hat-x0)-v||^2 (the realized endpoint as the chord), not the endpoint-error form. A6
adds no new shared nanovision symbol; the nanovision.unet shim it adds is owned by a05. A7
will promote cfm_loss and sample_timesteps when its DiT reuses the flow objective.
