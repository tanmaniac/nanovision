# assignments/a13_vla/ASSIGNMENT.md

```yaml
id: a13_vla
title: Vision-Language-Action capstone (flow-matching action head from pixels)
module: 4
type: Mixed
estimated_learner_hours: 8
depends_on: [a00_harness, a05_diffusion, a06_0_flow_matching, a12_world_models]
builds_into_shared_lib: []   # leaf assignment, last in the course; nothing imports it, so no nanovision shim
forbidden_imports:
  - gym / gymnasium          # not the robot env; the dm_control reacher is used directly in env.py
  - diffusers                # the DDPM head is built from the diffusion objective, not a library
  - robomimic / lerobot      # no prebuilt policy/dataset libraries
  - torchcfm / torchdiffeq   # the CFM objective and the ODE integrator are written from scratch
  - bare cross-assignment imports (from assignments...) — A13 owns its modules locally
  # The scan is over the HEAD files only (flow.py, bc.py, ddpm.py, and their solution copies). dm_control
  # is allowed in env.py / viz.py and isolated there; it must NOT appear in the head files.
fits_12gb: true
external_data: none (dm_control 'reacher' easy ships with dm_control; demos are collected locally)
```

## motivation

The 2023-2026 vision-language-action (VLA) pattern wires a perception/language backbone to a
dedicated action decoder: the backbone interprets the scene and instruction, the decoder generates
continuous, temporally coherent actions. This capstone builds the perception-to-action path on a
real (simulated) robot: a 2-link reacher controlled from a 64x64 camera image, with a CNN encoder
producing the conditioning vector and the pi0-line conditional flow-matching (CFM) action head
generating the joint-torque chunk, behavior-cloned from filtered analytic-expert demos. Single-step
BC is the compounding-error baseline, action chunking is the ACT fix, and a DDPM head is the
diffusion-vs-flow contrast. The full treatment is in the README.

## background

The action head is the generative model from the diffusion and flow-matching topics, re-conditioned
on a perception embedding $c = \operatorname{Encoder}(\text{obs})$ instead of a class label. CFM
convention: $t=0$ is noise $z_0 \sim \mathcal{N}(0, I)$, $t=1$ is the demonstrated action chunk $a$.
The straight path and its constant velocity target are

$$z_t = (1-t)\,z_0 + t\,a, \qquad v = a - z_0 \quad (\text{constant in } t).$$

The network $v_\theta(z_t, t, c)$ regresses onto $v$ by MSE; at inference, integrate
$z \leftarrow z + \tfrac{1}{n}\,v_\theta(z, t, c)$ from $t=0$ to $t=1$ with $n \approx 10$ Euler
steps. Action chunking predicts an $H$-step chunk executed open-loop. The DDPM contrast uses the
epsilon-prediction objective $a_t = \sqrt{\bar\alpha_t}\,a + \sqrt{1-\bar\alpha_t}\,\varepsilon$ and
the ancestral reverse chain.

Shapes: $a$, $z_0$, $z_t$ are $(B, H, 2)$; $t$ is $(B, 1, 1)$; the obs is $(B, 3, 64, 64)$ and the
encoder output $c$ is $(B, 128)$; chunks are $(B, T-H+1, H, 2)$.

## what_you_implement

- `cfm_target`: the straight-path interpolant $z_t$ and the t-independent velocity target $a - z_0$.
- `flow_loss`: sample $(z_0, t)$, build the target, regress $v_\theta$ onto it by MSE.
- `flow_sample`: Euler-integrate the ODE from noise to action over `n_steps`.
- `bc_loss`: plain MSE regression of the action chunk (the compounding-error baseline).
- `chunk_actions`, `de_chunk`, `receding_horizon_indices`: overlapping chunking, its specified
  inverse, and open-loop chunk start indices.
- `ddpm_loss`, `ddpm_sample`: the epsilon-prediction loss and the ancestral reverse chain (contrast).

## tasks

1. **Task 1 - cfm_target** (file: `flow.py`, symbol: `cfm_target`): given $a$, $z_0$ both
   $(B, H, 2)$ and $t$ $(B, 1, 1)$, return $z_t = (1-t)z_0 + t a$ and the velocity target $a - z_0$.
   The target is constant in $t$; a t-dependent target is wrong. Teaches the rectified-flow
   interpolant and its constant conditional velocity.

2. **Task 2 - flow_loss** (file: `flow.py`, symbol: `flow_loss`): sample
   $z_0 \sim \mathcal{N}(0, I)$ and $t \sim U(0, 1)$ of shape $(B, 1, 1)$, build $(z_t, v)$ with
   `cfm_target`, predict $v_\theta(z_t, t, c)$, return MSE. Unweighted velocity regression.

3. **Task 3 - flow_sample** (file: `flow.py`, symbol: `flow_sample`): from $z \sim \mathcal{N}(0,I)$
   of shape $(B, H, 2)$, take `n_steps` forward-Euler steps with $dt = 1/n$, $t = k\,dt$, updating
   $z \leftarrow z + dt\,v_\theta(z, t, c)$. Return $z$. The likely bug is starting at $t=1$ or a
   wrong $dt$.

4. **Task 4 - bc_loss** (file: `bc.py`, symbol: `bc_loss`): MSE between `policy(c)` and the
   demonstrated chunk. BC regresses the conditional mean, correct only when $p(a|c)$ is unimodal.

5. **Task 5 - chunk_actions / de_chunk / receding_horizon_indices** (file: `bc.py`): overlapping
   $H$-windows $(B, T, 2) \to (B, T-H+1, H, 2)$; the inverse (full first chunk, then the last action
   of each later chunk) that reconstructs the sequence exactly; the open-loop start indices
   $0, H, 2H, \dots$ clamped to $T-H$.

6. **Task 6 - ddpm_loss / ddpm_sample** (file: `ddpm.py`): the epsilon-prediction loss and the
   ancestral reverse chain re-conditioned on $c$ (the diffusion-vs-flow contrast).

## tests

Run with `/home/tanmay/miniconda3/envs/nanovision/bin/python -m pytest assignments/a13_vla/tests`.
Default mode fails cleanly at the holes; `NANOVISION_IMPL=solution` is green. The mechanism tests run
on CPU WITHOUT `dm_control`; only `test_env_smoke.py` needs it (and `MUJOCO_GL=egl`) and skips via
`pytest.importorskip`. Order: cfm_target, flow_sample_ode, chunking, shapes, gradcheck, overfit_bc,
overfit_flow, forbidden_imports, env_smoke.

- `tests/test_cfm_target.py` - reference-value: $z_t$ at $t=0$ is $z_0$, at $t=1$ is $a$; $v = a-z_0$
  exactly and identical at every $t$ (t-independence); shapes.
- `tests/test_flow_sample_ode.py` - exact oracle: a constant velocity field integrates from $z_0$ to
  $a^\star$ exactly at any step count (1, 5, 10, 50), checking the integrator wiring.
- `tests/test_chunking.py` - exact: `de_chunk(chunk_actions(x)) == x` for several $(T, H)$; the
  receding-horizon indices start at 0, stay in $[0, T-H]$, have no duplicates, and cover all $T$.
- `tests/test_shapes.py` - shape: the encoder gives $(B, 128)$; the three heads' forward and
  `flow_sample` give $(B, H, 2)$; the image-to-chunk path composes; `chunk_actions` gives
  $(B, T-H+1, H, 2)$.
- `tests/test_gradcheck.py` - float64 gradcheck: `cfm_target` wrt the action; the flow network wrt
  $z_t$ and $t$; `flow_loss` wrt the action with a fixed-seed generator.
- `tests/test_overfit_flow.py` - bounded training, robust: `flow_loss` drops to a small fraction of
  its untrained value (measured ratio ~0.08); a loose secondary bound on `flow_sample`
  reconstruction (measured MAE ~0.14). The loss does NOT reach zero by construction (see
  solution_notes).
- `tests/test_overfit_bc.py` - bounded training: `bc_loss` overfits one batch to near zero.
- `tests/test_forbidden_imports.py` - static tokenize scan over the head files (flow/bc/ddpm) and
  their solution copies for robot/diffusion/flow libraries and bare cross-assignment imports. Passes
  in both modes. Does NOT scan env.py/viz.py, which legitimately use dm_control.
- `tests/test_env_smoke.py` - `importorskip("dm_control")`: reset/step/render shapes ($(3,64,64)$ obs
  in $[0,1]$, 2D torque in $[-1,1]$), the analytic expert reaches on some seeds, demo collection
  filters to successes and pads with a validity mask.

The reach-success and chunk-size numbers are NOT unit tests; rollout success is init- and
seed-sensitive. They are in `viz.py` and the README with per-seed statistics against the
random-torque floor.

## provided_boilerplate

`env.py` (the dm_control reacher wrapper with lazy import, the analytic IK+PD expert, filtered
`collect_demos`, `render_obs`, `rollout_policy`, `random_reach_success`, and the point-mass
side-demo functions), `nets.py` (the 64x64 CNN `Encoder`), `config.py` (`VLAConfig`,
`GradcheckConfig`), `conftest.py`, `_train.py` (pixel-batch building, joint encoder+head training,
the real-env rollout-success metric, and the point-mass side-demo trainers), `viz.py` (the reacher
rollout / chunk-ablation / flow-path / point-mass-multimodal panels), the three head modules'
`__init__`/`forward`/`sinusoidal_embedding`/`make_schedule`.

## compute_notes

Mechanism tests run on CPU in seconds (overfit loops 300-1000 steps), no dm_control. Demo collection
runs the simulator: ~130s for 200 filtered demos (~270 episodes at ~75% expert reach). Training a
head + encoder is ~10-15s on an RTX 4080; a 48-episode rollout is ~30s; full `viz.py` is a few
minutes. Everything under 12GB. Episodes are ~20 steps; demos pad to T~34. A healthy `flow_loss`
drops from ~2.3 to a floor near ~0.18 and stays there (the floor is real, not a stuck run). Render
needs `MUJOCO_GL=egl`.

## stretch_goals

1. Language conditioning: pass a goal phrase through a frozen text encoder, concatenate with the
   image embedding, train only the head and projector.
2. Temporal ensembling: blend overlapping chunks with exponential weighting vs open-loop (pi0 found
   ensembling detrimental and dropped it).
3. A harder reacher (`reacher` hard or added observation noise) where single-step BC drifts more and
   the chunk-size effect should reappear.
4. World-model-conditioned VLA: condition the head on a learned world-models latent (model-based flavor).

## further_reading

- ACT / ALOHA, [arXiv 2304.13705](https://arxiv.org/abs/2304.13705) - action chunking with a CVAE.
- Diffusion Policy, [arXiv 2303.04137](https://arxiv.org/abs/2303.04137) - the DDPM action head.
- pi0, [arXiv 2410.24164](https://arxiv.org/abs/2410.24164) - the flow-matching VLA built here.
- OpenVLA, [arXiv 2406.09246](https://arxiv.org/abs/2406.09246) - the open discretized-token VLA.
- OpenVLA-OFT, [arXiv 2502.19645](https://arxiv.org/abs/2502.19645) - the same-backbone ablation
  showing the action-head design dominates the outcome.
- Octo, [arXiv 2405.12213](https://arxiv.org/abs/2405.12213) - the diffusion-head generalist policy.

## solution_notes

- `flow_loss` does NOT reach zero. The target is $a - z_0$, but near $t=1$ the input $z_t$ collapses
  onto $a$ and the network cannot recover $z_0$, leaving an irreducible residual. Measured floor at
  the test seed: final ~0.18 from start ~2.32 (ratio ~0.08). The test asserts the ratio, not a
  near-zero floor.
- `flow_sample` reconstruction MAE is ~0.14 on the overfit batch (loose secondary bound, 0.30 cap).
- `sinusoidal_embedding` must respect the input dtype (no `.float()` downcast) or the float64
  gradcheck on $t$ fails. The provided helper already does this.
- The encoder and head train jointly under one optimizer; each head gets its own encoder instance so
  the flow-vs-BC comparison is clean. `embed_dim=128` is the heads' `cond_in` on the reacher.
- Demo collection filters to episodes that reach (reward > 0.5), truncated at the reach step. The
  analytic expert reaches on ~75-83% of seeds; render the frame BEFORE stepping so the recorded obs
  is what the policy sees at that step.
- Reacher reach-success is unimodal in the image, so deterministic BC matches the flow head from
  pixels (measured flow 0.75, BC 0.75 at H=4, random floor ~0.06). Do not claim a flow-over-regression
  win from pixels. The generative win is in the point-mass goal-hidden side-demo: BC averaged-action
  magnitude ~0.001 (collapses to origin) vs flow sample spread ~0.015 (spreads to four directions).
- Measured viz numbers (RTX 4080): pixel reach success flow/BC/random = {0.75, 0.75, 0.06}; BC chunk
  sweep H={1,4,8} = {0.78, 0.74, 0.62} (does NOT rise with H on this short-horizon clean-demo
  reacher; chunking's compounding-error benefit needs a harder regime - frame it as a toy measurement,
  not a chunking-literature claim).
```
