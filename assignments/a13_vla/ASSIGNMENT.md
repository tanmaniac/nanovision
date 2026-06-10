# assignments/a13_vla/ASSIGNMENT.md

```yaml
id: a13_vla
title: Vision-Language-Action capstone (flow-matching action head)
module: 4
type: Mixed
estimated_learner_hours: 8
depends_on: [a00_harness, a05_diffusion, a06_0_flow_matching]
builds_into_shared_lib: []   # leaf assignment, last in the course; nothing imports it, so no nanovision shim
forbidden_imports:
  - gym / gymnasium          # the env is pure NumPy, written from scratch
  - diffusers                # the DDPM head is built from the diffusion objective, not a library
  - robomimic / lerobot / dm_control
  - torchcfm / torchdiffeq   # the CFM objective and the ODE integrator are written from scratch
  - bare cross-assignment imports (from assignments...) — A13 owns its modules locally
fits_12gb: true
external_data: none (scripted demonstrations from the local 2D point-mass env)
```

## motivation

The 2023-2026 vision-language-action (VLA) pattern wires a perception/language backbone to a
dedicated action decoder: the backbone interprets the scene and instruction, the decoder generates
continuous, temporally coherent actions. Text-token autoregression fits reasoning but not 50-100 Hz
continuous control, so the action head is a separate generative model. You build the pi0-line head:
conditional flow matching (CFM) over an action chunk, conditioned on robot state. Single-step
behavior cloning is the compounding-error baseline, action chunking is the fix that the
single-step-vs-chunk ablation measures, and a DDPM head is the diffusion-vs-flow contrast. The full
treatment (the discretized-token vs continuous-head split, the data engines, the conditioning
options) is in the README.

## background

The action head is the generative model from the diffusion and flow-matching topics, re-conditioned
on state instead of a class label. CFM convention: $t=0$ is noise $z_0 \sim \mathcal{N}(0, I)$,
$t=1$ is the demonstrated action chunk $a$. The straight path and its constant velocity target are

$$z_t = (1-t)\,z_0 + t\,a, \qquad v = a - z_0 \quad (\text{constant in } t).$$

The network $v_\theta(z_t, t, c)$ regresses onto $v$ by MSE; at inference, integrate
$z \leftarrow z + \tfrac{1}{n}\,v_\theta(z, t, c)$ from $t=0$ to $t=1$ with $n \approx 10$ Euler
steps. Action chunking predicts an $H$-step chunk executed open-loop. The DDPM contrast uses the
epsilon-prediction objective $a_t = \sqrt{\bar\alpha_t}\,a + \sqrt{1-\bar\alpha_t}\,\varepsilon$ and
the ancestral reverse chain.

Shapes: $a$, $z_0$, $z_t$ are $(B, H, 2)$; $t$ is $(B, 1, 1)$; $c$ is $(B, \text{cond\_in})$;
chunks are $(B, T-H+1, H, 2)$.

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
   `cfm_target`, predict $v_\theta(z_t, t, c)$, return MSE. Unweighted velocity regression. Teaches
   the CFM objective.

3. **Task 3 - flow_sample** (file: `flow.py`, symbol: `flow_sample`): from $z \sim \mathcal{N}(0,I)$
   of shape $(B, H, 2)$, take `n_steps` forward-Euler steps with $dt = 1/n$, $t = k\,dt$, updating
   $z \leftarrow z + dt\,v_\theta(z, t, c)$. Return $z$. Teaches few-step ODE sampling. The likely
   bug is starting at $t=1$ or a wrong $dt$.

4. **Task 4 - bc_loss** (file: `bc.py`, symbol: `bc_loss`): MSE between `policy(c)` and the
   demonstrated chunk. Teaches that BC regresses the conditional mean, which is correct only when
   $p(a|c)$ is unimodal.

5. **Task 5 - chunk_actions / de_chunk / receding_horizon_indices** (file: `bc.py`): overlapping
   $H$-windows $(B, T, 2) \to (B, T-H+1, H, 2)$; the inverse (full first chunk, then the last action
   of each later chunk) that reconstructs the sequence exactly; the open-loop start indices
   $0, H, 2H, \dots$ clamped to $T-H$. Teaches action chunking and receding-horizon execution.

6. **Task 6 - ddpm_loss / ddpm_sample** (file: `ddpm.py`): the epsilon-prediction loss and the
   ancestral reverse chain re-conditioned on $c$. Teaches the diffusion-vs-flow contrast (a noise
   schedule and many reverse steps vs velocity regression and a few Euler steps).

## tests

Run with `/home/tanmay/miniconda3/envs/nanovision/bin/python -m pytest assignments/a13_vla/tests`.
Default mode fails cleanly at the holes; `NANOVISION_IMPL=solution` is green. Order: cfm_target,
flow_sample_ode, chunking, shapes, gradcheck, overfit_bc, overfit_flow, forbidden_imports.

- `tests/test_cfm_target.py` - reference-value: $z_t$ at $t=0$ is $z_0$, at $t=1$ is $a$; $v = a-z_0$
  exactly and identical at every $t$ (t-independence); shapes.
- `tests/test_flow_sample_ode.py` - exact oracle: a constant velocity field integrates from $z_0$ to
  $a^\star$ exactly at any step count (1, 5, 10, 50), checking the integrator wiring.
- `tests/test_chunking.py` - exact: `de_chunk(chunk_actions(x)) == x` for several $(T, H)$; the
  receding-horizon indices start at 0, stay in $[0, T-H]$, have no duplicates, and cover all $T$.
- `tests/test_shapes.py` - shape: the three heads' forward and `flow_sample` give $(B, H, 2)$;
  `chunk_actions` gives $(B, T-H+1, H, 2)$.
- `tests/test_gradcheck.py` - float64 gradcheck: `cfm_target` wrt the action; the flow network wrt
  $z_t$ and $t$; `flow_loss` wrt the action with a fixed-seed generator.
- `tests/test_overfit_flow.py` - bounded training, robust: `flow_loss` drops to a small fraction of
  its untrained value (measured ratio ~0.08); a loose secondary bound on `flow_sample`
  reconstruction (measured MAE ~0.14). The loss does NOT reach zero by construction (see
  solution_notes).
- `tests/test_overfit_bc.py` - bounded training: `bc_loss` overfits one batch to near zero.
- `tests/test_forbidden_imports.py` - static tokenize scan over the top-level holed files and the
  solution for robot/diffusion/flow libraries and bare cross-assignment imports. Passes in both
  modes.

The headline ablation (chunking/flow beats single-step BC on rollout success) is NOT a unit test;
rollout success is init- and seed-sensitive. It is in `viz.py` and the README with per-seed
statistics against the straight-line-expert ceiling.

## provided_boilerplate

`env.py` (the 2D point-mass reacher, the scripted expert, `collect_demos` with both
`goal_conditioned` modes), `config.py` (`VLAConfig`, `GradcheckConfig`), `conftest.py`, `_train.py`
(chunk-batch building, the three training loops, the rollout-success metric), `viz.py` (the five
GPU-aware panels), the three head modules' `__init__`/`forward`/`sinusoidal_embedding`/`make_schedule`.

## compute_notes

Everything runs on CPU in seconds: the tests are exact oracles plus bounded overfit loops (300-500
steps). `viz.py` uses the GPU when present (`nanovision.determinism.default_device`); the full five
panels train ~10 small heads and take a few minutes on an RTX 4080. v_max is 0.05 so an episode is
~20 steps, long enough for the $H=16$ chunk; demos pad to $T=24$. A healthy `flow_loss` curve drops
from ~2.3 to a floor near ~0.18 and stays there (the floor is real, not a stuck run).

## stretch_goals

1. Language conditioning: pass the goal phrase ("top-right corner") through a frozen small text
   encoder and project to $c$; train only the flow head and the projector.
2. Temporal ensembling: blend overlapping chunks from consecutive queries with exponential weighting
   and compare to open-loop execution (pi0 found ensembling detrimental and dropped it).
3. The image observation path: render the state to a small image and condition the flow head on a
   ViT/CLIP-style encoder instead of the raw 2D state.
4. World-model-conditioned VLA: condition the action head on a learned latent from the world-models
   assignment (the model-based flavor).

## further_reading

- ACT / ALOHA, [arXiv 2304.13705](https://arxiv.org/abs/2304.13705) - action chunking with a CVAE.
- Diffusion Policy, [arXiv 2303.04137](https://arxiv.org/abs/2303.04137) - the DDPM action head.
- pi0, [arXiv 2410.24164](https://arxiv.org/abs/2410.24164) - the flow-matching VLA built here.
- OpenVLA, [arXiv 2406.09246](https://arxiv.org/abs/2406.09246) - the open discretized-token VLA.
- OpenVLA-OFT, [arXiv 2502.19645](https://arxiv.org/abs/2502.19645) - the same-backbone ablation
  showing the action-head design dominates the outcome.
- Open X-Embodiment, [arXiv 2310.08864](https://arxiv.org/abs/2310.08864) - the pooled dataset.

## solution_notes

- `flow_loss` does NOT reach zero. The target is $a - z_0$, but near $t=1$ the input $z_t$ collapses
  onto $a$ and the network cannot recover $z_0$, leaving an irreducible residual. Measured floor at
  the test seed: final ~0.18 from start ~2.32 (ratio ~0.08). The test asserts the ratio, not a
  near-zero floor; the overfit batch uses unit-scale actions so the residual is well above noise.
- `flow_sample` reconstruction MAE is ~0.14 on the overfit batch (loose secondary bound, 0.30 cap).
  It is the only learned-sampler assertion in pytest, kept loose on purpose.
- `sinusoidal_embedding` must respect the input dtype (no `.float()` downcast) or the float64
  gradcheck on $t$ fails. The provided helper already does this.
- Rollout evaluation must use `env.sample_start` (far-side starts matching the demo distribution).
  A start sampled uniformly over the whole square lands in near-goal states the demos never visited,
  which is out of distribution and reads as low success for reasons unrelated to the mechanism.
- In the goal-conditioned (default) mode $p(a|c)$ is unimodal, so deterministic BC matches or beats
  the flow head on rollout success (the flow head's sampling noise is pure downside there). Do not
  claim a flow-over-regression win in this mode. The flow head's value is in the goal-dropped
  (multimodal) mode: measured BC averaged-action magnitude ~0.002 (collapses to the origin) vs flow
  sample spread ~0.021 (spreads to the four goal directions).
- Measured viz numbers (RTX 4080, seeds 0-2): chunk ablation (BC) success H={1,4,16} =
  {0.93, 0.97, 1.00}; flow vs DDPM rollout = {0.23, 0.27} (both noisy in the unimodal mode); flow
  reconstruction MAE at {1,2,5,10} Euler steps = {0.067, 0.025, 0.013, 0.010}.
```
