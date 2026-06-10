# A13 - VLA capstone (flow-matching action head): build plan

Status: plan for expert review, then delegated build. Leaf assignment, last in the course (no
`nanovision` shim). Build target is a conditional flow-matching (CFM) action head in the pi0 line
(Black et al., 2024), with single-step behavior cloning and a DDPM head as labeled contrast.
Research note: `docs/research/a13_vla.md`.

## 1. What the student builds and why

A13 wires the course together: a vision/goal encoder feeds a generative action head trained by
behavior cloning on scripted demonstrations. The capstone teaches the 2023-2026 VLA pattern -
a perception/language backbone produces a conditioning vector, and a purpose-built action decoder
generates temporally coherent continuous actions - on a toy small enough to train in minutes and
verify by eye.

Four things the student implements and the lesson each carries:

1. Single-step behavior cloning (BC) - the compounding-error baseline. An MLP predicts the next
   action from the observation. At rollout, small errors push the point mass into states absent
   from the demos and the error compounds. This motivates everything after it.
2. Action chunking (ACT, Zhao et al., 2023) - predict an H-step action chunk and execute it
   before re-querying, cutting the decision frequency by H and forcing internally consistent
   trajectories. The single-step-vs-chunk ablation is the assessment vehicle.
3. Conditional flow matching action head (the build target) - learn a velocity field that
   transports a Gaussian sample to the demonstrated action chunk along a straight path, integrated
   with ~10 Euler steps at inference. This is the pi0 action-head mechanism.
4. A DDPM action head (contrast) - the same chunk generation as a denoising chain, to make the
   diffusion-vs-flow tradeoff concrete on one task (DDPM needs many steps and a noise schedule;
   flow matching trains by plain velocity regression and integrates in few steps).

Forward/back pointers by name: the action head is the generative model from the course's diffusion
and flow-matching topics, re-conditioned on robot state instead of a class label; the vision
encoder is a ViT/CLIP-style backbone; the optional language path reuses the VLM text encoder. The
2025-2026 framing: continuous generative action heads have displaced per-step discretized tokens
for new systems - flow matching (pi0, pi0.5) and diffusion transformers (RDT-1B) on one side,
discretized action tokens (RT-2, OpenVLA) on the other. Flow matching is the build target here;
DDPM is the diffusion baseline it is compared against (a plain DDPM denoiser, not a diffusion
transformer - RDT-1B is named only as a reading pointer, not built). The FAST tokenizer is the note
that the discrete branch is not dead.

### Shared-library decision

A13 is the final assignment and a leaf. Nothing imports from it, so no `nanovision/<mod>.py` shim;
it owns its modules locally like A11.5e. Flag for the expert: the research note says the action
head is "almost a direct reuse" of the flow-matching machinery, but the shared lib does not expose
a flow-matching module (the course built it inside the diffusion/flow assignments, not as a
`nanovision.*` export), so A13 reimplements the CFM objective locally in the action-conditioning
context. Confirm that is the intended scope and there is no shared CFM contract to import.

## 2. The toy task (provided, not a hole)

`env.py` - a deterministic 2D point-mass reacher in pure NumPy, no external RL/robot dependency.

- State: position `p = (x, y)` in the unit square `[0, 1]^2`. Action: a velocity `a = (vx, vy)`
  clipped to `[-v_max, v_max]` (v_max ~ 0.1). Step: `p <- clip(p + a, 0, 1)`.
- Goal: one of 4 fixed targets (the corners, inset from the edge), selected per episode. The goal
  is the conditioning signal, provided two ways the student can switch between: a one-hot of the 4
  goals (default) and the 2D goal coordinate. (The optional VLM path replaces this with a text
  embedding; see section 5.)
- Conditioning modes (this is what motivates a generative head over plain regression):
  - `goal_conditioned=True` (default): `c` carries the goal, so the expert action is a
    deterministic function of (state, goal) up to small jitter - `p(a | c)` is effectively
    UNIMODAL. Plain MSE regression fits this as well as a flow head. Here the flow head's role is
    mechanism demonstration (it learns a correct conditional generator and integrates in few
    steps), NOT beating regression - the lesson in this mode is compounding error and chunking.
  - `goal_conditioned=False`: the goal is hidden from `c` (condition on state only) while the goal
    still varies across episodes, so from a given state the demonstrated action points to one of
    several goals - `p(a | state)` is genuinely MULTIMODAL. A unimodal MSE regressor provably
    averages the modes (aims between goals, fails); the flow head samples a coherent mode. This is
    the mode that motivates flow matching over regression, and it is a viz panel + a README point.
- Demonstrations: a scripted expert that moves straight toward the goal at speed `v_max` with a
  small Gaussian jitter, stopping within `eps` of the goal. `collect_demos(n, seed)` returns ~200
  trajectories of `(state, goal, action)` where the action is the expert's next-step velocity.
  Episode length ~30 steps (enough to cross the square at `v_max`).
- Observation for the policy: the state `p` (2D) concatenated with the goal conditioning. A small
  image-render mode is provided for the optional CNN/ViT encoder path but is NOT the default (the
  default keeps the obs as the 2D state so CPU tests are instant).
- Success metric: fraction of rollouts that end within `eps` of the goal. A straight-line expert
  is the ceiling (100%); a single-step BC policy under compounding error is the contrast.

Why this task: it is the minimal setting where compounding error is visible (a single-step policy
drifts and overshoots), where action chunking measurably helps (committing to an H-step straight
segment), and where a generative head's multimodality matters only mildly (so the lesson is the
mechanism, not exploration). It trains in seconds on CPU.

## 3. Files and holes

Mirror `assignments/a11_5e_pred_planning` for structure (`conftest.py`, top-level holed files,
`solution/<file>.py` per holed file, `config.py`/`viz.py` top-level only, `tests/`, `README.md`,
`ASSIGNMENT.md`, `__init__.py`s). Exemplar to read cold: `assignments/a11_5e_pred_planning` and,
for the flow-matching objective, the course's flow-matching assignment
`assignments/a06_0_flow_matching` (named in the build guide as the canonical CFM exemplar).

Provided (no hole, top-level only): `env.py`, `config.py`, `viz.py`, the goal/observation encoder
(a small MLP from obs+goal to the conditioning vector `c` - provided plumbing), and the
training-loop scaffolding in `_train.py`.

Holed files (top-level hole + `solution/` answer key):

### `flow.py` - the conditional flow-matching action head

- `cfm_target(a_chunk, z0, t)`: hole. The straight-path interpolant and velocity target.
  `z_t = (1 - t) * z0 + t * a_chunk`; target velocity `v = a_chunk - z0` (constant along the path,
  independent of `t`). Shapes: `a_chunk, z0` are `(B, H, 2)`, `t` is `(B, 1, 1)` broadcast.
- `class FlowHead(nn.Module)`: an MLP/transformer velocity field `v_theta(z_t, t, c) -> (B, H, 2)`,
  conditioned on the chunk-state `z_t`, a sinusoidal/Fourier embedding of `t`, and the conditioning
  `c`. `__init__` and the network body are provided; the `forward` wiring of `[flatten(z_t),
  temb(t), c] -> v` is provided. The HOLE is `flow_loss(head, a_chunk, c)`: sample
  `z0 ~ N(0, I)`, `t ~ Uniform(0,1)`, build `z_t` and `v` via `cfm_target`, predict
  `v_theta(z_t, t, c)`, return `mse(pred, v)`.
- `flow_sample(head, c, H, n_steps)`: hole. Euler-integrate the ODE from `t=0` to `t=1`:
  `z <- z + (1/n_steps) * v_theta(z, t, c)` over `n_steps` (default 10), starting from
  `z ~ N(0, I)` of shape `(B, H, 2)`. Return the final `z` as the action chunk. (No external ODE
  library; the integrator is a short loop.)

### `bc.py` - the behavior-cloning baseline and action chunking

- `class BCPolicy(nn.Module)`: an MLP `c -> (H, 2)` predicting an H-step action chunk directly
  (deterministic regression). `H = 1` is the single-step baseline. Body provided.
- `bc_loss(policy, a_chunk, c)`: hole. `mse(policy(c), a_chunk)` - plain behavior cloning on the
  chunk. The contrast to `flow_loss`: BC regresses the conditional MEAN action. When `p(a|c)` is
  unimodal (goal-conditioned mode) the mean is correct and BC matches the flow head; when `p(a|c)`
  is multimodal (goal-dropped mode) the mean is a point between modes that the expert never takes,
  so BC fails and the flow head (which samples a mode) does not. Do not claim a flow-over-regression
  win in the unimodal mode.
- `chunk_actions(actions, H)`: hole (provided in `_train.py`? no - keep it a small tested hole).
  Given a per-step action sequence `(B, T, 2)`, build overlapping H-step chunks `(B, T-H+1, H, 2)`
  for training, and the receding-horizon execution helper that yields the next chunk. Exact-testable.

### `ddpm.py` - the DDPM action head (contrast, holed)

- `ddpm_loss(head, a_chunk, c, alphas_bar)`: hole. The course's diffusion epsilon-prediction
  objective re-conditioned on `c`: sample `t`, `eps ~ N(0,I)`, form
  `a_t = sqrt(abar_t) a_chunk + sqrt(1 - abar_t) eps`, predict `eps_theta(a_t, t, c)`, return
  `mse`. `ddpm_sample(head, c, H, schedule)`: the reverse chain. The schedule and the head body are
  provided. This exists so the README/viz can compare DDPM (many steps) vs flow (few steps) on the
  same task; it is a smaller, secondary hole.

### `config.py` (provided)

Tiny: `act_dim=2`, `chunk=4` (default H; ablation runs {1,4,16}), `cond_dim=64`, `hidden=128`,
`n_flow_steps=10`, `v_max=0.1`, `eps=0.05`, `n_goals=4`, `ddpm_T=50`. A `gradcheck` sub-config
shrinks hidden for float64 speed. State the production scale (pi0's 3B VLM + 300M action expert,
flow head over 7-DoF action chunks) in a comment so the toy reads as a mechanism isolator.

## 4. Tests (CPU, seconds; both modes)

Default mode fails cleanly at the holes; `NANOVISION_IMPL=solution` green. Run with
`/home/tanmay/miniconda3/envs/nanovision/bin/python -m pytest`. Prefer exact analytic oracles.

1. `test_cfm_target.py` (training-free, exact): `z_t` at `t=0` equals `z0`, at `t=1` equals
   `a_chunk`; the velocity target `v = a_chunk - z0` is exactly recovered and is `t`-independent;
   shapes `(B, H, 2)`.
2. `test_flow_sample_ode.py` (training-free, exact): with a velocity network REPLACED by the
   analytic constant field `v(z,t,c) = a* - z0_known` (a fixture), Euler integration from `z0_known`
   to `t=1` lands on `a*` to within the integrator's error bound; more robustly, for a constant
   target velocity the straight-line integrator is exact regardless of `n_steps`. This checks the
   integrator wiring without depending on training.
3. `test_shapes.py`: `FlowHead.forward` and `flow_sample` give `(B, H, 2)`; conditioning batch dim;
   `chunk_actions` produces `(B, T-H+1, H, 2)`.
4. `test_gradcheck.py` (float64, gradcheck sub-config): `flow_loss`'s velocity network is
   differentiable wrt `z_t` and `t`; `cfm_target` passes `torch.autograd.gradcheck`.
5. `test_chunking.py` (training-free, exact): `chunk_actions` round-trips under a SPECIFIED de-chunk
   rule (overlapping windows are redundant, so the inverse must be defined): take the full first
   chunk, then append the last action of each subsequent chunk; this reconstructs the original
   `(B, T, 2)` sequence exactly. The receding-horizon helper yields the right next-chunk indices.
   Exact, no training.
6. `test_overfit_flow.py` (bounded training, robust): the PRIMARY assertion is the loss drop -
   `flow_loss` drives below a pre-measured floor (target < 0.02, measure and set) on one fixed batch
   within <=300 steps. A SECONDARY, loose check is that `flow_sample` reproduces the batch's action
   chunk to within a generous tolerance (measure and floor it). The loss drop carries the test; the
   sampler reconstruction is a sanity bound, not a tight assertion (it is the only learned-sampler
   result in pytest, so keep its tolerance loose).
7. `test_overfit_bc.py` (bounded training): `bc_loss` overfits one batch to near zero (BC is plain
   regression, so this is a quick sanity check that the conditioning reaches the head).
8. `test_forbidden_imports.py`: one static tokenize scan over top-level + solution holed files;
   forbid any robot/diffusion library (`gym`, `gymnasium`, `diffusers`, `robomimic`, `lerobot`,
   `dm_control`) and any bare cross-assignment import. Passes in both modes.

Explicitly NOT a unit test (per the no-fragile-test gate): the headline ablation claim "action
chunking beats single-step BC on rollout success." Rollout success under compounding error is
init- and seed-sensitive; pinning a success-rate ordering into pytest is exactly the fragile
assertion the build-workflow gate forbids. Instead, `viz.py` runs the rollout ablation on GPU and
the README reports measured success rate and trajectory variance for H in {1, 4, 16} and for
flow-vs-BC-vs-DDPM, with the per-seed spread and the straight-line-expert ceiling. The unit suite
asserts only exact oracles (CFM target, ODE integrator, chunk round-trip) and robust overfit
behavior.

## 5. viz.py (GPU when present)

`from nanovision.determinism import default_device`. Panels to `out/`:
1. Rollout trajectories: 20 rollouts from varied starts for each of single-step BC, chunked BC,
   and the flow head, over the goal targets - the BC drift/overshoot vs the chunked/flow smoothness
   should be visible.
2. Chunk-size ablation: success rate and trajectory variance vs H in {1, 4, 16}, bar chart with
   per-seed spread, against the expert ceiling. This is where the headline ablation lives
   (measured, honestly framed), not in a unit test.
3. Flow vs DDPM: success/smoothness at matched training, and an inference-step sweep (flow at 1, 2,
   5, 10 Euler steps vs DDPM at 50 steps) showing flow's few-step advantage. Note in the panel
   caption that the TRAINED flow sampler is only approximately step-count-invariant (it degrades at
   1-2 steps because the learned field is not literally constant along a trajectory); only the
   analytic constant-field oracle in test 2 is exactly step-count-invariant. Do not state the
   trained sampler is exact at any step count.
4. Flow-matching path: for one conditioning, plot the ODE integration `z_t` from `t=0` (noise) to
   `t=1` (action) to show the straight-line transport.
5. Multimodality (goal-dropped mode): from a fixed state with the goal hidden from `c`, overlay the
   BC regressor's single predicted action (points between goals - the averaged mode the expert
   never takes) against a scatter of many flow-head samples (clustering on the distinct goal
   directions). This is the panel that shows WHY a generative head over plain regression, and it is
   the one place the flow head's distribution-modeling is actually load-bearing on this toy.

Optional VLM panel (only if cheap): a frozen small text encoder embeds the goal phrase
("top-right corner") into `c`; train only the flow head + a linear projector. Keep it optional so
the assignment does not hard-depend on the VLM assignment's weights.

## 6. README (lecture notes, per the skill)

Cover: the VLA paradigm (perception/language backbone + dedicated action decoder, why text-token
autoregression is a poor fit for continuous high-rate control); behavior cloning and compounding
error; why a generative action head over plain regression - when the conditional action
distribution is multimodal (several valid expert actions from one observation), an MSE regressor
converges to the mean of the modes, a point the expert never takes, while a flow head samples a
coherent mode (the goal-dropped panel demonstrates this measured failure of regression); action
chunking and temporal ensembling (define each at first use; note pi0 dropped temporal ensembling
for open-loop chunk execution); discretized action tokens (RT-2, OpenVLA) vs
continuous generative heads (Diffusion Policy, ACT, pi0) as the live split; diffusion vs flow
matching as action heads (DDPM's noise schedule and many steps vs CFM's velocity regression and
few-step ODE); conditioning options (prepend/FiLM vs the pi0 cross-attention action expert);
and the data story (Open X-Embodiment as the pooled dataset, DROID as the diversity argument, pi0's
proprietary hours as the scale argument - three distinct lessons). Real LaTeX for the CFM
interpolant and velocity target. FAST tokenizer and OpenVLA-OFT as reading notes (OFT is the clean
same-backbone ablation showing the action-head design dominates the fine-tuning outcome).

Keep the at-scale caveat: the 2D point-mass with a 4-goal one-hot is a mechanism isolator. Its
numbers demonstrate that chunking reduces compounding error and that a flow head learns a
conditional action distribution; they do not predict pi0-scale manipulation behavior. Do not let
any toy number read as overriding the VLA literature.

As the LAST assignment, the README should close the course thread: name how A13 reuses the
diffusion/flow generative model, the ViT/CLIP vision encoder, and the VLM language interface built
earlier, and point to the optional world-model-conditioned (DreamerV3-style) VLA as the model-based
flavor for students who built the world-models assignment.

Verify every arXiv id by fetching `https://arxiv.org/abs/<id>` and confirming the title (all of
these were checked in the expert review; re-confirm on fetch):
- ACT/ALOHA 2304.13705, Diffusion Policy 2303.04137, RT-2 2307.15818, OpenVLA 2406.09246,
  pi0 2410.24164, Octo 2405.12213, Open X-Embodiment 2310.08864.
- OpenVLA-OFT 2502.19645 (real title "Fine-Tuning Vision-Language-Action Models: Optimizing Speed
  and Success"; OFT is the method name, not the title), FAST 2501.09747 ("FAST: Efficient Action
  Tokenization for Vision-Language-Action Models"), RDT-1B 2410.07864 ("RDT-1B: a Diffusion
  Foundation Model for Bimanual Manipulation" - a diffusion transformer, cite it as such, not as
  flow matching).

Mandatory final step: the context-less README style-review subagent (skill step), before commit.

## 7. Risks and pre-measured expectations

- The overfit floors (flow, BC) and the `flow_sample` reconstruction tolerance must be MEASURED on
  the solution and the thresholds set from the measurement, then floored - do not thrash.
- The CFM velocity target is `t`-independent (`v = a - z0`); a student who makes it depend on `t`
  is wrong. The test asserts `t`-independence.
- `flow_sample` integrates `dt = 1/n_steps`; an off-by-one in the step count or starting at `t=1`
  instead of `t=0` is the likely bug. The analytic-field test (2) catches it.
- The chunking ablation is deliberately out of the asserted suite. If the build agent is tempted to
  assert "H=4 beats H=1 success," it must not - route it to README/viz with per-seed stats.
- DDPM is the secondary contrast, not the build target; keep its hole small and do not let it
  crowd out the flow head as the headline mechanism.

## 8. Build-workflow gate checklist (apply in expert review and on-disk verify)

- Build target is the 2025-consensus flow-matching action head, not a deprecated variant.
  Discretized action tokens and DDPM appear only as labeled contrast (historical / baseline), never
  as the implemented endpoint presented as current best.
- No test asserts a cherry-picked seed or a fragile training threshold. The one outcome-level claim
  (chunking/flow beats single-step BC on rollout success) is in the README/viz with per-seed
  statistics, not a pinned assertion. Every pytest assertion is an exact analytic oracle or a
  robust, implementation-independent inequality.
- The README keeps the at-scale caveat and never frames a toy number as overriding the VLA
  literature.
