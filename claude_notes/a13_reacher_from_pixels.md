# A13 upgrade: reacher control from pixels

Turn the committed A13 (2D point-mass toy, state conditioning) into authentic robot
control from pixels: dm_control reacher 'easy', 64x64 RGB observation, flow-matching
action head conditioned on a CNN image embedding, behavior-cloned from filtered expert
demonstrations.

## What is kept (mechanisms, correct already)
- `flow.py`: `cfm_target`, `flow_loss`, `flow_sample` (HOLES + solution). Take a
  conditioning vector `c` of width `cond_in`. Unchanged except `act_dim` is already
  config-driven (2 for reacher torque, same as the point-mass).
- `bc.py`: `bc_loss`, `chunk_actions`, `de_chunk`, `receding_horizon_indices`. Unchanged.
- `ddpm.py`: DDPM contrast head. Unchanged.
- All mechanism tests: cfm_target, flow_sample ODE, chunk round-trip, gradcheck, shapes,
  overfit, forbidden imports. These run on CPU WITHOUT dm_control.
- The point-mass multimodal demo (GOALS, expert_action, make_condition, sample_start,
  rollout_expert, collect_demos for the point mass) is kept in env.py as a self-contained
  2D side-demo for the "generative head vs regression" lesson (mode averaging).

## What changes
- The conditioning is now `c = Encoder(obs)` where obs is (3,64,64) in [0,1], instead of a
  state+goal vector. The heads are unchanged; only what feeds `c` changes.
- New `nets.py`: the 4-conv 64x64 Encoder (ported from a12 world-models solution/nets.py).
- env.py gains the dm_control reacher wrapper (lazy import inside functions), the IK+PD
  expert, filtered demo collection (keep episodes that reach, truncate at/after reach),
  a render helper, a random/scripted policy, and a reach-success metric.
- `_train.py`: a pixel path. Build (obs, action-chunk) batches from filtered image demos,
  encode obs -> c, train flow head + BC baseline jointly with the encoder, and a rollout
  that runs the policy in the real reacher (encode each frame -> c -> flow_sample a chunk
  -> execute) and measures reach success.
- config.py: reacher dims (act_dim=2, obs_ch=3, obs_size=64, embed_dim), keep flow
  hyperparameters.

## Encoder + head joint training
The image encoder and the action head train together end-to-end under BC. For the flow
head: `c = Encoder(obs); loss = flow_loss(head, a_chunk, c)`, one optimizer over both. For
BC baseline: `c = Encoder(obs); loss = bc_loss(policy, a_chunk, c)`. The encoder is shared
machinery, trained per head (each head gets its own encoder instance so the comparison is
clean).

## Isolation
dm_control is imported only inside env.py functions and in viz.py. The mechanism tests
import flow/bc/ddpm/config/nets only and never touch dm_control. The env smoke test uses
`pytest.importorskip("dm_control")`. `nets.py` is pure torch (no dm_control).

## Verification gate
Collect ~200 filtered successful demos, train flow + BC from pixels, roll out in the real
reacher, confirm reach success clearly beats a random-torque policy (random ~5-20%, BC
target >=50%). Report measured numbers. If BC does not beat random clearly, stop and report.

## Test modes
- `NANOVISION_IMPL=solution pytest` green (mechanism tests + env smoke if dm_control present).
- default mode fails at the holes (NotImplementedError).
- mechanism tests pass WITHOUT dm_control (CPU-only grading path).
