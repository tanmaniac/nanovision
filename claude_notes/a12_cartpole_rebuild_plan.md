# A12 rebuild - DreamerV3 on cartpole-from-pixels (continuous dynamics-backprop actor)

Status: rebuild plan. The original A12 toy was a discrete-action image gridworld; it could not
demonstrate imagination-to-real transfer (the policy exploited the inaccurate prior, and discrete
REINFORCE on a near-flat reward collapsed). After a fresh CV/RL-professor consult, the task is
changed to the canonical regime where Dreamer's "behaviors learned in imagination" demonstrably
transfers: dm_control cartpole-balance from 64x64 pixels, with a CONTINUOUS Tanh-Normal actor
trained by DYNAMICS BACKPROP (backprop the lambda-return through the differentiable world model),
the DreamerV3 path for continuous control. This rebuild replaces the (uncommitted) gridworld A12.

The authoritative, VERIFIED reference implementation is `/tmp/a12_cartpole.py` (a single-file
working trainer). It clears the random baseline and does not collapse: greedy real-env return
~300+ (best ~350) vs random ~214 vs optimal ~500, on a single 12GB GPU in ~1-2 hours. Port this
into the assignment structure; preserve its working code exactly (do not "improve" the
hyperparameters or the loss).

## 1. What changes vs the old gridworld A12, and what is reused unchanged

REUSED UNCHANGED (correct + already exact-tested - do not modify the solution logic):
- `solution/rssm.py` RSSMCell: forward_h, prior, posterior, observe, imagine, initial_state. ONE
  compatibility note: `forward_h` already accepts a continuous action when it is passed as a
  `(B, action_dim)` float (it only one-hots when `a.dim()==1`). With `action_dim=1` and continuous
  actions passed as `(B,1)` floats, it works as-is.
- `solution/nets.py`: symlog/symexp, twohot_encode/decode, twohot_loss, categorical_sample
  (straight-through + unimix), value_bins. The Encoder/Decoder in nets.py are resized to 64x64 (see
  below) but the scalar-encoding and categorical-sampler holes are identical.
- `solution/world_model.py` kl_loss (two separately-weighted KL terms, free bits on the summed KL).
- `solution/actor_critic.py` compute_lambda_returns, critic_loss, ReturnNormalizer, Critic.

CHANGED:
- Task/env: gridworld -> dm_control cartpole-balance, 64x64x3 pixel observations, continuous 1-D
  force action, dense reward in [0,1], 1000-step episodes (no early termination).
- Encoder/decoder: 16x16 2-conv -> 64x64 4-conv (channels 32/64/128/256; mirror decoder).
- Model size (config): h_dim 128->512, latent 16x16->32x32, embed 128->1024, horizon 15->8.
  Reward/continuation heads widen to 512 hidden.
- THE ACTOR (the headline pedagogical change): discrete Categorical + REINFORCE -> continuous
  Tanh-Normal + dynamics backprop. This is the heart of the rebuild.
- Training loop (`_train.py`): the collect-fit-imagine loop with the differentiable imagination and
  the dynamics-backprop actor update, a critic warmup, and a recurrent (state-carrying) eval policy.
- viz.py, README.md, ASSIGNMENT.md: rewritten for cartpole + dynamics backprop.

## 2. The headline new concept: dynamics backprop vs REINFORCE

The README must teach why the actor changes. DreamerV3 uses REINFORCE for discrete actions and the
reparameterized dynamics-backprop gradient for continuous actions. cartpole-balance has a near-flat
reward in the action (a balanced pole earns ~1 regardless of small force differences over a short
horizon), so the score-function (REINFORCE) estimator is dominated by variance and the policy
collapses. Because the world model is differentiable, the analytic gradient
$\nabla_\theta \mathbb{E}[\,\sum_t \gamma^t r_t\,]$ obtained by backpropagating the imagined
lambda-return THROUGH the learned dynamics into a reparameterized action is dense and low-variance.
That is the core payoff of model-based RL: a differentiable world model gives you an analytic policy
gradient. Define both estimators, state when each is used, and show (the measured contrast) that
dynamics backprop transfers here while discrete REINFORCE collapses.

The reparameterized Tanh-Normal action: actor outputs `(mean, log_std)`; `a = tanh(mean + std *
eps)`, `eps ~ N(0,1)`; gradient flows through `a` into the dynamics. `log_std` is clamped to a floor
(`log(0.1)`) so the policy cannot become a delta. Entropy bonus uses the pre-tanh Normal entropy,
small coefficient (1e-4).

## 3. Files and holes

Mirror the existing A12 file layout (it is already a leaf assignment). Provided (no hole): `env.py`
(dm_control wrapper + 64x64 render + episode collection), `config.py`, `viz.py`, `_train.py`
(the collect-fit-imagine loop, critic warmup, recurrent eval), the resized Encoder/Decoder bodies.

Holes (top-level + solution), most identical to the current A12:
- `nets.py`: symlog/symexp, twohot_encode/decode, categorical_sample (unchanged holes).
- `rssm.py`: forward_h, prior, posterior (unchanged holes; forward_h handles continuous actions).
- `world_model.py`: kl_loss, the ELBO assembly `loss` (recon MSE on symlog obs, two-hot reward,
  BCE continuation, KL). Unchanged in structure (encoder/decoder resized).
- `actor_critic.py`:
  - `compute_lambda_returns`, `critic_loss` (unchanged holes).
  - NEW HOLE `actor_loss_dynbackprop(returns, entropies, ent_coef, ret_range)`:
    `-(returns / ret_range).mean() - ent_coef * entropies.mean()`. The returns here are
    DIFFERENTIABLE w.r.t. the actor (they carry gradient through the imagined dynamics), so the
    loss is just the negative normalized return plus the entropy bonus - no log-prob/advantage.
    Contrast in a comment with the old REINFORCE form.
  - NEW provided class `ContActor` (Tanh-Normal head, `sample` returns `(action, entropy)`,
    reparameterized) - provide the body; the lesson is the loss + the differentiable imagination,
    not the MLP.
- `imagine` (in rssm.py or the training module): the DIFFERENTIABLE rollout. Per step: sample the
  reparameterized action from the actor at `(h,z)`, `h = forward_h(h,z,a)`, `z ~ prior(h)`; collect
  pre-action states `s_t` and post-action states `s_{t+1}`. Reward of `a_t` is decoded from the
  POST-action state `s_{t+1}` (the verified reward/action alignment fix). Do NOT wrap the reward /
  value decode in `no_grad` (the gradient must flow). This is a hole: the student writes the loop
  that keeps the graph attached so dynamics backprop works.

Keep the imagination start from DETACHED posterior states (encoded from a replay batch) and roll
forward under the prior. Keep the recurrent eval policy (carry `(h,z)` across the real episode:
reset to `initial_state`, each step `h=forward_h(h,z,a_prev); z~posterior(h, encode(obs))`).

## 4. Config (provided) - the verified working values

`obs_size=64, obs_ch=3, h_dim=512, n_cat=32, n_cls=32, embed_dim=1024, action_dim=1, unimix=0.01,
free_bits=1.0, kl_dyn_scale=0.5, kl_rep_scale=0.1, gamma=0.997, lam=0.95, ent_coef=1e-4,
ret_ema_decay=0.99, n_bins=255, bin_lo=-20, bin_hi=20, horizon=8`. Training: actor_lr=critic_lr=4e-5,
model_lr=1e-4, batch 16 x seq 50, ~100 updates per collected episode, critic warmup 5 iters, action
discretization NONE (continuous), action_repeat 2. These are DreamerV3-aligned and MEASURED to work;
do not change them.

## 5. Tests (CPU seconds, both modes; no fragile assertions)

Keep the existing mechanism tests (they pass and are exact): symlog/two-hot inverses, straight-
through gradient, RSSM shapes, gradcheck (forward_h+prior, kl_loss, lambda-returns), KL free-bits
and 0.5/0.1 weights, forbidden imports. Update/add:
- `test_cont_actor.py`: the Tanh-Normal `sample` is reparameterized - gradient of the action w.r.t.
  the actor params is nonzero (autograd), action is in (-1,1), log_std respects the floor.
- `test_imagine_differentiable.py`: a one-step imagined return has nonzero gradient w.r.t. the actor
  params THROUGH the dynamics (the dynamics-backprop path is connected). This is the test that
  guards the headline mechanism. Keep it a gradient-exists check, not a training-convergence assert.
- env smoke test: the dm_control wrapper resets/steps and renders `(3,64,64)`; mark it to skip
  cleanly if dm_control/MuJoCo is unavailable (it is a heavy optional dep) so the core suite still
  runs. The graded mechanism tests must NOT depend on dm_control.

NOT a unit test (gate): the transfer success number. It lives in viz.py + the README with the
measured greedy-vs-random-vs-optimal numbers over seeds. Do not pin a return threshold.

## 6. viz.py (GPU; needs dm_control)

`MUJOCO_GL=egl`. Panels: (1) the learning curve - greedy real-env return vs env steps, against the
random baseline and the ~500 optimal ceiling; (2) a reconstruction panel (real frames vs decoded);
(3) an imagined rollout filmstrip decoded from the prior under the trained actor; (4) the
discrete-REINFORCE-vs-continuous-dynamics-backprop contrast (the measured collapse vs transfer) as
the headline figure. Load the trained checkpoint the verification run produced if available, else
train briefly. Long full training is a documented multi-hour run, not run in CI.

## 7. README (lecture notes) - the honest result

Teach: RSSM, the sequence-VAE ELBO + KL (free bits, two weighted terms), categorical latents +
straight-through + unimix, symlog + two-hot, imagination as prior rollout, and actor-critic in
imagination with the dynamics-backprop-vs-REINFORCE distinction as the centerpiece. Report the
MEASURED transfer honestly: the imagination-trained policy reaches greedy ~300+ (best ~350) vs the
random baseline ~214 vs optimal ~500 - it demonstrably learns to balance the cartpole from a policy
trained purely on imagined rollouts (the "Dream to Control" result), while toy-scale training
(single GPU, ~hours, small model) does not reach optimal. Keep the at-scale caveat. Include the
honest contrast experiment: discrete REINFORCE on this task collapses below random (the measured
135), which is WHY continuous control uses the dynamics-backprop gradient. Verify arXiv ids
(PlaNet 1811.04551, DreamerV1 1912.01603, DreamerV2 2010.02193, DreamerV3 2301.04104). Real LaTeX.
Run the mandatory context-less style review before commit.

## 8. Gate checklist

- Build target is the working DreamerV3 continuous-control recipe; the discrete REINFORCE collapse
  is shown only as a labeled contrast experiment, never as the endpoint.
- No fragile/seed-pinned test. The transfer number is in viz/README with measured stats; every
  pytest assertion is an exact oracle or a gradient-exists/shape check.
- README reports the real measured numbers and keeps the at-scale caveat; the toy "~300 not 500" is
  framed as a scale/compute artifact, not as the mechanism failing.
- The dm_control dependency is isolated to env.py/viz.py; the graded mechanism tests run without it.
