# assignments/a12_world_models/ASSIGNMENT.md

```yaml
id: a12_world_models
title: World models (RSSM and DreamerV3)
module: 4
type: Core
estimated_learner_hours: 8
depends_on: [a00_harness, a01_transformer]
builds_into_shared_lib: []   # leaf: A12 owns its modules locally, no nanovision shim (see README)
forbidden_imports:
  - gym
  - gymnasium
  - dreamerv3
  - stable_baselines3
  - tianshou
fits_12gb: true
external_data: none          # dm_control/MuJoCo is an optional dep for the env/viz only
```

## motivation

DreamerV3 is the latent-control world model that reaches strong performance across many domains with
one fixed hyperparameter set. This assignment builds its core: an RSSM trained as a sequence VAE, then a
continuous Tanh-Normal actor trained entirely on imagined latent rollouts by dynamics backprop. The
task is dm_control cartpole-balance from 64x64 pixels; the headline is that a policy learned purely
in imagination transfers to the real env. Full history, the pixel-space / JEPA contrast, and the
measured numbers are in the README.

## background

RSSM state: deterministic $h_t = \mathrm{GRU}(h_{t-1}, [z_{t-1}, a_{t-1}])$ plus categorical
$z_t$ (`n_cat=32` heads x `n_cls=32` classes), `h_dim=512`. Posterior $q(z_t \mid h_t, o_t)$ trains;
prior $p(z_t \mid h_t)$ imagines. ELBO = recon (MSE on symlog target) + reward (two-hot CE) +
continuation (BCE) + KL. KL is two weighted terms with stop-gradients:
$\beta_{\text{dyn}}\max(1, \mathrm{KL}[\mathrm{sg}(q)\|p]) + \beta_{\text{rep}}\max(1, \mathrm{KL}[q\|\mathrm{sg}(p)])$,
$\beta_{\text{dyn}}=0.5$, $\beta_{\text{rep}}=0.1$, free bits 1 nat on the summed-over-heads KL.
Two-hot bins are `linspace(-20, 20, 255)` in symlog space; encode pushes the target through symlog,
decode is symexp of the bin expectation. Lambda-returns:
$R_t = r_t + \gamma c_t((1-\lambda)V_{t+1} + \lambda R_{t+1})$, $R_H = V_H$. Continuous actor:
Tanh-Normal $a = \tanh(\mu + \sigma\varepsilon)$, log-std floor $\log(0.1)$, trained by DYNAMICS
BACKPROP - the imagined return is differentiable through the world model, loss
$-\mathbb{E}[R]/\max(1, S) - \eta\mathcal{H}$ with no log-prob. Shapes: obs $(B, T, 3, 64, 64)$, $h$
$(B, h\_dim)$, $z$ $(B, n\_cat\cdot n\_cls)$, action $(B, 1)$ float, returns $(B, H)$, values
$(B, H+1)$.

## what_you_implement

- `symlog` / `symexp`, `twohot_encode` / `twohot_decode`, `categorical_sample` (straight-through + unimix).
- `RSSMCell.forward_h` / `prior` / `posterior` (forward_h handles continuous $(B, 1)$ float actions).
- `kl_loss` (two weighted terms, free bits on summed KL) and `WorldModel.loss` (ELBO assembly).
- `compute_lambda_returns`, `critic_loss` (two-hot), `imagine_dynamics` (differentiable rollout),
  `actor_loss_dynbackprop` (dynamics-backprop actor loss).

## tasks

- **Task 1 - symlog/symexp** (file: `nets.py`, symbols: `symlog`, `symexp`): $\mathrm{sign}(x)\ln(|x|+1)$
  and its exact inverse $\mathrm{sign}(x)(e^{|x|}-1)$.
- **Task 2 - two-hot** (file: `nets.py`, symbols: `twohot_encode`, `twohot_decode`): encode pushes
  $y$ through symlog and splits over the two bracketing bins of `linspace(-20,20,255)` (weights
  linear in distance, sum to 1); decode is $\mathrm{symexp}(\sum_k p_k b_k)$. Exact round-trip.
- **Task 3 - categorical_sample** (file: `nets.py`, symbol: `categorical_sample`): reshape to
  $(B, n\_cat, n\_cls)$, blend $p=(1-u)\,\mathrm{softmax}+u/n\_cls$, draw one-hot (argmax if greedy,
  else multinomial), return $z=(\text{onehot}-p).\mathrm{detach}()+p$ flattened, plus $p$.
- **Task 4 - RSSM cell** (file: `rssm.py`, symbols: `forward_h`, `prior`, `posterior`): GRU step on
  `in_proj([z, a])` (one-hot $a$ if integer, pass $(B, action\_dim)$ float through); prior MLP on
  $h$; posterior MLP on $[h, \text{embed}]$; each head calls `categorical_sample`.
- **Task 5 - kl_loss** (file: `world_model.py`, symbol: `kl_loss`): per-head categorical KL summed
  over heads (use the provided `_categorical_kl`); $\mathcal{L}_{\text{dyn}}=\max(fb, \mathrm{KL}[\mathrm{sg}(q)\|p])$,
  $\mathcal{L}_{\text{rep}}=\max(fb, \mathrm{KL}[q\|\mathrm{sg}(p)])$, clipped on the summed scalar
  then meaned; return $0.5 L_{\text{dyn}}+0.1 L_{\text{rep}}$ and a parts dict.
- **Task 6 - ELBO** (file: `world_model.py`, symbol: `WorldModel.loss`): `encode_seq` -> `observe`
  -> decode states (MSE vs `symlog(obs)`), reward head (two-hot CE), cont head (BCE), plus `kl_loss`.
- **Task 7 - lambda-returns** (file: `actor_critic.py`, symbol: `compute_lambda_returns`): backward
  recursion bootstrapping on `values[:, t+1]`, $R_H = \text{values}[:, H]$; $c_t$ zeros bootstrap
  and tail. rewards/conts $(B, H)$, values $(B, H+1)$.
- **Task 8 - critic loss** (file: `actor_critic.py`, symbol: `critic_loss`):
  `twohot_loss(logits, returns.detach(), bins).mean()`.
- **Task 9 - imagine_dynamics** (file: `actor_critic.py`, symbol: `imagine_dynamics`): roll the
  prior `cfg.horizon` steps with `actor.sample` in the loop (reparameterized action), step
  `forward_h`, sample `prior`; collect pre- and post-action $(h, z)$; reward/cont of $a_t$ from the
  POST-action state `states[:, 1:]`; values over all `horizon+1` states; differentiable
  `compute_lambda_returns`. NO `no_grad`. Return `returns (B, H)`, `entropies (B, H)`, `H_h`, `H_z`.
- **Task 10 - actor_loss_dynbackprop** (file: `actor_critic.py`, symbol: `actor_loss_dynbackprop`):
  $-(\text{returns}/\text{ret}_{\text{range}}).\mathrm{mean}() - \eta\,\text{entropies}.\mathrm{mean}()$. The
  returns are differentiable through the dynamics - no log-prob, no detach.

## tests

- `tests/test_symlog_twohot.py` - reference-value: symexp(symlog)=id; two-hot round-trip; label
  two-nonzero-summing-to-1; loss minimized at matching logits; bins symlog-spaced. (Task 1, 2)
- `tests/test_straight_through.py` - reference-value: one-hot forward; ST grad == blended-prob grad;
  unimix floor. (Task 3)
- `tests/test_shapes.py` - shape: cell / observe / encoder-decoder / ContActor sample. (Task 4)
- `tests/test_gradcheck.py` - gradcheck (float64, greedy): forward_h+prior, categorical KL,
  lambda-returns. (Task 4, 5, 7)
- `tests/test_lambda_returns.py` - reference-value: hand 3-step return; $c=0$ termination; random
  vs reference. (Task 7)
- `tests/test_kl_balancing.py` - reference-value: free-bits clamp + zero grad below floor; 1 nat on
  summed (not n_cat); doubling a beta doubles only that side's gradient. (Task 5)
- `tests/test_cont_actor.py` - reparameterized: action in $(-1,1)$; nonzero action grad to actor
  params; log-std floor $\log(0.1)$. (provided ContActor; guards the actor contract)
- `tests/test_imagine_differentiable.py` - gradient-exists: the imagined return has nonzero gradient
  w.r.t. the actor params THROUGH the dynamics. Guards the headline mechanism. (Task 9, 10)
- `tests/test_overfit_world_model.py` - overfit-one-batch (synthetic, no dm_control): recon < 0.05
  in 400 steps; each KL term in a 0.8-6 nat band. (Task 6)
- `tests/test_imagination.py` - relative (synthetic, no dm_control): prior rollout finite +
  decoder-free; next state depends on the action. (Task 4)
- `tests/test_env_smoke.py` - dm_control wrapper resets/steps/renders $(3,64,64)$; `importorskip`
  skips when dm_control is unavailable.
- `tests/test_forbidden_imports.py` - static scan: no RL library, no cross-assignment import.

Run order: symlog_twohot -> straight_through -> shapes -> gradcheck -> lambda_returns ->
kl_balancing -> cont_actor -> imagine_differentiable -> overfit_world_model -> imagination.

## provided_boilerplate

`env.py` (dm_control cartpole wrapper, 64x64 render, recurrent `run_episode`, `random_return`),
`config.py` (DreamerV3 dims + symlog-space bins), `nets.py` Encoder/Decoder (64x64, 4-conv) and
`twohot_loss`, `rssm.py` `initial_state`/`observe`/`imagine` (the old discrete imagine kept for
reference), `world_model.py` `__init__`/`encode_seq`/`encode_start` and `_categorical_kl`,
`actor_critic.py` `ContActor`/`Critic`/`ReturnNormalizer` and the discrete `Actor`/`actor_loss`
(REINFORCE contrast), all of `_train.py` (the collect-fit-imagine loop, critic warmup, recurrent
eval, the CPU `smoke_train`), and `viz.py`.

## compute_notes

Mechanism tests: CPU, seconds; gradcheck/overfit use shrunk configs; no GPU, no dm_control needed.
The env smoke test needs dm_control (skips otherwise). The full collect-fit-imagine run needs a CUDA
GPU and a MuJoCo GL backend (`MUJOCO_GL=egl`): ~400 iterations, ~1-2 hours on a 12GB GPU, not run in
CI. Healthy world-model curve: recon ~0.001, KL terms near the 1-nat free-bits floor; a KL running
to tens of nats or collapsing to 0 signals a sign error.

## stretch_goals

1. Multinomial training sampler (ST gradient test still passes).
2. Gaussian latent (DreamerV1) vs categorical, compare reconstruction.
3. Re-add the discrete REINFORCE actor (discretize the force) and reproduce the measured collapse.
4. Lengthen the horizon and watch the early imagined-return inflation settle as the critic learns.

## solution_notes

- The actor is CONTINUOUS (Tanh-Normal, action_dim=1) trained by DYNAMICS BACKPROP, not REINFORCE.
  This is the headline change and the reason transfer works. `forward_h` already passes a continuous
  $(B, 1)$ float through unchanged and one-hots only an integer action (`a.dim() == 1`); buffer
  actions are stored as floats.
- The imagined return must stay differentiable: NO `no_grad` around the reward / value / cont decode
  in `imagine_dynamics`. The reward of $a_t$ is read from the POST-action state $s_{t+1}$
  (`states[:, 1:]`); reading it from $s_t$ breaks credit assignment (measured).
- Two-hot bins are `linspace(-20, 20, 255)` in SYMLOG space (encode symlog, decode symexp).
  Value-space bins blow up the imagined-reward decode (outer buckets at ~5e8); the symlog-space
  convention keeps decoded rewards $O(1)$ and the round-trip exact.
- KL weights are 0.5 / 0.1 (DreamerV3), not 0.8 / 0.2 (DreamerV2). Free bits = 1 nat clamps the
  summed-over-heads KL per term, not per head. The KL gradcheck checks the raw `_categorical_kl`
  (both args live), since finite-diff gradcheck ignores `.detach()`.
- Config is the verified DreamerV3-aligned set: `h_dim=512, n_cat=n_cls=32, embed_dim=1024,
  horizon=8, gamma=0.997, lam=0.95, ent_coef=1e-4, free_bits=1.0, kl_dyn=0.5, kl_rep=0.1,
  unimix=0.01, n_bins=255, bin +/-20, ret_ema=0.99`; LRs model=1e-4, actor=critic=4e-5; batch 16 x
  seq 50, ~100 updates/episode, critic warmup 5 iters, action_repeat 2. Measured to work; do not
  retune.
- Measured transfer (single 12GB GPU, ~1-2 h): greedy real return >300 (best ~350-375 across seeds) vs random ~214
  vs optimal ~500; discrete REINFORCE collapses to ~135 (below random) on this near-flat reward.
  Reported in viz/README with stats; never pinned as a pytest threshold. The toy reaches "clearly
  beats random," not "optimal" - a scale/compute artifact, not the mechanism failing.
