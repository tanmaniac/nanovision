# A12 - world models (RSSM / DreamerV3): build plan

Status: plan for expert review, then delegated build. Leaf assignment (no `nanovision` shim;
see "Shared-library decision"). Build target is DreamerV3 (Hafner et al., Nature 2025;
arXiv:2301.04104). Research note: `docs/research/a12_world_models.md`.

## 1. What the student builds and why

The student implements the buildable core of a latent-control world model: an RSSM (recurrent
state-space model) trained as a sequence VAE, then an actor-critic trained entirely on
trajectories imagined inside that model. The pedagogy is the four-part DreamerV3 loop:

1. RSSM cell - a deterministic GRU recurrence `h_t` plus a stochastic categorical latent `z_t`,
   with a posterior (sees the observation, used in training) and a prior (no observation, used in
   imagination).
2. World-model ELBO - reconstruction + reward + continuation prediction, minus a KL that trains
   the prior to match the posterior, with free bits and KL balancing.
3. Imagination - roll the dynamics forward using only the prior, no decoder, no environment.
4. Actor-critic in imagination - lambda-returns, a two-hot critic, a REINFORCE actor with an
   entropy bonus and percentile return normalization.

The DreamerV3-specific mechanisms that make one fixed config work across reward scales are
first-class holes, not footnotes: categorical latents with the straight-through estimator and
unimix; symlog and two-hot target encoding; KL balancing and free bits separately.

The leaf goal is not peak performance. It is a working mechanism: the world model overfits a
short rollout to near-zero reconstruction, imagination decodes to coherent grid images, and an
actor trained only in imagination beats a random policy on the toy gridworld.

### Shared-library decision

A12 is a leaf. The only downstream dependant is A13 (VLA capstone), and the build checklist marks
that dependency parenthetical (`deps: A5/A6,A8,(A12)`). A13 is not built yet, and nothing in the
shared `nanovision.*` contract needs the RSSM today. So A12 owns its modules locally with no
`nanovision/<mod>.py` shim, exactly like A11.5e. If A13's plan turns out to import RSSM
components, add the shim during A13. Flag for the expert: confirm this leaf decision is right, or
name the specific symbols A13 will need so we expose them now.

## 2. The toy environment (provided, not a hole)

`env.py` - a deterministic gridworld in pure NumPy, no external RL dependency.

- Grid: 6x6 cells. A fixed interior wall splits it into two rooms connected by a one-cell doorway,
  so reaching the goal requires routing through the door (the dynamics are not trivially "walk
  diagonally"). Start cell fixed at `(0,0)`, goal cell fixed at `(5,5)` in the far room.
- Actions: 4 discrete (up, down, left, right). Moving into a wall or boundary is a no-op (agent
  stays).
- Reward: `+1` on the step that enters the goal, `0` otherwise. Episode ends on goal reach or at
  50 steps. Continuation flag `c_t = 1 - done`.
- Observation: an RGB render of the grid state at `obs_size x obs_size` (default 16x16, 3
  channels), agent drawn as one color block, goal another, walls a third, floor background. The
  render is a fixed deterministic rasterization (each cell is a `obs_size/6` block). 16x16 keeps
  CPU encoder/decoder cheap; viz can re-render larger for display.
- API: `reset(seed) -> obs`, `step(action) -> (obs, reward, done, info)`, plus
  `render(state) -> obs` and `optimal_action(state)` (a BFS shortest-path oracle used only to
  generate the offline replay buffer and to compute the success ceiling - never called by the
  model). Deterministic transitions, so a fixed seed gives a fixed trajectory.
- `collect_random_episodes(env, n, seed)` and `collect_episodes(env, policy, n, seed)` return a
  replay buffer of `(obs, action, reward, cont)` sequences for training.

Why this environment: sparse reward + a doorway bottleneck makes the world model actually have to
predict a multi-step consequence (you do not get reward by reacting to the current frame), and the
fixed layout means imagination has a single correct geometry to reproduce, which is checkable by
eye in viz and by reconstruction error in tests.

## 3. Files and holes

Mirror `assignments/a11_5e_pred_planning` for structure (`conftest.py`, top-level holed files,
`solution/<file>.py` per holed file, `config.py`/`viz.py` top-level only, `tests/`, `README.md`,
`ASSIGNMENT.md`, `__init__.py`s). Exemplar to read cold: `assignments/a11_5e_pred_planning` and,
for the VAE/ELBO framing, `assignments/a05_diffusion` is a useful cross-reference.

Provided (no hole, top-level only): `env.py`, `config.py`, `viz.py`, the CNN encoder/decoder
module bodies (the conv plumbing is not the lesson), and the training-loop scaffolding in
`_train.py` (data iteration, optimizer, logging) that calls into the holed loss functions.

Holed files (top-level hole + `solution/` answer key):

### `nets.py` - scalar-target encodings and the straight-through sampler

- `symlog(x)` / `symexp(x)`: holes. `symlog(x) = sign(x) * log(|x| + 1)`,
  `symexp(x) = sign(x) * (exp(|x|) - 1)`. Exact inverses.
- `twohot_encode(y, bins)` -> soft label over `len(bins)` buckets; `twohot_decode(probs, bins)` ->
  expected value `sum probs * bins`. Holes. The bins are in VALUE space, not symlog space:
  `bins = symexp(linspace(bin_lo, bin_hi, n_bins))` (DreamerV3 eq. 10 builds the 255 bin POSITIONS
  by pushing a linear `[-20, 20]` grid through symexp once, so the buckets are exponentially spaced
  in value space). This is provided in `config.py` as the `bins` vector. `twohot_encode`/
  `twohot_decode` operate directly on the raw target over these value-space bins, so they are exact
  inverses with NO extra symlog/symexp at encode or decode (this avoids the
  `symexp(sum p*s) != sum p*symexp(s)` non-commutation bug). For target `y` between adjacent bins
  `b_i <= y <= b_{i+1}`, weight `(b_{i+1}-y)/(b_{i+1}-b_i)` on `i`, complement on `i+1`; clamp
  outside the range to the end bucket. `twohot_decode(probs, bins) = sum(probs * bins)`.
- `twohot_loss(logits, target, bins)`: cross-entropy of `log_softmax(logits)` against
  `twohot_encode(target, bins)`. Helper used by reward/critic/continuation heads. Provided wrapper
  that calls the holed `twohot_encode`. (Named without the `symexp_` prefix since the symexp now
  lives in the bin construction, not the loss.)
- `categorical_sample(logits, unimix, n_cat, n_cls)`: hole. Reshape logits to
  `(B, n_cat, n_cls)`, softmax, blend with uniform: `probs = (1-unimix)*softmax + unimix/n_cls`
  (unimix default 0.01). Draw a one-hot sample by argmax (greedy is fine for the deterministic
  toy; the plan uses argmax + unimix for reproducible tests, not multinomial - see note). Return
  the straight-through sample `z = (onehot - probs).detach() + probs`, flattened to
  `(B, n_cat*n_cls)`, plus `probs` for the KL.
  - Note for the expert: DreamerV3 samples multinomially from the categorical. For a deterministic,
    gradcheck-able, seed-stable test suite I default the *training* sampler to a reparameterized
    multinomial via `torch.multinomial` seeded by the global generator (so it is stochastic but
    reproducible under a fixed seed), and expose a `greedy=True` path used by the shape/gradcheck
    tests. The straight-through estimator is identical either way. Confirm this does not distort
    the lesson.

### `rssm.py` - the RSSM cell

- `class RSSMCell(nn.Module)` with `h_dim`, `n_cat`, `n_cls`, `action_dim`, `embed_dim`.
  Uses only `nn.GRUCell`, `nn.Linear`, `F.one_hot`, and `nets.categorical_sample`.
- `forward_h(self, h, z, a) -> h_new`: hole. Concatenate `z` (flattened categorical) and the
  one-hot action, project to GRU input, step `nn.GRUCell`. `h_new = GRU(cat([z,a]) -> in, h)`.
- `prior(self, h) -> (logits, z, probs)`: hole. MLP `h -> logits (n_cat*n_cls)`, then
  `categorical_sample`. The transition prior `p(z_t | h_t)`.
- `posterior(self, h, embed) -> (logits, z, probs)`: hole. MLP `cat([h, embed]) -> logits`, then
  `categorical_sample`. The posterior `q(z_t | h_t, o_t)`.
- `initial_state(B)`: zeros for `h` and `z`. Provided.
- `observe(self, embeds, actions, h0, z0)`: provided loop that, per step, calls `forward_h` then
  `posterior`, returning the sequences `h_{1..T}, z_{1..T}, prior_logits, post_logits`. (Provided
  because it is just the unrolled loop; the per-step holes are the lesson.)
- `imagine(self, policy, h0, z0, horizon)`: provided loop that, per step, samples an action from
  `policy(h,z)`, calls `forward_h` then `prior` (no observation), returns the imagined
  `h, z, actions, action_logprobs, entropies`.

### `world_model.py` - the ELBO

- `class WorldModel(nn.Module)`: holds encoder, decoder, `RSSMCell`, and reward/continuation
  heads (two-hot reward, Bernoulli-logit continuation). `__init__` provided.
- `kl_loss(post_logits, prior_logits, free_bits, beta_dyn, beta_rep)`: hole. This is the exact
  DreamerV3 form (verified against arXiv:2301.04104 eq. 2-3, not the DreamerV2 0.8/0.2 balance).
  Two separately-weighted KL terms, each with `sg` = stop-gradient (`.detach()`):
  - dynamics loss `L_dyn = max(free_bits, KL[sg(q) || p])` - trains the prior `p` toward the
    posterior;
  - representation loss `L_rep = max(free_bits, KL[q || sg(p)])` - trains the posterior `q` toward
    the prior.
  The categorical `KL[.||.]` is summed over the `n_cat` heads FIRST (the factorized posterior
  makes the joint KL the sum of per-head KLs), THEN the `max(free_bits, .)` free-bits clip is
  applied to that single summed scalar per term (free_bits = 1 nat; clip on the total, NOT
  per-head). Return `beta_dyn * L_dyn + beta_rep * L_rep` with `beta_dyn = 0.5`, `beta_rep = 0.1`.
  The 5:1 weight ratio is what makes the prior move faster than the posterior.
- `loss(self, batch) -> (total, parts)`: hole (the headline assembly). Encode the obs sequence,
  `observe` to get states + prior/post logits, decode `(h,z)` to obs reconstruction, predict
  reward and continuation. Total = `recon + reward_ce + cont_bce + kl_loss(...)`, where `kl_loss`
  already carries the `0.5`/`0.1` weights so there is no separate beta to tune (free bits +
  per-term weighting remove it). DreamerV3's `beta_pred = 1.0` is the implicit weight on the
  reconstruction/reward/continuation block. Reconstruction is MSE on symlog targets:
  `mse(decode, symlog(obs))`, with a symexp at viz time. Return the total and a dict of parts for
  logging.

### `actor_critic.py` - behavior learning in imagination

- `compute_lambda_returns(rewards, values, continues, gamma, lam)`: hole. The DreamerV3
  lambda-return backward recursion (verified against arXiv:2301.04104 eq. 5 - it bootstraps on the
  NEXT-state value `V_{t+1}`, not `V_t`):
  `R_t = r_t + gamma*c_t * ((1-lam)*V_{t+1} + lam*R_{t+1})`, with `R_H = V_H`. The `c_t` (=0 at an
  episode end) zeros both the bootstrap and the recursion tail at termination. Pure tensor op,
  exact-testable against a hand-computed 3-step example.
- `class Actor(nn.Module)`: MLP `(h,z) -> 4 action logits`; `forward` returns a `Categorical`.
  Body provided; the loss is the hole.
- `class Critic(nn.Module)`: MLP `(h,z) -> 255 two-hot logits`; value = `twohot_decode(softmax,
  bins)` (the bins are already in value space, so no symexp at decode). Body provided; the loss is
  the hole.
- `actor_loss(logprobs, returns, values, entropies, ent_coef, ret_range)`: hole. REINFORCE on the
  normalized advantage: `adv = (returns - values).detach() / ret_range`; loss =
  `-(logprobs * adv).mean() - ent_coef * entropies.mean()`. (DreamerV3 uses REINFORCE for both
  discrete AND continuous actions; the discrete toy here matches the V3 path exactly.) `ret_range`
  is `max(1.0, S)` where `S` is an exponentially-moving-average (decay 0.99) of the per-batch
  5th-95th-percentile spread of returns, per DreamerV3 eq. 6-7 - the EMA, not the instantaneous
  batch spread. `S` is tracked in `_train.py` via a provided helper (`return_normalizer`) and
  passed in; the actor only divides by `max(1, S)`.
- `critic_loss(logits, returns, bins)`: hole. `twohot_loss(logits, returns.detach(), bins)`. The
  EMA in DreamerV3 is the critic regularizing toward an exponentially-moving-average copy of its
  OWN weights (not a separate DQN-style frozen target net): the lambda-returns are computed from
  the CURRENT critic's values, and `_train.py` adds the small regularization term pulling the
  current critic's outputs toward the slow copy. The student writes the two-hot return regression;
  the EMA bookkeeping is provided in `_train.py`.

### `config.py` (provided)

Tiny dims so CPU tests run in seconds and viz fits the RTX 4080 easily:
`obs_size=16`, `obs_ch=3`, `h_dim=128`, `n_cat=16`, `n_cls=16` (256 latent dims, reduced from
DreamerV3's 32x32=1024), `embed_dim=128`, `action_dim=4`, `enc_ch=(16,32)`, `horizon=15`,
`gamma=0.997`, `lam=0.95`, `free_bits=1.0`, `kl_dyn_scale=0.5`, `kl_rep_scale=0.1` (DreamerV3's
beta_dyn/beta_rep), `unimix=0.01`, `ent_coef=3e-4`, `ret_ema_decay=0.99`, `n_bins=255`,
`bin_lo=-20`, `bin_hi=20`. Provide the value-space two-hot bins as
`bins = symexp(linspace(bin_lo, bin_hi, n_bins))` (a cached tensor on the config or built in
`nets.py` from these fields). A `gradcheck` sub-config shrinks to
`h_dim=8, n_cat=4, n_cls=4, obs_size=8` for float64 speed. State the production DreamerV3 sizing
(h=512..4096, 32x32 latent) in a comment so the toy reads as a mechanism isolator.

## 4. Tests (CPU, seconds; both modes)

Default mode fails cleanly at the holes (NotImplementedError); `NANOVISION_IMPL=solution` green.
Run with `/home/tanmay/miniconda3/envs/nanovision/bin/python -m pytest`. Prefer exact analytic
oracles over training-convergence asserts.

1. `test_symlog_twohot.py` (training-free, exact): `symexp(symlog(x)) == x` to 1e-6 over a spread
   of magnitudes including large values; with value-space bins `symexp(linspace(-20,20,255))`,
   `twohot_decode(twohot_encode(y)) == y` to 1e-5 for raw `y` inside the bin range (exact inverse,
   no extra symlog/symexp); two-hot label has exactly two nonzero entries summing to 1; the
   `twohot_loss` minimum is at `logits` matching the two-hot target.
2. `test_straight_through.py` (training-free): forward sample is one-hot per categorical (argmax
   matches, rows sum to 1); the straight-through output's gradient w.r.t. logits equals the
   softmax-probability gradient (autograd check: `z.sum().backward()` gives the same grad as
   `probs.sum().backward()` up to the unimix scaling). Unimix floor: every probability `>=
   unimix/n_cls`.
3. `test_shapes.py`: `prior`/`posterior`/`forward_h` output shapes for a batch; `observe` and
   `imagine` sequence shapes; encoder/decoder round-trip shape.
4. `test_gradcheck.py` (float64, gradcheck sub-config): single-step `forward_h` + `prior` is
   differentiable w.r.t. inputs and parameters; `kl_loss` and `compute_lambda_returns` pass
   `torch.autograd.gradcheck`. Use the greedy categorical path so the graph is deterministic.
5. `test_lambda_returns.py` (training-free, exact): hand-built 3-step `rewards/values/continues`
   with a known closed-form `R_t`; assert equality to 1e-6. Include a `cont=0` (episode end) case
   that must zero the bootstrap.
6. `test_kl_balancing.py` (training-free): free bits clamp - when the summed `KL < free_bits`, each
   loss term equals `free_bits` (= 1 nat for the whole summed term, NOT `n_cat * free_bits`) and
   has zero gradient to the logits; when summed `KL > free_bits`, the gradient is nonzero. Weighted
   terms - on a controlled input where both terms are above the free-bits floor, the gradient
   magnitude onto the prior logits (from `beta_dyn * L_dyn`) is `beta_dyn/beta_rep = 0.5/0.1 = 5`
   times that onto the posterior logits (from `beta_rep * L_rep`); assert the prior moves ~5x more
   than the posterior. (Do not assert 4x or 10x - the V3 weights are 0.5/0.1.)
7. `test_overfit_world_model.py` (bounded training, robust): overfit one fixed 12-step rollout;
   reconstruction MSE drops below a pre-measured floor (target < 0.02, measure and set) within
   <=600 steps, and each summed KL term settles near its 1-nat free-bits floor (a loose band such
   as 0.8-4 nats - measure and set), NOT collapsed to 0 and not diverging. Assert the recon drop
   and a loose KL band only; do not pin an exact KL.
8. `test_imagination.py`: from a real encoded start state, `imagine` for `horizon` steps using the
   prior produces finite states, calls no decoder, and (with the overfit model from a fixture or a
   short train) the decoded imagined first step matches the true next obs better than a
   shuffled-action baseline. Keep this a relative check, not an absolute threshold.
9. `test_forbidden_imports.py`: one static tokenize scan over top-level + solution holed files;
   forbid any RL/world-model library (`gym`, `gymnasium`, `dreamerv3`, `stable_baselines3`,
   `tianshou`) and any bare cross-assignment import. Passes in both modes.

Explicitly NOT a unit test (per the no-fragile-test gate): the full "actor trained in imagination
solves the gridworld >= 30%" claim. Policy transfer depends on a multi-thousand-step training run
and is init-sensitive; pinning a success-rate threshold into pytest would be exactly the fragile,
seed-dependent assertion the build-workflow gate forbids. Instead, `viz.py` runs the full loop on
GPU and the README reports the measured success rate over N seeds against the random-policy
baseline (~1/50 = 2%) and the BFS-oracle ceiling. The unit suite asserts only the robust,
mechanism-level truths above (exact encodings, exact returns, KL behavior, reconstruction
overfit). The critic/actor losses are covered by gradcheck and by a short "actor loss decreases on
a fixed imagined batch" smoke check that is relative, not thresholded.

## 5. viz.py (GPU when present)

`from nanovision.determinism import default_device`. Panels, each saved to `out/`:
1. Replay vs reconstruction: a real rollout's frames over the decoder's symexp reconstruction.
2. Imagination filmstrip: decode a prior-only rollout from a fixed start under the learned actor;
   the agent should be seen routing through the doorway toward the goal.
3. KL and reconstruction curves over world-model training (the KL settling at the free-bits floor).
4. Policy transfer: success rate of the imagination-trained actor in the real env vs random vs the
   BFS oracle ceiling, as a bar chart with the per-seed spread. This is where the headline RL
   result lives (measured, honestly framed), not in a unit test.

## 6. README (lecture notes, per the skill)

Cover, in the research note's order: why both deterministic `h` and stochastic `z` (PlaNet's
argument), the sequence-VAE ELBO and why the KL trains the prior for imagination, the two
separately-weighted KL terms (dynamics loss `beta_dyn=0.5` for prior accuracy, representation loss
`beta_rep=0.1` for posterior non-collapse) plus free bits (1 nat on the summed KL), categorical
latents + straight-through + unimix, symlog + two-hot as the scale-invariance mechanism (elevated,
not a footnote), imagination as prior-mode rollout, and actor-critic in imagination (lambda-returns
bootstrapping on `V_{t+1}`, two-hot critic with self-EMA regularization, percentile return
normalization). Note that DreamerV3 unified the actor gradient on REINFORCE for BOTH discrete and
continuous actions (DreamerV1/V2 used dynamics-backprop for continuous); do not state that V3 uses
dynamics-backprop for continuous. Real LaTeX throughout.

Landscape and forward pointers (reading-only, do not build): the pixel-space generative branch
(GAIA, Genie / Genie 2, DIAMOND, DreamerV4), JEPA-style prediction-in-latent-space as the
reconstruction-free contrast, and the Sora "world simulator" debate (plausible video is not a
queryable causal model). Name DreamerV4 as where the RSSM lineage goes (transformer + flow-matching
dynamics). Verify every arXiv id by fetching `https://arxiv.org/abs/<id>`:
- PlaNet 1811.04551, DreamerV1 1912.01603, DreamerV2 2010.02193, DreamerV3 2301.04104 (Nature
  2025, exact title "Mastering diverse domains through world models" - NOT "control tasks";
  confirm the Nature title string when citing), DIAMOND 2405.12399, DreamerV4 2509.24527, V-JEPA
  (verify id during build).

Keep the at-scale caveat: this 6x6 / 16x16 toy is a mechanism isolator. The toy's small latent
(16x16 vs 32x32), short training, and single fixed layout mean its numbers demonstrate the loop
works, not that they predict DreamerV3's at-scale behavior. Do not let any toy number read as
contradicting the paper.

Mandatory final step: the context-less README style-review subagent (skill step), applied before
commit.

## 7. Risks and pre-measured expectations

- The overfit reconstruction floor and KL band must be MEASURED on the solution and the test
  thresholds set from the measurement, then floored - do not thrash (build guide rule). Report the
  numbers.
- Straight-through gradient equality: the unimix blend means the backward grad is through
  `(1-unimix)*softmax + unimix/n_cls`, so the test compares against that blended prob, not raw
  softmax. State this in the test so a correct implementation is not failed.
- gradcheck must use the greedy categorical path and float64; multinomial sampling breaks gradcheck
  determinism. Confirm the sampler exposes `greedy=True`.
- Policy transfer is deliberately out of the asserted suite. If the build agent is tempted to add a
  success-rate assertion, it must not - route it to the README/viz with per-seed stats.
- Two-hot bins are `symexp(linspace(-20, 20, 255))`, so the outer buckets sit at
  `+/- symexp(20) ~ 4.85e8`; the toy's returns are `O(1)`, well inside, so no clipping artifacts.
  Keep the wide range anyway (it is the DreamerV3 default and the lesson is scale invariance).

## 8. Build-workflow gate checklist (apply in expert review and on-disk verify)

- Build target is the 2025-consensus DreamerV3, not a deprecated variant. Deprecated/contrasting
  approaches (Gaussian latents, scalar-MSE reward, pixel-space generation) appear only as labeled
  contrast in the README, never as the implemented endpoint.
- No test asserts a cherry-picked seed or a fragile training threshold. The one outcome-level claim
  (policy transfer) is in the README/viz with per-seed statistics, not a pinned assertion. Every
  pytest assertion is an exact analytic oracle or a robust, implementation-independent inequality.
- The README keeps the at-scale caveat and never frames a toy number as overriding the paper.
