# assignments/a11_5e_pred_planning/ASSIGNMENT.md

```yaml
id: a11_5e_pred_planning
title: Unified perception -> prediction -> planning
module: 3.5
type: Core
estimated_learner_hours: 6
depends_on: [a00_harness, a01_transformer, a11_5a_camera_geometry_bev, a11_5b_lift_splat_shoot, a11_5c_bevformer, a11_5d_occupancy]
builds_into_shared_lib: []   # assignment-local: nothing imports predict.py, so no nanovision shim
forbidden_imports:
  - nn.MultiheadAttention      # attention must be the A1 MultiHeadAttention from nanovision.attention
  - nn.Transformer             # and the Transformer* convenience modules
  - F.scaled_dot_product_attention
  # F.grid_sample IS ALLOWED (the bilinear-sampling substrate for roi_align_bev).
fits_12gb: true
external_data: none (tests run on the synthetic pred_toy_scene)
```

## motivation

The future of an agent is multimodal: the same observed state is consistent with several distinct
intentions (turn left, go straight, turn right). A single trajectory regressor trained on mean
error is pulled to the conditional mean of those futures, a path through the middle of every
option that matches none of them. This is mode averaging. The fix is to predict K trajectory
hypotheses and supervise only the closest one per sample, the min-of-N (winner-take-all) loss.
You build one stage of the end-to-end driving stack: a multimodal motion-prediction head that
maps BEV features at an agent's location to K future trajectories. The full treatment, the
end-to-end stack (UniAD -> VAD -> DriveTransformer), and the open-loop-metric critique are in the
README.

## background

A dense BEV feature grid `(C, nx, ny)` is pooled around each agent by a small RoI-align
(`F.grid_sample`) into a fixed-length token set. K learned mode queries cross-attend over those
tokens through a few decoder layers (built from the A1 `MultiHeadAttention`), and each mode
regresses per-step displacements that integrate (cumsum) to an absolute agent-centric trajectory
plus a confidence score. Training uses winner-take-all: the closest mode by endpoint distance
(minFDE) is the winner, only it gets regression gradient (hard path), and the score head is
trained by cross-entropy to point at the winner. Hard WTA can leave spare modes that never win
without gradient (dead-mode collapse); a soft path with weights `softmax(-FDE/temperature)` gives
every mode gradient, and annealing the temperature to 0 recovers the hard loss after every mode
has spread out.

Shapes: `bev_feat (C, nx, ny)`; `centers (N, 2)` fractional cell coords; RoI tokens
`(N, roi_size**2, C)`; trajectories `(B, K, T, 2)` with `B = N` agents; scores `(B, K)`; ground
truth `(B, T, 2)` agent-centric.

## what_you_implement

- `roi_align_bev`: bilinear RoI-align of a shared BEV grid around N agent centers.
- `MultimodalTrajectoryHead`: mode-query decoder -> K trajectories + K scores.
- `wta_loss`: the hard min-of-N and the soft/annealed regression paths plus the classification
  term.
- `min_ade`, `min_fde`, `miss_rate`: best-of-K oracle metrics.

## tasks

1. **Task 1 - roi_align_bev** (file: `predict.py`, symbol: `roi_align_bev`): build an
   `out_size x out_size` grid of sample points in CELL units (`linspace(-radius, +radius,
   out_size)`) around each center, normalize per axis with `g = 2*(cell+0.5)/S - 1`
   (`align_corners=False`), SWAP to `(g_w=ny, g_h=nx)` for `grid_sample`, sample the N-expanded
   `bev_feat` bilinear with `padding_mode="border"`, return `(N, out_size**2, C)`. Teaches the
   per-agent RoI pooling and the `grid_sample` axis convention.

2. **Task 2 - MultimodalTrajectoryHead.forward** (file: `predict.py`, symbol:
   `MultimodalTrajectoryHead.forward`): RoI-align -> project to `dim` -> expand the DISTINCT mode
   queries over `B = N` agents -> `n_layers` of (mode self-attn, cross-attn over RoI tokens, MLP,
   residual + pre-norm) -> per-mode `Linear(dim -> horizon*2)` reshaped and cumsum-ed to absolute
   positions `(B, K, T, 2)`; per-mode score logits `(B, K)`. Teaches the mode-query decoder.

3. **Task 3 - wta_loss** (file: `predict.py`, symbol: `wta_loss`): winner = detached
   `argmin_k ||traj_k[-1] - gt[-1]||`. Hard (`temperature is None`): MSE of the gathered winner
   trajectory. Soft (`temperature` a float): `reg = sum_k softmax(-FDE/temperature)_k * MSE_k`.
   Classification = `cross_entropy(raw scores, winner)` in both. `total = reg + cls_weight*cls`.
   Teaches min-of-N and the soft/annealed fix for dead modes.

4. **Task 4 - metrics** (file: `predict.py`, symbols: `min_ade`, `min_fde`, `miss_rate`):
   best-of-K by minFDE; `min_ade` is the mean-over-time L2 of that mode, `min_fde` the endpoint
   L2, `miss_rate` the fraction whose best endpoint error exceeds `thresh`. Teaches the
   Argoverse/nuScenes-style oracle metrics.

## tests

- `tests/test_shapes.py` - `roi_align_bev -> (N, roi_size**2, C)`; head `-> (B, K, T, 2)` and
  `(B, K)`.
- `tests/test_gradcheck.py` - float64 gradcheck on `roi_align_bev` at fractional interior centers,
  and on both `wta_loss` paths (a hard case with a strict unique winner, and the soft path).
- `tests/test_overfit.py` - `shared_context=False`; distinct inputs drive `min_ade` below 0.1 m.
- `tests/test_wta_beats_single_mode.py` - `shared_context=True`; train K=1, K=6 hard, K=6 annealed
  and assert single > hard > annealed `min_ade` (measured 1.168 / 0.554 / 0.048 m at seed 2).
- `tests/test_modes_specialize.py` - after annealed K=6, `>= n_intentions` modes win with
  mutually separated endpoints (measured 3 modes, separation > 4.3 m).
- `tests/test_forbidden_imports.py` - tokenize scan bans the three high-level attention APIs;
  positive check that attention comes from `nanovision.attention`.

## provided_boilerplate

`config.py` (`PredConfig` with the tiny dims and the annealing recipe), `conftest.py`, `_train.py`
(the shared training loop with the hard / soft / annealed recipes), `viz.py` (GPU-aware demo with
the modes figure and the three-recipe comparison), the `MultimodalTrajectoryHead.__init__` module
list, and the `pred_toy_scene` toy (provided and verified - do not modify).

## compute_notes

Everything runs on CPU in seconds. The headline test trains three small heads for 400 steps each;
the overfit test trains one for 800 steps. The single-mode floor is seed-independent at 1.168 m
(the conditional mean of three balanced intentions). Hard-WTA's dead-mode collapse is
SEED-DEPENDENT with the real attention head (hard `min_ade` ranges ~0.01 to ~0.55 across seeds);
the headline test fixes seed 2, where hard WTA collapses to 2 winning modes, to demonstrate the
lesson. Annealed WTA is robustly near 0 (~0.005-0.05) on every seed tried.

## stretch_goals

1. Replace the K mode queries with K learned intention points (2D anchors) initialized from a
   k-means of training endpoints, decode one trajectory per anchor, and compare the minADE to the
   query-only head (the minimal TNT/MTR goal-conditioned variant).
2. Add EWTA (evolving WTA: regress the top-m winners with m decayed from K to 1) and compare its
   coverage to the temperature-annealed soft loss.
3. Add agent-to-agent attention over multiple agents' mode queries (the joint-prediction step,
   MotionFormer-style) and check whether predicted trajectories stop overlapping.
4. Swap the per-step-delta + cumsum parameterization for direct absolute-position regression and
   measure the effect on the overfit step count.

## further_reading

- VectorNet, [arXiv 2005.04259](https://arxiv.org/abs/2005.04259) - vectorized HD-map and agent
  encoding, the standard predictor input.
- MTR (Motion Transformer), [NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/file/2ab47c960bfee4f86dfc362f26ad066a-Paper-Conference.pdf) -
  intention-query anchors + local refinement; the design template for goal-based heads.
- UniAD, [arXiv 2212.10156](https://arxiv.org/abs/2212.10156) - planning-oriented end-to-end stack
  with query passing between tasks.
- VAD, [arXiv 2303.12077](https://arxiv.org/abs/2303.12077) - vectorized scene, drops the dense
  BEV grid; VADv2, [arXiv 2402.13243](https://arxiv.org/abs/2402.13243) - probabilistic planning.
- DriveTransformer, [arXiv 2503.07656](https://arxiv.org/abs/2503.07656) - parallel task queries.
- "Is Ego Status All You Need for Open-Loop End-to-End Autonomous Driving?",
  [arXiv 2312.03031](https://arxiv.org/abs/2312.03031) - the open-loop-metric critique.

## solution_notes

- The single-mode floor is exactly 1.168 m and seed-independent: the conditional mean of three
  balanced arcs sits slightly short of straight (both arcs trade forward progress for lateral), so
  it is far from all three endpoints. The README must not claim the centroid equals the straight
  mode.
- Measurement-driven deviation from the plan's clean per-seed ordering. With the real attention
  head (not the free-vector floor), hard WTA does NOT reliably plateau: across seeds 0-11 it lands
  at `min_ade` ~0.01 when it keeps 3 modes alive and ~0.55 (once 1.168, seed 9) when modes die.
  Annealed is robustly near 0 everywhere. The headline test therefore fixes seed 2 (a collapse
  seed) and pins single > 1.0, hard < 0.9, annealed < 0.2 with the strict ordering, rather than
  asserting hard always plateaus. This is the dead-mode lesson on the seed where it occurs, with
  the at-scale caveat stated in the README.
- Distinct mode-query init (`randn * 0.02`) is required, not optional: identical queries with a
  shared trajectory MLP make all K outputs identical at step 0 and the winner arbitrary.
- The loss selects the winner by Euclidean FDE but regresses with squared error; the mismatch is
  intended (squared error is smooth at 0 for gradcheck). Report meters through `min_ade`/`min_fde`.
- The oracle metrics reward coverage and can be gamed by redundant near-identical modes; they do
  not penalize redundancy.
```
