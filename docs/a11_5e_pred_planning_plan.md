# A11.5e build plan - unified perception -> prediction -> planning

Capstone of the AV module. Build one stage end-to-end: a multimodal motion-prediction
head that maps BEV features at an agent's location to K future trajectory hypotheses,
trained with a winner-take-all loss. Survey the rest of the end-to-end stack
(UniAD -> VAD -> DriveTransformer) and the open-loop-metric critique in the README.
Full planning is read, not built.

Source research note: `docs/research/a115e_pred_planning.md`. Dependency line in
`BUILD_ORDER.md`: deps A11.5b/c, A11.5d. This is a leaf - nothing imports from it, so no
new `nanovision.*` shim.

## What the student builds (`assignments/a11_5e_pred_planning/predict.py`)

All holed (raise `NotImplementedError("...")`); the answer key lives in `solution/predict.py`.

1. `roi_align_bev(bev_feat, centers, out_size=3, radius=1.0)` -> `(N, out_size*out_size, C)`.
   Bilinear-sample an `out_size x out_size` grid of points around each agent's continuous
   BEV-grid location (`centers` in fractional cell coords), spanning `+/- radius` cells in
   each axis, and return the sampled `C`-dim features as a flat token sequence per agent.
   Implemented with `F.grid_sample` (allowed - A11.5c/d already use it). This is the RoI-align
   that turns a dense BEV map into a fixed-length per-agent token set, standing in for what
   MotionFormer does around each tracked agent. gradcheck-able.

   grid_sample convention (pin these exactly - this is the single most likely silent bug):
   - `bev_feat` is stored `(C, nx, ny)` with `nx` = x/forward, `ny` = y/left (house style from
     `bev_toy_scene`). Fed to `grid_sample` as `(1, C, H, W)` that means `H = nx`, `W = ny`.
     `grid_sample` reads the last grid dim as `x = width FIRST`, so the sampling grid's last
     dimension must be `(g_y, g_x)` = `(normalized ny-coord, normalized nx-coord)`. `centers`
     are `(N, 2)` as `(x_cell, y_cell)` fractional indices, so you must SWAP: width-coord comes
     from `y_cell` (over `ny`), height-coord from `x_cell` (over `nx`).
   - Build the `out_size` offsets in CELL units (`linspace(-radius, +radius, out_size)`), add to
     the agent cell, THEN normalize per axis with `g = 2*(cell + 0.5)/S - 1`, `S = nx` or `ny`,
     `align_corners=False` (matches A11.5c/d). Radius stays in cells, independent of grid size.
   - `padding_mode="border"` (edge agents sample the boundary feature, not spurious zeros),
     `mode="bilinear"`.
   - Batch contract: there is ONE shared `bev_feat` and `N` agents; expand the single BEV to the
     `N` agent grids (`bev_feat[None].expand(N, ...)`), one `out_size x out_size` grid per agent.

2. `MultimodalTrajectoryHead(nn.Module)`:
   - `__init__(self, in_ch, dim=64, n_modes=6, horizon=12, n_layers=2, n_heads=4, roi_size=3, radius=1.0)`.
   - `K = n_modes` learned mode-query embeddings `(n_modes, dim)`, initialized DISTINCT (e.g.
     `nn.Parameter(torch.randn(n_modes, dim) * 0.02)` - NOT zeros / not all-equal). If every mode
     query starts identical and the trajectory MLP is shared, all K outputs are identical at step 0,
     the minFDE winner is arbitrary, and the modes collapse - `test_modes_specialize` then becomes a
     coin flip. Distinct init is required, not optional.
   - a linear `in_ch -> dim` projection for the RoI tokens.
   - Batch contract: `B = N` agents are the batch; all agents index the same `bev_feat`. The head
     takes `(bev_feat (C,nx,ny), centers (N,2))` and `roi_align_bev` broadcasts the one BEV over the
     `N` centers, so the head/loss `(B, ...)` shapes connect directly to the generator outputs.
   - `n_layers` decoder layers, each: mode-query self-attention (modes see each other, so they
     can spread out), then cross-attention of mode queries over the agent's RoI tokens, then an
     MLP - all built from `nanovision.attention.MultiHeadAttention` (forward `(x, kv=None, mask=None)`,
     `kv` given = cross-attention) and `nanovision.primitives` MLP/norm. No causal mask (the modes
     and the trajectory are not autoregressive here).
   - trajectory head: `Linear(dim -> horizon*2)` per mode, reshaped to per-step displacements and
     `cumsum`-ed along time to absolute agent-centric POSITIONS `(B, K, T, 2)`. Predicting
     per-step deltas then integrating keeps targets small and well-scaled; state this in the README.
     The head returns absolute positions and `gt` is absolute, so the loss and metrics are in
     position space - do not regress raw deltas against absolute `gt` (silent test failure).
   - score head: `Linear(dim -> 1)` per mode -> `(B, K)` mode logits.
   - `forward(bev_feat, centers) -> (trajs (B, K, T, 2), scores (B, K))`.

3. `wta_loss(trajs, scores, gt, *, temperature=None, cls_weight=1.0, return_components=False)`:
   - `gt`: `(B, T, 2)` in agent-centric coordinates (one observed future per agent).
   - regression term, two paths selected by `temperature`:
     - `temperature is None` (hard winner-take-all, the canonical min-of-N): winner index per
       sample = `argmin_k || trajs[:, k, -1] - gt[:, -1] ||` (minFDE selection by Euclidean
       endpoint distance - the endpoint carries most of the uncertainty; the standard, not minADE).
       Compute the winner index ONCE, detached (`argmin` has no gradient), so only the winning
       mode's trajectory carries regression gradient. Regression = mean over batch of mean-over-time
       squared error of the winning mode's full trajectory. (Squared error, not Euclidean, so it is
       smooth at 0 for gradcheck; report the meters-unit error through `min_ade`/`min_fde`, not the
       loss value. Selection uses Euclidean FDE while regression uses squared error - the mismatch
       is intended; the README must not claim the loss selects and regresses with the same metric.)
     - `temperature` a float (soft assignment / annealed min-of-N): per-mode weight
       `w = softmax(-FDE / temperature)` over the K modes; regression = mean over batch of
       `sum_k w_k * MSE_k` (every mode gets gradient, weighted toward the closer ones). This is the
       fix for the WTA dead-mode problem (see the lesson below); fully smooth, also gradcheck-able.
   - classification term = `cross_entropy(scores, winner_index)` where `winner_index` is the hard
     `argmin` over FDE (the committed mode) in BOTH paths, and `scores` are RAW logits
     (cross-entropy applies log-softmax internally - do not pre-softmax). Trains the score head to
     point at whichever mode won, so at inference the top-scored mode is the model's committed guess.
   - total = regression + `cls_weight` * classification.

   The dead-mode lesson (measured on the toy; drives the test and viz design). In `shared_context`
   mode every agent has identical input, so the head emits K trajectories independent of the agent -
   the predictor is literally K free trajectory vectors fit by the loss. Measured outcomes
   (free-parameter sim, deterministic), with K=6 modes over 3 balanced intentions:
   - single mode (K=1): min_ade 1.17 m - forced to the conditional mean of the three intentions
     (a trajectory through the middle of all of them, physically impossible). This is mode averaging.
   - hard WTA from random init (K=6): min_ade 0.55 m - better than one mode, but stuck at 2 winning
     modes; the spare modes never win, get no gradient, and stay dead (the WTA dead-mode collapse;
     K=3 hard collapses all the way back to ~1.17). Distinct mode-query init is necessary just to
     avoid total collapse, but does not by itself fix dead modes.
   - soft->hard annealed WTA (K=6, linear `temperature` decay from `tau0=3.0` to 0 over ~60% of
     training, then hard): min_ade ~0 with all 3 intentions covered by distinct modes, robust across
     init seeds. This is the headline result: K modes recover the multimodal future when trained so
     every mode gets early gradient.

4. Oracle metrics (no grad): `min_ade(trajs, gt)` (mean-over-time L2 of the best-of-K),
   `min_fde(trajs, gt)` (endpoint L2 of the best-of-K), `miss_rate(trajs, gt, thresh=2.0)`. For
   miss rate, select the per-sample best mode by minFDE (consistent with the WTA winner), then
   count a miss when that best mode's endpoint error exceeds 2.0 m - do not pick a different
   best-of-K for the miss count than FDE uses. Label these "Argoverse/nuScenes-style
   minADE_K/minFDE_K/MR@2m" in the README, not "the nuScenes metric" (nuScenes prediction uses
   K=5/10; this toy uses K=6 with the Argoverse 2 m threshold). README states the "oracle" caveat:
   K near-identical modes can score well, so these reward coverage but do not penalize redundancy.

## Toy data: `pred_toy_scene` in `nanovision/data/toy.py` (I author + verify before delegating)

Ego BEV conventions match the rest of the module (x forward, y left, z up; grid `(nx, ny)` over
`bev_x`/`bev_y` at `res`). One generator, two regimes selected by `shared_context`:

- `shared_context=True` (the mode-averaging demonstration): all agents sit at the SAME BEV cell
  with the SAME painted feature (which encodes only initial speed, not turn direction), but their
  futures fan out over `n_intentions` distinct intentions (default 3: left / straight / right).
  Given identical input, the future is genuinely ambiguous - a single-mode regressor must predict
  the conditional mean of the intentions and is wrong on every one; a K-mode WTA head can cover them.
  Generate BALANCED intention counts (equal agents per intention) so the conditional mean stays
  centered; an unbalanced batch shifts the mean and breaks the "forced to the centroid" framing. The
  mean of a left arc and a right arc is NOT exactly the straight trajectory (both arcs trade forward
  progress for lateral), so it sits slightly short of straight - still far from all three endpoints,
  so single-mode min_ade stays high; the README must not claim the centroid equals the straight mode.
- `shared_context=False` (identifiability / RoI test): agents are spread to distinct cells with
  per-agent features AND a spread of distinct initial yaws, so the head maps each distinct input to
  its own future and overfits to ~0, and the agent-centric rotation is load-bearing (a wrong rotation
  sign then actually fails the round-trip test instead of passing on a near-identity).

Motion model: every agent starts heading along its `yaw` (`+x`/ego forward when yaw=0) at `speed`
m/s; intention bends the path laterally over the horizon (straight = no bend; left = arc toward the
agent's +left; right = arc toward its +right). Pin the lateral arc displacement at the horizon to
`>= 3 m` (comfortably above the 2 m miss threshold and the modes-specialize endpoint-distance
threshold) so the three intention endpoints are well-separated regardless of seed. Agent-centric
frame is the initial-heading frame (agent at origin, +x = heading), so `futures_local` is the ego
future minus the start, rotated by `-yaw`; viz maps back with `+yaw`.

BEV feature grid `bev_feat (C, nx, ny)`: a `C`-channel grid (default `C=8`) that is ~0 in free
space with a Gaussian feature blob splatted at each agent cell; the blob's channel vector encodes
`[speed_norm, context_onehot...]` - position and speed are observable, intention is not. `centers`
returned in fractional-cell coords for `roi_align_bev`.

Returns dict: `bev_feat (C, nx, ny)`, `centers (N, 2)` fractional cell coords, `agents_xy (N, 2)`
ego meters, `agent_yaw (N,)`, `futures_local (N, T, 2)` agent-centric (training target),
`futures_ego (N, T, 2)` absolute ego (viz), `intention (N,) long`, `dt`, `horizon`, `bev_x`,
`bev_y`, `res`, `C`. Deterministic per seed; verified standalone (shapes, that intention
endpoints are well separated, that `shared_context=True` gives byte-identical features across
agents, and that `futures_local` round-trips back to `futures_ego` through `agent_yaw`).

## Tests (`tests/`)

- `test_shapes.py`: head outputs `(B, K, T, 2)` and `(B, K)` for `K=6, T=12`; `roi_align_bev`
  returns `(N, roi_size**2, C)`.
- `test_gradcheck.py`: double-precision `gradcheck` on `roi_align_bev` and on `wta_loss` w.r.t.
  `trajs`/`scores`. For `roi_align_bev`, place the sample centers at FRACTIONAL interior locations
  (not on integer cells, not at the border) - bilinear sampling is non-differentiable exactly at
  grid nodes. For `wta_loss`, construct `gt`/`trajs` so one mode is strictly the minFDE winner (off
  the tie boundary, so the detached `argmin` is stable).
- `test_overfit.py`: `shared_context=False`, drive `min_ade` below ~0.1 m within the step budget
  (distinct inputs -> exact fit).
- `test_wta_beats_single_mode.py` (headline lesson): `shared_context=True`. Train three configs on
  the same ambiguous batch and assert the ordering single > hard-WTA > annealed, each pinned after
  the build agent measures the real head (with margin; the free-vector floor is 1.17 / 0.55 / ~0):
  (a) `K=1` -> high `min_ade` (forced to the conditional mean of the intentions); (b) `K=6` hard
  WTA from random init -> lower but plateaued (dead modes); (c) `K=6` soft->hard annealed
  (`temperature` linear `tau0=3.0`->0 over ~60% of steps) -> near-zero `min_ade`. This builds the
  WORKING mechanism (c) and demonstrates why (a) and (b) fall short - it is not just (b).
- `test_modes_specialize.py`: after the ANNEALED `K=6` training, at least `n_intentions` modes win,
  and the winning modes' endpoints are mutually separated above a threshold (the modes cover the
  distinct intentions rather than collapsing). Use the annealed recipe, not hard WTA, here - hard
  WTA's dead modes would make this a coin flip (that is the point of the headline test, not this one).
- `test_forbidden_imports.py`: ban `nn.MultiheadAttention`, `nn.Transformer*`,
  `F.scaled_dot_product_attention` (the attention must come from `nanovision.attention`).

`config.py`: tiny dims (`dim=64`, `n_modes=6`, `horizon=12`, `n_heads=4`, `n_layers=2`) so the CPU
tests run in seconds. `viz.py`: GPU-aware (`--device`, CPU default) - render the BEV grid, the agent,
the GT future, and all K predicted modes colored by score; a second panel comparing K=6 vs K=1 on
the ambiguous scene to show mode averaging visually. `conftest.py`: insert assignment dir + impl dir
on `sys.path`, `NANOVISION_IMPL` switch, same pattern as A11.5d.

## README (lecture-notes standard) and ASSIGNMENT - the survey half

Teaching order from the research note, by concept name not assignment number:
1. The multi-future problem and mode averaging - why mean-L2 over one output is physically a
   trajectory through the middle of every option; motivate K hypotheses + min-of-N before any stack.
2. The build: RoI-align agent features -> mode-query decoder -> K trajectories -> min-of-N(minFDE)
   loss, in agent-centric coordinates. Teach the full arc with the toy's three measured numbers:
   single mode averages (1.17 m), hard WTA improves but leaves dead modes (0.55 m, only 2 of 6 modes
   alive; K=3 collapses fully), soft->hard annealed assignment gives every mode early gradient and
   recovers full coverage (~0). Frame hard WTA correctly: it is the workhorse min-of-N loss in real
   forecasting (MTR, MultiPath++), where diverse per-scene context keeps modes alive; the toy uses
   ONE shared ambiguous input, the extreme case that isolates the dead-mode collapse - so the toy
   exposes a real failure mode of WTA without implying "hard WTA is broken, always anneal" (it is
   not broken at scale). Soft assignment / EWTA / goal anchors (TNT, MTR) are the real mitigations.
   Marginal (one agent at a time) vs joint prediction named: social interaction (agent-to-agent
   attention, MotionFormer) is the next step, not built here.
3. The end-to-end stack: the modular pipeline (sensor -> det -> track -> pred -> plan) with
   non-differentiable hand-offs, then UniAD's replacement of each hand-off with soft query passing
   (track -> motion -> occ -> plan queries over a shared BEV backbone); VAD's vectorized scene
   (drops the dense BEV occupancy grid); DriveTransformer's parallel task queries (drops the
   sequential ordering). The planning head is a regression/imitation MLP, not RL or optimization.
4. The open-loop-metric critique as a primary point, not a footnote: AD-MLP ("Is Ego Status All
   You Need", CVPR 2024) - an ego-status-only MLP (ego velocity/acceleration/past ego trajectory,
   no perception input - NOT "velocity-only") matches perception-conditioned planners on nuScenes
   open-loop L2 because 73.9% of nuScenes is straight driving; so nuScenes open-loop is a
   debugging tool, not a measure of planning quality. Name NAVSIM and Bench2Drive as the
   closed-loop standard the field moved to; this build is necessarily open-loop (no simulator).
5. Bridge: A12 world models are a sensory model of the future (predict observations given actions),
   complementary to this behavioral model; A13 VLA / VLM planners (DriveLM, DriveVLM) trade
   geometric precision for instruction following - a different tradeoff, one paragraph.

Toy-doesn't-override-scale guard (state all of these explicitly in the README):
- The single-vs-multi-mode and dead-mode numbers are a toy demonstration of mode averaging and the
  WTA collapse failure, not evidence about real-world prediction accuracy.
- The dead-mode collapse here is amplified by the toy's single shared ambiguous input. At scale,
  hard min-of-N WTA is the standard, working loss because diverse per-scene context keeps modes
  alive; do not read the toy as "hard WTA is broken." It exposes a real risk, it does not refute it.
- The oracle metrics reward coverage and can be gamed by redundant modes.
- nuScenes open-loop numbers (including any this course could produce) do not measure planning
  quality.

arXiv ids to verify by fetching before writing them into the README: UniAD 2212.10156,
VAD 2303.12077, VADv2 2402.13243, AD-MLP 2312.03031, VectorNet 2005.04259,
DriveTransformer 2503.07656. (MTR is a NeurIPS proceedings PDF, not arXiv - cite as the
proceedings link from the research note.)

## Verification gates

- solution mode: `NANOVISION_IMPL=solution pytest` all green.
- default mode: every test fails only at a `NotImplementedError` hole (no import/collection errors).
- shim: none (leaf). Confirm `predict.py` imports attention only via `nanovision.attention`.
- forbidden-imports grep test passes.
- README style review (context-less) before commit; checklist box + commit last.
