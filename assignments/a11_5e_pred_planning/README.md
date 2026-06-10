# Unified perception -> prediction -> planning

This assignment builds one stage of the end-to-end driving stack: a multimodal motion-prediction head that
maps a BEV feature grid at an agent's location to K future trajectory hypotheses, trained with a
winner-take-all loss. The dense BEV grid is pooled around the agent with a small RoI-align, K
learned mode queries cross-attend over those features through a few decoder layers, and each mode
regresses a full trajectory plus a confidence score. The rest of the stack (the modular pipeline,
UniAD, VAD, DriveTransformer) and the open-loop-metric critique are surveyed here, not built.

## Mode averaging, the problem this head exists to solve

Start with one agent at a known position and speed, approaching an intersection. It can turn left,
go straight, or turn right. Nothing in the observation says which: position and speed are the same
under all three. The future is genuinely multimodal given the input.

Train a single trajectory regressor on this with mean squared error. The minimizer of expected
squared error is the conditional mean, so the network learns the average of the three futures: a
path that goes a little forward and almost nowhere lateral, ending in the middle of the fan. That
average matches none of the three real futures. Worse, it is often not even a legal trajectory: the
mean of a hard left and a hard right is a path that drives straight into whatever the agent was
turning to avoid. This is mode averaging, and it is the reason a single regressor is the wrong
model for prediction.

The toy makes this exact. `pred_toy_scene` with `shared_context=True` places several agents at one
BEV cell with one feature vector (which encodes only speed, not intention) but fans their futures
over three balanced intentions (left, straight, right) with the horizon lateral displacement pinned
above 3 m so the endpoints are well separated. A single-mode head trained on this batch reaches
`min_ade = 1.168 m` and cannot go lower: that is the distance from the conditional mean to the
nearest real future. The number is seed-independent because it is set by the geometry of the three
arcs, not by optimization. Note that the centroid of a left arc and a right arc is not the straight
trajectory: both arcs trade forward progress for lateral motion, so the mean sits slightly short of
straight and is still far from all three endpoints.

The fix is to stop predicting one trajectory. Predict K of them, and at training time supervise
only the one closest to the observed future. Each hypothesis is then free to specialize on a
different intention without being pulled toward the others' average. This is the min-of-N, or
winner-take-all (WTA), loss, and it is the core mechanism of this assignment.

## What to implement

### RoI-align: a dense BEV grid to a fixed per-agent token set

A BEV backbone (the lift-splat or BEVFormer stage earlier in the module) produces a dense feature
grid `bev_feat (C, nx, ny)`, with `nx` along x/forward and `ny` along y/left. A prediction head
does not consume the whole grid; it pools the features in a small window around each agent. That
pooling is RoI-align: `roi_align_bev(bev_feat, centers, out_size, radius)` builds an
`out_size x out_size` grid of sample points spanning $\pm$`radius` cells around each agent's
(fractional) cell, bilinearly samples `bev_feat` at those points with `F.grid_sample`, and returns
`(N, out_size**2, C)` tokens per agent. This stands in for what a production motion decoder
(MotionFormer in UniAD) does around each tracked agent: deformable-attention-pool the BEV around
the agent box into a fixed query set.

The one line that silently breaks is the `grid_sample` axis order. `bev_feat (C, nx, ny)` is fed as
`(1, C, H, W)`, so `H = nx` and `W = ny`. `grid_sample` reads the last grid dimension as
$(x{=}\text{width}, y{=}\text{height})$, which is the opposite order from how `centers` store
$(x\_cell, y\_cell)$. So the width coordinate must come from `y_cell` (it indexes `ny`) and the
height coordinate from `x_cell` (it indexes `nx`) - a swap. Offsets are built in cell units, added
to the agent cell, then each axis is normalized with the `align_corners=False` cell-center rule
$g = 2(\text{cell} + 0.5)/S - 1$, $S = nx$ or $ny$, with `padding_mode="border"` so an edge agent
samples the boundary feature instead of spurious zeros. `config.py` documents this convention; it
matches the `grid_sample` usage in the BEVFormer and occupancy assignments.

### A mode-query decoder

`MultimodalTrajectoryHead` holds `K = n_modes` learned mode-query embeddings. They are initialized
distinct (`randn * 0.02`), not zero and not all-equal. This matters: if every query starts
identical and the per-mode trajectory MLP is shared, all K outputs are byte-identical at step 0,
the min-of-N winner is an arbitrary tie-break, and the modes never separate. Distinct init breaks
that symmetry. It is necessary but not sufficient; it does not by itself keep all modes alive (see
the dead-mode lesson below).

Each decoder layer runs three steps from the A1 `MultiHeadAttention` and the shared MLP primitive:
mode-query self-attention (the K modes see each other, so they can spread out and avoid duplicating
a hypothesis), then cross-attention of the mode queries over the agent's RoI tokens (each mode
reads the agent's context), then an MLP. There is no causal mask: neither the modes nor the
trajectory steps are autoregressive here, they are all produced in one shot.

Each mode's trajectory head is a `Linear(dim -> horizon*2)` reshaped to per-step displacements,
which are cumsum-ed along time into absolute agent-centric positions $(B, K, T, 2)$. Predicting
displacements and integrating keeps the regression targets small and well-scaled (a per-step delta
is sub-meter; an absolute position at the horizon is several meters), and the cumsum is a fixed
linear map so it adds no parameters. The head returns absolute positions, and the ground-truth
future is absolute, so the loss and metrics live in position space. A score head
`Linear(dim -> 1)` produces a per-mode logit $(B, K)$.

The batch is the agents: `B = N` agents share one `bev_feat`, and `roi_align_bev` broadcasts that
single BEV over the N centers, so the head and loss shapes connect directly to the generator's
outputs.

### The winner-take-all loss and its two paths

`wta_loss(trajs, scores, gt, temperature, ...)` selects a winner mode per sample and supervises it.
The winner is the mode whose endpoint is closest to the ground-truth endpoint, the minFDE selection
$\arg\min_k \lVert \text{traj}_k[-1] - \text{gt}[-1] \rVert$. The endpoint carries most of the
trajectory uncertainty, so endpoint distance is the standard selection (minFDE), not mean-over-time
distance (minADE).

The hard path (`temperature=None`) is the canonical min-of-N. Compute the winner index once,
detached (argmin has no gradient), and regress that one mode's full trajectory with mean squared
error. Only the winning mode carries regression gradient. Selection is by Euclidean FDE while
regression is squared error; the mismatch is intended (squared error is smooth at 0, which the
gradcheck needs), and the meters-unit error is reported through `min_ade`/`min_fde`, not through the
loss value.

The soft path (`temperature` a float) replaces the hard winner with a softmax over modes:
$w_k = \mathrm{softmax}(-\text{FDE}_k / \tau)$, and the regression is $\sum_k w_k \cdot \text{MSE}_k$.
Every mode now gets gradient, weighted toward the closer ones. As $\tau \to 0$ the softmax becomes
a one-hot at the winner and the soft loss converges to the hard loss.

The classification term is the same in both paths: `cross_entropy(scores, winner)` with the hard
argmin winner and raw logits (cross-entropy applies log-softmax internally, so do not pre-softmax).
This trains the score head to point at whichever mode committed, so at inference the top-scored mode
is the model's single committed guess. The total loss is
$\text{regression} + \text{cls}_{\text{weight}} \cdot \text{classification}$.

### The dead-mode lesson, measured

The reason for the soft path is a failure mode of hard WTA. A mode that never wins gets no
regression gradient, so it never moves toward any data, so it keeps never winning. It is dead. With
K modes and only a few intentions in the batch, the surviving modes can carve up the intentions
between them and leave the rest dead.

On the toy's shared-context scene (one ambiguous input, three balanced intentions), measured with
the real attention head and the tiny config here, over 8 random init seeds (0-7, 250 steps):

| recipe | min_ade | what happens |
|---|---|---|
| single mode (K=1) | 1.168 m, seed-independent | forced to the conditional mean of the three intentions |
| hard WTA (K=6) | 0.01-0.02 m on 5 of 8 seeds, 0.56 m on 3 of 8 | sometimes all 3 modes survive, sometimes 2 die |
| soft -> hard annealed (K=6) | 0.01-0.06 m on every seed | all 3 intentions covered by distinct modes |

The annealed recipe decays $\tau$ linearly from `tau0 = 3.0` to 0 over the first ~60% of steps,
then runs hard. Early on every mode gets gradient and the modes spread to cover the fan; by the time
the loss goes hard the modes are already specialized, so no mode is left to die. This is the
headline result: K modes recover the multimodal future when trained so every mode gets early
gradient, robustly across init.

The honest shape of the hard-WTA result is that it is FRAGILE, not uniformly broken. With the real
attention head (as opposed to a bag of K free trajectory vectors), whether the dead-mode collapse
happens depends on the init: across seeds 0-7 it collapsed to 2 modes (`min_ade` ~0.56) on 3 seeds
and kept all 3 alive (`min_ade` ~0.01) on the other 5. The mode self-attention gives the modes a
path to spread out that the free-vector model lacks, so the real head often escapes the trap on its
own. The headline test therefore asserts only the robust, init-independent facts - single-mode
averages (~1.17 m), annealed reaches full coverage (<0.15 m on every seed it tries), coverage beats
averaging by a wide margin - and leaves the intermittent hard-WTA collapse to this measured table
and the `compare_recipes` figure rather than pinning a single "collapse seed" into an assertion. The
lesson is that hard WTA can dead-mode and annealing removes that risk, not that hard WTA always fails.

### Toy results do not override the at-scale picture

State these plainly so the toy is not over-read:

- The single-vs-multi-mode and dead-mode numbers are a toy demonstration of mode averaging and the
  WTA collapse failure. They are not evidence about real-world prediction accuracy.
- Hard min-of-N WTA is the workhorse loss at scale. It is the standard, working loss in real motion
  forecasting (MTR, MultiPath++), because diverse per-scene context keeps modes alive: different
  scenes activate different modes, so every mode gets gradient across the dataset even though any
  single scene only supervises one. The toy uses one shared ambiguous input, the amplified extreme
  that isolates the dead-mode collapse. It exposes a real risk of WTA; it does not show that hard
  WTA is broken, and the takeaway is not "always anneal." Soft assignment, evolving WTA (EWTA), and
  goal anchors (TNT, MTR) are the real mitigations when the risk does bite.
- The oracle metrics below reward coverage and can be gamed by redundant modes; they do not measure
  planning quality, and neither do nuScenes open-loop numbers (see the critique section).

### Oracle metrics

`min_ade`, `min_fde`, and `miss_rate` are best-of-K oracle metrics: they pick the single mode
closest to the ground truth and score it. `min_fde` is the endpoint L2 of the best-of-K mode;
`min_ade` is the mean-over-time L2 of that same mode (best chosen by minFDE, consistent with the WTA
winner); `miss_rate` at threshold 2.0 m is the fraction of agents whose best-of-K endpoint error
exceeds 2 m. These are the Argoverse/nuScenes-style $\text{minADE}_K$ / $\text{minFDE}_K$ / MR@2m
metrics. This toy uses K=6 with the Argoverse 2 m threshold; nuScenes prediction reports K=5/10, so
these are that family of metric, not "the nuScenes metric" verbatim. They are called oracle metrics
because they assume an oracle that picks the right mode: K near-identical modes can score well, so
they reward coverage but do not penalize redundancy. The score head exists precisely so that at
inference, without the oracle, the model still commits to one mode.

## The end-to-end stack

Marginal vs joint prediction names the scope. This head predicts one agent at a time, conditioned on
its own context: marginal prediction. The agents do not see each other, so two predicted futures can
collide. Joint prediction adds agent-to-agent attention (the modes of different agents attend to
each other) so the predictions are mutually consistent; UniAD's MotionFormer is the joint version.
That is the next step, not built here.

The classical stack is modular: sensors feed detection, detection feeds tracking, tracking feeds
prediction, prediction feeds planning. Each hand-off is a hard, non-differentiable interface - a
list of boxes, a set of tracks, a set of trajectories. Errors compound across the stages and no
gradient flows backward, so the perception modules cannot learn what the planner actually needs.

UniAD ([arXiv 2212.10156](https://arxiv.org/abs/2212.10156), "Planning-oriented Autonomous
Driving") replaces each hard hand-off with soft query passing over a shared BEV backbone. Track
queries become motion queries become occupancy queries become planning queries; every stage is a
transformer reading the previous stage's queries, so the whole stack is one differentiable network
and every task is trained toward the final planning objective. The RoI-align-then-mode-query-decoder
built here is the shape of UniAD's MotionFormer in miniature.

VAD ([arXiv 2303.12077](https://arxiv.org/abs/2303.12077), "Vectorized Scene Representation for
Efficient Autonomous Driving") drops the dense BEV occupancy grid in favor of a vectorized scene:
agents and map elements are polylines and instance vectors, not pixels. The input representation
traces back to VectorNet ([arXiv 2005.04259](https://arxiv.org/abs/2005.04259)), which established
polyline encoding of HD maps and agent histories. VADv2
([arXiv 2402.13243](https://arxiv.org/abs/2402.13243)) turns the planner into a probabilistic model
over a discrete vocabulary of planning actions. DriveTransformer
([arXiv 2503.07656](https://arxiv.org/abs/2503.07656)) drops the sequential task ordering: agent,
map, and planning queries run in parallel and attend to each other directly, instead of the
track-then-motion-then-plan chain.

The planning head in all of these is a regression or imitation MLP: it predicts the ego trajectory
to match the human demonstration. It is not reinforcement learning and not an explicit optimizer
over a cost function. That choice is what the next section criticizes.

## The open-loop metric is a debugging tool, not a score

A planner trained by imitation is usually evaluated open-loop: replay a logged scene, predict the
ego trajectory, and measure its L2 distance to what the human actually drove. The trouble is that
"Is Ego Status All You Need for Open-Loop End-to-End Autonomous Driving?"
([arXiv 2312.03031](https://arxiv.org/abs/2312.03031), the AD-MLP result) showed an MLP that sees
only the ego status - its own velocity, acceleration, and past trajectory, with no perception input
at all - matches the perception-conditioned planners on nuScenes open-loop L2. Not "velocity only":
the input is the ego state, and that alone is enough. The reason is that 73.9% of nuScenes is
near-straight driving, so extrapolating the ego's current motion is a strong predictor of the next
few seconds and perception barely changes the answer. nuScenes open-loop L2 therefore does not
measure planning quality; it mostly measures whether you can extrapolate a constant-velocity ego.
Treat it as a debugging tool that catches gross errors, not as a ranking.

The field moved to closed-loop evaluation in a simulator, where the planner's outputs actually drive
the car and errors compound: NAVSIM (a non-reactive but metric-faithful nuScenes-based benchmark)
and Bench2Drive (closed-loop in CARLA) are the current standards. This assignment is necessarily
open-loop, since the course has no driving simulator; the metrics here are the oracle
forecasting metrics, with the same caveat that they reward coverage rather than driving quality.

## Bridge to what comes next

This head is a behavioral model of the future: given the scene, predict what agents will do. The
world-models assignment builds a sensory model of the future instead: given the current observation
and an action, predict the next observation. The two are complementary; a full driver wants both,
the behavioral prediction of other agents and the sensory rollout of its own actions.

The VLA / VLM assignment trades geometric precision for instruction following: planners like DriveLM
and DriveVLM reason over language ("yield to the pedestrian", "the light is red") and emit
higher-level decisions, accepting coarser trajectories in exchange for the ability to follow
instructions and explain themselves. That is a different point on the precision-vs-generality
tradeoff than the metric-trajectory head built here.

## Running it

```
make test A=a11_5e_pred_planning          # CPU, seconds
python -m assignments.a11_5e_pred_planning.viz            # writes out/modes_annealed.png and out/compare_recipes.png
python -m assignments.a11_5e_pred_planning.viz --device cuda   # GPU if available
```

`out/compare_recipes.png` shows the three recipes side by side on the ambiguous scene: K=1 collapses
to one short averaged path, hard K=6 covers some intentions while the dead modes wander off, and
annealed K=6 covers all three ground-truth arcs.
