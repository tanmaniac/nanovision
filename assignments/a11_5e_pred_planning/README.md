# A11.5e - unified perception, prediction, planning

The future of a driving agent is multimodal: the same observed state (position, speed) is
consistent with several distinct intentions, turn left, go straight, turn right. A single
trajectory regressor trained on mean error is pulled to the conditional mean of those futures, a
path through the middle of every option that matches none of them. The fix is to predict $K$
trajectory hypotheses and supervise only the closest one per sample, the winner-take-all loss.

Build one stage of the end-to-end driving stack: a multimodal motion-prediction head that maps a
BEV feature grid at an agent's location to $K$ future trajectory hypotheses. The dense BEV grid is
pooled around the agent with a small RoI-align, $K$ learned mode queries cross-attend over those
features through a few decoder layers, and each mode regresses a full trajectory plus a confidence
score, trained with the winner-take-all loss and its soft-to-hard annealed variant. The rest of the
stack (the modular pipeline, UniAD, VAD, DriveTransformer) and the open-loop-metric critique are
context, not part of the build.

Required reading before starting:
- Hu et al. 2023, "Planning-oriented Autonomous Driving" (UniAD),
  [arXiv:2212.10156](https://arxiv.org/abs/2212.10156).
- Shi et al. 2022, "Motion Transformer with Global Intention Localization and Local Movement
  Refinement" (MTR), [arXiv:2209.13508](https://arxiv.org/abs/2209.13508) (the multimodal
  trajectory head and the winner-take-all loss at scale).

## Lecture notes

### Mode averaging

Start with one agent at a known position and speed, approaching an intersection. It can turn left,
go straight, or turn right. Nothing in the observation says which: position and speed are the same
under all three. The future is genuinely multimodal given the input.

Train a single trajectory regressor on this with mean squared error. The minimizer of expected
squared error is the conditional mean, so the network learns the average of the three futures: a
path that goes a little forward and almost nowhere lateral, ending in the middle of the fan. That
average matches none of the three real futures. Worse, it is often not even a legal trajectory: the
mean of a hard left and a hard right is a path that drives straight into whatever the agent was
turning to avoid. This is mode averaging, and it is the reason a single regressor is the wrong model
for prediction. The centroid of a left arc and a right arc is not the straight trajectory either:
both arcs trade forward progress for lateral motion, so the mean sits slightly short of straight and
is still far from all three endpoints.

The fix is to stop predicting one trajectory. Predict $K$ of them, and at training time supervise
only the one closest to the observed future. Each hypothesis is then free to specialize on a
different intention without being pulled toward the others' average. This is the min-of-N, or
winner-take-all (WTA), loss.

### RoI-align from a dense grid to a per-agent token set

A BEV backbone (the Lift-Splat-Shoot or BEVFormer stage earlier in the module) produces a dense
feature grid $(C, n_x, n_y)$, with $n_x$ along x/forward and $n_y$ along y/left. A prediction head
does not consume the whole grid; it pools the features in a small window around each agent. That
pooling is RoI-align: build an $\text{out}\times\text{out}$ grid of sample points spanning $\pm$
radius cells around each agent's fractional cell, bilinearly sample the BEV at those points with
`grid_sample`, and return a fixed token set per agent. This stands in for what a production motion
decoder (MotionFormer in UniAD) does around each tracked agent: deformable-attention-pool the BEV
around the agent box into a fixed query set.

The line that silently breaks is the `grid_sample` axis order. The BEV grid $(C, n_x, n_y)$ is fed
as $(1, C, H, W)$, so $H = n_x$ and $W = n_y$. `grid_sample` reads the last grid dimension as
$(x{=}\text{width}, y{=}\text{height})$, the opposite order from how the agent centers store
$(x_{\text{cell}}, y_{\text{cell}})$. So the width coordinate must come from $y_{\text{cell}}$ (it
indexes $n_y$) and the height coordinate from $x_{\text{cell}}$ (it indexes $n_x$), a swap. Each
axis is normalized with the align_corners=False cell-center rule $g = 2(\text{cell} + 0.5)/S - 1$,
$S = n_x$ or $n_y$, and `padding_mode="border"` so an edge agent samples the boundary feature
instead of spurious zeros.

### A mode-query decoder

The head holds $K$ learned mode-query embeddings, initialized distinct (small random), not zero and
not all-equal. If every query starts identical and the per-mode trajectory MLP is shared, all $K$
outputs are byte-identical at step 0, the min-of-N winner is an arbitrary tie-break, and the modes
never separate. Distinct init breaks that symmetry. It is necessary but not sufficient; it does not
by itself keep all modes alive.

Each decoder layer runs three steps from the transformer's multi-head attention and an MLP:
mode-query self-attention (the $K$ modes see each other, so they can spread out and avoid
duplicating a hypothesis), then cross-attention of the mode queries over the agent's RoI tokens
(each mode reads the agent's context), then an MLP. There is no causal mask: neither the modes nor
the trajectory steps are autoregressive, they are all produced in one shot.

Each mode's trajectory head predicts per-step displacements that are cumsum-ed along time into
absolute agent-centric positions $(B, K, T, 2)$. Predicting displacements and integrating keeps the
regression targets small and well-scaled (a per-step delta is sub-meter; an absolute position at the
horizon is several meters), and the cumsum is a fixed linear map so it adds no parameters. A score
head produces a per-mode logit $(B, K)$.

### The winner-take-all loss

The winner is the mode whose endpoint is closest to the ground-truth endpoint, the minFDE selection
$\arg\min_k \lVert \text{traj}_k[-1] - \text{gt}[-1] \rVert$. The endpoint carries most of the
trajectory uncertainty, so endpoint distance is the standard selection (minFDE), not mean-over-time
distance (minADE).

The hard path is the canonical min-of-N. Compute the winner index once, detached (argmin has no
gradient), and regress that one mode's full trajectory with mean squared error. Only the winning
mode carries regression gradient. Selection is by Euclidean FDE while regression is squared error;
the mismatch is intended (squared error is smooth at 0, which a gradcheck needs), and the meters-unit
error is reported through the minADE/minFDE metrics, not through the loss value.

The soft path replaces the hard winner with a softmax over modes,
$w_k = \mathrm{softmax}(-\text{FDE}_k / \tau)$, and the regression is $\sum_k w_k \cdot \text{MSE}_k$.
Every mode now gets gradient, weighted toward the closer ones. As $\tau \to 0$ the softmax becomes a
one-hot at the winner and the soft loss converges to the hard loss.

The classification term is the same in both paths: a cross-entropy of the score logits against the
hard argmin winner, with raw logits (cross-entropy applies log-softmax internally). This trains the
score head to point at whichever mode committed, so at inference the top-scored mode is the model's
single committed guess. The total loss is
$\text{regression} + \text{cls}_{\text{weight}} \cdot \text{classification}$.

### The dead-mode failure and annealing

The reason for the soft path is a failure mode of hard WTA. A mode that never wins gets no
regression gradient, so it never moves toward any data, so it keeps never winning. It is dead. With
$K$ modes and only a few intentions in the batch, the surviving modes can carve up the intentions
between them and leave the rest dead.

The annealed recipe decays $\tau$ linearly from a positive start to 0 over the first part of
training, then runs hard. Early on every mode gets gradient and the modes spread to cover the fan;
by the time the loss goes hard the modes are already specialized, so no mode is left to die. Soft
assignment, evolving WTA (EWTA), and goal anchors (TNT, MTR) are the standard mitigations when the
dead-mode risk bites.

Hard min-of-N WTA is the workhorse loss at scale (MTR, MultiPath++), because diverse per-scene
context keeps modes alive: different scenes activate different modes, so every mode gets gradient
across the dataset even though any single scene only supervises one. The dead-mode collapse is a
real risk, not a claim that hard WTA is broken; it bites hardest when the input is a single
ambiguous context with no per-scene diversity to keep spare modes fed.

### Oracle metrics

minADE, minFDE, and miss-rate are best-of-K oracle metrics: they pick the single mode closest to the
ground truth and score it. minFDE is the endpoint L2 of the best-of-K mode; minADE is the
mean-over-time L2 of that same mode (best chosen by minFDE, consistent with the WTA winner);
miss-rate at a threshold is the fraction of agents whose best-of-K endpoint error exceeds it. These
are the Argoverse/nuScenes-style $\text{minADE}_K$ / $\text{minFDE}_K$ / MR metrics. They are called
oracle metrics because they assume an oracle that picks the right mode: $K$ near-identical modes can
score well, so they reward coverage but do not penalize redundancy. The score head exists precisely
so that at inference, without the oracle, the model still commits to one mode.

### The end-to-end stack

Marginal versus joint prediction names the scope. This head predicts one agent at a time,
conditioned on its own context: marginal prediction. The agents do not see each other, so two
predicted futures can collide. Joint prediction adds agent-to-agent attention (the modes of
different agents attend to each other) so the predictions are mutually consistent; UniAD's
MotionFormer is the joint version.

The classical stack is modular: sensors feed detection, detection feeds tracking, tracking feeds
prediction, prediction feeds planning. Each hand-off is a hard, non-differentiable interface, a list
of boxes, a set of tracks, a set of trajectories. Errors compound across the stages and no gradient
flows backward, so the perception modules cannot learn what the planner actually needs.

UniAD replaces each hard hand-off with soft query passing over a shared BEV backbone. Track queries
become motion queries become occupancy queries become planning queries; every stage is a transformer
reading the previous stage's queries, so the whole stack is one differentiable network and every
task is trained toward the final planning objective. The RoI-align-then-mode-query-decoder built here
is the shape of UniAD's MotionFormer in miniature.

VAD (Jiang et al. 2023, [arXiv:2303.12077](https://arxiv.org/abs/2303.12077)) drops the dense BEV
occupancy grid in favor of a vectorized scene: agents and map elements are polylines and instance
vectors, not pixels. The input representation traces back to VectorNet (Gao et al. 2020,
[arXiv:2005.04259](https://arxiv.org/abs/2005.04259)), which established polyline encoding of HD
maps and agent histories. VADv2 (Chen et al. 2024,
[arXiv:2402.13243](https://arxiv.org/abs/2402.13243)) turns the planner into a probabilistic model
over a discrete vocabulary of planning actions. DriveTransformer (Jia et al. 2025,
[arXiv:2503.07656](https://arxiv.org/abs/2503.07656)) drops the sequential task ordering: agent,
map, and planning queries run in parallel and attend to each other directly, instead of the
track-then-motion-then-plan chain.

The planning head in all of these is a regression or imitation MLP: it predicts the ego trajectory
to match the human demonstration. It is not reinforcement learning and not an explicit optimizer
over a cost function.

### The open-loop metric is a debugging tool, not a score

A planner trained by imitation is usually evaluated open-loop: replay a logged scene, predict the
ego trajectory, and measure its L2 distance to what the human drove. "Is Ego Status All You Need for
Open-Loop End-to-End Autonomous Driving?" (Zhai et al. 2023,
[arXiv:2312.03031](https://arxiv.org/abs/2312.03031), the AD-MLP result) showed an MLP that sees only
the ego status, its own velocity, acceleration, and past trajectory, with no perception input at
all, matches the perception-conditioned planners on nuScenes open-loop L2. The reason is that 73.9%
of nuScenes is near-straight driving, so extrapolating the ego's current motion is a strong predictor
of the next few seconds and perception barely changes the answer. nuScenes open-loop L2 therefore
does not measure planning quality; it mostly measures whether the ego's constant-velocity motion can
be extrapolated.

The field moved to closed-loop evaluation in a simulator, where the planner's outputs actually drive
the car and errors compound: NAVSIM (a non-reactive but metric-faithful nuScenes-based benchmark) and
Bench2Drive (closed-loop in CARLA) are the current standards.

### Where this goes next

This head is a behavioral model of the future: given the scene, predict what agents will do. The
world-models assignment builds a sensory model of the future instead: given the current observation
and an action, predict the next observation. A full driver wants both, the behavioral prediction of
other agents and the sensory rollout of its own actions. The VLA/VLM assignment trades geometric
precision for instruction following: planners like DriveLM and DriveVLM reason over language ("yield
to the pedestrian", "the light is red") and emit higher-level decisions, accepting coarser
trajectories in exchange for following instructions and explaining themselves.

## The assignment

Fill these holes, in order. Each is one `NotImplementedError` with a matching test; the docstring in each file gives the signature, shapes, and constraints.

1. [`roi_align_bev()`](predict.py) in `predict.py`
2. [`MultimodalTrajectoryHead.forward()`](predict.py) in `predict.py`
3. [`wta_loss()`](predict.py) in `predict.py`
4. [`min_ade()`](predict.py) in `predict.py`
5. [`min_fde()`](predict.py) in `predict.py`
6. [`miss_rate()`](predict.py) in `predict.py`

### Running and validating

Activate the environment (`conda activate nanovision`), then:

```
make test     A=a11_5e_pred_planning   # run the tests against the top-level files (the holes)
make verify   A=a11_5e_pred_planning   # run the same tests against the reference solution/
make viz      A=a11_5e_pred_planning   # render the figures from the reference solution
make viz-mine A=a11_5e_pred_planning   # render the figures from your own code (holes filled)
```

`make test` runs the suite in `assignments/a11_5e_pred_planning/tests/` against the top-level
`predict.py`, red until the holes are filled and green once correct. `make verify` runs the identical
suite against the reference `solution/` by setting `NANOVISION_IMPL=solution`, so it is green from
the start and shows the target. The tests cover shapes, a gradcheck on the differentiable pieces,
that the modes specialize, an overfit, and the headline contrast that a $K$-mode annealed head beats
a single mode.

What you should see when you run this. The toy is CPU only, seconds per test. `pred_toy_scene` with
`shared_context=True` places several agents at one BEV cell with one feature vector (which encodes
only speed, not intention) but fans their futures over three balanced intentions, with the horizon
lateral displacement pinned above 3 m so the endpoints are well separated. Measured with the real
attention head and the tiny config here, over random init seeds:

| recipe | minADE | what happens |
|---|---|---|
| single mode (K=1) | 1.168 m, seed-independent | forced to the conditional mean of the three intentions |
| hard WTA (K=6) | 0.01-0.02 m on 5 of 8 seeds, 0.56 m on 3 of 8 | sometimes all 3 modes survive, sometimes 2 die |
| soft-to-hard annealed (K=6) | 0.01-0.06 m on every seed | all 3 intentions covered by distinct modes |

The single-mode 1.168 m is seed-independent because it is the geometric distance from the conditional
mean to the nearest of the three arcs, set by the scene, not by optimization. The hard-WTA collapse
is fragile, not uniform: with the real attention head the mode self-attention gives the modes a path
to spread out, so they often escape the dead-mode trap, and whether they do depends on init. The
headline test asserts only the robust, init-independent facts, that single-mode averages near
1.17 m, the annealed recipe reaches under 0.15 m on every seed it tries, and coverage beats averaging
by a wide margin, and leaves the intermittent hard-WTA collapse to the measured table and the
`compare_recipes` figure. `make viz` writes `modes_annealed.png` and `compare_recipes.png` to `out/`;
the comparison shows K=1 collapsed to one short averaged path, hard K=6 covering some intentions
while dead modes wander off, and annealed K=6 covering all three arcs.

These are toy demonstrations of mode averaging and the WTA collapse on one shared ambiguous input,
the amplified worst case. They are not evidence about real-world prediction accuracy, and the oracle
metrics reward coverage rather than driving quality, the same caveat the open-loop-metric section
makes about nuScenes L2.

## Further reading

- Hu et al. 2023, UniAD, [arXiv:2212.10156](https://arxiv.org/abs/2212.10156).
- Shi et al. 2022, MTR, [arXiv:2209.13508](https://arxiv.org/abs/2209.13508).
- Gao et al. 2020, VectorNet, [arXiv:2005.04259](https://arxiv.org/abs/2005.04259).
- Jiang et al. 2023, VAD, [arXiv:2303.12077](https://arxiv.org/abs/2303.12077).
- Chen et al. 2024, VADv2, [arXiv:2402.13243](https://arxiv.org/abs/2402.13243).
- Jia et al. 2025, DriveTransformer, [arXiv:2503.07656](https://arxiv.org/abs/2503.07656).
- Zhai et al. 2023, "Is Ego Status All You Need for Open-Loop End-to-End Autonomous Driving?",
  [arXiv:2312.03031](https://arxiv.org/abs/2312.03031).
