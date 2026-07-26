# A11.5e - unified perception, prediction, planning

The future of a driving agent is multimodal: the same observed state (position, speed) is
consistent with several distinct intentions, turn left, go straight, turn right. A single
trajectory regressor trained on mean error is pulled to the conditional mean of those futures, a
path through the middle of every option that matches none of them. The fix is to predict $K$
trajectory hypotheses and supervise only the closest one per sample, the winner-take-all loss.

Build one stage of the end-to-end driving stack: a multimodal motion-prediction head that maps a
bird's-eye-view (BEV) feature grid at an agent's location to $K$ future trajectory hypotheses. The
dense BEV grid is pooled around the agent with a small region-of-interest align (RoI-align), $K$
learned mode queries cross-attend over those features through a few decoder layers, and each mode
regresses a full trajectory plus a confidence score, trained with the winner-take-all loss and its
soft-to-hard annealed variant. The rest of the stack (the modular pipeline, UniAD, VAD,
DriveTransformer) and the open-loop-metric critique are context, not part of the build.

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

Train a single trajectory regressor $f$ on this with mean squared error. That it lands in the
middle of the fan is a one-line consequence of the objective, not an empirical tendency. Fix the
input $x$, write $y$ for the observed future, and let $\mu = \mathbb{E}[y \mid x]$ be the
conditional mean. Adding and subtracting $\mu$ inside the expected squared error splits it:

$$\mathbb{E}\big[\lVert f(x) - y\rVert^2 \mid x\big]
= \lVert f(x) - \mu \rVert^2 + \mathbb{E}\big[\lVert y - \mu\rVert^2 \mid x\big],$$

the cross term vanishing because $\mathbb{E}[y - \mu \mid x] = 0$. The second term is the spread of
the true futures and does not involve $f$ at all, so the objective is minimized exactly at
$f(x) = \mu$ and its floor is that spread. This is the same decomposition that makes the
conditional mean the minimum-mean-square estimator in classical estimation. A squared-error
regressor with enough capacity has no option other than to average.

Averaging is wrong here in a specific way. The toy scene rolls each intention out under a
constant-speed, constant-turn-rate model at $v = 1.5$ m/s over a 6 s horizon, with
$\omega = \pm 1/6$ rad/s for the turns and $\omega = 0$ for straight. In the agent frame ($+x$
forward, $+y$ left) the three endpoints are $(7.57, +4.14)$, $(9.00, 0.00)$ and $(7.57, -4.14)$
meters. Their mean is $(8.05, 0.00)$: 0.95 m short of the straight endpoint and 4.16 m from either
turn. Averaging a hard left and a hard right does not recover the straight path, because both arcs
trade forward progress for lateral motion, so the mean falls short of straight while still missing
both turns. On a real intersection the averaged path is often not even drivable, running through
whatever the agent was turning to avoid.

The fix is to stop predicting one trajectory. Predict $K$ of them, and at training time supervise
only the one closest to the observed future. Each hypothesis is then free to specialize on a
different intention without being pulled toward the others' average. This is the min-of-N, or
winner-take-all (WTA), loss.

### Measuring the distance between two trajectories

"Closest" needs a number. A trajectory here is a sequence of $T$ two-dimensional positions
$y_1, \dots, y_T$ in the agent's own frame, one per future timestep. Comparing a prediction
$\hat y$ against an observed future $y$ means reducing $T$ per-step Euclidean errors to one scalar,
and the prediction literature uses two reductions, both in meters:

$$\text{ADE}(\hat y, y) = \frac{1}{T}\sum_{t=1}^{T} \lVert \hat y_t - y_t \rVert_2,
\qquad \text{FDE}(\hat y, y) = \lVert \hat y_T - y_T\rVert_2.$$

ADE is the average displacement error over the whole horizon; FDE is the final displacement error,
the endpoint alone. Time is 1-based throughout: step $t$ is $t\,\Delta t$ seconds into the future,
stored at array index $t-1$, and $\Delta t = 0.5$ s here.

FDE is the standard way to decide which hypothesis matches an observed future, because intentions
separate over time and the endpoint is where the separation is largest. In this toy all three
futures start from the same point with the same heading; at the first step, 0.5 s in, the left and
right arcs are 0.06 m apart, and at the last step, 6 s in, they are 8.27 m apart. An error
averaged over the horizon dilutes that separation with the early steps where every intention looks
alike. The endpoint does not.

Three different quantities are easy to confuse here, so the code keeps them apart deliberately.
FDE selects (which mode is the winner, which mode is the best). Squared error drives the
regression gradient. ADE and FDE in meters are what gets reported as a metric.

### Min-of-N as k-means with input-conditional centers

The min-of-N objective over a dataset of (context, future) pairs is

$$\mathcal{L}(\theta) = \mathbb{E}_{(x,y)}\Big[\min_{k \in \{1,\dots,K\}}
d\big(f_k(x;\theta),\, y\big)\Big],$$

with $f_1, \dots, f_K$ the $K$ hypothesis heads and $d$ a trajectory distance. The inner minimum
removes the averaging: a hypothesis is charged only for the futures it is closest to, so the three
intentions can be split across three heads instead of merged into one.

This objective is older than trajectory prediction. Strip out the input dependence, make each
$f_k$ a constant vector $\mu_k$, and take $d$ to be squared Euclidean distance; what is left is
$\sum_i \min_k \lVert y_i - \mu_k\rVert^2$, the k-means clustering objective on the set of observed
futures. Min-of-N is k-means with the cluster centers replaced by input-conditional functions.

The training procedure inherits the same structure. Lloyd's algorithm alternates two steps: assign
each point to its nearest center, then move each center toward the points assigned to it. Hard WTA
runs the same alternation once per minibatch, with the exact mean update replaced by one gradient
step and the fixed centers replaced by network outputs. Several properties of k-means carry over
unchanged, and each of them shows up later in this build.

The assignment step carries no derivative. $\arg\min_k$ is integer-valued and piecewise constant in
the parameters: perturb the weights slightly and the winner does not change, except on the
measure-zero set where two hypotheses tie exactly, where it jumps. There is no useful gradient to
take through it, so the assignment is computed, frozen, and treated as a constant for that step. In
PyTorch, freezing is `.detach()`, which returns a tensor sharing the same storage but cut out of
the autograd graph, so no gradient flows back through the computation that produced it.

The objective is non-convex and the result depends on initialization, the same way k-means lands in
different local minima from different seeds. And a center that is nearest to no point receives no
update at all. In k-means that is the empty-cluster problem, and implementations paper over it by
reseeding empty clusters; here it is the dead mode, and there is no reseeding step.

Adding a score head over the $K$ hypotheses makes the model a mixture: $K$ trajectories plus a
distribution over which one applies. Trained with hard assignment it is an approximation to a
mixture density network, a network that outputs the parameters of a mixture distribution instead of
a point estimate. The soft variant below is the corresponding soft-assignment version, which has
the shape of the E-step of expectation-maximization, with the caveat that the weights below are
built from distance rather than squared distance and so are not literally a Gaussian posterior.

### Pooling BEV features around an agent

RoI stands for region of interest, and RoI-align comes from two-stage detection. Fast R-CNN's
RoIPool took a proposal box on a feature map, rounded its real-valued corners to integer cells,
split the result into a fixed grid of bins, rounded those boundaries too, and pooled each bin,
producing a fixed-size descriptor whatever the box size. The two roundings shift the pooled
features relative to the true box by up to half a cell, which classification tolerates and
pixel-accurate masks do not. Mask R-CNN (He et al. 2017,
[arXiv:1703.06870](https://arxiv.org/abs/1703.06870)) removed both roundings by sampling the
feature map at exact fractional locations with bilinear interpolation instead, and called the
result RoIAlign.

This head does the simplest version of that. A BEV backbone (the lift-splat-shoot or BEVFormer
stage earlier in the module) produces a dense feature grid $(C, n_x, n_y)$, with $n_x$ along
$x$/forward and $n_y$ along $y$/left. The head does not consume the whole grid. For each agent it
lays down an $\text{out}\times\text{out}$ grid of sample points spanning $\pm$ radius cells around
that agent's fractional cell, reads the BEV at those points by bilinear interpolation, and returns
the $\text{out}^2$ sampled feature vectors as that agent's token set. The window is specified in
cell units, so it covers the same number of cells at any grid resolution. One shared BEV grid is
broadcast over all $N$ agents, so the token sets differ only in where they were sampled. The agent
count is also the batch dimension everything downstream sees, so $B = N$ in the shapes below.

The sampling is bilinear rather than nearest-neighbor, for the reason the BEV projection stage
gives: nearest-neighbor output is piecewise constant in the sample location, with zero derivative
almost everywhere, while bilinear interpolation is piecewise linear and passes gradient back into
the features being sampled.

PyTorch's `grid_sample` does the sampling, and its coordinate conventions are where this function
breaks quietly. The grid is fed as $(1, C, H, W)$ with $H = n_x$ and $W = n_y$. `grid_sample` reads
the last dimension of the sampling grid as $(x{=}\text{width}, y{=}\text{height})$, the opposite
order from how the agent centers store $(x_{\text{cell}}, y_{\text{cell}})$. So the width
coordinate has to come from $y_{\text{cell}}$ (it indexes $n_y$) and the height coordinate from
$x_{\text{cell}}$ (it indexes $n_x$), a swap. Get it backwards and the function still returns
correctly shaped tokens sampled from real features; it reads the transposed location, which no
shape check can see.

Each axis is normalized to $[-1, 1]$ under the `align_corners=False` convention. Under that
convention cell $i$ of $S$ occupies the interval $[i, i+1)$ in edge units with its center at
$i + 0.5$, and the full extent $[0, S]$ maps to $[-1, 1]$, so

$$g = \frac{2\,(\text{cell} + 0.5)}{S} - 1, \qquad S = n_x \ \text{or}\ n_y.$$

`padding_mode="border"` clamps reads that fall outside the grid to the nearest boundary feature, so
an agent near the edge of the BEV samples a real feature on the outside of its window rather than a
zero that would read as empty space.

A production motion decoder pools more cleverly. UniAD's MotionFormer uses deformable attention,
the mechanism from the BEVFormer stage: each query predicts a few sampling offsets around a
reference point plus a softmax weight per offset, samples there, and weight-sums the results, so
the sampling pattern is learned and agent-dependent instead of a fixed square window. The fixed
window here isolates the interface, a dense grid in and a fixed-length token set out.

### Mode queries and the decoder

The detection assignment introduced object queries: a fixed set of learned embedding vectors, each
a slot the decoder fills with one detection. Mode queries are the same construction one level up.
The head
holds $K$ mode-query embeddings, each a slot for one trajectory hypothesis. A query is a parameter,
not a function of the input; the scene enters only through cross-attention. So the query is what
the mode is before it has seen anything, and cross-attention specializes it to this agent.

The queries have to be initialized distinct (here `randn(K, dim) * 0.02`). Consider what happens if
they are all equal. Every operation downstream of the query is shared across modes and equivariant
to permuting the mode axis: self-attention over the $K$ modes, cross-attention over the same RoI
tokens, the same MLP, the same trajectory and score heads. Feed $K$ identical queries into that and
the $K$ outputs are identical, the min-of-N winner is an arbitrary tie-break, and the gradient
arriving at each query is identical too, so the queries stay equal at every subsequent step.
Distinct initialization breaks the symmetry. It is a precondition and nothing more; on its own it
does not keep all modes alive.

Each decoder layer runs three sublayers, each a residual update on a LayerNorm-ed input,
$q \leftarrow q + \text{sublayer}(\text{LayerNorm}(q))$. First self-attention over the $K$ mode
queries, which is how the modes see each other and can spread out instead of duplicating a
hypothesis. Then cross-attention of the $K$ queries over the agent's $\text{out}^2$ RoI tokens,
which is where scene context enters. Then a per-mode MLP. There is no causal mask, since neither
the modes nor the trajectory steps are produced in sequence: the whole $(K, T, 2)$ output comes out
in one shot.

Each mode's trajectory head emits $T$ per-step displacements $\Delta_1, \dots, \Delta_T$, which are
cumulatively summed along time into absolute agent-centric positions,
$\hat y_t = \sum_{s \le t} \Delta_s$. In matrix form $\hat y = L\Delta$ with $L$ the
lower-triangular matrix of ones, a fixed linear map that adds no parameters. Two things follow from
predicting displacements rather than positions. The quantity the final linear layer has to produce
stays small and uniform in scale over the horizon: at 1.5 m/s and 0.5 s per step a displacement is
0.75 m at every $t$, while an absolute position grows to about 9 m by the 6 s horizon. Because $L$
is triangular, a gradient on the position at step $t$ reaches every displacement at or before $t$,
so an error late in the trajectory corrects the whole prefix instead of only the last step.

A separate score head maps each mode query to one number, giving $(B, K)$ raw logits, unnormalized
scores that become a distribution over modes after a softmax.

### The winner-take-all loss

Both paths of the loss share the same per-mode endpoint distances and the same classification term.
For a batch of $B$ agents with predictions $\hat y \in \mathbb{R}^{B \times K \times T \times 2}$
and observed futures $y \in \mathbb{R}^{B \times T \times 2}$,

$$\text{FDE}_{b,k} = \lVert \hat y_{b,k,T} - y_{b,T}\rVert_2,
\qquad k^{\star}_b = \arg\min_k \text{FDE}_{b,k},$$

and $k^{\star}$ is detached, for the reason the k-means section gives: it is an integer index with
no derivative worth taking.

The hard path is the canonical min-of-N. Regress the winner's whole trajectory with mean squared
error,

$$\mathcal{L}_{\text{reg}} = \frac{1}{2BT}\sum_{b=1}^{B}
\big\lVert \hat y_{b,k^{\star}_b} - y_b \big\rVert_F^2 .$$

Only the winning mode appears in that expression, so only it carries regression gradient; the other
$K-1$ modes receive nothing from this term on this step.

Selection uses Euclidean distance while regression uses squared error, and the mismatch is
deliberate. The Euclidean norm $\lVert u \rVert$ has gradient $u/\lVert u \rVert$, which is
undefined at $u = 0$, exactly the point a converging prediction approaches. Squared error is smooth
there. The test suite includes a gradcheck, which compares the analytic gradient autograd computes
against a finite-difference estimate obtained in double precision by perturbing each input by a
small $\epsilon$ and re-evaluating the function; a kink makes the two disagree and the check fails.
The cost of using squared error is that the loss value is in $\text{m}^2$ and cannot be read as an
error in meters, which is why minADE and minFDE are reported separately from it.

The classification term trains the score head. It is the cross-entropy of the score logits against
the hard winner index,

$$\mathcal{L}_{\text{cls}} = -\frac{1}{B}\sum_{b=1}^{B}
\log \mathrm{softmax}(s_b)_{k^{\star}_b},$$

with $s_b \in \mathbb{R}^K$ the raw logits. They go in raw because cross-entropy applies log-softmax
internally; pre-softmaxing applies it twice, which flattens the distribution and the gradient with
it. The term teaches the score head to name whichever mode committed to the observed future, so at
inference, with no ground truth available to select with, the top-scored mode is the model's single
guess. The total is $\mathcal{L}_{\text{reg}} + \lambda_{\text{cls}} \mathcal{L}_{\text{cls}}$,
with $\lambda_{\text{cls}}$ the `cls_weight` argument.

### Soft assignment and the temperature

The soft path replaces the hard winner with a distribution over modes:

$$w_{b,k} = \mathrm{softmax}_k\big(-\text{FDE}_{b,k}/\tau\big),
\qquad \ell_b = \sum_{k} w_{b,k}\,\text{MSE}_{b,k},
\qquad \mathcal{L}_{\text{reg}} = \frac{1}{B}\sum_{b} \ell_b,$$

where $\text{MSE}_{b,k}$ is the mean squared error of mode $k$'s full trajectory against $y_b$. The
temperature $\tau$ controls how peaked $w$ is, the same knob as in the contrastive and
self-distillation losses earlier in the course. As $\tau \to 0$ the weights go one-hot at the
smallest FDE and the soft loss becomes the hard loss. As $\tau \to \infty$ they go uniform, every
mode is regressed toward the same target, and the head is back to mode averaging with $K$ copies.
In between, every mode gets gradient, weighted toward the ones already close.

The weights are not detached, unlike the winner index, so $w$ itself carries gradient and the
regression term has a second effect beyond pulling each mode toward the target. Fixing one agent
and dropping the $b$ index, write $s_k$ for mode $k$'s MSE and $\bar s = \sum_k w_k s_k$ for the
weighted average. Differentiating $\ell = \sum_k w_k s_k$ through the softmax, holding the $s_k$
fixed to isolate the weight path, gives

$$\frac{\partial \ell}{\partial \text{FDE}_j} = -\frac{w_j}{\tau}\,(s_j - \bar s).$$

A mode doing worse than the weighted average has a negative derivative here, so a gradient step
increases its endpoint distance: it is pushed away from the target and its share of the
responsibility shrinks. A mode doing better than average is pulled in and its share grows. The soft
path therefore sharpens its own assignment as it trains, on top of the direct pull each
$\text{MSE}_k$ term applies.

The classification term is identical in both paths: cross-entropy against the hard $\arg\min$
winner, whatever the regression term is doing.

### Dead modes and the annealing recipe

Hard WTA inherits the empty-cluster problem. A mode that never wins gets no regression gradient, so
it never moves toward any data, so it keeps never winning. Its regression gradient is not merely
small; it is exactly zero. The classification term still touches it, through its own logit, but
that only teaches its score head to say "not me" and does nothing about where its trajectory sits.
With $K$ modes and only a few distinct intentions in the batch, the winners carve up the intentions
between them and the rest stay where they were initialized.

The annealed recipe decays $\tau$ linearly from `tau0` to 0 over the first `anneal_frac` of
training (3.0 and 0.6 in `config.py`), then runs hard for the remainder. Early on every mode gets
gradient and the modes spread out to cover the fan of futures; by the time the loss goes hard, each
mode already owns a region of the future space, so none is left to starve. This is soft k-means
annealed into hard k-means, the same move as deterministic annealing in vector quantization.

Two other mitigations are standard. Evolving WTA (Makansi et al. 2019,
[arXiv:1906.03631](https://arxiv.org/abs/1906.03631)) keeps the assignment hard but supervises the
top $M$ modes instead of the top 1, decaying $M$ from $K$ down to 1 over training, so early steps
feed every mode and late steps recover min-of-N. Goal anchors attack the feedback loop from the
other side, by making the assignment independent of what the network currently predicts. TNT (Zhao
et al. 2020, [arXiv:2008.08294](https://arxiv.org/abs/2008.08294)) samples candidate target points
off the lane graph, scores them, and regresses a trajectory conditioned on each. MTR runs k-means
over the endpoints of every ground-truth trajectory in the training set to get a fixed set of
intention points (64 per agent category), attaches one query to each, and assigns a training
example to the query whose intention point is nearest to the observed endpoint. That assignment
reads the anchor and the label, never the current prediction, so a mode cannot starve itself out:
whenever an example lands in its region of the endpoint space it is supervised, however badly it is
predicting.

Hard assignment is therefore the standard choice at scale rather than a known-broken one. Large
prediction systems get both anchoring and per-scene diversity for free: different scenes activate
different modes, so every mode receives gradient somewhere in the dataset even though any single
scene supervises one. The collapse shown here is a real risk that bites hardest in the amplified
case this toy sets up, a single ambiguous context repeated with no per-scene variation and no
anchors.

### Best-of-K oracle metrics

The reported metrics all take a minimum over the $K$ hypotheses before averaging over agents:

$$\text{minADE}_K = \min_k \text{ADE}_k, \qquad
\text{minFDE}_K = \min_k \text{FDE}_k, \qquad
\text{MR}_K = \mathbb{1}\big[\text{minFDE}_K > \delta\big],$$

with the miss threshold $\delta = 2$ m by default here. These are the Argoverse/nuScenes-style
$\text{minADE}_K$, $\text{minFDE}_K$ and miss-rate metrics.

One implementation detail departs from the literal definitions. `min_ade` picks the mode by minFDE
and reports that mode's ADE, rather than taking the minimum of ADE over modes. The two agree
whenever the same mode is best under both, and where they differ this version is the larger of the
two, so it never flatters the model. It is written that way to score the same mode the
winner-take-all loss trained.

They are called oracle metrics because the minimum assumes an oracle that picks the right mode
after the answer is known. Two consequences follow. They reward coverage but do not penalize
redundancy: $K$ near-identical modes score as well as one mode, and a model that hedges is graded
on its luckiest hypothesis. And they say nothing about whether the model can pick the right
hypothesis without the oracle, which is the only thing a downstream planner can use. The score head
exists for that: at inference the top-scored mode is the committed answer, and these metrics do not
measure its error.

### Marginal and joint prediction

This head predicts one agent at a time, conditioned on its own pooled context, so it models
$p(y_i \mid x)$ separately for each agent $i$. That is a marginal. What a planner needs is the
joint $p(y_1, \dots, y_n \mid x)$, and a product of marginals is not the joint whenever the agents'
futures are dependent, which is precisely the case that matters: two cars cannot occupy the same
piece of road at the same instant. Sampling one future per agent from independent marginals can
therefore produce a scene where the predicted trajectories intersect, a configuration with
essentially zero probability under the true joint. Joint prediction adds agent-to-agent attention,
so the modes of different agents read each other and the model can express the dependence. UniAD's
MotionFormer is the joint version.

### The modular stack and UniAD's query passing

The classical stack is modular: sensors feed detection, detection feeds tracking, tracking feeds
prediction, prediction feeds planning. Each hand-off is a hard, non-differentiable interface, a
list of boxes, a set of tracks, a set of trajectories. Thresholding a detection score to decide
what enters the track list is a step function with no derivative to send backward. Errors compound
across the stages, and since no gradient flows back, each perception module is trained against its
own proxy objective and cannot learn what the planner actually needs from it.

UniAD replaces each hard hand-off with query passing over a shared BEV backbone. Nothing is
thresholded into a list; the interface between stages is a set of query vectors, the same kind of
learned slot as the mode queries here. Track queries feed motion queries feed occupancy queries
feed planning queries, every stage a transformer reading the previous stage's queries, so the whole
stack is one differentiable network and every task is trained toward the final planning objective.
The RoI-align and mode-query decoder built here is MotionFormer in miniature.

### From dense grids to vectorized scenes

VAD (Jiang et al. 2023, [arXiv:2303.12077](https://arxiv.org/abs/2303.12077)) drops the dense BEV
occupancy grid in favor of a vectorized scene: agents and map elements are polylines and instance
vectors, not pixels. A polyline is a lane boundary, a crosswalk or an agent's past track written as
an ordered sequence of connected line segments; the encoder turns each polyline into a single
vector and lets the vectors attend to each other, instead of rasterizing the map into image
channels and running a convolutional network over mostly empty pixels. VectorNet (Gao et al. 2020,
[arXiv:2005.04259](https://arxiv.org/abs/2005.04259)) established that representation for HD maps
and agent histories, and MultiPath++ (Varadarajan et al. 2022,
[arXiv:2111.14973](https://arxiv.org/abs/2111.14973)) made the same move on the prediction side.
VADv2 (Chen et al. 2024, [arXiv:2402.13243](https://arxiv.org/abs/2402.13243)) turns the planner
into a probabilistic model over a discrete vocabulary of planning actions. DriveTransformer (Jia et
al. 2025, [arXiv:2503.07656](https://arxiv.org/abs/2503.07656)) drops the sequential task ordering:
agent, map, and planning queries run in parallel and attend to each other directly, instead of the
track-then-motion-then-plan chain.

### The open-loop metric is a debugging tool, not a score

The planning head in all of these is a regression or imitation MLP: it predicts the ego trajectory
to match the human demonstration. That is behavior cloning, plain supervised learning on (state,
expert action) pairs. There is no reward, no rollout, and no explicit optimization over a cost
function at inference; the network is trained to reproduce the log.

Behavior cloning has a failure that appears only once the policy drives. Its training states come
from the human's driving, but at test time the policy's own actions determine the next state, so a
small error moves it to a state the training distribution did not cover, where it is less reliable,
and the deviation grows over the episode. That distribution shift is invisible to any evaluation
which never lets the policy act.

Open-loop evaluation is exactly that kind of evaluation: replay a logged scene, predict the ego
trajectory once, and measure its L2 distance to what the human drove. Two 2023 results showed how
little the number means on nuScenes. AD-MLP (Zhai et al. 2023,
[arXiv:2305.10430](https://arxiv.org/abs/2305.10430)) built an MLP that sees only ego status, its
own velocity, acceleration and past trajectory, with no camera or lidar input at all, and matched
the perception-conditioned planners on nuScenes open-loop planning, cutting average L2 by about 20%
while doing worse on collision rate. "Is Ego Status All You Need for Open-Loop End-to-End
Autonomous Driving?" (Li et al. 2023,
[arXiv:2312.03031](https://arxiv.org/abs/2312.03031)) traced the cause: 73.9% of nuScenes is
straight-line driving, so holding the current velocity and heading answers most of the benchmark
and perception barely changes the result. nuScenes open-loop L2 therefore mostly measures whether
the ego's current motion extrapolates.

Two replacements are in use. NAVSIM (Dauner et al. 2024,
[arXiv:2406.15349](https://arxiv.org/abs/2406.15349)) stays non-reactive, meaning the logged agents
replay their recorded behavior and never respond to the ego, but it drops displacement error in
favor of metrics computed by simulating the ego's own predicted trajectory forward in a BEV
abstraction: at-fault collision, drivable-area compliance, progress, time-to-collision and comfort,
combined into a single score. It is built on OpenScene, a resampled redistribution of the nuPlan
dataset. Bench2Drive runs fully closed-loop in the CARLA simulator, where the planner's outputs
drive the car and errors compound over the episode.

### Where this goes next

This head is a behavioral model of the future: given the scene, predict what agents will do. The
world-models assignment builds a sensory model of the future instead: given the current observation
and an action, predict the next observation. A full driver wants both, the behavioral prediction of
other agents and the sensory rollout of its own actions. The vision-language-action assignment
trades geometric precision for instruction following: planners like DriveLM and DriveVLM reason
over language ("yield to the pedestrian", "the light is red") and emit higher-level decisions,
accepting coarser trajectories in exchange for following instructions and explaining themselves.

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
the start and shows the target. The tests cover output shapes, a gradcheck on `roi_align_bev` and on
both `wta_loss` paths, that the modes specialize, an overfit on distinct per-agent contexts, the
headline contrast that a $K$-mode annealed head beats a single mode, and a scan confirming the
attention comes from the course's own multi-head attention rather than a `torch.nn` shortcut.

What you should see when you run this. The toy is CPU only, seconds per test. `pred_toy_scene` with
`shared_context=True` places several agents at one BEV cell with one feature vector (which encodes
only speed, not intention) but fans their futures over three balanced intentions. The horizon
lateral displacement is $\pm 4.14$ m and the closest pair of intention endpoints is 4.38 m apart, so
the three futures are well separated at the endpoint the winner selection uses. Measured with the
real attention head and the tiny config here, over random init seeds:

| recipe | minADE | what happens |
|---|---|---|
| single mode (K=1) | 1.168 m, seed-independent | forced to the conditional mean of the three intentions |
| hard WTA (K=6) | 0.01-0.02 m on 5 of 8 seeds, 0.56 m on 3 of 8 | sometimes all 3 modes survive, sometimes 2 die |
| soft-to-hard annealed (K=6) | 0.01-0.06 m on every seed | all 3 intentions covered by distinct modes |

The single-mode 1.168 m is seed-independent because it is fixed by the scene rather than by
optimization. With $K=1$ the minimum over modes is vacuous, so minADE is just that one mode's ADE,
and once it has converged to the conditional mean of the three intentions the geometry pins the
number: 1.610 m for each of the four turning agents and 0.283 m for each of the two straight ones,
1.168 m averaged over the batch. The hard-WTA collapse is fragile rather than uniform: with the real
attention head, mode self-attention gives the modes a path to spread out, so they often escape the
dead-mode trap, and whether they do depends on init. The headline test asserts only the
init-independent facts, that single-mode minADE sits above 1 m, that the annealed recipe reaches
under 0.15 m on every seed it tries, and that coverage beats averaging by a wide margin; it leaves
the intermittent hard-WTA collapse to the measured table and the `compare_recipes` figure. `make viz`
writes `modes_annealed.png` and `compare_recipes.png` to `out/`; the comparison shows K=1 collapsed
to one short averaged path, hard K=6 covering some intentions while dead modes wander off, and
annealed K=6 covering all three arcs.

These are toy demonstrations of mode averaging and the WTA collapse on one shared ambiguous input,
the amplified worst case. They are not evidence about real-world prediction accuracy, and the oracle
metrics reward coverage rather than driving quality, the same caveat the open-loop-metric section
makes about nuScenes L2.

## Further reading

- Hu et al. 2023, UniAD, [arXiv:2212.10156](https://arxiv.org/abs/2212.10156).
- Shi et al. 2022, MTR, [arXiv:2209.13508](https://arxiv.org/abs/2209.13508).
- He et al. 2017, Mask R-CNN (RoIAlign), [arXiv:1703.06870](https://arxiv.org/abs/1703.06870).
- Makansi et al. 2019, "Overcoming Limitations of Mixture Density Networks" (EWTA),
  [arXiv:1906.03631](https://arxiv.org/abs/1906.03631).
- Zhao et al. 2020, TNT, [arXiv:2008.08294](https://arxiv.org/abs/2008.08294).
- Gao et al. 2020, VectorNet, [arXiv:2005.04259](https://arxiv.org/abs/2005.04259).
- Varadarajan et al. 2022, MultiPath++, [arXiv:2111.14973](https://arxiv.org/abs/2111.14973).
- Jiang et al. 2023, VAD, [arXiv:2303.12077](https://arxiv.org/abs/2303.12077).
- Chen et al. 2024, VADv2, [arXiv:2402.13243](https://arxiv.org/abs/2402.13243).
- Jia et al. 2025, DriveTransformer, [arXiv:2503.07656](https://arxiv.org/abs/2503.07656).
- Zhai et al. 2023, "Rethinking the Open-Loop Evaluation of End-to-End Autonomous Driving in
  nuScenes" (AD-MLP), [arXiv:2305.10430](https://arxiv.org/abs/2305.10430).
- Li et al. 2023, "Is Ego Status All You Need for Open-Loop End-to-End Autonomous Driving?",
  [arXiv:2312.03031](https://arxiv.org/abs/2312.03031).
- Dauner et al. 2024, NAVSIM, [arXiv:2406.15349](https://arxiv.org/abs/2406.15349).
