# A11.5e - unified perception, prediction, planning

A self-driving car has to know where the other road users will be over the next few seconds before
it can plan its own path. A motion-prediction head answers that question: for each agent (another
car, a cyclist, a pedestrian) it reads the scene around the agent and outputs where the agent will
be at each future timestep. The difficulty is that the future of a driving agent is multimodal:
the same observed state (position, speed) is consistent with several distinct intentions, such as
turn left, go straight, or turn right. A single trajectory regressor trained on mean error is
pulled to the conditional mean of those futures, a path through the middle of every option that
matches none of them. The fix is to predict $K$ trajectory hypotheses and supervise only the
closest one per sample, the winner-take-all loss.

Build one stage of the end-to-end driving stack: a multimodal motion-prediction head that maps a
bird's-eye-view (BEV) feature grid at an agent's location to $K$ future trajectory hypotheses. The
dense BEV grid is pooled around the agent with a small region-of-interest align (RoI-align), $K$
learned mode queries cross-attend over those features through a few decoder layers, and each mode
regresses a full trajectory plus a confidence score, trained with the winner-take-all loss and its
soft-to-hard annealed variant. The rest of the stack (the modular pipeline, UniAD, VAD,
DriveTransformer) and the critique of open-loop metrics are context, not part of the build. All
graded tests run on CPU in seconds.

Required reading before starting:
- Hu et al. 2023, "Planning-oriented Autonomous Driving" (UniAD),
  [arXiv:2212.10156](https://arxiv.org/abs/2212.10156).
- Shi et al. 2022, "Motion Transformer with Global Intention Localization and Local Movement
  Refinement" (MTR), [arXiv:2209.13508](https://arxiv.org/abs/2209.13508) (a multimodal
  trajectory head at scale, trained with hard assignment of each example to one mode).

## Lecture notes

### What a motion-prediction head is for

#### The task

The earlier stages of this module turn camera images into a BEV feature grid: a top-down grid
around the ego car (the car doing the driving) with one feature vector per patch of ground. The
[Lift-Splat-Shoot](../a11_5b_lift_splat_shoot/README.md) and
[BEVFormer](../a11_5c_bevformer/README.md) view transforms produce that grid, and the
[occupancy](../a11_5d_occupancy/README.md) head reads it to say which space is filled now. A planner
needs more than the present. To choose a path for the next few seconds it needs to know where every
other agent will be over those seconds.

A concrete instance: a car is 10 m ahead of the ego car, approaching an intersection at 1.5 m/s.
The prediction head is asked for that car's position every 0.5 s for the next 6 s, twelve 2-D
points in the car's own frame. The input is the BEV feature grid, which encodes where the car is
and how fast it moves. The output feeds the planner, which must avoid every place the car might
plausibly be.

#### Why one regressor fails

Nothing in the observation says whether the car will turn left, go straight, or turn right:
position and speed are the same under all three. The future is multimodal given the input. Train a
single trajectory regressor $f$ on such data with mean squared error, and it lands in the middle of
the three futures. This follows from the objective in one line; it is not an empirical tendency.
Fix the input $x$, write $y$ for the observed future, and let $\mu = \mathbb{E}[y \mid x]$ be the
conditional mean. Adding and subtracting $\mu$ inside the expected squared error splits it:

$$\mathbb{E}\big[\lVert f(x) - y\rVert^2 \mid x\big]
= \lVert f(x) - \mu \rVert^2 + \mathbb{E}\big[\lVert y - \mu\rVert^2 \mid x\big],$$

the cross term vanishing because $\mathbb{E}[y - \mu \mid x] = 0$. The second term is the spread of
the true futures and does not involve $f$ at all, so the objective is minimized exactly at
$f(x) = \mu$ and its floor is that spread. This is the same decomposition that makes the
conditional mean the minimum-mean-square estimator in classical estimation. A squared-error
regressor with enough capacity has no option other than to average.

For a unimodal future, such as a Kalman filter's Gaussian prediction, the mean is the right answer.
For three separated futures it is wrong, and the next section shows how wrong on numbers.

#### Predicting several trajectories

Stop predicting one trajectory. Predict $K$ of them, and at training time supervise only the one
closest to the observed future. Each hypothesis is then free to specialize on a different intention
without being pulled toward the others' average. This is the winner-take-all (WTA) loss, also
called the min-of-N loss. Each of the $K$ predicted trajectories is called a mode, and the one
closest to the observed future is the winner. A score head adds one number per mode, trained to
say which mode will be the winner, so that at inference, with no observed future to compare
against, the model can still name its best guess.

#### Why it matters

Multimodal heads trained with hard assignment of each example to one mode are the standard
design for motion prediction. MTR (Shi et al. 2022) has 64 intention queries per agent category
and assigns each example to the query whose fixed intention point is nearest to the observed
endpoint (a variant described later), and UniAD's motion stage
(MotionFormer) is a mode-query decoder of the same kind inside a network that is trained end to
end for planning. The head built here is that decoder at toy scale.

### Our three-intention fan (one agent at an intersection)

To see the mechanism by hand we use the scene the tests use, reduced to one agent. We call it our
three-intention fan. The agent starts at the origin of its own frame ($+x$ forward, $+y$ left) and
drives at $v = 1.5$ m/s for a 6 s horizon under a constant-speed, constant-turn-rate model, with
turn rate $\omega = \pm 1/6$ rad/s for the turns and $\omega = 0$ for straight. A left turn
follows $x(t) = (v/\omega)\sin(\omega t)$, $y(t) = (v/\omega)(1 - \cos \omega t)$; a right turn is
its mirror image; straight is $x = vt$, $y = 0$. The three endpoints are $(7.57, +4.14)$,
$(9.00, 0.00)$ and $(7.57, -4.14)$ meters. The three intentions are equally likely.

Their mean is $(8.05, 0.00)$: 0.95 m short of the straight endpoint and 4.16 m from either turn.
Averaging a hard left and a hard right does not recover the straight path, because both arcs trade
forward progress for lateral motion, so the mean falls short of straight while still missing both
turns. On a real intersection the averaged path can also run over whatever lies between the
options, such as a curb or a median.

![Our three-intention fan. Left, the three true futures and their conditional mean, which ends at (8.05, 0) on none of them. Right, a two-mode configuration in which one mode follows the right turn and the other sits on the mean of the left turn and straight.](assets/fan.png)

The left panel shows the single-mode answer. The right panel shows a configuration that comes up
again below: two modes in use, one on the right turn and one on the average of the other two
intentions. Winner-take-all training can get stuck there, as a later section explains.

Our three-intention fan keeps what this assignment is about: identical input with several futures,
the averaging a single mode is forced into, and the competition between modes for the futures. It
drops the scene itself (lanes, other agents, the history of the agent's motion), which in a real
dataset often does say which way the agent will go, and the variety of contexts across a dataset.
The toy scene in the tests is this fan with six agents, two per intention, all placed in the same
BEV cell with the same feature vector.

> **Exercise 1 (what a single mode scores).** A single-mode head on our three-intention fan
> converges to the conditional mean. Using the endpoints above, compute its final displacement
> error (the endpoint distance) for an agent that turns and for one that goes straight, and the
> average over a balanced set of agents.

<details>
<summary>Answer</summary>

The mean endpoint is $(8.05, 0)$. For a turn, the endpoint distance is
$\sqrt{(8.05 - 7.57)^2 + 4.14^2} = \sqrt{0.476^2 + 4.137^2} \approx 4.16$ m. For straight it is
$9.00 - 8.05 = 0.95$ m. Two of every three agents turn, so the average is
$(2 \cdot 4.16 + 0.95)/3 \approx 3.09$ m. A planner reading this prediction is wrong by more than
4 m about two thirds of the agents, and even for the straight-going third it is wrong by almost a
meter, because the mean of three futures is not one of them.

</details>

### Overview of the rest of the notes

The sections below build the head and its loss, then place it in the driving stack.

1. Distances between trajectories. ADE and FDE reduce a trajectory error to one number; FDE decides
   which mode is the winner.
2. Winner-take-all as k-means. The loss is the k-means objective with input-dependent centers, and
   training is online k-means, one gradient step per minibatch. Its failure mode, a center that
   never wins, is the empty-cluster problem.
3. The head. RoI-align pools the BEV grid around the agent by bilinear sampling, as in inverse
   perspective mapping. Mode queries are the object queries of the detection assignment's
   set-prediction detector (DETR), one per trajectory hypothesis, and a transformer decoder fills
   them in.
4. The loss. The hard winner-take-all loss, the soft version weighted by a softmax over distances
   (shaped like the E-step of expectation-maximization), and the annealing schedule from soft to
   hard, borrowed from deterministic annealing, a clustering method that starts with soft
   assignments at a high temperature and lowers it.
5. Evaluation. Best-of-$K$ metrics and what they do not measure.
6. The driving stack. Marginal and joint prediction, UniAD's query passing, vectorized scenes, and
   why open-loop planning metrics mislead.

Five exercises, placed after the sections they test, are meant to be worked by hand.

### Measuring the distance between two trajectories

"Closest" needs a number. A trajectory here is a sequence of $T$ two-dimensional positions
$y_1, \dots, y_T$ in the agent's own frame, one per future timestep. Comparing a prediction
$\hat y$ against an observed future $y$ means reducing $T$ per-step Euclidean errors to one scalar,
and the prediction literature uses two reductions, both in meters:

$$\text{ADE}(\hat y, y) = \frac{1}{T}\sum_{t=1}^{T} \lVert \hat y_t - y_t \rVert_2,
\qquad \text{FDE}(\hat y, y) = \lVert \hat y_T - y_T\rVert_2.$$

ADE is the average displacement error over the whole horizon; FDE is the final displacement error,
the endpoint alone. Time is 1-based throughout: step $t$ is $t\,\Delta t$ seconds into the future,
stored at array index $t-1$, and $\Delta t = 0.5$ s here, so $T = 12$.

FDE is the standard way to decide which mode matches an observed future, because intentions
separate over time and the endpoint is where the separation is largest. In our three-intention fan
all three futures start from the same point with the same heading; at the first step, 0.5 s in, the
left and right arcs are 0.06 m apart, and at the last step, 6 s in, they are 8.27 m apart. An error
averaged over the horizon dilutes that separation with the early steps where every intention looks
alike. The endpoint does not.

Three different quantities are easy to confuse here, so the notes keep them apart. FDE selects
(which mode is the winner). Squared error drives the regression gradient. ADE and FDE in meters are
what gets reported as a metric.

### Winner-take-all as k-means with input-conditional centers

The winner-take-all objective over a dataset of (context, future) pairs is

$$\mathcal{L}(\theta) = \mathbb{E}_{(x,y)}\Big[\min_{k \in \{1,\dots,K\}}
d\big(f_k(x;\theta),\, y\big)\Big],$$

with $f_1, \dots, f_K$ the $K$ mode outputs of the network and $d$ a trajectory distance. The inner
minimum removes the averaging: a mode is charged only for the futures it is closest to, so the
three intentions can be split across three modes instead of merged into one.

This objective is older than trajectory prediction. Strip out the input dependence, make each
$f_k$ a constant vector $\mu_k$, and take $d$ to be squared Euclidean distance; what is left is
$\sum_i \min_k \lVert y_i - \mu_k\rVert^2$, the k-means clustering objective on the set of observed
futures. Winner-take-all is k-means with the cluster centers replaced by input-conditional
functions.

The training procedure has the same structure. Lloyd's algorithm, the usual batch k-means
algorithm, alternates two steps over the whole dataset: assign each point to its nearest center,
then move each center to the mean of the points assigned to it. Online k-means (MacQueen 1967)
instead takes one point at a time, assigns it, and moves only its center a small step toward it;
Bottou and Bengio (1995) showed this is stochastic gradient descent on the k-means objective, and
mini-batch k-means (Sculley 2010) does the same on a small batch. Hard winner-take-all training is
mini-batch k-means with the centers replaced by network outputs: each step assigns every example
in the batch to its winner and takes one gradient step that moves the winners toward their
examples. One difference: the assignment reads only the endpoint, while the update regresses the
whole trajectory. Three properties of k-means carry over.

First, the assignment step has no useful derivative. $\arg\min_k$ is integer-valued and piecewise
constant in the parameters: perturb the weights slightly and the winner does not change, except on
the measure-zero set where two modes tie exactly, where it jumps. So the assignment is computed,
frozen, and treated as a constant for that step. In PyTorch, freezing is `.detach()`, which returns
a tensor sharing the same storage but cut out of the autograd graph, so no gradient flows back
through the computation that produced it.

Second, the objective is non-convex and the result depends on initialization, the same way k-means
lands in different local minima from different starting centers.

Third, a center that is nearest to no point receives no update at all. In k-means that is the
empty-cluster problem, and implementations work around it by reseeding empty clusters. Here it is
what we call a dead mode, and there is no reseeding step. Unlike a k-means center, a dead mode
still moves, as the section on dead modes explains. (A dead mode is different from what the
literature usually calls mode collapse, where several hypotheses converge on the same output.)

Adding a score head over the $K$ modes makes the model a mixture: $K$ trajectories plus a
distribution over which one applies. Trained with hard assignment it is an approximation to a
mixture density network, a network that outputs the parameters of a mixture distribution instead of
a point estimate. The soft variant below is the corresponding soft-assignment version, which has
the shape of the E-step of expectation-maximization, with the caveat that its weights are built
from distance rather than squared distance and so are not literally a Gaussian posterior.

> **Exercise 2 (a dead center in k-means).** Keep only the lateral endpoint of each intention in
> our three-intention fan: the points $-4.14$ (right), $0$ (straight), $+4.14$ (left) on a line.
> Run Lloyd's algorithm by hand with three centers initialized at $0.5$, $1.0$, and $10$. Where do
> the centers end up, what is the final objective $\sum_i \min_k (y_i - \mu_k)^2$, and how does the
> result compare with the right panel of the fan figure?

<details>
<summary>Answer</summary>

Assignment: $-4.14$ is nearest to $0.5$ (distance 4.64), $0$ is nearest to $0.5$, and $4.14$ is
nearest to $1.0$ (distance 3.14, against 5.86 for the center at 10). Update: the first center moves
to the mean of $\{-4.14, 0\}$, which is $-2.07$; the second moves to $4.14$; the third has no
points and stays at $10$. On the next assignment nothing changes ($0$ is 2.07 from $-2.07$ and 4.14
from $4.14$), so the algorithm has converged. The objective is $2 \cdot 2.07^2 \approx 8.57$,
against $0$ for the solution with one center on each point. The center at 10 is dead, and one live
center averages two intentions. This is the right panel of the fan figure in one dimension, with
the sides mirrored: one mode on one turn, one mode on the mean of the other turn and straight.
Lloyd's algorithm never increases the objective, and here it has reached a fixed point, so it
cannot leave this configuration.

</details>

### Pooling BEV features around an agent

The head needs a fixed-size description of each agent's surroundings. It gets one by sampling the
BEV grid in a small window around the agent.

RoI stands for region of interest, and RoI-align comes from two-stage detection. Fast R-CNN's
RoIPool took a proposal box on a feature map, rounded its real-valued corners to integer cells,
split the result into a fixed grid of bins, rounded those boundaries too, and pooled each bin,
producing a fixed-size descriptor whatever the box size. The two roundings shift the pooled
features relative to the true box by up to half a cell, which classification tolerates and
pixel-accurate masks do not. Mask R-CNN (He et al. 2017,
[arXiv:1703.06870](https://arxiv.org/abs/1703.06870)) removed both roundings by sampling the
feature map at exact fractional locations with bilinear interpolation instead, and called the
result RoIAlign.

This head does the simplest version of that. A BEV backbone (the Lift-Splat-Shoot or BEVFormer
stage earlier in the module) produces a dense feature grid $(C, n_x, n_y)$, with $n_x$ along
$x$/forward and $n_y$ along $y$/left. The head does not consume the whole grid. For each agent it
lays down an $\text{out}\times\text{out}$ grid of sample points spanning $\pm$ radius cells around
that agent's fractional cell, reads the BEV at those points by bilinear interpolation, and returns
the $\text{out}^2$ sampled feature vectors as that agent's token set (a token is one feature vector
that attention reads). The window is specified in cell units, so it covers the same number of cells
at any grid resolution. One shared BEV grid is broadcast over all $N$ agents, so the token sets
differ only in where they were sampled. The agent count is also the batch dimension everything
downstream sees, so $B = N$ in the shapes below.

The sampling is bilinear rather than nearest-neighbor for the reason given in the
[camera-geometry](../a11_5a_camera_geometry_bev/README.md) notes on inverse perspective mapping:
nearest-neighbor output is piecewise constant in the sample location, with zero derivative almost
everywhere, while bilinear interpolation is piecewise linear and passes gradient back into the
features being sampled.

PyTorch's `grid_sample` does the sampling, and an error in its coordinate conventions produces no
error message. The grid is fed as $(1, C, H, W)$ with $H = n_x$ and $W = n_y$. `grid_sample` reads
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

> **Exercise 3 (normalized sample coordinates).** A BEV grid has $n_x = 8$ and $n_y = 16$. An
> agent sits at $(x_{\text{cell}}, y_{\text{cell}}) = (2.5, 10.0)$, and the window has radius 1
> cell and $\text{out} = 3$. Give the normalized pair (width, height) that goes into the sampling
> grid for the center sample and for the sample one cell forward and one cell left of it. Then
> take the bug the code invites, stacking the two normalized coordinates as (height, width) instead
> of (width, height), and say which cell the center sample reads.

<details>
<summary>Answer</summary>

Center sample: width $g = 2(10.0 + 0.5)/16 - 1 = 0.3125$ (from $y_{\text{cell}}$ over $n_y$) and
height $g = 2(2.5 + 0.5)/8 - 1 = -0.25$ (from $x_{\text{cell}}$ over $n_x$), so the pair is
$(0.3125, -0.25)$. One cell forward and one left is
$(x_{\text{cell}}, y_{\text{cell}}) = (3.5, 11.0)$: width $2(11.5)/16 - 1 = 0.4375$, height
$2(4.0)/8 - 1 = 0$, so $(0.4375, 0)$. Stacked in the wrong order, the center pair is
$(-0.25, 0.3125)$: `grid_sample` takes $-0.25$ as the width coordinate and $0.3125$ as the height
coordinate. Inverting $g = 2(\text{cell} + 0.5)/S - 1$ gives $y_{\text{cell}} = 0.75 \cdot 8 - 0.5
= 5.5$ over $n_y = 16$ and $x_{\text{cell}} = 1.3125 \cdot 4 - 0.5 = 4.75$ over $n_x = 8$. The read
lands at $(4.75, 5.5)$ instead of $(2.5, 10.0)$, about 5 cells away and inside the grid. The output
has the right shape and contains real features, from the wrong place, so no shape check or range
check catches it.

</details>

### Mode queries and the decoder

The [detection assignment](../a11_detection_segmentation/README.md) introduced object queries: a
fixed set of learned embedding vectors, each a slot the decoder fills with one detection. Mode
queries are the same construction one level up. The head holds $K$ mode-query embeddings, each a
slot for one trajectory hypothesis. A query is a parameter, not a function of the input; the scene
enters only through cross-attention. So the query is what the mode is before it has seen anything,
and cross-attention specializes it to this agent.

The queries have to be initialized distinct (here `randn(K, dim) * 0.02`). Consider what happens if
they are all equal. Every operation downstream of the query is shared across modes and equivariant
to permuting the mode axis: self-attention over the $K$ modes, cross-attention over the same RoI
tokens, the same MLP, the same trajectory and score heads. Feed $K$ identical queries into that and
the $K$ outputs are identical, the winner is an arbitrary tie-break, and the gradient arriving at
each query is identical too, so the queries stay equal at every subsequent step. Distinct
initialization breaks the symmetry. It is a precondition and nothing more; on its own it does not
keep all modes alive.

Each decoder layer runs three sublayers, each a residual update on a LayerNorm-ed input,
$q \leftarrow q + \text{sublayer}(\text{LayerNorm}(q))$, as in the transformer assignment. First
self-attention over the $K$ mode queries, which is how the modes see each other and can spread out
instead of duplicating a hypothesis. Then cross-attention of the $K$ queries over the agent's
$\text{out}^2$ RoI tokens, which is where scene context enters. Then a per-mode MLP. There is no
causal mask, since neither the modes nor the trajectory steps are produced in sequence: the whole
$(K, T, 2)$ output comes out in one shot.

Each mode's trajectory head emits $T$ per-step displacements $\Delta_1, \dots, \Delta_T$, which are
cumulatively summed along time into absolute agent-centric positions,
$\hat y_t = \sum_{s \le t} \Delta_s$. In matrix form $\hat y = L\Delta$ with $L$ the
lower-triangular matrix of ones, a fixed linear map that adds no parameters. Predicting
displacements rather than positions has two effects. The quantity the final linear layer has to
produce stays small and uniform in scale over the horizon: at 1.5 m/s and 0.5 s per step a
displacement is 0.75 m at every $t$, while an absolute position grows to about 9 m by the 6 s
horizon. And because $L$ is triangular, a gradient on the position at step $t$ reaches every
displacement at or before $t$, so an error late in the trajectory corrects the whole prefix instead
of only the last step.

A separate score head maps each mode query to one number, giving $(B, K)$ raw logits, unnormalized
scores that become a distribution over modes after a softmax.

### The winner-take-all loss

The loss has a hard path and a soft path. Both share the same per-mode endpoint distances and the
same classification term. For a batch of $B$ agents with predictions
$\hat y \in \mathbb{R}^{B \times K \times T \times 2}$ and observed futures
$y \in \mathbb{R}^{B \times T \times 2}$,

$$\text{FDE}_{b,k} = \lVert \hat y_{b,k,T} - y_{b,T}\rVert_2,
\qquad k^{\star}_b = \arg\min_k \text{FDE}_{b,k},$$

and the winner $k^{\star}$ is detached, for the reason the k-means section gives: it is an integer
index with no derivative worth taking.

The hard path is the canonical winner-take-all loss. Regress the winner's whole trajectory with
mean squared error,

$$\mathcal{L}_{\text{reg}} = \frac{1}{2BT}\sum_{b=1}^{B}
\big\lVert \hat y_{b,k^{\star}_b} - y_b \big\rVert_F^2 .$$

The 2 in the denominator counts the two coordinates, so this is the mean over all $B$ agents, $T$
steps and both coordinates. Only the winning mode appears in that expression, so only it carries
regression gradient; the errors of the other $K-1$ modes do not enter this term on this step.

Selection uses Euclidean distance while regression uses squared error, and the mismatch is
deliberate. The Euclidean norm $\lVert u \rVert$ has gradient $u/\lVert u \rVert$, which is
undefined at $u = 0$, exactly the point a converging prediction approaches. Squared error is smooth
there. A gradcheck, which compares the analytic gradient autograd computes against a
finite-difference estimate obtained in double precision by perturbing each input by a small
$\epsilon$ and re-evaluating the function, needs that smoothness: a kink makes the two disagree.
In the hard path FDE enters only through the detached $\arg\min$, so its kink does not matter. The
soft path below does differentiate through FDE, through its weights, so it has the same kink at
$\text{FDE} = 0$; its gradcheck uses random trajectories whose endpoints are far from the target.
The cost of using squared error is that the loss value is in $\text{m}^2$ and cannot be read as an
error in meters, which is why minADE and minFDE are reported separately from it.

The classification term trains the score head. It is the cross-entropy of the score logits against
the hard winner index,

$$\mathcal{L}_{\text{cls}} = -\frac{1}{B}\sum_{b=1}^{B}
\log \mathrm{softmax}(s_b)_{k^{\star}_b},$$

with $s_b \in \mathbb{R}^K$ the raw logits. They go in raw because cross-entropy applies log-softmax
internally; applying a softmax first runs it twice, which flattens the distribution and the gradient
with it. The term teaches the score head to name whichever mode committed to the observed future,
so at inference, with no ground truth available to select with, the top-scored mode is the model's
single guess. The total is $\mathcal{L}_{\text{reg}} + \lambda_{\text{cls}} \mathcal{L}_{\text{cls}}$,
with $\lambda_{\text{cls}}$ a weight.

### Soft assignment and the temperature

The soft path replaces the hard winner with a distribution over modes:

$$w_{b,k} = \mathrm{softmax}_k\big(-\text{FDE}_{b,k}/\tau\big),
\qquad \ell_b = \sum_{k} w_{b,k}\,\text{MSE}_{b,k},
\qquad \mathcal{L}_{\text{reg}} = \frac{1}{B}\sum_{b} \ell_b,$$

where $\text{MSE}_{b,k}$ is the mean squared error of mode $k$'s full trajectory against $y_b$. The
temperature $\tau$ controls how peaked $w$ is, the same knob as in the
[self-supervised learning](../a03_0_ssl/README.md) and [CLIP](../a04_clip/README.md) losses. As
$\tau \to 0$ the weights go one-hot at the smallest FDE and the soft loss becomes the hard loss. As
$\tau \to \infty$ they go uniform, every mode is regressed toward the same target, and the head is
back to mode averaging with $K$ copies. In between, every mode gets gradient, weighted toward the
ones already close.

The weights are not detached, unlike the winner index, so $w$ itself carries gradient and the
regression term has a second effect beyond pulling each mode toward the target. Fixing one agent
and dropping the $b$ index, write $s_k$ for mode $k$'s MSE and $\bar s = \sum_k w_k s_k$ for the
weighted average. Differentiating $\ell = \sum_k w_k s_k$ through the softmax, holding the $s_k$
fixed to isolate the weight path, gives

$$\frac{\partial \ell}{\partial \text{FDE}_j} = -\frac{w_j}{\tau}\,(s_j - \bar s).$$

A mode doing worse than the weighted average has a negative derivative here, so along this path a
gradient step increases its endpoint distance: the weight path pushes it away from the target and
its share of the responsibility shrinks. A mode doing better than average is pulled in and its
share grows. The direct term $w_j \nabla \text{MSE}_j$ still pulls every mode toward the target,
with strength $w_j$, so the net motion of a worse-than-average mode depends on the balance of the
two. The weight path sharpens the assignment as training proceeds.

The classification term is identical in both paths: cross-entropy against the hard $\arg\min$
winner, whatever the regression term is doing.

> **Exercise 4 (soft weights at two temperatures).** One agent has three modes with endpoint
> distances $\text{FDE} = (0.5, 2.0, 4.0)$ m and trajectory errors
> $s = (0.1, 1.0, 6.0)$ $\text{m}^2$. Compute the weights $w$, the weighted average $\bar s$, and
> the sign of $\partial \ell / \partial \text{FDE}_j$ for each mode at $\tau = 3$ and at
> $\tau = 0.5$. What happens to mode 2?

<details>
<summary>Answer</summary>

At $\tau = 3$: the logits are $(-0.167, -0.667, -1.333)$, so $w \approx (0.521, 0.316, 0.162)$
and $\bar s \approx 0.052 + 0.316 + 0.974 \approx 1.343$. The derivatives
$-\tfrac{w_j}{\tau}(s_j - \bar s)$ are about $(+0.216, +0.036, -0.252)$. Modes 1 and 2 are better
than average and the weight path pulls them in; it pushes mode 3 away.

At $\tau = 0.5$: the logits are $(-1, -4, -8)$, so $w \approx (0.952, 0.047, 0.001)$ and
$\bar s \approx 0.148$. The derivatives are about $(+0.091, -0.081, -0.010)$. Mode 2, with
$s_2 = 1.0$, is now worse than the weighted average, which is dominated by mode 1, so the weight
path pushes it away, while the direct term still pulls it in with weight only 0.047. At high
temperature every mode shares the target; as the temperature falls the target is handed to the
closest mode and the weight path pushes the others off it, which is the transition from soft to
hard assignment.

</details>

### Dead modes and the annealing recipe

Hard winner-take-all inherits the empty-cluster problem. For a mode that never wins, no loss term
measures its error against any future, so nothing pulls it toward the data, and it keeps losing.
It is not frozen: the trajectory head and the decoder weights are shared by all modes, and its
query receives gradient through mode self-attention, so it drifts as those weights change for the
winners. Nothing steers that drift toward an unclaimed future. The classification term touches it
through its own logit, but that only teaches the score head to give it a low score. With $K$ modes
and only a few distinct intentions in the batch, the winners carve up the intentions between them
and the rest drift without a target. If two intentions
are both nearest to the same mode early on, that mode settles on their average, the configuration
in the right panel of the fan figure and in Exercise 2, and no live mode is left to split them.

The annealed recipe decays $\tau$ linearly from an initial value $\tau_0$ to 0 over the first part
of training, then runs hard for the remainder. Early on every mode gets gradient and the modes
spread out to cover the fan of futures; by the time the loss goes hard, each mode already owns a
region of the future space, so none is left without data. This is soft k-means annealed into hard
k-means. The idea comes from deterministic annealing (Rose, Gurewitz and Fox 1990), a clustering
method for vector quantization: it minimizes the expected distortion under soft assignments
weighted by $\exp(-d/T)$ at a temperature $T$, starting hot, where every center sits at the data
mean, and cooling, so that the centers split apart one by one as $T$ falls instead of being placed
by a random initialization.

Two other mitigations are standard. Evolving WTA (Makansi et al. 2019,
[arXiv:1906.03631](https://arxiv.org/abs/1906.03631)) keeps the assignment hard but supervises the
top $M$ modes instead of the top 1, decaying $M$ from $K$ down to 1 over training, so early steps
feed every mode and late steps recover the winner-take-all loss. Goal anchors attack the feedback
loop from the other side, by making the assignment independent of what the network currently
predicts. TNT (Zhao et al. 2020, [arXiv:2008.08294](https://arxiv.org/abs/2008.08294)) samples
candidate target points off the lane graph, scores them, and regresses a trajectory conditioned on
each. MTR runs k-means over the endpoints of every ground-truth trajectory in the training set to
get a fixed set of intention points (64 per agent category), attaches one query to each, and
assigns a training example to the query whose intention point is nearest to the observed endpoint.
That assignment reads the anchor and the label, never the current prediction, so a mode cannot
starve itself out: whenever an example lands in its region of the endpoint space it is supervised,
however badly it is predicting.

Hard assignment is therefore the standard choice at scale. Large prediction systems have
anchoring and per-scene diversity: different scenes activate different modes, so every mode
receives gradient somewhere in the dataset even though any single scene supervises one. Dead modes
are most likely in the case our three-intention fan sets up, a single ambiguous context repeated
with no per-scene variation and no anchors.

### Best-of-K oracle metrics

The reported metrics all take a minimum over the $K$ modes before averaging over agents:

$$\text{minADE}_K = \min_k \text{ADE}_k, \qquad
\text{minFDE}_K = \min_k \text{FDE}_k, \qquad
\text{MR}_K = \mathbb{1}\big[\text{minFDE}_K > \delta\big],$$

with the miss threshold $\delta = 2$ m by default here. These are the Argoverse/nuScenes-style
$\text{minADE}_K$, $\text{minFDE}_K$ and miss-rate metrics.

The minADE used in this course departs from the literal definition: it picks the mode by FDE and
reports that mode's ADE, rather than taking the minimum of ADE over modes. The two agree whenever
the same mode is best under both, and where they differ this version is the larger of the two, so
it never flatters the model. It scores the same mode the winner-take-all loss trained.

They are called oracle metrics because the minimum assumes an oracle that picks the right mode
after the answer is known. Two consequences follow. They reward coverage but do not penalize
redundancy: $K$ near-identical modes score as well as one mode, and a model that hedges is graded
on its luckiest hypothesis. And they say nothing about whether the model can pick the right
hypothesis without the oracle, which is the only thing a downstream planner can use. The score head
exists for that: at inference the top-scored mode is the committed answer, and these metrics do not
measure its error.

> **Exercise 5 (when ADE and FDE pick different modes).** The observed future is
> $y = \big((1, 0), (2, 0), (3, 0)\big)$. Mode A is $\big((1, 1), (2, 1), (3, 0.2)\big)$ and mode B
> is $\big((1, 0), (2, 0), (3, 0.5)\big)$. Compute ADE and FDE of each mode, the literal
> $\text{minADE}_2$ and $\text{minFDE}_2$, the ADE of the mode selected by FDE, and the miss
> indicator at $\delta = 2$ m.

<details>
<summary>Answer</summary>

Mode A: per-step errors $(1, 1, 0.2)$, so ADE $= 2.2/3 \approx 0.733$ m and FDE $= 0.2$ m. Mode B:
errors $(0, 0, 0.5)$, so ADE $\approx 0.167$ m and FDE $= 0.5$ m. The literal minADE is 0.167 m
(mode B) and minFDE is 0.2 m (mode A). FDE selects mode A, whose ADE is 0.733 m, the larger value.
The miss indicator is 0, since $0.2 < 2$. The literal metric mixes two different modes into one
report; selecting by FDE scores one mode consistently, the one the loss would have trained.

</details>

### Marginal and joint prediction

This head predicts one agent at a time, conditioned on its own pooled context, so it models
$p(y_i \mid x)$ separately for each agent $i$. That is a marginal. What a planner needs is the
joint $p(y_1, \dots, y_n \mid x)$, and a product of marginals is not the joint whenever the agents'
futures are dependent, which is the case that matters: two cars cannot occupy the same piece of
road at the same instant. Sampling one future per agent from independent marginals can therefore
produce a scene where the predicted trajectories intersect, a configuration with essentially zero
probability under the true joint. Joint prediction adds agent-to-agent attention, so the modes of
different agents read each other and the model can express the dependence. UniAD's MotionFormer is
the joint version.

### The modular stack and UniAD's query passing

The classical stack is modular: sensors feed detection, detection feeds tracking, tracking feeds
prediction, prediction feeds planning. Each hand-off is a hard, non-differentiable interface, a
list of boxes, a set of tracks, a set of trajectories. Thresholding a detection score to decide
what enters the track list is a step function with no derivative to send backward. Errors compound
across the stages, and since no gradient flows back, each perception module is trained against its
own proxy objective and cannot learn what the planner needs from it.

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
(high-definition maps, prebuilt maps of lane geometry, crosswalks and signs accurate to
centimeters) and agent histories, and MultiPath++ (Varadarajan et al. 2022,
[arXiv:2111.14973](https://arxiv.org/abs/2111.14973)) made the same move on the prediction side.
VADv2 (Chen et al. 2024, [arXiv:2402.13243](https://arxiv.org/abs/2402.13243)) turns the planner
into a probabilistic model over a discrete vocabulary of planning actions. DriveTransformer (Jia et
al. 2025, [arXiv:2503.07656](https://arxiv.org/abs/2503.07656)) drops the sequential task ordering:
agent, map, and planning queries run in parallel and attend to each other directly, instead of the
track-then-motion-then-plan chain.

### Open-loop evaluation of planners

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
[world-models assignment](../a12_world_models/README.md) builds a sensory model of the future
instead: given the current observation and an action, predict the next observation. A full driver
wants both, the behavioral prediction of other agents and the sensory rollout of its own actions.
The [vision-language-action assignment](../a13_vla/README.md) trades geometric precision for
instruction following: planners like DriveLM and DriveVLM reason over language ("yield to the
pedestrian", "the light is red") and emit higher-level decisions, accepting coarser trajectories in
exchange for following instructions and explaining themselves.

## The assignment

Fill these holes, in order. Each is one `NotImplementedError` with a matching test. The docstring
in each function gives the signature, shapes, and constraints.

| # | Function | File | Concept from the notes |
|---|---|---|---|
| 1 | [`roi_align_bev()`](predict.py) | `predict.py` | RoI-align around each agent, the axis swap, and the normalization $g$ |
| 2 | [`MultimodalTrajectoryHead.forward()`](predict.py) | `predict.py` | mode queries, the decoder layers, displacements summed into positions |
| 3 | [`wta_loss()`](predict.py) | `predict.py` | the hard and soft winner-take-all loss and the classification term |
| 4 | [`min_ade()`](predict.py) | `predict.py` | $\text{minADE}_K$, with the mode selected by FDE |
| 5 | [`min_fde()`](predict.py) | `predict.py` | $\text{minFDE}_K$ |
| 6 | [`miss_rate()`](predict.py) | `predict.py` | $\text{MR}_K$ at threshold $\delta$ |

Everything is in `predict.py`. The attention layers come from `nanovision.attention` (the
transformer assignment's `MultiHeadAttention`), and the MLP from `nanovision.primitives`. Provided,
not holes: `config.py` (`PredConfig`), the shared training loop `_train.py`, and the
`pred_toy_scene` generator in `nanovision/data/toy.py`.

Notes on specific holes:
- `roi_align_bev` builds offsets with `linspace(-radius, radius, out_size)` in cell units, adds them
  to the agent centers, normalizes each axis, stacks the grid's last dimension as (width from
  $y_{\text{cell}}$, height from $x_{\text{cell}}$), and calls `F.grid_sample` with
  `mode="bilinear"`, `padding_mode="border"`, `align_corners=False` on the BEV expanded over the
  $N$ agents. Tokens come out row-major over the $(x_{\text{cell}}, y_{\text{cell}})$ grid.
- `MultimodalTrajectoryHead.forward` expands the mode queries over the batch, runs each layer as
  `q = q + sa(norm(q))`, `q = q + ca(norm(q), kv=tokens)`, `q = q + mlp(norm(q))`, reshapes the
  trajectory head's output to $(B, K, T, 2)$, and applies `torch.cumsum` over $T$.
- `wta_loss` computes the winner once with `argmin` and detaches it, gathers the winner's trajectory
  for the hard path (`temperature=None`), and uses `F.cross_entropy(scores, winner)` on the raw
  logits in both paths. The `cls_weight` argument is $\lambda_{\text{cls}}$.
- `min_ade` selects each agent's mode by FDE, the same selection the loss uses, and reports that
  mode's ADE. `miss_rate` counts an agent as a miss when its minFDE exceeds `thresh`.
- `_train.train_head` implements the three recipes: `"hard"` (temperature `None` throughout),
  `"soft"` (fixed temperature), and `"annealed"` (temperature decays linearly from `tau0` to 0 over
  the first `anneal_frac` of the steps, then hard). `config.py` sets `tau0` to 3.0 and
  `anneal_frac` to 0.6. In every recipe the Adam learning rate (5e-3) stays constant for the
  first `anneal_frac` of the steps and then decays linearly to 0. Without that decay, hard-phase
  training on the shared scene keeps oscillating as the winner for one intention flips between two
  nearby modes, and the final minADE depended on where the last step landed: annealed seed 0
  ended at 0.463 m with 2 CPU threads and 0.0055 m with 11. The single-mode baseline is $K = 1$
  with `"hard"`.

### Running and validating

Activate the environment (`conda activate nanovision`), then:

```
make test     A=a11_5e_pred_planning   # run the tests against the top-level files (the holes)
make verify   A=a11_5e_pred_planning   # run the same tests against the reference solution/
make viz      A=a11_5e_pred_planning   # render the figures from the reference solution
make viz-mine A=a11_5e_pred_planning   # render the figures from your own code (holes filled)
```

`make test` runs the suite in `assignments/a11_5e_pred_planning/tests/` against the top-level
`predict.py`, red until the holes are filled and green once correct. `make verify` runs the
identical suite against the reference `solution/` by setting `NANOVISION_IMPL=solution`, so it is
green from the start and shows the target.

The tests check:
- `test_shapes.py`: output shapes of `roi_align_bev` and of the head.
- `test_gradcheck.py`: a float64 gradcheck of `roi_align_bev` at fractional interior centers, and of
  both `wta_loss` paths (the hard one with a planted clear winner, so the detached `argmin` does not
  change under the finite-difference perturbation).
- `test_modes_specialize.py`: after annealed $K = 6$ training on the shared-context scene, at least
  three modes win and their endpoints are more than 2 m apart.
- `test_overfit.py`: on the distinct-context scene (agents in different cells with different yaws
  and feature vectors), the head overfits to minADE below 0.1 m. A wrong axis swap or rotation
  fails here.
- `test_wta_beats_single_mode.py`: the headline contrast. Single-mode minADE sits above 1 m, the
  annealed $K = 6$ head reaches under 0.15 m on each of the two init seeds it tries, and annealed
  stays below 0.3 times the single-mode value.
- `test_forbidden_imports.py`: a scan confirming the attention comes from the course's own
  multi-head attention rather than `nn.MultiheadAttention`, `nn.Transformer*`, or
  `F.scaled_dot_product_attention`.

`make viz` writes two figures to `out/`:
- `modes_annealed.png`: the annealed $K = 6$ head on the shared-context scene, with the true futures
  and the six modes colored by score.
- `compare_recipes.png`: $K = 1$, hard $K = 6$, and annealed $K = 6$ on the same scene, all from
  init seed 2.

The figure in these notes is generated by `assets/make_fan_figure.py`, which writes the
constant-turn-rate rollout in numpy and does not use the assignment code.

What you should see when you run this. The toy is CPU only, seconds per test. `pred_toy_scene` with
`shared_context=True` is our three-intention fan with six agents: all at one BEV cell with one
feature vector (which encodes only speed, not intention), their futures fanned over three balanced
intentions. The horizon lateral displacement is $\pm 4.14$ m and the closest pair of intention
endpoints is 4.38 m apart, so the three futures are well separated at the endpoint the winner
selection uses. Measured with the reference solution and the tiny config here, 400 training steps,
over init seeds 0 to 15:

| recipe | minADE | what happens |
|---|---|---|
| single mode (K=1) | 1.168 m on every seed | forced to the conditional mean of the three intentions |
| hard WTA (K=6) | 0.005-0.014 m on 6 of 16 seeds (5 of seeds 0-7) | three modes win, all 3 intentions covered |
| | 0.554-0.558 m on 9 of 16 seeds (3 of seeds 0-7) | two modes win; one sits on the mean of two intentions |
| | 1.168 m on 1 of 16 seeds (seed 9) | one mode wins, the same answer as K=1 |
| soft-to-hard annealed (K=6) | 0.007-0.029 m on every seed | all 3 intentions covered by distinct modes |

These numbers are at PyTorch's default CPU thread count (11 on the machine used). The annealed
recipe also stays at or below 0.03 m on seeds 0-7 with 1 and with 2 threads.

The single-mode 1.168 m is fixed by the scene rather than by optimization. With $K=1$ the minimum
over modes is vacuous, so minADE is just that one mode's ADE, and once it has converged to the
conditional mean of the three intentions the geometry pins the number: 1.610 m for each of the four
turning agents and 0.283 m for each of the two straight ones, 1.168 m averaged over the batch. Its
minFDE is 3.09 m, the value from Exercise 1.

Hard WTA getting stuck is intermittent: on this toy it reaches full coverage on some
initializations and gets stuck on others, and which one happens depends on the init seed. The
0.554 m runs are the configuration in the right panel of the fan figure. Placing one mode exactly on
one turn and the other on the mean of the remaining two intentions gives minADE 0.554 m and minFDE
1.459 m by direct computation, and the stuck runs measure 0.554-0.558 m and 1.459-1.460 m. The
headline test asserts only the init-independent facts (single-mode minADE above 1 m, the annealed
recipe under 0.15 m on every seed it tries, and coverage beating averaging by a wide margin); it
leaves the intermittent stuck runs of hard WTA to the measured table and the `compare_recipes` figure,
whose seed 2 is one of the stuck runs. The comparison shows $K=1$ reduced to one short averaged
path, hard $K=6$ covering one intention with one mode and averaging the other two with another, and
annealed $K=6$ covering all three arcs.

These are toy demonstrations of mode averaging and dead modes on one shared ambiguous input,
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
- MacQueen 1967, "Some methods for classification and analysis of multivariate observations",
  Fifth Berkeley Symposium on Mathematical Statistics and Probability (online k-means).
- Bottou and Bengio 1995, "Convergence Properties of the K-Means Algorithms", NIPS 7.
- Sculley 2010, "Web-scale k-means clustering", WWW 2010 (mini-batch k-means).
- Rose, Gurewitz and Fox 1990, "Statistical mechanics and phase transitions in clustering",
  Physical Review Letters 65 (deterministic annealing).
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
