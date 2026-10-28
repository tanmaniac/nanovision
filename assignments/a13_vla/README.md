# A13 - vision-language-action capstone (flow-matching action head from pixels)

A vision-language-action (VLA) model is a robot policy: it takes camera images and a language
instruction, such as "pick up the red block", and outputs the motor commands that carry the
instruction out. The VLAs of 2023-2026 share one design. A pretrained network that understands
images and text (the backbone) interprets the scene and the instruction, and a separate, smaller
network attached to its output (the action head) turns that interpretation into continuous motor
commands that vary smoothly from one control step to the next. The action head of pi0 (Black et
al. 2024), the first VLA at production scale to use flow matching, generates a short sequence of
future actions at once by flow matching, the same noise-to-data generative method covered earlier
in the course.

This capstone builds the perception-to-action path of such a model on a simulated robot: a 2-link
arm, the reacher, controlled from a 64x64 camera image, with no access to joint angles or to the
target position. A convolutional encoder turns the frame into a conditioning vector, standing in for
pi0's backbone. A conditional flow-matching (CFM) action head, standing in for pi0's action head, is
trained by behavior cloning (fitting the policy to recorded demonstrations) on filtered
demonstrations from a scripted expert, a hand-written controller, and generates the joint-torque
sequence that drives the arm's fingertip to the target. The baseline is a plain regression policy
that predicts several future actions per query (action chunking, defined in the notes), and the
diffusion contrast is a head built on a denoising diffusion probabilistic model (DDPM). The policy
that reads only pixels reaches the target far more often than random torques do. Everything is small
enough to collect demonstrations and train in minutes, so the mechanism is the focus, not the scale.

Required reading before starting:
- Black et al. 2024, "$\pi_0$: A Vision-Language-Action Flow Model for General Robot Control" (pi0),
  [arXiv:2410.24164](https://arxiv.org/abs/2410.24164).
- Zhao et al. 2023, "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware" (ACT),
  [arXiv:2304.13705](https://arxiv.org/abs/2304.13705).
- Chi et al. 2023, "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion",
  [arXiv:2303.04137](https://arxiv.org/abs/2303.04137).

## Lecture notes

### What a VLA action head is for

#### The task

The reacher is a simulated arm with two links in a plane, a shoulder joint and an elbow joint
(called the wrist in the simulator). A small target is placed at a random spot within reach. A
camera above the arm produces a 64x64 color picture at every control step. The policy sees only
that picture and must output two joint torques per step, between $-1$ and $1$, that move the
fingertip onto the target, typically within about 20 steps. From the picture alone the policy has
to work out where the arm is, where the target is, and which torques move one onto the other.

The same task at full scale is a robot arm on a table with a camera, told "put the cup in the sink".
The picture is larger, there may be several cameras and an instruction, the arm has about seven
motors, and commands go out 50 to 100 times per second, but the structure is the same: pixels (and
words) in, a stream of continuous motor commands out.

#### The obvious approach, and why it fails

Nobody can write down the function from pixels to torques, but a controller that performs the task,
called the expert, can. Here the expert is a hand-written program, the scripted expert, that cheats:
it reads the true joint angles and the target position out of the simulator and computes torques
with textbook inverse kinematics and a proportional-derivative (PD) law. Record what it does,
(picture, torque) at every step, and fit a network to predict the torque from the picture by least
squares. This is called behavior cloning, and it is the starting point of every method in these
notes.

Plain behavior cloning has two problems.

The first is drift. The network is trained on pictures from the expert's own runs. Once the
network is in control, its small errors put the arm in configurations the expert never
visited, the pictures look unlike anything in training, the next prediction is worse, and the errors
grow. Predicting several future actions at once and executing them before looking again (action
chunking) reduces the number of decisions at which this can happen.

The second is averaging. Least squares returns the average of the demonstrated actions for a given
input. When the demonstrations contain several valid actions for the same input (go around an
obstacle on the left on some runs and on the right on others), the average is an action between
them that no expert ever took, such as driving straight into the obstacle.

#### The idea

Replace the regression network with a generative model of actions, a network that outputs a random
sample from the distribution of demonstrated actions given the picture rather than their average.
Each sample is one of the valid actions, so averaging never happens. The generative model of choice
is flow matching: draw Gaussian noise of the same shape as the action sequence, and move it step by
step along a learned velocity until it becomes an action sequence. The picture enters as an extra
input to the velocity network. Training is the plain velocity regression of the flow-matching notes,
with the picture's encoding added as an input.

#### Why this design

The alternative that came first makes the backbone emit actions as text-like tokens, one number at a
time. That gives coarse actions and slow sequential decoding. A flow-matching action head produces a
whole chunk of continuous actions with about 10 network evaluations, fast enough for 50 Hz control,
and it represents several valid actions without averaging them. pi0 introduced this design, and the
2025-2026 literature has largely followed it.

### Our four-corner example (a point mass with the goal hidden)

The reacher cannot show the averaging problem, because the picture shows the target, so from a given
picture there is essentially one correct action. To see the problem, we use a second, smaller task,
which we call our four-corner example. A point mass moves in the unit square. Four goals sit near the
corners, at $(0.15, 0.15)$, $(0.85, 0.15)$, $(0.15, 0.85)$, and $(0.85, 0.85)$. Each demonstration
picks one goal at random and drives the point straight toward it at speed $v_{\max} = 0.05$ per step.
The policy is told the point's position but not which goal this run is heading for (we call this the
goal-hidden mode).

Look at the center $(0.5, 0.5)$. Every demonstration passes near it, and the demonstrated action
there is one of four vectors of length $0.05$ pointing along the diagonals, each with components
$\pm 0.05/\sqrt{2} \approx \pm 0.035$. The conditional distribution of the action given the
position has four separate lumps. Their average is exactly the zero vector, an action that moves the
point mass nowhere. That is what least squares returns, and what a regression policy does at the
center.

![Our four-corner example. Left, the four demonstrated actions at the center and their mean, the zero vector. Right, in units rescaled so the actions sit at (plus or minus 1, plus or minus 1), Gaussian noise samples carried to the four actions by the exact flow-matching velocity](assets/fourcorner.png)

The left panel shows the four demonstrated actions and their mean. The right panel shows what a
flow-matching head does instead, with the actions rescaled by $0.035$ so they sit at
$(\pm 1, \pm 1)$. Each gray dot is a Gaussian noise sample, and each curve is the path that sample
follows under the exact velocity that flow matching learns (derived in the conditional flow matching
section and in Exercise 5). Every sample ends on one of the four actions, and by symmetry the exact
velocity sends a quarter of the noise to each. No sample ends at the mean.

Our four-corner example stands for any situation where the input does not determine which of several
valid demonstrated actions to take: an expert who passes an obstacle on either side, grasps a mug
by either side, or approaches from either direction. It keeps the averaging and the sampling. It drops
images, dynamics, and time: the action is a single step, and the input is a position rather than a
picture.

### Overview of the rest of the notes

Most of the parts below are ideas from estimation and control applied to a new problem.

1. Vocabulary. The terms of imitation learning (policy, demonstration, behavior cloning), what
   pretraining and a vision-language model are, and what a VLA adds.
2. The reacher and its scripted expert, which is two-link inverse kinematics plus a
   PD law, both standard robotics.
3. The image encoder. A small convolutional network that turns the picture into a vector, the same
   encoder the world model used.
4. Behavior cloning and drift. Supervised regression on demonstrations, and why its errors compound
   in closed loop, which is open-loop drift in a new form. Action chunking reduces the damage, and it
   has the shape of model-predictive control without the model.
5. Two kinds of action head. Actions as text tokens, and continuous action heads.
6. Why the head must be generative. Least squares returns the conditional mean, the same fact behind
   every least-squares estimator, and the mean of a multimodal distribution is a bad action.
7. Diffusion and flow matching as action heads, the two generative models from earlier in the course,
   now conditioned on the picture's encoding.
8. Why the flow loss stays above zero. The leftover conditional variance, the same irreducible floor
   a least-squares estimator leaves on a noisy measurement.

Five exercises, placed after the sections they test, are meant to be worked by hand (a calculator
helps for some). Each has a worked answer that can be expanded.

### The vocabulary of imitation learning

The reacher is a DeepMind Control Suite task, the same simulator family as the cartpole the world
model learned to balance, and the vocabulary carries over. An *environment* is the simulated system;
it advances one discrete step when handed an *action* and returns a scalar *reward* along with the
*observation*. A run from a reset until the environment terminates or a step cap is hit is an
*episode*. A *policy* is the controller, a map from the observation to an action. Running a policy in
the environment for one episode is a *rollout*.

Reinforcement learning searches for a policy that maximizes accumulated reward by trying actions and
keeping what works. None of that happens in this capstone. The reward here is a success detector and
nothing more: it is 1 on the steps where the fingertip overlaps the target and 0 otherwise, and it is
read after the fact to score a rollout. It never enters a loss and is never maximized.

The training signal comes from *demonstrations*. A demonstration is one recorded episode of a
different controller, the *expert*, driving the same system: the observations the learner would have
seen, paired with the actions the expert took. *Imitation learning* is the family of methods that
fit a policy to demonstrations. Its simplest member, and the only one used here, is *behavior
cloning* (BC): treat the (observation, action) pairs as a supervised regression dataset and fit the
policy to them under a squared-error loss. No reward appears in that loss, and the environment is
not touched during training.

Two properties of the expert used here matter later. It is *privileged*: it reads the true joint
angles, joint velocities, and target position out of the simulator, none of which the learner gets.
And it is *scripted*, a hand-written controller rather than a learned one, so demonstrations cost
only simulator time. Both properties are ordinary in the VLA literature, where demonstrations
normally come from a human teleoperating the arm and the same asymmetry holds: the human expert sees
the whole workspace with their own eyes, the policy sees a camera feed.

### Pretraining, tokens, and vision-language models

*Pretraining* is training a large network on a large generic corpus, under an objective that needs
no hand-applied labels, and then reusing the resulting weights as the starting point for a much
smaller fit on the task actually wanted. The reused part is the *backbone*; the small task-specific
piece attached to its output is a *head*. This works because the task-specific dataset is tiny
compared with the pretraining corpus, so the backbone supplies representations the task data could
never have produced on its own.

A language model works on *tokens*. Text is chopped into a fixed vocabulary of subword pieces, each
piece is an integer index, and the model is trained to predict the next index given all the previous
ones. Generation is *autoregressive*: sample one token, append it to the sequence, feed the extended
sequence back in, sample the next. Every token costs one full forward pass through the network, and
the passes cannot be run in parallel, because each input contains the previous output.

A *vision-language model* (VLM) attaches an image path to that language model, through the
visual-token interface covered in the VLM notes: a vision encoder turns the image into a sequence of
vectors, a connector maps those vectors into the language model's embedding space, and the language
model consumes them as if they were text tokens. Trained on image-caption and instruction data, the
result *grounds* language in pixels, meaning the model ties words to the image content they name
rather than to text statistics alone. Asked about "the red block on the left", its answer depends on
the image region that phrase picks out. A robot policy needs this property, because an instruction
like "pick up the red block" is useless unless "red block" resolves to a location in the current
frame.

### What a vision-language-action model is

A VLA takes a pretrained VLM and attaches an action head (also called an action decoder). The two
train together on robot demonstration data. The VLM interprets the scene and instruction; the action
head turns that interpretation into motor commands.

Why split the model this way instead of having the VLM emit actions directly as text? Text
generation is discrete, sequential, and slow. Robot actions are real-valued, correlated in time, must
be smooth enough not to jerk the hardware, and are issued at 50-100 Hz. A purpose-built action head
lets the backbone do the interpretation while the head produces continuous action sequences in one
shot.

In pi0 the action head is called the *action expert*: a second, smaller set of transformer weights
attached to the backbone, which reads the backbone's features and outputs continuous action chunks
(short sequences of future actions, defined in the action chunking section). Backbone and action
expert are trained jointly on robot data, so "expert" in this name refers to a component of the
network, not the controller that produced the demonstrations. These notes call pi0's component the
action head, and call the hand-written controller that produces this assignment's demonstrations
the *scripted expert*.

On the reacher, the conditioning vector $c$ that the action head reads comes from a small
convolutional encoder applied to the camera frame instead of from a VLM. The encoder stands in for
the VLM backbone; the action head is the same object as pi0's action expert.

### The reacher and the scripted expert

The reacher is a 2-link arm in the plane; both links have length $L_0 = L_1 = 0.12$. The action is a
2D joint torque $(\tau_{\text{shoulder}}, \tau_{\text{wrist}})$ clipped to $[-1, 1]$. The reward is
1 while the fingertip overlaps the small target and 0 otherwise; a *reach* is a step with reward
above 0.5, and the *reach success* of a policy is the fraction of episodes in which it reaches the
target before the step cap. The observation is the 64x64 RGB frame from the fixed overhead camera,
normalized to $[0, 1]$ and laid out channel-first as $(3, 64, 64)$. The learned policy sees only
this image, so it has to recover from pixels both where the arm is and where the target is.

The scripted expert reads the target's world position $(g_x, g_y)$, measured from the shoulder, and
the joint state straight out of the simulator. It solves the 2-link inverse kinematics for the joint
angles that would put the fingertip on the target. Here $\theta_1$ is the shoulder angle measured
from the $x$-axis, $\theta_2$ is the elbow angle measured relative to the first link, and $r$ is the
distance from the shoulder to the target:

$$\theta_2 = \arccos\!\left(\frac{r^2 - L_0^2 - L_1^2}{2 L_0 L_1}\right), \qquad
\theta_1 = \operatorname{atan2}(g_y, g_x) - \operatorname{atan2}\!\big(L_1 \sin\theta_2,\; L_0 + L_1 \cos\theta_2\big),$$

with $r = \lVert (g_x, g_y) \rVert$ clamped into the workspace $[0.002,\, L_0 + L_1 - 0.002]$ and
the cosine clamped to $[-1, 1]$ so the arccos is always defined. The first formula is the law of
cosines on the triangle formed by the two links and the line to the target. Since $\arccos$ returns
a value in $[0, \pi]$, this picks one of the two mirror-image solutions consistently. Call the
resulting pair $\theta^\star = (\theta_1, \theta_2)$. The expert then drives the joints to
$\theta^\star$ with a proportional-derivative law on the wrapped angle error,

$$\tau = \operatorname{clip}\big(k_p\,(\theta^\star - \theta) - k_d\,\dot\theta,\; -1,\; 1\big),
\qquad k_p = 12,\; k_d = 0.8,$$

where $\theta$ and $\dot\theta$ are the current joint angles and velocities and the error is wrapped
into $(-\pi, \pi]$ so the joint always turns the short way round. Nothing in this expert is learned,
and nothing in it is available to the policy.

Demonstrations are filtered. The scripted expert runs on consecutive random seeds; only episodes
that reach the target are kept, and each kept episode is truncated at the step where the reach
happened. Collection stops once enough successful episodes have accumulated. Behavior cloning
therefore sees clean successful trajectories and never sees a failure, which is the normal setup
with human demonstrations too and which has consequences taken up under compounding error below.

> **Exercise 1 (the scripted expert by hand).** Take the target at $(g_x, g_y) = (0.12, 0.12)$.
> (a) Compute $\theta^\star$ and check that the fingertip lands on the target. (b) Compute the
> torque when the arm is at rest at $\theta = (0.3, 1.2)$. (c) Compute it again at
> $\theta = (0.05, 1.5)$ with $\dot\theta = (0.2, 0)$. What does the difference between (b) and (c)
> say about the actions the policy has to learn?

<details>
<summary>Answer</summary>

(a) $r = 0.12\sqrt{2} \approx 0.1697$, so $r^2 = 0.0288 = L_0^2 + L_1^2$, the cosine is $0$, and
$\theta_2 = \pi/2 \approx 1.5708$. Then
$\theta_1 = \operatorname{atan2}(0.12, 0.12) - \operatorname{atan2}(0.12, 0.12) = \pi/4 - \pi/4 = 0$.
The first link points along $x$ to $(0.12, 0)$, the second turns $90°$ and points along $y$, so the
fingertip is at $(0.12, 0.12)$, the target.

(b) The error is $\theta^\star - \theta = (-0.3,\ 0.3708)$ and the velocity is zero, so the
unclipped torque is $12 \cdot (-0.3, 0.3708) = (-3.6,\ 4.45)$, clipped to $(-1, 1)$.

(c) The error is $(-0.05,\ 0.0708)$, and the damping term subtracts $0.8 \cdot 0.2 = 0.16$ from the
shoulder: $(-0.6 - 0.16,\ 0.85) = (-0.76,\ 0.85)$, inside the limits.

Far from the target the expert's torques sit at the limits, $\pm 1$; only close to the target do
they take intermediate values, and those depend on the exact angles and on the joint velocities. The
policy must read small angle differences from a 64x64 picture, and a single frame shows no velocity
at all, so near the target the demonstrated torque is only partly determined by what the policy sees.

</details>

### The image encoder as conditioning

The action heads read a conditioning vector $c$ of fixed width and do not care where it comes from.
On the reacher, $c$ is the output of a convolutional (CNN) encoder applied to the frame:

$$c = \operatorname{Encoder}(\text{obs}), \qquad \text{obs} \in \mathbb{R}^{(B,\,3,\,64,\,64)}, \quad
c \in \mathbb{R}^{(B,\,128)},$$

where $B$ is the batch size. The encoder is four convolutions, each with stride 2, so each one halves
the spatial resolution while raising the channel count: $64 \to 32 \to 16 \to 8 \to 4$ pixels on a
side, with channel widths 32, 64, 128, 256. The result is a $256 \times 4 \times 4$ block of
features, which is flattened to a 4096-vector and passed through a single linear layer to give the
128-wide *embedding*. Embedding here means only a fixed-width real vector standing in for a large
input, whose coordinates carry no assigned individual meaning and are whatever the training
objective made useful. It is the same 64x64 pixel encoder the world model used, producing a
conditioning vector here rather than a latent state.

Encoder and action head train together end to end: the gradient of the action-prediction loss flows
back through the head into the encoder's convolution weights, with one optimizer covering both sets
of parameters. There is no separate representation-learning stage. Nothing tells the encoder what a
target or a fingertip is, and the only pressure shaping it is that the action head has to predict
expert torques from its output.

In a full-scale VLA the conditioning would instead come from a pretrained vision transformer, often
one trained by contrastive image-text matching in the style of CLIP, feeding a language model. Here
it is a small CNN on the camera frame so the whole system trains in minutes.

```mermaid
flowchart LR
  obs["camera frame<br/>(B,3,64,64)"] --> enc["CNN encoder<br/>4 stride-2 convs"]
  enc --> c["conditioning c<br/>(B,128)"]
  c --> head["flow / BC head"]
  head --> chunk["torque chunk<br/>(B,H,2)"]
  chunk -->|"execute open-loop"| robot["reacher"]
  robot -->|"next frame"| obs
```

### Behavior cloning and compounding error

Behavior cloning flattens the demonstrations into a pile of (observation, action) pairs and fits a
network to them by least squares. That is all the training loop does. It is supervised learning, and
it inherits supervised learning's one guarantee: low error on data drawn from the same distribution
as the training set.

A controller cannot rely on that guarantee. The training observations were produced by the
*expert's* closed loop. The test observations are produced by the *learner's* closed loop, and those
two loops visit different states as soon as their actions differ at all. This mismatch between the
input distribution at training time and at test time is called *covariate shift*, and unlike the
usual train/test gap it is self-inflicted: the policy's own errors are what move the input
distribution.

The mechanism will be familiar to anyone who has watched an open-loop trajectory drift. At step $k$
the policy's torque is slightly wrong. The arm arrives at step $k+1$ in a configuration a little off
the demonstrated path, so the camera frame is a little unlike anything in the training set, so the
next prediction is a little worse than the last one, which puts the arm further off at step $k+2$.
Errors accumulate rather than cancel, because each one changes the input to the next decision. The
filtering makes this sharper: demonstrations are truncated at the reach and failures are thrown
away, so the training set contains no examples of recovering from a bad configuration. The policy
has never been shown what to do once it is off the path.

Ross et al. (2011), introducing DAgger, made the horizon dependence explicit. Their analysis shows
that a behavior-cloned policy with per-step error $\varepsilon$ can accumulate cost growing like
$\varepsilon T^2$ over a $T$-step episode, against $\varepsilon T$ if the policy were accurate on
its own state distribution, with the extra factor of $T$ coming from exactly the drift above.
DAgger's fix is to run the learner, ask the expert to label the states the learner actually visited,
and retrain, repeating until the two distributions agree. That needs an expert available on demand
for relabeling, which a human teleoperator is not. The VLA line took a different route.

### Action chunking

Action chunking, from ACT (Zhao et al. 2023), has the policy predict a sequence of $H$ future
actions (a *chunk*, with $H$ the *chunk length*) and execute the whole chunk before looking at a new
observation. Anyone who has written a model-predictive controller will recognize the shape: solve
for a short plan, execute a prefix of it, re-solve. The difference is that there is no model and no
online optimization here. One forward pass of the network emits the whole plan, and the plan is
executed in full rather than only its first step.

Three things change. The number of decisions per episode drops by a factor of $H$, so there are $H$
times fewer opportunities for an error to feed into the next input. Within a chunk the policy
commits to a segment that was produced in one shot and is internally consistent, instead of
re-deciding at every step under whatever drift has accumulated. And the number of steps over which
covariate shift can compound is effectively $T/H$ rather than $T$, which is the quantity that
entered the horizon bound above. ACT reported that chunking reduces compounding error, and chunked
action heads became standard because of it.

ACT also introduced *temporal ensembling*, which is optional and not used here. If chunks are
queried every step rather than every $H$ steps, then at any given step several past queries have
each predicted an action for that step, made from $0, 1, 2, \dots$ steps of stale observation.
Temporal ensembling averages them with a weight $w_i \propto \exp(-m\,i)$ on the prediction made $i$
queries ago, so recent predictions dominate and the executed torque changes smoothly instead of
jumping at chunk boundaries. The constant $m$ sets how fast old predictions are discounted. It can
hurt: pi0 found it detrimental on their evaluation and dropped it, executing chunks open-loop, as
this assignment does.

Training needs chunks as targets, so a per-step action sequence is cut into overlapping windows.
Given a demonstration of $T$ steps, the window starting at step $i$ holds actions
$i, i+1, \dots, i+H-1$, so there are $T - H + 1$ windows, and each window is paired with the frame
observed at its first step. Consecutive windows overlap in $H-1$ actions, which makes the
representation redundant, so the inverse has to be defined rather than guessed: take the full first
window, then the last action of each subsequent window. That last action is the one new step each
window introduced, so the reconstruction is exact. With $T = 6$ and $H = 3$ the windows are
$(0,1,2), (1,2,3), (2,3,4), (3,4,5)$, and the inverse reads off $0,1,2$ from the first, then
$3, 4, 5$ as the last entries of the remaining three.

Execution uses a different set of windows. Running chunks back to back means starting them at
$0, H, 2H, \dots$, and the last start is clamped to $T - H$ so the final chunk does not run past the
end of the sequence. With $T = 7$ and $H = 4$ the starts are $0$ and $3$: the second chunk is pulled
back by one step and re-covers step 3, which is the cost of not running off the end. Clamping can
make the last two starts identical, and the duplicate is dropped.

> **Exercise 2 (what chunk length costs and saves).** A filtered reacher demonstration is about 20
> steps long; take $T = 18$. For $H = 1$, $4$, and $8$, find (a) the number of training windows the
> demonstration provides and (b) the chunk start steps when executing back to back, and so the
> number of policy queries. What does each column of numbers trade against the other?

<details>
<summary>Answer</summary>

(a) $T - H + 1$ windows: $18$, $15$, and $11$.

(b) For $H = 1$ the starts are $0, 1, \dots, 17$, which is 18 queries. For $H = 4$ the unclamped
starts are $0, 4, 8, 12, 16$, and $16$ is clamped to $T - H = 14$, giving $0, 4, 8, 12, 14$: 5
queries. For $H = 8$ they are $0, 8, 16 \to 10$, giving $0, 8, 10$: 3 queries.

Longer chunks mean fewer decisions, so fewer points at which drift feeds back into the input. They
also mean fewer training targets from the same demonstrations, about 40% fewer at $H = 8$ here, and
longer stretches of open-loop execution in which the policy cannot correct a mistake. On short,
forgiving episodes the second effect can outweigh the first.

</details>

### Discretized action tokens

The first of the two families of action head makes actions look like text. Each continuous actuator
value is quantized into one of about 256 bins, each bin is assigned a token in the language model's
vocabulary, and the tokens are generated with the same autoregressive loop that generates text. RT-2
(Brohan et al. 2023, [arXiv:2307.15818](https://arxiv.org/abs/2307.15818)) and OpenVLA (Kim et al.
2024, [arXiv:2406.09246](https://arxiv.org/abs/2406.09246)) both work this way.

The appeal is that no new machinery is needed: any VLM can emit actions the moment its vocabulary
has action tokens in it, and the whole pretrained next-token apparatus is inherited unchanged. Three
costs come with it. Quantization error is fixed by the bin width; 256 bins over a normalized
$[-1, 1]$ range give a resolution of $2/256 \approx 0.008$, and no amount of training gets finer
than that, which matters on motions that need fine positioning. Decoding is sequential, one forward
pass per number: a 7-actuator arm emitting a 50-step chunk costs 350 passes, so real-time control at
50 Hz becomes an engineering problem. And nothing in the token representation says that neighboring
bins are nearby values, so the model has to learn the ordering of its own action alphabet from data.

### Continuous action heads

The second family generates raw continuous actions with a small generative model conditioned on the
backbone's output. There is no discretization, so no quantization floor, and the whole chunk comes
out of one parallel decode rather than $H \times D$ sequential passes, with $D$ the number of
actuators. As of 2025 this is the dominant approach for contact-rich and dexterous manipulation.

Three constructions have been used, and they are the three generative-modeling families the course
has covered.

ACT used a *conditional variational autoencoder* (CVAE). Two networks train together: an encoder
that sees both the conditioning $c$ and the demonstrated chunk $a$ and outputs the mean and variance
of a Gaussian over a small vector $z$ called the latent code, and a decoder that sees $c$ and $z$
and reconstructs $a$. The loss is reconstruction error plus a Kullback-Leibler (KL) penalty pulling
the encoder's Gaussian toward the standard normal, the same evidence lower bound the world model's
representation is trained with. The latent $z$ absorbs whatever the conditioning does not determine,
which for demonstration data is which of several valid ways of doing the task this particular run
took. At test time the encoder is discarded, $z$ is drawn from the standard normal, and the decoder
turns it into a chunk. Its weakness is the standard variational-autoencoder one: the KL weight
trades reconstruction sharpness against how well the latent actually matches the prior it will be
sampled from, and the balance has to be tuned.

Diffusion Policy (Chi et al. 2023) used a denoising diffusion probabilistic model (DDPM), and pi0
used flow matching. Both are covered below, after the question of why the head needs to be
generative at all.

Flow matching is now the more common of the two for the action head, for two reasons that the
sections below make concrete: it trains by plain regression with no noise schedule to
choose, and it produces a sample in a handful of integration steps rather than a long chain.

### Why a generative action head instead of a regressor

Suppose the head is a plain network $f$ trained to minimize $\mathbb{E}\lVert a - f(c)\rVert^2$ over
the demonstration set. Add and subtract the conditional mean inside the square and expand:

$$\mathbb{E}\big\lVert a - f(c)\big\rVert^2
= \mathbb{E}\big\lVert a - \mathbb{E}[a \mid c]\big\rVert^2
+ \mathbb{E}\big\lVert \mathbb{E}[a \mid c] - f(c)\big\rVert^2.$$

The cross term vanishes: condition on $c$ first, and the second factor is fixed while the first has
conditional mean zero. The first term on the right does not involve $f$ at all, so minimizing over
$f$ minimizes only the second, and the minimizer is $f(c) = \mathbb{E}[a \mid c]$. Least squares
returns the conditional mean and nothing else. This is the same fact that makes a least-squares
estimator recover the noise-free conditional mean of a noisy measurement, with the noise leaving an
irreducible floor.

The conditional mean is the right action when the conditioning determines the action. It is the
wrong one when it does not. A conditional distribution $p(a \mid c)$ is *multimodal* when its
probability mass sits in several separated lumps, each a *mode*, with low probability between them.
The mean of such a distribution lands in the gap, and the gap is where no expert ever acted.

In our four-corner example, the four modes at the center have components
$\pm 0.05/\sqrt{2} \approx \pm 0.035$ and average to the zero vector. A regressor trained on that
data can do no better; it is fitting the mean, and the mean is zero.

A generative head avoids this by not producing a single answer. It produces a *sample* from a
learned conditional distribution: draw a random seed, push it through the head, get one chunk. Draw
again, get a different chunk. The head is trained so that the distribution of those samples matches
$p(a \mid c)$, so each individual sample lands on a mode and drives toward one goal, and the
averaging never happens because no average is ever formed.

> **Exercise 3 (the regressor on unequal modes).** In our four-corner example, suppose the
> demonstrations head for the goals $(0.15, 0.15)$, $(0.85, 0.15)$, $(0.15, 0.85)$, $(0.85, 0.85)$
> with probabilities $0.1, 0.2, 0.3, 0.4$ instead of a quarter each. At the center, compute the
> regressor's output, its length, and the irreducible loss floor
> $\mathbb{E}\lVert a - \mathbb{E}[a \mid c]\rVert^2$. Compare with the equal-probability case.

<details>
<summary>Answer</summary>

Write $m = 0.05/\sqrt{2} \approx 0.0354$. The four actions are $(-m, -m)$, $(m, -m)$, $(-m, m)$,
$(m, m)$. The mean has $x$-component $m(-0.1 + 0.2 - 0.3 + 0.4) = 0.2m \approx 0.0071$ and
$y$-component $m(-0.1 - 0.2 + 0.3 + 0.4) = 0.4m \approx 0.0141$. Its length is $0.0158$, under a
third of the demonstrated speed $0.05$, and it points at about $63°$, a direction none of the four
goals lies in (they are at $\pm 45°$ and $\pm 135°$).

Every action has squared length $0.05^2 = 0.0025$, so the floor is
$\mathbb{E}\lVert a\rVert^2 - \lVert \mathbb{E}a\rVert^2 = 0.0025 - 0.0158^2 = 0.00225$. With equal
probabilities the mean is zero and the floor is the full $0.0025$.

Making the modes unequal does not rescue the regressor. Its output moves toward the likeliest goal
but stays short and off every demonstrated direction, so the point mass creeps in a direction no
demonstration went. The floor is the part of the loss no regressor can remove; a generative head
uses exactly that spread to produce different actions on different draws.

</details>

### Generative models as transport from noise

Both remaining heads rest on one idea. Sampling from a complicated distribution is hard; sampling
from $\mathcal{N}(0, I)$ is a call to a random number generator. So learn a map that carries the
easy distribution onto the hard one, and sample by drawing noise and pushing it through the map. If
$z \sim \mathcal{N}(0, I)$ and $G$ is the map, training has to make the distribution of $G(z)$ match
the data. Conditioning is one more argument, $G(z, c)$: one map per conditioning vector, all sharing
weights.

Neither construction learns $G$ in one shot. Both build it as an integration over an artificial time
axis: start at the noise sample and take many small steps until arriving at a data sample. Diffusion
takes stochastic steps along a chain of noise levels fixed in advance; flow matching takes
deterministic steps along a learned velocity field. Training in both cases fits a network that says
what one small step should be.

That artificial time is not the robot's time, and the two appear in the same tensors here, so the
notes keep them apart. A chunk is an array of shape $(H, 2)$: index $h = 0, \dots, H-1$ is robot
time, $H$ consecutive control steps. The *flow time* $t$ is a separate axis along which the whole
$(H, 2)$ array is transported as a single point, all $2H$ numbers at once. With $H = 4$ the head
treats the chunk as one 8-dimensional point and moves it from noise to data, so the chunk's steps
are produced jointly and are consistent with each other by construction.

### Diffusion as an action head

A diffusion model (DDPM, Ho et al. 2020, [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)) fixes
the forward direction in advance and learns only the reverse. The forward direction destroys a chunk
by adding Gaussian noise in $N$ stages. Because a sum of Gaussians is Gaussian, the whole forward
process reduces to one closed form: with a *cumulative signal level* $\bar\alpha_t$ decreasing from
near 1 toward 0 as the integer stage $t$ runs from $0$ to $N-1$,

$$a_t = \sqrt{\bar\alpha_t}\,a + \sqrt{1 - \bar\alpha_t}\,\varepsilon, \qquad
\varepsilon \sim \mathcal{N}(0, I),$$

so $\bar\alpha_t$ is the fraction of the chunk's variance still carried by the signal at stage $t$.
The schedule $\bar\alpha_t$ is built by picking per-stage noise increments $\beta_t$ and taking
$\bar\alpha_t = \prod_{s \le t}(1 - \beta_s)$; the head built here uses $\beta$ linear from
$10^{-4}$ to $2 \times 10^{-2}$ over $N = 50$ stages. Note that $t$ here is an integer stage index,
not the continuous $[0, 1]$ flow time of the flow section below.

Training is a regression, and the target is the noise. Sample a stage $t$ uniformly, sample
$\varepsilon$, build $a_t$ by the formula above, and fit $\varepsilon_\theta(a_t, t, c)$ to
$\varepsilon$ under mean squared error. Predicting the noise rather than the clean chunk is a change
of variable, not of information: knowing $a_t$ and $\varepsilon$ gives the clean-chunk estimate
$\hat a_0 = (a_t - \sqrt{1-\bar\alpha_t}\,\varepsilon_\theta)/\sqrt{\bar\alpha_t}$ immediately. It
is the parameterization that keeps the target unit-scale at every noise level.

Sampling runs the reverse chain from $t = N-1$ down to $0$, one stage at a time. At each stage,
recover $\hat a_0$ from the current $a_t$ as above, then take the *ancestral* step, a draw from the
exact posterior of the forward process given the current sample and that estimate of the clean
chunk:

$$a_{t-1} = \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\,\hat a_0
          + \frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,a_t
          + \sqrt{\tilde\beta_t}\,z,$$

with $\alpha_t = \bar\alpha_t/\bar\alpha_{t-1}$, $\beta_t = 1-\alpha_t$, posterior variance
$\tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\,\beta_t$, and $z \sim \mathcal{N}(0,I)$
for $t>0$ with no noise added at $t=0$ (take $\bar\alpha_{-1} := 1$). The mean is a weighted average
of the estimated clean chunk and the current noisy one, with the weights set entirely by the
schedule; the added $z$ makes the reverse process stochastic, so two runs from the same starting
noise give different chunks. This is the same ancestral sampler derived in the diffusion notes, with
the conditioning $c$ carried along as an extra network input.

The cost is visible in the loop: $N$ network evaluations per sample, sequential, because each stage
needs the previous one's output. Diffusion Policy showed that a DDPM denoiser conditioned on images
produces stable multimodal action distributions, which is why diffusion became a serious action head
at all. RDT-1B (Liu et al. 2024, [arXiv:2410.07864](https://arxiv.org/abs/2410.07864)) scaled that
idea up with a transformer denoiser for bimanual manipulation; it is a reading pointer, not the
plain MLP denoiser built here.

One caveat about the small configuration used here. With $N = 50$ and $\beta$ linear over the range
above, $\bar\alpha_{N-1} \approx 0.60$, so the forward chain stops well short of pure noise while the
sampler still starts from $\mathcal{N}(0, I)$. The original DDPM used the same $\beta$ endpoints
over $N = 1000$ stages, where $\bar\alpha_{N-1} \approx 4 \times 10^{-5}$ and the mismatch is
negligible. The DDPM head here is a mechanism contrast against flow matching, not a tuned baseline,
and the reacher results are all from the flow and regression heads.

> **Exercise 4 (how far the short chain gets from the data).** Use
> $\log(1 - \beta) \approx -\beta$ to estimate $\bar\alpha_{N-1}$ for $N = 50$ and for $N = 1000$
> with $\beta$ linear from $10^{-4}$ to $2 \times 10^{-2}$. For unit-variance chunks, what is the
> correlation between the last noised chunk $a_{N-1}$ and the clean chunk $a$ in each case?

<details>
<summary>Answer</summary>

$\log \bar\alpha_{N-1} \approx -\sum_s \beta_s$, and the sum of a linear ramp is $N$ times its
average: $\sum_s \beta_s = N \cdot (10^{-4} + 2 \times 10^{-2})/2 = 0.01005\,N$. For $N = 50$ that is
$0.5025$, so $\bar\alpha_{49} \approx e^{-0.5025} \approx 0.605$ (the exact product is $0.603$). For
$N = 1000$ it is $10.05$, so $\bar\alpha_{999} \approx e^{-10.05} \approx 4.3 \times 10^{-5}$ (exact
$4.0 \times 10^{-5}$).

With unit-variance $a$ and independent unit-variance $\varepsilon$, $a_t$ has variance
$\bar\alpha_t + (1 - \bar\alpha_t) = 1$ and covariance $\sqrt{\bar\alpha_t}$ with $a$, so the
correlation is $\sqrt{\bar\alpha_t}$: about $0.78$ for $N = 50$ and $0.006$ for $N = 1000$. The
50-stage chain ends at a point that still mostly resembles the clean chunk, but sampling starts from
pure noise, which has zero correlation with any chunk, so the first reverse steps see inputs unlike
anything in training. With 1000 stages the last forward stage is essentially pure noise and the two
match.

</details>

### Conditional flow matching

Flow matching (Lipman et al. 2022, [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)) replaces
the noise chain with an ordinary differential equation (ODE). Define a time-varying velocity field
$v(z, t)$ over the space of chunks, start a point at a Gaussian sample at $t = 0$, and integrate
$\mathrm{d}z/\mathrm{d}t = v(z, t)$ up to $t = 1$. Where the point lands is the generated sample.
Any smooth velocity field defines such a transport; the training problem is finding one whose
endpoints at $t=1$ are distributed like the demonstrated chunks.

The convention matches the flow-matching notes: $t = 0$ is noise $z_0 \sim \mathcal{N}(0, I)$, and
$t = 1$ is the demonstrated action chunk $a$. Pair each demonstrated chunk with a fresh noise sample
and take the straight line between them as the path that chunk should travel:

$$z_t = (1 - t)\,z_0 + t\,a, \qquad v = \frac{\mathrm{d}z_t}{\mathrm{d}t} = a - z_0.$$

The velocity target $a - z_0$ is the same at every $t$: a point on the straight line moves in one
direction at one speed for the whole trip from $t = 0$ to $t = 1$. An implementation whose target
changes with $t$ has the wrong target.

The word *conditional* in "conditional flow matching" names what makes this trainable. The field
actually needed is the *marginal* one, the field that transports the whole Gaussian onto the whole
distribution of chunks, and there is no formula for it: many different $(z_0, a)$ pairs pass through
the same point $z_t$ at the same time $t$ with different velocities, and the marginal field at that
point is the average of them, weighted by how likely each pair was to arrive there. What is
available is the *conditional* velocity $a - z_0$, attached to one particular chunk, which is one
line of algebra. Regressing the network on the conditional target recovers the marginal field
anyway, by the conditional-mean fact used above: least squares returns
$\mathbb{E}[\,a - z_0 \mid z_t, t, c\,]$, and that conditional average is exactly the marginal
field. The flow-matching notes carry the derivation. The consequence is that a per-sample target,
which is easy to compute, trains the network to the marginal field, which has no closed form.

This is where the generative head differs from the regressor, although both are trained by least
squares. The regressor's input is only $c$, so it learns the average action. The flow head's inputs
also include $z_t$ and $t$, and given where a point currently is, the average velocity is a weighted
vote among the modes that could have produced that point. Early in the flow, near $t = 0$, every mode
is equally plausible and the velocity points toward the mean, just like the regressor. As $t$ grows,
the point is closer to one mode than the others, the vote becomes one-sided, and the point commits
to that mode. The right panel of the four-corner figure shows this: trajectories first drift toward
the center and then turn toward a corner.

So the loss is plain velocity regression, with no noise schedule and no weighting:

$$\mathcal{L}_{\text{CFM}} = \mathbb{E}_{z_0 \sim \mathcal{N}(0,I),\; t \sim U(0,1)}
\left\lVert v_\theta\big((1-t)z_0 + t\,a,\; t,\; c\big) - (a - z_0) \right\rVert^2,$$

with a fresh $z_0$ and a fresh $t$ drawn at every optimization step. The network takes three inputs:
the current point $z_t$ flattened to $2H$ numbers, the time $t$, and the conditioning $c$. The time
enters through a sinusoidal embedding, the same construction as the transformer's positional
encoding, which maps the scalar $t$ to a vector of sines and cosines at a geometric spread of
frequencies. A scalar fed directly into a linear layer gives the network one degree of freedom to
work with; the sinusoidal vector gives it a representation in which both coarse and fine differences
in $t$ are linearly readable.

At inference, integrate the ODE from $z \sim \mathcal{N}(0, I)$ at $t = 0$ to $t = 1$ with $n$
forward-Euler steps:

$$z \leftarrow z + \tfrac{1}{n}\,v_\theta(z, t, c), \qquad t = 0, \tfrac{1}{n}, \tfrac{2}{n}, \dots$$

Few steps are enough because the conditional paths are straight. Forward Euler is exact for a
constant field at any step count, and the learned field is close to constant along a trajectory
wherever the paths do not cross, so $n = 10$ suffices here against the 50 sequential evaluations the
DDPM chain needs. In the multimodal case the field is not constant: the four-corner trajectories
turn sharply late in the flow, when each point commits to a mode, so 10 Euler steps approximate
those turns rather than follow them exactly.

```mermaid
flowchart LR
  z0["z0 ~ N(0,I)<br/>(B,H,2)"] -->|"t: 0 -> 1, n Euler steps"| zt["z_t"]
  c["conditioning c<br/>(B,128)"] --> vnet["v_theta(z_t, t, c)"]
  t["time t<br/>sinusoidal embed"] --> vnet
  zt --> vnet
  vnet -->|"z += (1/n) v"| a["action chunk a<br/>(B,H,2)"]
```

> **Exercise 5 (the flow head on our four-corner example).** Use the rescaled units of the
> four-corner figure, where the four actions are $(\pm 1, \pm 1)$ with equal probability, and a
> single action step ($H = 1$). On the straight path, $z_t$ given the action $a$ is Gaussian with
> mean $t\,a$ and standard deviation $1 - t$ in each coordinate, and the conditional velocity
> written in terms of the current point is $(a - z_t)/(1 - t)$. (a) At $t = 0.5$ and
> $z_t = (0.25, -0.6)$, compute the posterior probability of each of the four actions and the
> marginal velocity. (b) Compute the marginal velocity at the same point at $t = 0$. (c) Compare
> both with what the regressor outputs.

<details>
<summary>Answer</summary>

(a) The posterior weight of action $a$ is proportional to
$\exp\big(-\lVert z_t - t\,a\rVert^2 / (2(1-t)^2)\big)$, with $2(1-t)^2 = 0.5$. The four actions
are the four combinations of independent $\pm 1$ coordinates, so the weight factors per coordinate.
In the $x$-coordinate, the ratio of the $+1$ weight to the $-1$ weight is
$\exp\big(((0.25 + 0.5)^2 - (0.25 - 0.5)^2)/0.5\big) = e^{1}$, so $P(a_x = +1) = \sigma(1) = 0.731$,
with $\sigma$ the sigmoid. In the $y$-coordinate the ratio is
$\exp\big(((-0.6 + 0.5)^2 - (-0.6 - 0.5)^2)/0.5\big) = e^{-2.4}$, so
$P(a_y = +1) = \sigma(-2.4) = 0.083$. Multiplying, the posteriors are $0.670$ for $(+1, -1)$,
$0.247$ for $(-1, -1)$, $0.061$ for $(+1, +1)$, and $0.022$ for $(-1, +1)$.

The posterior mean of the action is $(2 \cdot 0.731 - 1,\ 2 \cdot 0.083 - 1) = (0.462, -0.834)$,
which is $\big(\tanh(0.5), \tanh(-1.2)\big)$, the formula
$\mathbb{E}[a \mid z] = \tanh\big(t z/(1-t)^2\big)$ of the $\pm 1$ example in the flow-matching
notes, applied to each coordinate. The conditional velocity is linear in $a$, so the marginal
velocity is
$(\mathbb{E}[a \mid z] - z)/(1 - t) = \big((0.462 - 0.25)/0.5,\ (-0.834 + 0.6)/0.5\big) = (0.424, -0.467)$.
It points toward $(+1, -1)$, the most probable action.

(b) At $t = 0$ the noise carries no information about the action, the posterior is uniform, the
posterior mean is $(0, 0)$, and the marginal velocity is $(0 - z)/1 = (-0.25, 0.6)$: straight toward
the origin.

(c) The regressor outputs the mean, $(0, 0)$, for every input. At $t = 0$ the flow head heads toward
the same point. By $t = 0.5$ it has mostly committed to the mode nearest its current position, and it
keeps the other three only in proportion to their posterior weight. Different noise samples commit to
different modes, which produces the four clusters in the figure.

</details>

### The residual the flow loss keeps

The flow loss does not go to zero, and a training curve that flattens well above zero is not
evidence of a bug. Two effects put a floor under it, and the conditional-mean identity from above
names both.

The best any network can do is $v_\theta = \mathbb{E}[\,a - z_0 \mid z_t, t, c\,]$, and at that
optimum the loss equals the leftover conditional variance
$\mathbb{E}\,\mathrm{Var}[\,a - z_0 \mid z_t, t, c\,]$. Wherever several demonstrated chunks are
consistent with the same $(z_t, t, c)$, that variance is nonzero and no network removes it. That is
the multimodal case, and the residual there is the same spread that lets the head produce different
samples on different draws.

Even when the conditioning pins down the chunk, a second effect appears near $t = 1$. Inverting the
interpolant gives $z_0 = (z_t - t\,a)/(1-t)$, whose gain grows without bound as $t \to 1$. A finite
network fitted by least squares does not represent an unbounded map, so near $t = 1$ it falls back
toward its best bounded guess and the residual concentrates there. The linear-path CFM loss in the
flow-matching notes carries the same floor for the same reason.

The scale of the loss at the start can be estimated in advance. An untrained network outputs roughly
zero, so the loss starts at the mean square of the target: assuming unit-variance chunks,
$\mathrm{Var}(a) + \mathrm{Var}(z_0) = 2$ per component. This is an estimate; an untrained network's
output is not exactly zero, and the measured start is about 2.3. On a fixed batch, training brings
it down to a floor well above zero rather than to zero.

### How this capstone reuses the rest of the course

The action head is the generative model from the diffusion and flow-matching notes, re-conditioned
on a perception embedding in place of a class label. The CFM interpolant, the velocity target, and
the Euler ODE sampler are the same machinery as the flow-matching build; the DDPM head is the
epsilon-prediction objective and the ancestral sampler from the diffusion build. The image encoder
is the four-convolution 64x64 pixel encoder from the world model, and the reacher runs in the same
DeepMind Control Suite the world model used. The vision-language backbone this capstone stands in
for is the one built in the VLM notes.

## The assignment

Fill these holes, in order. Each is one `NotImplementedError`; the docstring in each file gives the
signature, shapes, and constraints. All but the two DDPM holes have a matching test (see below).

| # | Function | File | Concept from the notes |
|---|---|---|---|
| 1 | [`cfm_target()`](flow.py) | `flow.py` | the straight-path interpolant $z_t$ and the constant target $a - z_0$ |
| 2 | [`flow_loss()`](flow.py) | `flow.py` | the CFM loss $\mathcal{L}_{\text{CFM}}$ |
| 3 | [`flow_sample()`](flow.py) | `flow.py` | forward-Euler integration of the flow ODE |
| 4 | [`bc_loss()`](bc.py) | `bc.py` | behavior cloning by least squares |
| 5 | [`chunk_actions()`](bc.py) | `bc.py` | overlapping training windows |
| 6 | [`de_chunk()`](bc.py) | `bc.py` | the inverse of the windowing |
| 7 | [`receding_horizon_indices()`](bc.py) | `bc.py` | back-to-back execution starts, last clamped to $T - H$ |
| 8 | [`ddpm_loss()`](ddpm.py) | `ddpm.py` | the DDPM noise-prediction loss |
| 9 | [`ddpm_sample()`](ddpm.py) | `ddpm.py` | the ancestral reverse chain |

Notes on specific holes:
- `flow_loss` draws a fresh $z_0$ and a fresh $t \sim U(0, 1)$ per batch element, with $t$ of shape
  $(B, 1, 1)$ so it broadcasts over the chunk; pass the optional `generator` to both draws so the
  gradient check in `test_gradcheck.py` (described below) can make the loss deterministic.
- `flow_sample` starts from its own $z \sim \mathcal{N}(0, I)$ of shape $(B, H, 2)$, drawn with the
  `generator` argument, and evaluates the head at $t = k/n$ for $k = 0, \dots, n-1$.
- `ddpm_sample` uses the schedule from the provided `make_schedule`; the stage index passed to the
  head is an integer tensor of shape $(B,)$.
- The encoder (`nets.py`), the reacher wrapper and scripted expert (`env.py`), and the training and
  rollout helpers (`_train.py`) are provided. The code calls our four-corner example the point-mass
  side-demo; its functions are at the bottom of `env.py`.

### Running and validating

Activate the environment (`conda activate nanovision`), then:

```
make test     A=a13_vla   # run the tests against the top-level files (the ones with holes)
make verify   A=a13_vla   # run the same tests against the reference solution/
make viz      A=a13_vla   # render the figures from the reference solution
make viz-mine A=a13_vla   # render the figures from your own code (once the holes are filled)
```

`make test` is the command to run while working. It runs the suite in `assignments/a13_vla/tests/`
against the top-level files. The tests fail while the holes raise `NotImplementedError` and pass as
the holes are filled in correctly. `make verify` runs the identical suite against the reference
answer key in `solution/`: it sets `NANOVISION_IMPL=solution`, so the tests import the reference
instead of the top-level files. It passes from the start, which confirms the tests and environment
work. The goal is to make `make test` pass as `make verify` does.

The tests check:
- `test_cfm_target.py`: the interpolant at $t = 0$, $0.5$, and $1$, and that the velocity target is
  $a - z_0$ at every $t$.
- `test_flow_sample_ode.py`: a constant velocity field integrates from $z_0$ to the target exactly
  at 1, 5, 10, and 50 steps (the integrator wiring, no training).
- `test_chunking.py`: `de_chunk(chunk_actions(x))` reconstructs $x$ exactly, and the
  receding-horizon starts begin at 0, stay within $[0, T - H]$, have no duplicates, and cover every
  step.
- `test_shapes.py`: the encoder gives $(B, 128)$; the flow, BC, and DDPM heads' forward passes and
  `flow_sample` give $(B, H, 2)$; the image-to-chunk path composes.
- `test_gradcheck.py`: a float64 gradcheck of `cfm_target`, of the flow network, and of the flow
  network inside the loss. Gradcheck compares each analytic gradient against a finite-difference
  estimate in double precision and fails if they disagree.
- `test_overfit_bc.py`: the BC loss overfits one batch to near zero.
- `test_overfit_flow.py`: the flow loss drops to a small fraction of its untrained value on one
  batch, with a loose secondary bound on `flow_sample` reconstruction.
- `test_env_smoke.py`: the reacher wrapper resets, steps, and renders, and the demo collector keeps
  only successful episodes and pads them with a validity mask. Skipped when dm_control is absent.
- `test_forbidden_imports.py`: a static scan of the holed files rejects prebuilt robot, diffusion,
  and flow-matching libraries, so the heads are built from scratch.

Two gaps to know about. The suite checks `DDPMHead`'s forward shape but never calls `ddpm_loss` or
`ddpm_sample`, so nothing fails if those two holes are wrong; check them against
`solution/ddpm.py` by reading. And the reach-success and chunk-size numbers are not unit tests,
because rollout success is sensitive to initialization and seed; they live in the visualizations and
below.

dm_control is isolated to `env.py` and `viz.py`; the only test that imports it skips via
`pytest.importorskip("dm_control")`, so the graded mechanism tests run on CPU without the robot
library.

`make viz` renders from the reference solution, so it works on a fresh checkout before any holes are
filled and shows the target figures. `make viz-mine` runs the same script against the top-level
code, to check a finished implementation by eye. The figures need dm_control with headless rendering
(`MUJOCO_GL=egl`); dm_control is not in `environment.yml` by default because it is heavy and the
graded tests skip it, so install it with `pip install dm_control` (or uncomment the `dm_control` line
in `environment.yml`) before running viz. Both write PNG files to `out/` using matplotlib's headless
Agg backend, so they work over SSH, in WSL, and in CI with no display, and the figures are viewable
inline in VSCode. Add `SHOW=1` (for example `make viz-mine A=a13_vla SHOW=1`) to also open
interactive windows when a display is available. The figures are:
- `reacher_rollouts.png`: one episode each of the flow and BC policies from pixels, with each row
  labeled reached or missed.
- `chunk_ablation.png`: the BC chunk-size sweep.
- `multimodal.png`: the point-mass goal-hidden side-demo (our four-corner example, trained).
- `flow_path.png`: the ODE path from noise to the torque chunk.

The four-corner figure in the notes is generated by `assets/make_fourcorner_figure.py` from the
closed-form marginal velocity of Exercise 5. It does not use the assignment code.

What you should see when you run this. The mechanism tests run on CPU in seconds. Demo collection
runs the simulator and renders, so it is the slow part, about 130 seconds to gather 200 filtered
successful demos (roughly 270 episodes at a 75% expert reach rate). Training a flow or BC head with
its encoder is about 10-15 seconds on an RTX 4080; a 48-episode rollout in the simulator is about 30
seconds; the full visualization run is a few minutes. Everything fits well under 12GB. A healthy
`flow_loss` curve drops from about 2.3 to a floor near 0.18 and stays there; a flat curve that never
leaves about 2 means the target or the conditioning is wrong, not that training is slow. The
fixed-batch overfit test in `tests/test_overfit_flow.py` measures the same start near 2.3 and floor
near 0.18, and asserts the ratio between them rather than a near-zero final value.

The headline result is that a flow policy reading only 64x64 pixels reaches the target far more
often than a random-torque policy. The reference run measured, over 48 rollout episodes:

| policy (from pixels) | reach success |
| --- | --- |
| flow head, $H = 4$ | 0.75 |
| BC regressor, $H = 4$ | 0.75 |
| random torque | 0.06-0.07 |

![One reacher episode per policy from 64x64 pixels, frames at chunk queries: the flow policy reaches the target after 12 queries, the BC policy misses within the 80-step cap](assets/reacher_rollouts.png)

The figure shows a single episode (seed 2001) per policy, with frames taken at chunk queries, one
query per $H = 4$ environment steps. In this episode the flow policy reaches the target after 12
queries and the BC policy misses within the 80-step cap. One episode is an illustration only; the
reach-success numbers in the table come from the 48-episode rollouts.

The flow head and the deterministic BC regressor match here, for a reason specific to this task. The
image fixes the target, so from a given frame the expert action is essentially determined and
$p(a \mid \text{image})$ is unimodal, where the conditional mean a regressor learns is the correct
action. This is a measurement of this toy, not a flow-over-regression claim from pixels.

The reference chunk-size sweep on this reacher measured (mean over 2 seeds, 48 episodes each):

| chunk size $H$ | BC reach success |
| --- | --- |
| 1 | 0.78 |
| 4 | 0.74 |
| 8 | 0.62 |

![Chunk-size sweep: BC from pixels against the random-torque floor](assets/chunk_ablation.png)

On this small reacher the sweep does not rise with $H$, and the reason is specific to the setup, not
a contradiction of ACT. The episodes are short (about 20 steps), the demos are filtered to clean
successes, and the policy re-queries every chunk, so single-step BC already stays close to the
demonstrated path and reaches at 0.78. Longer chunks also lose late-episode training windows and
commit longer open-loop, which costs accuracy here (Exercise 2 counts both effects). Chunking helps
in ACT's compounding-error regime of long horizons and contact-rich dynamics where a single-step
policy drifts, and a 2D reacher with 20-step episodes is too forgiving to show it. Treat these
numbers as a measurement of this toy, not as evidence about the chunking literature.

The reacher cannot show the generative-versus-regression lesson, because the image fixes the target
and $p(a \mid \text{image})$ is unimodal. The point-mass side-demo, our four-corner example, isolates
it. In the goal-hidden mode the conditioning carries only the state, so from a fixed state the
demonstrated action points to any of the four goals and $p(a \mid c)$ is multimodal, with each mode
about $0.035$ per component. From the center state the reference run measured the BC regressor's
predicted action collapsing toward the origin (magnitude 0.0014, the average of four opposing
directions), while the flow-head samples cluster on the four diagonal goal directions: of 200
samples, 94% land within 0.01 of one of the four demonstrated actions, split 47, 45, 44, and 52
across the four goals, with a per-component standard deviation of about 0.035, the spread of the
four modes themselves.

![Point-mass side-demo: goal hidden, the BC regressor collapses to the origin while flow samples land on the four goal directions](assets/multimodal.png)

The side-demo trains both heads on actions divided by their standard deviation (about 0.035) and
multiplies the outputs back. Without that rescaling the demonstrated actions are about 30 times
smaller than the $\mathcal{N}(0, I)$ noise the flow starts from, and the trained flow head's samples
form a single blob around the origin: only 2% to 7% land within 0.01 of a mode, whether the head
is trained for 4000 or 12000 steps and sampled with 10 or 50 Euler steps. The reacher torques
already lie in $[-1, 1]$, so the pixel path needs no rescaling.

The flow-matching ODE path on the reacher, for one image conditioning, transports the Gaussian
sample to the torque chunk over the 10 Euler steps:

![Flow-matching ODE path: noise transported to the torque chunk](assets/flow_path.png)

These toy numbers demonstrate that a flow head reaches from pixels and that a flow head learns a
conditional action distribution; they do not predict pi0-scale manipulation behavior. A 2-link
reacher and a four-goal point mass are mechanism isolators, not evidence about the VLA literature.

## Further reading

- ACT / ALOHA, [Zhao et al. 2023](https://arxiv.org/abs/2304.13705) - action chunking with a CVAE
  head; the source of the chunking idea and temporal ensembling.
- DAgger, Ross et al. 2011 - the horizon analysis of behavior cloning and the interactive relabeling
  fix that chunking sidesteps.
- Diffusion Policy, [Chi et al. 2023](https://arxiv.org/abs/2303.04137) - the DDPM action head that
  made diffusion a serious option.
- pi0, [Black et al. 2024](https://arxiv.org/abs/2410.24164) - the flow-matching VLA at production
  scale; the head built here.
- OpenVLA, [Kim et al. 2024](https://arxiv.org/abs/2406.09246) - the open-source discretized-token
  VLA, the other branch of the field.
- OpenVLA-OFT, [Kim et al. 2025](https://arxiv.org/abs/2502.19645) - a same-backbone ablation
  showing the action-head design dominates the outcome; adds chunking and a continuous head for a
  large throughput and success gain.
- Octo, [Octo team 2024](https://arxiv.org/abs/2405.12213) - a generalist policy with a diffusion
  head, trained across many robots, sitting between Diffusion Policy and pi0 in the lineage.
- Open X-Embodiment, [Padalkar et al. 2023](https://arxiv.org/abs/2310.08864) - demonstrations
  pooled across many different robot bodies, the dataset behind the data scaling of recent VLAs.
- Flow Matching for Generative Modeling, [Lipman et al. 2022](https://arxiv.org/abs/2210.02747) -
  the CFM objective the action head reuses.
