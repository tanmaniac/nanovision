# A11.5e — Unified perception -> prediction -> planning: validation report

## 1. Key concepts the student must learn

### Motion forecasting: agent-centric vs scene-centric

In an **agent-centric** formulation, the scene is rotated and translated so the target agent sits at the origin with its heading aligned to the x-axis. Each agent gets its own normalized view of the world. This is pose-invariant and generalizes well, but in a multi-agent batch you must re-normalize the scene once per target, so inference time scales linearly with agent count - a real-time problem.

In a **scene-centric** formulation, all agents share a single global (or vehicle-body) coordinate frame. One forward pass handles the whole scene, which is necessary for the joint prediction problem. The cost is that the model must learn to handle arbitrary positions and headings in the input. Most production-grade systems (SceneTransformer, MTR++, QCNet) now operate in scene-centric or query-centric frames for efficiency. The student should understand both and know why the field migrated toward scene-centric joint prediction.

### The multi-future problem and multimodal trajectory prediction

A pedestrian at an intersection will go left, right, or straight. There is no single correct future. A model trained to minimize mean L2 over a single output learns to predict the average of all plausible futures - a physically impossible trajectory passing through the middle of every option. This **mode averaging** problem is why the field uses **multimodal** prediction: the model outputs K trajectory hypotheses (typically K = 6 for benchmarks) with associated scores or probabilities.

The two families of training loss for multimodal heads are:
- **Winner-take-all (WTA) / min-of-N**: at each training step, only the hypothesis closest to the ground-truth future receives gradient. This avoids mode averaging but risks mode collapse where all K modes converge. It is the workhorse loss for regression heads.
- **Likelihood-based (GMM / Laplace NLL)**: model the distribution over futures explicitly as a mixture; minimize the negative log-likelihood of the ground truth under the mixture. More principled but requires calibrating mixture weights.

Evaluation metrics encode the same idea: **minADE_K** and **minFDE_K** measure the best (minimum) error over the K predictions, rewarding models that cover the future distribution rather than committing to one mode. **MissRate** counts predictions where the best prediction at the horizon still exceeds a threshold (2 m on nuScenes/Argoverse). Understanding that these "oracle" metrics are necessary but incomplete - a model can score well by outputting K near-identical modes - is a required nuance.

### Anchor/goal-based vs regression prediction

**Regression heads** directly predict K trajectories as (x, y) sequences via an MLP or transformer decoder applied to scene features. Simple to implement; the WTA loss trains them straightforwardly.

**Goal-conditioned / anchor-based heads** (TNT, DenseTNT, MTR) decompose prediction into two steps: (1) predict a distribution over candidate endpoints or intention points; (2) decode a full trajectory conditioned on each selected goal. This factorization concentrates the multimodal uncertainty in the endpoint distribution, which is easier to model because it is lower-dimensional. MTR uses a small set of K learnable "motion intention queries" (roughly 64) as spatial anchors covering the mode space, refines them with local attention, and regresses trajectory offsets per anchor. Anchor-based models consistently lead motion-forecasting leaderboards.

### Query-based agent representation

UniAD and its lineage represent each tracked agent as a **persistent query vector** that carries the agent's state through the module stack. In UniAD's TrackFormer, each detected agent is assigned a track query embedding. Those queries are then passed directly into MotionFormer as motion queries, which attend over BEV features, map features, and other agent queries to model social interactions. The final output is decoded from the motion query via an MLP head.

The key insight is that the query is a latent summary of everything the model knows about an agent. Passing queries between modules (track -> motion -> occupancy -> plan) enables end-to-end gradient flow without needing hard intermediate outputs like detected bounding boxes. It also lets MotionFormer know "what the tracker believed about this agent" rather than working from raw sensor features again.

### How end-to-end differentiable AV stacks chain detection -> tracking -> prediction -> planning

A modular stack passes hard discrete outputs between stages: the tracker consumes detections, the predictor consumes tracks, the planner consumes predicted trajectories. Each hand-off is non-differentiable. Errors in upstream modules cannot be corrected by downstream gradient signal.

UniAD's key design was to replace each hand-off with **soft query-based communication**: the outputs of TrackFormer are query vectors fed into MapFormer's cross-attention; those updated queries flow into MotionFormer; MotionFormer outputs BEV occupancy for OccFormer; OccFormer and MotionFormer outputs condition the Planner. All stages are transformer decoders sharing the same BEV feature backbone. Because every link is a differentiable attention operation, the end-to-end loss (planning L2 + auxiliary perception/prediction losses) backpropagates through the full stack. The planning loss can in principle pull better features out of the BEV encoder even though detection and prediction are not the final objective.

VAD pushes this further by replacing rasterized BEV occupancy maps with **fully vectorized scene elements** (agents as trajectory sequences, map as polylines). This eliminates the intermediate dense grid entirely, cutting memory and compute while preserving instance-level structure.

---

## 2. Mechanisms to implement from scratch

### 2.1 Multimodal trajectory head with WTA loss (the primary build target)

**What to build:** Given a BEV feature map (real or synthetic), extract a fixed-length agent feature vector via RoI-align or a simple bilinear sample at the agent's BEV location. Feed that vector through a small transformer decoder (2-4 layers) with K=6 learned mode queries as initial decoder queries. Each mode query decodes to a trajectory of T=12 steps at 0.5 s intervals (6 s total, matching nuScenes). Apply a linear head to each decoded query to get (x, y) offsets at each step, giving shape `[K, T, 2]`. Train with WTA loss: compute FDE from each of K predictions to the ground-truth endpoint, select the winning mode, and apply L2 regression loss only on that mode's full trajectory.

**Minimal verifiable tasks:**
1. Shape test: feed a batch of agent BEV features `[B, C]`; output `[B, K, T, 2]` with K=6, T=12.
2. `torch.autograd.gradcheck` on the WTA loss with double precision on a single sample.
3. Overfit one scene: given a single agent's BEV crop and GT trajectory, drive training loss below 0.05 m after 500 steps. Visualize all K modes; check that at least one tracks the ground truth and others are plausible (not all collapsed to the same prediction after convergence).

**Where nuScenes mini fits in:** Use the nuScenes prediction split (annotation-based GT future trajectories at 2 Hz, 6 s horizon) on 5-10 agents from 1-2 scenes. The input BEV features can come from a pre-trained BEV encoder frozen weights from the earlier BEV module (A11b/c) - the head is trained in isolation on these frozen features. This keeps GPU memory and training time minimal.

**Optional extension (anchor/goal-based variant):** Replace the K mode queries with K learned intention points (2D positions) initialized from a k-means of training endpoints. For each intention point, concatenate its coordinate to the agent feature and decode a single trajectory. Add a classification head to score each intention point. Train with cross-entropy on the correct intention class plus regression loss on the winning trajectory. This is the minimal version of the TNT/MTR idea and can be verified against the regression-head version.

### 2.2 Written survey note: mapping the E2E stack to the student's prior mental model

This is explicitly a non-code deliverable. The note should:
- Draw the modular pipeline the student already knows (sensor -> det -> track -> pred -> plan) and annotate where the hand-offs were non-differentiable.
- Overlay the UniAD query-passing design and show which arrows became soft attention.
- Explain the planning head: UniAD's planner is a simple MLP that takes the ego BEV query plus predicted agent occupancy and outputs a sequence of ego waypoints. It is not an RL planner or an optimization-based planner - it is a regression head with an imitation loss against human driving.
- Note what is NOT built: OccFormer, the full tracker, differentiable map encoding. The student builds one stage (prediction head) and reads the rest.

---

## 3. Assessment of the draft scope

### What is right

The split between a build part (trajectory head) and a survey part (E2E stack) is correct. The 12-GB constraint makes building the full UniAD chain impractical, so isolating the prediction head is the right call. The written note bridging old and new mental models is a valuable learning device the draft correctly identifies.

### What is missing or under-specified

**The open-loop metric caveat must be a primary teaching point, not a footnote.** Li et al., "Is Ego Status All You Need for Open-Loop End-to-End Autonomous Driving?" (CVPR 2024) showed that a trivial MLP (AD-MLP) taking only ego vehicle velocity and heading achieves competitive L2 / collision-rate scores on nuScenes open-loop planning. 73.9% of nuScenes driving is straight-ahead; a constant-velocity extrapolation beats most perception-conditioned planners on L2. This is not a minor caveat: it means the standard nuScenes planning benchmark does not measure what it claims to measure, and any method (UniAD included) that reports competitive numbers on it should be interpreted with caution. The student must learn this before treating nuScenes planning scores as meaningful.

**The distinction between open-loop and closed-loop evaluation** needs explicit treatment. Open-loop: roll out one prediction step and compare to logged ground truth. Closed-loop: the policy controls the vehicle in a simulator; other agents react to the ego. The field's shift to NAVSIM, Bench2Drive, and CARLA-based closed-loop eval (2024-2025) is a direct response to the AD-MLP critique. The build exercise is necessarily open-loop (no simulator), so the student must be told explicitly what that limitation means.

**The multimodal head loss deserves precise specification.** The draft says "winner-take-all / min-of-N" which is correct but needs to specify the metric used to select the winner (minFDE at the endpoint, not minADE over the trajectory - this is the standard because endpoints carry most uncertainty), and should mention that the regression targets should be in ego-relative (agent-centric) coordinates to keep the regression range small.

**Joint vs. marginal prediction** is not mentioned. The head described is marginal (one agent at a time). Scene-level social interaction - other agents' predictions affecting ego-agent prediction - is the next step and is what MotionFormer handles via agent-to-agent cross-attention. The survey note should at minimum name this distinction.

**The BEV feature dependency should be made concrete.** The draft says "BEV features" but does not specify what the feature is at each agent location. The practical choice is: (a) RoI-align a patch around the agent's BEV position, then flatten+project to a feature vector; or (b) sample a single BEV cell at the agent's location. Option (a) is more realistic and closer to what MotionFormer does. This should be specified so the student is not left guessing.

### What is outdated or mis-emphasized

**UniAD's rasterized BEV occupancy** is an acknowledged limitation; VAD (ICCV 2023) and SparseDrive (2024) both move to vectorized or sparse representations that are faster and more scalable. The survey note should flag UniAD as the foundational formulation but note that production-direction architectures dropped the dense BEV occupancy intermediate.

**DriveTransformer (ICLR 2025)** further challenges the sequential module chain by running perception, prediction, and planning queries in parallel with shared attention, removing the ordering assumption entirely. It achieves state-of-the-art on both open-loop (nuScenes) and closed-loop (Bench2Drive) at the time of writing. This should be named in the survey as the current representative architecture.

**VLM/LLM planners** (DriveVLM, DriveLM, GPT-Driver) are a parallel development where a language model reasons over scene descriptions and outputs waypoints or high-level commands. These are real and growing in adoption (DriveLM at ECCV 2024; DriveVLM from Tsinghua MARS lab). They are architecturally different from the query-chain formulation and belong in the adjacent-topics note pointing toward A13 (VLA), not in the main survey - they would distract from teaching the prediction mechanism. The note should exist, but it should be one paragraph.

**Trajectory diffusion models** (Diffusion Policy applied to driving) are a 2024-2025 trend that the draft does not mention. They are not core to this topic but worth one sentence in the 2024-2026 update.

### Suggested reorder/reframe

1. Teach the multi-future / WTA loss problem with a toy example (one agent, K modes) before showing it in the context of E2E stacks.
2. Show the hand-written modular pipeline and the UniAD query-passing version side by side.
3. Build the multimodal regression head on frozen BEV features.
4. Survey UniAD -> VAD -> DriveTransformer as the rasterized -> vectorized -> parallel-query lineage.
5. Flag the open-loop metric problem and point to closed-loop benchmarks.
6. Bridge to A13 (VLA) and A12 (world models) in the closing note.

---

## 4. Dependencies and connections

**Depends on:**
- BEV encoding (A11b/c): the trajectory head's input is a BEV feature map. Without a working BEV encoder the input must be synthetic, which is fine for the implement-the-mechanism philosophy but the connection should be made explicit so the student sees this as the prediction stage of the BEV stack.
- Occupancy prediction (A11d): OccFormer in UniAD sits between MotionFormer and the Planner. Understanding occupancy prediction - what it gives the planner that trajectory predictions do not (dense free-space map vs. per-agent modes) - is necessary to understand why UniAD has both modules. The student can survey this without building it.
- Transformer decoder mechanics (A01/A02): the mode-query decoder is a standard transformer decoder. The student must already understand cross-attention to follow MotionFormer's design.

**This topic is the capstone of the AV module.** It assembles camera -> BEV (A11b), BEV -> occupancy (A11d), and now BEV -> agent trajectories -> ego plan into a single chain. It is the place where the student sees how all the intermediate representations (occupancy, tracked queries, map polylines) serve the downstream planning objective. The planning orientation - every prior module exists to enable better planning - is UniAD's central thesis and should be stated explicitly.

**Conceptual adjacency:**
- A12 (driving world models / video prediction): world models predict future sensor observations conditioned on ego actions, which can substitute for or augment the trajectory prediction head when the future scene is uncertain. The WoVogen / GAIA / DriveWorld-VLA line of work is the bridge. The student should be told: "the prediction head you built is a behavioral model; a world model is a sensory model of the future - different outputs, complementary uses."
- A13 (VLA / language-conditioned planning): VLM planners replace the implicit route-following behavior learned from imitation with an instruction-conditioned policy. The query-chain design (UniAD, VAD) has no language interface; VLM planners (DriveLM, DriveVLM) have weak geometric prediction but strong scene understanding and instruction following. The student should see this as a different tradeoff, not a direct successor.

---

## 5. Must-read sources

**UniAD** - Hu et al., "Planning-Oriented Autonomous Driving," CVPR 2023 (Best Paper). The reference architecture for query-based E2E stacks; defines TrackFormer -> MapFormer -> MotionFormer -> OccFormer -> Planner. https://arxiv.org/abs/2212.10156

**VAD** - Jiang et al., "VAD: Vectorized Scene Representation for Efficient Autonomous Driving," ICCV 2023. Replaces UniAD's dense BEV occupancy with fully vectorized agent and map tokens; 2.5x faster with lower collision rate. https://arxiv.org/abs/2303.12077

**VADv2** - Chen et al., "VADv2: End-to-End Vectorized Autonomous Driving via Probabilistic Planning," arXiv 2024. Extends VAD with a discrete planning vocabulary and probabilistic distribution over trajectory tokens; state-of-the-art closed-loop on Bench2Drive. https://arxiv.org/abs/2402.13243

**AD-MLP / "Ego Status" critique** - Li et al., "Is Ego Status All You Need for Open-Loop End-to-End Autonomous Driving?" CVPR 2024. Shows a velocity-only MLP matches perception-conditioned planners on nuScenes open-loop metrics; introduces road-boundary adherence metric. Required reading before interpreting any nuScenes planning number. https://arxiv.org/abs/2312.03031

**MTR / Motion Transformer** - Shi et al., "Motion Transformer with Global Intention Localization and Local Movement Refinement," NeurIPS 2022 (winning Waymo 2022 challenge). The cleanest anchor/goal-based regression architecture; intention queries + local refinement is the design template for the build exercise's optional extension. https://proceedings.neurips.cc/paper_files/paper/2022/file/2ab47c960bfee4f86dfc362f26ad066a-Paper-Conference.pdf

**VectorNet** - Gao et al., "VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized Representation," CVPR 2020. The foundational paper establishing vectorized (polyline) encoding of HD maps and agent histories as input to trajectory prediction; still the standard input representation in most modern predictors. https://arxiv.org/abs/2005.04259

**DriveTransformer** - Jia et al., "DriveTransformer: Unified Transformer for Scalable End-to-End Autonomous Driving," ICLR 2025. Drops the sequential module chain in favor of parallel task queries (agent/map/plan) attending jointly to sensor features at every layer; removes cumulative-error problem of sequential stacks; SOTA on nuScenes and Bench2Drive. https://arxiv.org/abs/2503.07656

---

## 6. 2024-2026 developments that change how this should be taught

### The open-loop planning benchmark is effectively invalidated for perception-conditioned methods

The AD-MLP result (CVPR 2024) is the most important update for this course. Any nuScenes planning score should come with the caveat that it measures imitation of recorded human driving on a dataset dominated by straight-ahead motion, not generalizable planning behavior. The teaching should frame open-loop nuScenes as a development/debugging tool, not a measure of real planning quality. The student who does not learn this caveat will misread the literature.

### Closed-loop benchmarks are now the primary evaluation target

NAVSIM (non-reactive, real-world clips from nuPlan) and Bench2Drive (CARLA-based, fully reactive) are the 2024-2025 standard evaluation environments. NAVSIM has an ECCV 2024 challenge and a growing leaderboard. Bench2Drive has 10,000 CARLA clips and 220 closed-loop evaluation routes. VADv2 and DriveTransformer both report on Bench2Drive. The build exercise cannot use these (no simulator), but the student should know the field has moved evaluation there.

### End-to-end imitation is challenged by RL and hybrid approaches

Several 2024-2025 papers apply offline RL or RLHF-style fine-tuning to E2E driving models to improve closed-loop safety beyond what pure imitation achieves. This is early-stage but directionally important: pure BC (behavior cloning from demonstrations) tends to fail in long-tail situations the training data does not cover. For the course scope, this is a one-paragraph note.

### The sequential stack assumption is under pressure

DriveTransformer (ICLR 2025) empirically demonstrates that running all task queries in parallel (rather than sequentially from perception to planning) produces more stable training and better performance. The rationale for the sequential design in UniAD was practical (training stability) rather than principled; parallel attention removes it. Teaching should note UniAD as the first clean E2E formulation and DriveTransformer as the direction the field is moving architecturally.

### VLM/LLM planners are a parallel track, not a replacement

GPT-Driver, DriveLM (ECCV 2024), DriveVLM, and OmniDrive have all shown that language-model backbones can serve as planners using chain-of-thought reasoning over scene tokens. They trade geometric precision and latency for interpretability and instruction following. As of mid-2026, they have not displaced query-chain E2E models on closed-loop benchmarks, but they are the research frontier for human-AI interaction and edge-case reasoning. Teaching should connect this to A13 without conflating it with the trajectory-regression mechanism being built.

### Trajectory diffusion models are emerging

Several papers (2024-2025) apply score-matching diffusion to multi-agent trajectory prediction, learning the joint distribution over futures rather than regressing K modes. Results on Argoverse 2 are competitive. This is not a replacement for the WTA regression head as a teaching mechanism - the latter is simpler and more pedagogically direct - but it is worth one sentence noting that the multi-future problem can also be framed as density estimation.

### nuScenes mini remains an appropriate substrate

Despite the benchmark limitations above, nuScenes v1.0-mini is still the correct dataset for the build exercise: it provides GT future trajectories for agents at 2 Hz, HD map annotations (lane boundaries, crosswalks), and a manageable scale (10 scenes, ~350 annotated agent trajectories). The prediction head can overfit a handful of agents from 2-3 scenes in minutes on a 12 GB GPU, which is exactly the verify-the-mechanism goal.
