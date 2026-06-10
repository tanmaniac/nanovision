# A12 — World models (RSSM / Dreamer): Validation Report

## 1. Key concepts a student must learn

### The RSSM: deterministic + stochastic latent state

The Recurrent State Space Model (RSSM), introduced in PlaNet (Hafner et al., NeurIPS 2019), is the architectural backbone of the Dreamer family. Its core insight is that a purely stochastic recurrent model is hard to optimize (gradients through samples are noisy), while a purely deterministic one cannot represent multiple plausible futures. RSSM solves this by running two components in parallel:

- A **deterministic recurrent state** `h_t` computed by a GRU: `h_t = f(h_{t-1}, z_{t-1}, a_{t-1})`. This carries memory across timesteps and is differentiable end-to-end.
- A **stochastic latent state** `z_t` sampled from a distribution `q(z_t | h_t, o_t)` (posterior, used during training) or `p(z_t | h_t)` (prior, used during imagination). In DreamerV3 this is a vector of 32 categorical distributions each over 32 classes (1024 discrete dimensions per step).

The full model state at each step is the concatenation `(h_t, z_t)`. All predictors (reward, continuation flag, decoder, actor, critic) condition on this concatenation.

The student should understand why both components are needed: `h_t` holds the deterministic context, `z_t` represents the uncertainty. This design cleanly separates "what happened up to now" from "what is likely now given ambiguity."

### The ELBO / reconstruction + KL objective

The world model is trained as a sequence VAE. The ELBO over a sequence decomposes as:

```
L = E_q [ sum_t log p(o_t | h_t, z_t)   # reconstruction
                + log p(r_t | h_t, z_t)   # reward prediction
                + log p(c_t | h_t, z_t)   # continuation prediction
        ] - beta * KL[q(z_t | h_t, o_t) || p(z_t | h_t)]  # dynamics regularization
```

The KL term pushes the learned prior `p(z_t | h_t)` toward the posterior. At imagination time (no observation), the prior is used to generate `z_t`, so training the prior to match the posterior ensures imagined rollouts stay realistic.

The reconstruction and KL are the two halves of the ELBO. The student should be clear that the KL is not a regularizer on the latent space shape (as in a beta-VAE for generation); it is specifically training the transition prior to be predictive so it can drive imagination later.

### KL balancing and free bits

These are two separate techniques that DreamerV3 combines:

**Free bits** (introduced in the original Dreamer): replace `KL` with `max(free_bits_threshold, KL)` in the loss. This prevents the model from spending capacity on trivial state variables - if the KL is already below the threshold (roughly 1 nat), the KL gradient is zeroed so the model focuses on reconstruction. In DreamerV3, free bits = 1 nat.

**KL balancing** (introduced in DreamerV2): the KL divergence between categorical distributions `q` and `p` can be written as the sum of `H_cross(q, p) - H(q)`. KL balancing applies different gradient scales to these two terms: a weight of 0.8 on the prior cross-entropy (pushing `p` toward `q`) and 0.2 on the posterior entropy (pushing `q` away from degeneracy). The key intuition is that you want the prior to move faster than the posterior so imagination stays grounded.

Together, free bits + KL balancing eliminate the need to tune the beta coefficient. The student should implement both and observe how removing either one degrades stability on a toy task.

### Categorical latents and straight-through gradients

In DreamerV2, Hafner et al. found that categorical latents work better than Gaussian latents for discrete observations. In DreamerV3, the latent is 32 vectors each drawn from a 32-class categorical, giving 32 x 32 = 1024 binary dimensions per timestep.

Sampling from a categorical is not differentiable. The **straight-through gradient estimator** handles this: in the forward pass, use the one-hot sample `z = onehot(argmax(logits))`; in the backward pass, pass gradients through as if the sample were the softmax probabilities. In code this is `z = (z_hard - probs).detach() + probs`.

DreamerV3 adds **unimix**: blend the categorical output with 1% uniform distribution before sampling. This prevents any logit from going to -inf and ensures exploration in the categorical space. Both tricks should be implemented and their effect on gradcheck verified.

### symlog and two-hot encoding

DreamerV3 introduces two tricks that together eliminate the need for reward normalization or clipping, making the same hyperparameters work across environments with wildly different reward scales (from games with rewards in [0,1] to Minecraft with rewards that can reach hundreds).

**symlog**: `symlog(x) = sign(x) * ln(|x| + 1)`. This compresses large magnitudes while being approximately identity near zero. All targets fed to neural predictors (rewards, values, reconstructed observations) are transformed with symlog. The model learns `symlog(target)` instead of `target`. At inference, predictions are inverted with `symexp(x) = sign(x) * (exp(|x|) - 1)`.

**Two-hot encoding**: instead of regressing a scalar, represent the target as a soft label over a fixed bucket sequence. For a target `y` that falls between adjacent buckets `b_i` and `b_{i+1}`, the encoding assigns weight `(b_{i+1} - y) / (b_{i+1} - b_i)` to bucket `i` and the complement to bucket `i+1`. The model outputs a softmax over 255 buckets spanning [-20, +20] in symlog space, and the loss is cross-entropy against the two-hot label. This turns regression into classification, making it robust to outliers and scale changes.

The student should implement both, understand why scalar MSE breaks under large reward variance, and verify that symlog + two-hot prediction is consistent: predict the bucket distribution, take the expected value in symlog space, apply symexp, compare to the original target.

### Imagination rollouts

Once the world model is trained, "imagination" means running the dynamics entirely inside the model without any real observations. The procedure:

1. Encode a real observation to get the starting state `(h_0, z_0)`.
2. For `t = 1..H` (horizon, typically 15): sample an action from the actor, compute `h_t = GRU(h_{t-1}, z_{t-1}, a_t)`, sample `z_t ~ p(z_t | h_t)` (the prior, not the posterior).
3. Collect the imagined trajectory `{(h_t, z_t, a_t, r_t, c_t)}`.

No decoder is called during imagination; reward and value are predicted directly from `(h_t, z_t)`. This makes imagination cheap: one GRU step + a categorical sample + a small MLP per step.

The student should verify that imagined sequences from the toy environment produce visually coherent decoded observations (using the decoder at every imagined step) and that reward predictions along the imagined path are consistent with actual rewards collected from the environment.

### Actor-critic in latent imagination

The actor `pi(a_t | h_t, z_t)` and critic `V(h_t, z_t)` are trained entirely on imagined trajectories. No real environment steps are needed during their training; they only need world model rollouts.

Critic training: compute lambda-returns along the imagined trajectory (bootstrapping with `V(h_H, z_H)` at the end of each horizon), then train the critic to predict these returns. DreamerV3 uses the critic's own exponential moving average (EMA) as the regression target to stabilize training. The critic head outputs a two-hot distribution over 255 symlog-scale buckets.

Actor training: maximize expected lambda-returns via REINFORCE plus a straight-through entropy bonus. Return normalization divides by the running 5th-95th percentile range (minimum 1) to keep the policy gradient scale invariant to return magnitude.

The student should check that the policy learned in imagination transfers to the real environment at all, even on a very small task. The goal is not peak performance but a proof that imagined actor-critic training produces a non-trivial policy.

---

## 2. Mechanisms to implement from scratch

The target environment should be a custom 4x4 or 5x5 gridworld: a few rooms, a goal square, a start square, sparse reward (+1 at goal, 0 elsewhere), episode length 50. Observations are small RGB images rendered from the grid state (32x32 or 64x64). The environment fits in a few hundred lines of NumPy code with no external dependencies.

### Component 1: RSSM cell

Implement `RSSMCell(h_dim, z_dim, z_classes, action_dim)` using only `nn.GRUCell`, `nn.Linear`, and `F.one_hot`. The cell should expose:
- `prior(h, z, a) -> (h_new, z_dist)` - transition using the learned prior (imagination).
- `posterior(h, z, a, o_enc) -> (h_new, z_dist)` - transition conditioning on the encoder output (training).

Verifiable tasks:
- Shape test: given batch of `(h, z, a)`, verify `h_new.shape == (B, h_dim)` and `z_sample.shape == (B, z_classes)` for each of the 32 categorical heads.
- `gradcheck` on a small single-step forward pass (use float64, small batch, reduce z_dim and z_classes to 4x4 for speed).
- Overfit one batch: on 8 short sequences from the gridworld, train the RSSM + decoder to overfit to zero reconstruction loss. Verify that imagined sequences from the starting state decode to visually plausible grid images.

### Component 2: World model training (ELBO)

Implement the full sequence training loop:
- CNN encoder: observation `o_t` -> `e_t` (small ConvNet, 3 layers, channels [32, 64, 128]).
- RSSM: encode sequence, compute posteriors.
- CNN decoder: `(h_t, z_t)` -> reconstructed `o_t`.
- MLP predictors: reward head, continuation head, both with symlog + two-hot outputs.
- Loss: reconstruction (MSE on symlog-decoded predictions, or cross-entropy on two-hot) + KL with free bits and KL balancing.

Verifiable tasks:
- Reconstruction loss should decrease to near zero on a single 10-step sequence within a few hundred gradient steps.
- KL should converge to near the free-bits threshold (not collapse to zero, not diverge).
- Imagined rollouts from a fixed starting state should decode to images that track the correct room layout.

### Component 3: Actor-critic in imagination

Implement actor `pi(a | h, z)` (MLP -> softmax over 4 actions) and critic `V(h, z)` (MLP -> 255-bucket two-hot distribution). Train them only on imagined trajectories from the world model.

Verifiable tasks:
- Lambda-return computation: write `compute_lambda_returns(rewards, values, continues, gamma, lambda_)` and verify on a hand-crafted sequence with known ground truth.
- Policy transfer: run the trained actor in the real gridworld for 100 episodes. It should solve the task (reach goal at least 30% of the time) while a random policy achieves roughly `1/episode_length` success rate (~2%).
- Latent rollout visualization: decode imagined trajectories from 5 different starting positions and verify that the agent appears to navigate toward the goal in imagination.

### Tiny-model sizing (fits 12GB RTX 4080 comfortably)

- h_dim = 256, z_dim = 32 x 16 (32 categoricals, 16 classes each - reduced from the full 32x32 for speed)
- Encoder: 3 conv layers, 32/64/128 channels, 4x4 kernels, stride 2, 64x64 input -> 8x8 feature map -> 2048-dim flatten -> 512-dim linear
- Decoder: transpose conv mirror
- Actor/critic: 2-layer MLP, 256 hidden units
- Imagination horizon: 15 steps
- Batch: 16 sequences x 16 steps; training: ~10k world model steps, ~5k actor-critic steps

---

## 3. Assessment of the draft scope

### What the draft gets right

The draft correctly identifies DreamerV3 as the implementation target. The choice is well-founded: it is the current definitive latent-control world model with a single fixed configuration that works across many tasks, published in Nature (2025). The core loop - RSSM + ELBO + imagination + actor-critic - is the right four-part decomposition.

The dependency links (A0/A1, A5/A7) are correct. The toy gridworld framing is appropriate.

### What is missing or under-specified

**symlog + two-hot are under-weighted in the draft.** The draft lists them as a parenthetical "(symlog/two-hot tricks)" at the end of the concept list. These are not tricks; they are the mechanism that makes DreamerV3 work across diverse reward scales and are one of its two main technical contributions over DreamerV2. They should be a first-class concept with their own implementation exercise.

**KL balancing and free bits need to be separated.** The draft groups them as "KL balancing and free bits" without distinguishing that they address different failure modes (prior accuracy vs. posterior collapse). A student who conflates them will mis-implement both.

**Unimix is absent.** The 1% uniform mixture on all categoricals is a simple but important stabilizer. It warrants a one-paragraph treatment and should be part of the implementation.

**Actor-critic return normalization is absent.** Normalizing by the running percentile range is a DreamerV3-specific contribution that prevents the actor gradient from collapsing or exploding. It belongs in the concept list.

**The straight-through estimator should be a named mechanism.** The draft mentions categorical latents but does not name the straight-through trick. This is worth an explicit two-line derivation since students will need to write the detach() call correctly.

### The video world model question

The draft's scope implicitly positions RSSM as the only type of world model worth knowing. This is incomplete given the 2024-2026 context.

**What changed:** The period 2023-2025 produced a second branch of world models that operate in pixel/video space and scale with data and model size: GAIA-1 (Wayve, 2023, driving), Genie (DeepMind, 2024, 2D interactive environments), Genie 2 (DeepMind, Dec 2024, 3D), DIAMOND (NeurIPS 2024, diffusion-based Atari agent), DreamerV4 (Hafner et al., Sep 2025, latent video diffusion + transformer for Minecraft), and GAIA-3 (Wayve, 2025). These models are primarily trained on large offline video corpora and generate pixel-level outputs, rather than learning a compact latent dynamics model for online RL.

**Why RSSM is still the right thing to BUILD:** Video world models require billions of parameters, terabytes of training data, and multi-GPU clusters that far exceed a 12GB RTX 4080. DreamerV3's full model for DMControl fits in under 2GB. The RSSM is the buildable, inspectable core of model-based RL. The objective (learn to act, not just predict video) is cleaner pedagogically. Every concept in the RSSM - ELBO, latent transition, imagination, actor-critic in latent space - has a direct mathematical reason to exist.

**What should be added as a reading note, not a build target:** Genie/Genie 2 (action-labeled video -> interactive world model), DIAMOND (diffusion world model for RL), DreamerV4 (scaling RSSM to video diffusion for Minecraft), and GAIA (driving video generation). Students should understand the distinction: latent control world models (RSSM/Dreamer) optimize for planning efficiency; pixel-space generative world models optimize for visual fidelity and data-scale generalization. The two families are converging in DreamerV4 but are still distinct design points.

**JEPA as a third alternative:** V-JEPA (Bardes et al., 2024) and related JEPA-style models represent a third approach: learn latent dynamics by predicting future embeddings rather than reconstructing observations. This avoids the "decode every frame" cost and is theoretically cleaner (LeCun's argument that good world models should predict in feature space, not pixel space). The student should read about this contrast but not build it in A12. A brief comparison note - RSSM uses reconstruction as a training signal, JEPA-style models discard reconstruction and predict only in latent space - would help students place RSSM in the broader landscape.

**Sora and the "world simulator" debate**: Sora (OpenAI, 2024) and the debate over whether large video generation models are world simulators is worth one paragraph. The short answer: Sora and similar models generate plausible-looking video but do not represent causal world state; they cannot answer "what happens if I take action X" in a controllable, repeatable way. RSSM-based models can, which is why they remain the right design for model-based RL despite being far less visually impressive.

### Reordering

Suggested concept order:
1. The RSSM: deterministic + stochastic state (the "why both" argument)
2. The ELBO / reconstruction + KL objective (the training signal)
3. KL balancing (prior accuracy) and free bits (posterior non-collapse) - separate treatments
4. Categorical latents and straight-through gradients (with unimix)
5. symlog and two-hot encoding (scale invariance - should be elevated, not listed last)
6. Imagination rollouts (prior-mode inference)
7. Actor-critic in imagination (lambda-returns, return normalization)

---

## 4. Dependencies and connections

**Depends on A0 (PyTorch fundamentals, autograd):** The straight-through estimator requires manually writing the detach() pattern, which requires understanding that `.detach()` breaks the autograd graph. gradcheck on the RSSM cell requires float64 and careful handling of discrete samples. These are non-trivial autograd exercises that assume A0.

**Depends on A1 (transformers / sequence models):** The GRU in the RSSM is a simpler version of what A1 covers. Students who have implemented multi-head attention and positional embeddings in A1 will find the GRU straightforward. The sequence-level ELBO training loop is a direct application of the training infrastructure built in A1.

**Conceptual link to A5 (VAEs) and A7 (latent diffusion):** The ELBO in the world model is exactly the sequence VAE objective. The KL term, the posterior vs. prior distinction, and the reconstruction loss all appeared in A5. The latent space concept - compress observations to a compact code, operate in that code space - is shared with A7. The difference is that A5/A7 are generative models for static data, while A12 adds dynamics: the latent evolves over time conditioned on actions.

**Feeds A13 (VLA as latent dynamics):** Vision-language-action models like RT-2, OpenVLA, and more recent architectures treat robot control as a sequence-to-sequence problem, but the framing of "learn a latent world model, plan in latent space, decode to actions" is increasingly appearing in embodied AI. DreamerV4 and similar work show how to scale the latent dynamics idea to realistic visual environments. A student who has built an RSSM in A12 can understand A13's latent planning architectures without confusion.

---

## 5. Must-read sources

1. **PlaNet — "Learning Latent Dynamics for Planning from Pixels"** (Hafner et al., NeurIPS 2019, arXiv:1811.04551). Introduces the RSSM, the ELBO training objective for sequence models, and latent overshooting. The foundational paper; read before Dreamer.

2. **DreamerV1 — "Dream to Control: Learning Behaviors by Latent Imagination"** (Hafner et al., ICLR 2020, arXiv:1912.01603). Adds the actor-critic trained in imagination on top of PlaNet's world model. The first paper to show that imagined rollouts are sufficient for competitive policy learning.

3. **DreamerV3 — "Mastering Diverse Domains through World Models"** (Hafner et al., Nature 2025; arXiv:2301.04104, originally posted Jan 2023). The primary build target. Introduces KL balancing, free bits, categorical latents, symlog, two-hot, unimix, and the single-configuration approach. The Nature publication (Apr 2025) is the definitive version; the arXiv is freely accessible.

4. **DreamerV2 — "Mastering Atari with Discrete World Models"** (Hafner et al., ICLR 2021, arXiv:2010.02193). Bridges V1 and V3. Introduces categorical latents and KL balancing on Atari. Worth reading as an intermediate step but can be skipped if time is short; V3 supersedes it technically.

5. **Genie 2** (Bruce et al., Google DeepMind, Dec 2024, blog post / technical report). The representative large-scale video world model: 3D interactive environments generated from a single image, trained on web-scale video. Read to understand what pixel-space generative world models look like at scale, and why they are not the right build target for a 12GB GPU.

6. **DIAMOND — "Diffusion for World Modeling: Visual Details Matter in Atari"** (Alonso et al., NeurIPS 2024, arXiv:2405.12399). Replaces the RSSM latent with a diffusion model, achieving human-normalized score 1.46 on Atari 100k. A useful contrast: shows that pixel-space diffusion world models are now competitive with latent methods on specific benchmarks, but are also larger and slower.

7. **DreamerV4 — "Training Agents Inside of Scalable World Models"** (Hafner, Yan, Lillicrap, Sep 2025, arXiv:2509.24527). Shows where the RSSM lineage is heading: a latent video diffusion model (masked autoencoder tokenizer + flow-matching transformer) trained on 2500 hours of Minecraft video, then fine-tuned for agent control. Read as a "where does A12 lead" paper, not as a build target.

---

## 6. 2024-2026 developments that change how this should be taught

### DreamerV3 published in Nature (April 2025)

The arXiv preprint is from January 2023, but the peer-reviewed Nature publication appeared in April 2025 with the title "Mastering diverse control tasks through world models." This is now the canonical citation. The content is the same as the arXiv, but instructors should cite the Nature version.

### DreamerV4 (September 2025)

DreamerV4 is a significant architectural departure: the GRU-based RSSM is replaced by a block-causal transformer with flow-matching dynamics, trained offline on 2500 hours of Minecraft video and then fine-tuned with RL. It achieves Minecraft diamond collection far more reliably than V3. Teaching implications: the RSSM is now a "V1-V3 era" architecture; students should be aware that the latent dynamics field has moved toward transformer + diffusion backends. However, DreamerV3 remains the right build target because its components (GRU, categorical VAE, actor-critic) are each independently graspable and do not require a large pre-training dataset.

### DIAMOND (NeurIPS 2024)

DIAMOND replaces the RSSM with a denoising diffusion model in observation space and trains an RL agent inside imagined diffusion trajectories. It achieves state-of-the-art on Atari 100k with a mean human-normalized score of 1.46. This is the first paper to show diffusion models are viable as the dynamics backbone for RL. It should be mentioned as a reading note after A12 to show the diversity of world model architectures.

### Genie / Genie 2 and the "generative world model" family

Genie (2024) and Genie 2 (December 2024) from Google DeepMind train on large video corpora to produce action-controllable interactive worlds. Genie 2 generates 3D environments from a single image at interactive frame rates. These models are not designed for RL agent training in the sense that Dreamer is; they are closer to interactive video generation with action conditioning. The practical distinction to teach: RSSM models are optimized for control (compact latent, fast imagination, explicit reward model), while Genie-style models are optimized for visual fidelity and generalization across diverse environments. Both are called "world models" in the literature, which causes confusion; the course should explicitly define the distinction.

### JEPA-style latent prediction

V-JEPA (Bardes et al., Feb 2024) and the broader JEPA family propose learning world models by predicting future latent representations rather than reconstructing observations. LeCun has argued strongly that reconstruction is unnecessary and that good representations should be learned by prediction alone. The RSSM does reconstruction (to keep the latent grounded in observations); JEPA-style models do not. The debate is ongoing and unresolved as of mid-2026. For A12, the practical recommendation is: teach the RSSM as the buildable mechanism and note the JEPA alternative as a conceptual contrast - reconstruction-based vs. prediction-based world modeling - without building it.

### Sora and the "world simulator" debate

OpenAI's Sora (February 2024) was introduced with a framing as a "world simulator." The claim was contested: Sora generates visually plausible video but does not maintain consistent world state or support reliable action conditioning. Sora 2 (2025/2026) improved physics fidelity but the fundamental critique remains. Students should understand this distinction: a world model for control requires a queryable causal model (if I take action a, what happens next?); a video generation model is a conditional distribution over pixel sequences without an explicit action interface. RSSM addresses the former; Sora addresses the latter.

### Robustness of DreamerV3 tricks in downstream work

Multiple 2023-2024 papers (including "Reward Scale Robustness for Proximal Policy Optimization via DreamerV3 Tricks," NeurIPS 2023) showed that DreamerV3's symlog + two-hot tricks transfer to other RL algorithms beyond Dreamer itself. This suggests these are general tools worth understanding independently of the world model context. Teaching them as standalone reward processing techniques (applicable even to model-free RL) is worth a brief note.
