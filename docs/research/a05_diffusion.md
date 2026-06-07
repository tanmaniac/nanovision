# A5 — Diffusion (DDPM / DDIM): validation report

**Date:** 2026-06-06  
**Scope reviewed:** Forward noising schedule (linear + cosine); time-embedded U-Net; epsilon-prediction objective; ancestral (DDPM) + DDIM sampling; classifier-free guidance; unconditional MNIST/shapes generation. Stretch: v-prediction, varying DDIM steps, class-conditional CFG.

---

## 1. Key concepts a student must learn

### Forward process and the noising schedule

The forward process is a Markov chain q(x_t | x_{t-1}) = N(x_t; sqrt(1-beta_t) x_{t-1}, beta_t I). The key closed form q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) x_0, (1-alpha_bar_t) I) lets you jump to any noise level in one shot; this is what makes training tractable. The student must derive this closed form and understand why it follows from the product of Gaussians.

Two schedules matter:
- **Linear schedule** (Ho et al. 2020): beta_t ramps linearly from ~0.0001 to ~0.02 over T=1000 steps. Simple but destroys signal too fast at low resolution.
- **Cosine schedule** (Nichol & Dhariwal 2021): alpha_bar_t = cos^2(pi/2 * (t/T + s)/(1+s)) / cos^2(pi/2 * s/(1+s)), clipped to avoid singularity. Slower signal decay; better for 32x32 and below. This is the current default.

### Reverse process and the variational objective

The reverse process p_theta(x_{t-1} | x_t) is parameterized as a Gaussian. The ELBO decomposes into per-step KL divergences between the true reverse posterior q(x_{t-1} | x_t, x_0) - which is analytic - and the model. Ho et al. show the mean of q(x_{t-1} | x_t, x_0) can be written as a function of x_0, which can itself be expressed in terms of the added noise epsilon. This is the bridge between the variational and the noise-prediction view.

### The score function and its connection to epsilon prediction

A score function is the gradient of the log data density: s(x) = grad_x log p(x). At noise level t, the score of the noised distribution is grad_{x_t} log p_t(x_t). By Tweedie's formula, the posterior mean E[x_0 | x_t] = (x_t + (1 - alpha_bar_t) * s_theta(x_t, t)) / sqrt(alpha_bar_t). Since x_t = sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) epsilon, the score satisfies:

    s*(x_t, t) = -epsilon / sqrt(1 - alpha_bar_t)

So a network trained to predict epsilon is, up to a scaling factor, approximating the score. This identity is not incidental - it is why DDPM's noise prediction objective is equivalent to denoising score matching (Vincent 2011). The student should derive this equivalence, not just accept it.

### Three prediction parameterizations

All three predict different but algebraically equivalent targets from a noisy x_t:

| Name | Target | Notes |
|------|--------|-------|
| epsilon-prediction | the noise added | Original DDPM default; numerically unstable at very low noise (t~0) |
| x0-prediction | the clean image | Numerically unstable at high noise (t~T); can help with perceptual quality |
| v-prediction | v = sqrt(alpha_bar_t) epsilon - sqrt(1-alpha_bar_t) x_0 | Salimans & Ho 2022; numerically well-conditioned across all t; required for progressive distillation; used in Stable Diffusion 2.x and beyond |

As of 2025, v-prediction has become the practical default in production systems. Teaching only epsilon-prediction and calling v-prediction a "stretch" understates how current the field is.

### Sampling: DDPM (ancestral) and DDIM

**DDPM sampling** uses the learned posterior mean plus a noise term with learned or fixed variance. It requires ~1000 steps.

**DDIM** (Song et al. 2020) rewrites the process as a non-Markovian one whose marginals match DDPM's. The reverse ODE step is:

    x_{t-1} = sqrt(alpha_bar_{t-1}) * x0_hat + sqrt(1 - alpha_bar_{t-1} - sigma_t^2) * epsilon_theta + sigma_t * z

With sigma_t = 0 this is fully deterministic; with sigma_t = sqrt((1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)) * sqrt(1 - alpha_bar_t / alpha_bar_{t-1}) it recovers DDPM. The deterministic DDIM path is a probability flow ODE - the same ODE derived from the score SDE by removing the stochastic term (Song et al. 2021). DDIM allows sub-sampling (skip timesteps), giving 10-50x speedup with little quality loss. The student should understand why this is possible: the ODE formulation has a consistent latent space, so you can choose any subset of timesteps.

### Classifier-free guidance (CFG)

A single model is trained on both conditional and unconditional data by randomly dropping the class/text condition during training (typically 10-20% of batches replaced with a null embedding). At inference, the score is extrapolated:

    epsilon_guided = epsilon_uncond + w * (epsilon_cond - epsilon_uncond)

where w > 1 sharpens the distribution at the cost of diversity. The student must implement the dropout, the combined forward pass, and understand that CFG is sampling from an implicit distribution proportional to p(x | c)^w / p(x)^(w-1), which is not a valid density for w != 1 - the trade-off between fidelity and diversity is real and sometimes produces artifacts at high w.

### Score-SDE perspective (unification)

Song et al. (2021) show that DDPM is a discretization of the VP-SDE dx = -beta(t)/2 x dt + sqrt(beta(t)) dW, and the reverse SDE is dx = [-beta(t)/2 x - beta(t) s_theta(x, t)] dt + sqrt(beta(t)) dW_bar. The ODE variant (no dW_bar term, halved diffusion coefficient) is the probability flow ODE underlying DDIM. This perspective:

1. Unifies all discrete-time DDPM/DDIM variants under one framework
2. Shows that the epsilon network is a scaled score estimator at each t
3. Directly motivates flow matching (A6): replace the SDE with an ODE and parameterize the velocity field directly

A student can understand DDPM without this view, but understanding the SDE connection makes A6 (flow matching) a natural extension rather than a separate topic.

### EDM preconditioning and loss weighting

Karras et al. (2022) observe that naive epsilon-prediction has poorly conditioned training: the network's input scale, output scale, and target scale all vary dramatically with noise level sigma. EDM introduces preconditioning to normalize these:

    D_theta(x; sigma) = c_skip(sigma) x + c_out(sigma) F_theta(c_in(sigma) x; c_noise(sigma))

where c_skip, c_out, c_in are chosen so the network F_theta operates on unit-scale inputs and produces unit-scale outputs for all sigma. EDM also shows that the effective loss weight across noise levels matters: naive MSE over epsilon underweights medium noise levels. Min-SNR weighting (Hang et al. 2023) caps the weight at low-SNR timesteps, yielding ~3.4x faster convergence with no architecture change.

---

## 2. Mechanisms to implement from scratch, with verifiable tasks

All implementations use pure PyTorch autograd, no diffusers/schedulers.

### 2.1 Forward noising schedule

**Task:** Implement `linear_schedule(T)` and `cosine_schedule(T)` returning `(betas, alphas_bar)`. Write shape tests: `alphas_bar.shape == (T,)`, `alphas_bar[0] ~= 1`, `alphas_bar[-1] ~= 0`. Plot alpha_bar vs t for both schedules. Verify that `q_sample(x0, t, eps)` returns a tensor with the correct mean and variance by checking `E[x_t | x0]` and `Var[x_t | x0]` on a batch.

**Minimal verifiable task:** Given a batch of MNIST images, compute x_t for t in {0, 250, 500, 750, 999} and visually confirm the progression from sharp to pure noise. Also run `torch.autograd.gradcheck` on the closed-form q_sample function.

### 2.2 Time-embedded U-Net

**Task:** Build a minimal U-Net (3-4 resolution levels, ResNet blocks with group norm, self-attention at the bottleneck, no pretrained weights). Inject timestep as a sinusoidal embedding projected through an MLP; add it to each ResNet block via feature-wise shift (AdaGN or simple additive injection into channels). The class/null condition for CFG is an embedding lookup added to the same time projection.

**Minimal verifiable task:** Shape test: input `(B, 1, 32, 32)` + time `(B,)` + class `(B,)` -> output `(B, 1, 32, 32)`. Overfit-one-batch: train the U-Net to memorize the noise on a single (B=4) batch - loss should reach near-zero within a few hundred steps.

### 2.3 Epsilon-prediction training loop

**Task:** Implement the DDPM training objective: sample t ~ Uniform(0, T-1), sample eps ~ N(0, I), compute x_t = sqrt(alpha_bar_t) * x0 + sqrt(1-alpha_bar_t) * eps, predict eps_theta(x_t, t), compute MSE loss.

**Minimal verifiable task:** Overfit-one-batch test - a model trained on 4 MNIST images should reconstruct those images via DDPM sampling after ~500 steps. Track the training loss; it should decrease monotonically.

### 2.4 DDPM ancestral sampler

**Task:** Implement `ddpm_sample(model, shape, T, betas)` from scratch, stepping from t=T-1 to t=0 using the posterior mean formula and adding noise at all steps except t=0.

**Minimal verifiable task:** Run on the overfit model from 2.3 and confirm it reproduces roughly the training images. Run on a fully trained model and generate a grid of 64 samples. Qualitative check: samples should look like digits.

### 2.5 DDIM sampler (deterministic)

**Task:** Implement `ddim_sample(model, shape, timesteps_subset)` where `timesteps_subset` is a list of T values to use (e.g., 50 out of 1000). The step follows the DDIM update rule with sigma=0.

**Minimal verifiable task:** Given the same starting noise z, compare DDIM with 50 steps vs DDPM with 1000 steps on the same model. Outputs should be visually similar. Time both; confirm >10x wall-clock speedup.

### 2.6 Classifier-free guidance

**Task:** Extend training to randomly drop the class label with p=0.1, replacing with a null token. Implement the guided sampler that runs two forward passes per step and extrapolates the score with guidance scale w. Train a class-conditional model on MNIST (10 classes).

**Minimal verifiable task:** Generate class-conditional samples for all 10 digits. Vary w in {1.0, 3.0, 7.5} and observe the fidelity/diversity trade-off qualitatively. At w=1.0 output should match unconditional quality; at w=7.5 digits should be sharper but possibly more uniform.

### 2.7 Stretch: v-prediction

**Task:** Replace the epsilon-prediction target with v = sqrt(alpha_bar_t) * eps - sqrt(1-alpha_bar_t) * x0. Update the training loss and the DDIM step formula (which requires converting predicted v back to predicted x0).

**Minimal verifiable task:** Compare v-prediction and epsilon-prediction training curves on the same architecture and data. v-prediction should converge faster or comparably with better behavior at extreme timesteps.

---

## 3. Assessment of the draft scope

### What is right

- Forward noising schedule (linear + cosine): correct and necessary. Both schedules must be implemented.
- Time-embedded U-Net: correct. The sinusoidal time embedding and its injection via AdaGN or additive shift is the core architectural contribution of DDPM-era models.
- Epsilon-prediction objective: correct as the starting point, but see below.
- Ancestral DDPM + DDIM sampling: correct. These two samplers cover the stochastic and deterministic extremes.
- Classifier-free guidance: correct and important. It is used in essentially every downstream application.
- MNIST/shapes unconditional generation that visibly improves: correct as the target task.

### What is missing

**Score/SDE connection should be taught, not optional.** The connection s(x_t) = -epsilon_theta / sqrt(1 - alpha_bar_t) is one derivation, not a separate topic. It should be a derivation exercise in the notebook, not an advanced aside. Without it, A6 (flow matching) will seem unrelated.

**EDM preconditioning and loss weighting are missing entirely.** The EDM framework (Karras et al. NeurIPS 2022) is now standard reference material. Its c_skip/c_out/c_in preconditioning explains why training is stable; the noise weighting discussion explains why Min-SNR (Hang et al. ICCV 2023) works. A student who implements raw epsilon-prediction MSE and moves to production will be immediately confused by why everything real uses weighted objectives. These concepts take one lecture sub-section and one extra loss function, not a full separate module.

**The probability flow ODE link between DDIM and SDE is missing.** DDIM is presented as "a faster sampler." It is actually an ODE integrator for the probability flow ODE that underlies all deterministic sampling. This link - and the observation that sigma controls stochasticity continuously - sets up A6 properly.

**Improved DDPM (Nichol & Dhariwal 2021)** introduced the cosine schedule (which the draft scope already includes) and also learned reverse variances. The cosine schedule is in scope but learned variance is not mentioned. Learned variance is important context for why fixed variance is a simplification.

### What is mis-emphasized or outdated

**V-prediction is core, not stretch.** Salimans & Ho (2022) introduced v-prediction specifically to fix numerical instability at extreme SNR and to enable progressive distillation. Stable Diffusion 2.x, all of Stable Diffusion 3, and most modern flow-based models use v-prediction or the flow equivalent. Calling it stretch means students learn the framework that was already being phased out in 2022. It should be a required implementation variant with a side-by-side comparison.

**Epsilon-prediction should not be presented as the default without qualification.** It is the historical default and correct pedagogically as the starting point. But the scope should explicitly name v-prediction as the current practical standard and include it in the required path.

**The SDE/score lens should be integrated, not separate.** The draft says "the variational/score view" without specifying whether to teach both or one. The answer is: derive the ELBO (variational view), show it reduces to noise prediction MSE, then in one paragraph derive the score connection via Tweedie's formula. This covers both in sequence without a separate module.

### Should the SDE/EDM lens replace the DDPM discrete-time derivation?

No - but both should be present. The DDPM derivation is concrete and motivates the algorithm directly. The SDE/EDM lens is what makes the result interpretable, connects it to flow matching, and explains why noise weighting matters. The right pedagogy is: DDPM first (2-3 pages of math), then "here is the SDE it discretizes, here is the score connection, here is why v-prediction and EDM preconditioning follow naturally." This adds perhaps 30 minutes to the lecture and one extra derivation exercise, but pays dividends for every subsequent topic.

### Should diffusion-flow matching unification be set up here?

Yes, but briefly. The scope should end with a single paragraph or exercise: "The probability flow ODE of DDPM is x_dot = f(x, t) - g^2(t)/2 * score(x, t). If you replace the score with a directly parameterized velocity field v_theta(x, t) and use a straight-line interpolation path, you get flow matching (A6). The difference is path choice and whether you propagate through an SDE or ODE." This two-sentence bridge makes A6 feel like a natural simplification rather than a different topic.

### Reordered/revised scope recommendation

**Required (in order):**
1. Forward process, linear and cosine schedules; closed-form q(x_t | x_0)
2. ELBO derivation; posterior q(x_{t-1} | x_t, x_0); noise prediction objective (epsilon)
3. Score connection: s = -epsilon/sqrt(1-alpha_bar); Tweedie's formula; connection to denoising score matching
4. Time-embedded U-Net implementation
5. DDPM training loop and ancestral sampler
6. DDIM sampler (deterministic, sub-sampled steps); link to probability flow ODE
7. V-prediction: derivation of v target, why it is better conditioned, update to training and sampling
8. CFG: joint training with null token, guided sampler, guidance scale effect
9. (Brief) EDM preconditioning concept and loss weighting (Min-SNR); students implement Min-SNR as a one-line change to the loss

**Stretch:**
- Varying DDIM steps (already in stretch - keep)
- Class-conditional CFG (move from stretch to required: it is the natural use case)
- Learned reverse variance (true stretch)
- Consistency distillation (beyond scope but worth naming)

---

## 4. Connections to downstream topics

### A6 — Flow matching

DDPM's probability flow ODE is the direct mathematical ancestor of flow matching. The key insight: if you remove the stochastic term from the reverse SDE and reparameterize time, you get an ODE whose velocity field can be learned directly from data via regression on straight-line interpolants (Lipman et al. 2022 / Liu et al. 2022 Rectified Flow). V-prediction is the bridge: the "velocity" in flow matching is structurally the same as v in v-prediction; both predict a direction in (x0, noise) space. A student who understands v-prediction and the probability flow ODE will find flow matching's training objective immediately recognizable.

### A7 — Latent DiT

The DiT (Peebles & Xie, ICCV 2023) replaces the U-Net backbone with a transformer operating on latent patches, applying diffusion in the latent space of a pretrained VAE (following Rombach et al. 2022 LDM). Everything from A5 - the noise schedule, the epsilon/v prediction objective, CFG, DDIM sampling - carries over unchanged. The only new element is the architecture (transformer blocks with adaLN-Zero conditioning instead of a U-Net). A student who has built the time-embedded U-Net in A5 will understand why adaLN-Zero is the natural transformer analog of the ResNet time injection.

### A13 — VLA action head

Diffusion Policy (Chi et al., RSS 2023 / IJRR 2024) applies DDPM to robot action sequences: the action trajectory is the "image," and the denoising network is conditioned on vision and language embeddings. CFG becomes action-conditioned guidance. The score connection to v-prediction matters here because action diffusion models are often trained with flow matching (rectified flow) in later VLA systems, and the conceptual bridge from A5 makes this swap transparent.

---

## 5. Must-read sources

1. **Ho, Jain, Abbeel. "Denoising Diffusion Probabilistic Models." NeurIPS 2020.** arXiv:2006.11239. The original DDPM paper. Derives the ELBO, establishes the epsilon-prediction objective, and reports CIFAR-10 FID 3.17. Read before anything else.

2. **Song, Meng, Ermon. "Denoising Diffusion Implicit Models." ICLR 2021.** arXiv:2010.02502. Introduces the non-Markovian generalization that yields DDIM and the probability flow ODE. The paper that made diffusion models practical for inference.

3. **Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole. "Score-Based Generative Modeling through Stochastic Differential Equations." ICLR 2021 (Oral).** arXiv:2011.13456. Unifies DDPM, NCSN, and other score-based models under a continuous-time SDE framework. Establishes the VP-SDE, VE-SDE, and the probability flow ODE. Essential for understanding the score-epsilon connection and for bridging to flow matching.

4. **Nichol, Dhariwal. "Improved Denoising Diffusion Probabilistic Models." ICML 2021.** arXiv:2102.09672. Introduces the cosine schedule (required reading if the scope includes cosine) and learned reverse variances. The cosine schedule is now the standard starting point; this paper explains why.

5. **Ho, Salimans. "Classifier-Free Diffusion Guidance." NeurIPS 2022 Workshop.** arXiv:2207.12598. The technique behind every practical conditional diffusion model. Short paper; read in full.

6. **Salimans, Ho. "Progressive Distillation for Fast Sampling of Diffusion Models." ICLR 2022.** arXiv:2202.00512. Introduces v-prediction parameterization as a prerequisite for stable distillation. Explains why epsilon-prediction is numerically problematic at extreme SNR. This paper establishes why v-prediction is the right default.

7. **Karras, Aittala, Aila, Laine. "Elucidating the Design Space of Diffusion-Based Generative Models." NeurIPS 2022.** arXiv:2206.00364. Separates noise schedule, network preconditioning, and loss weighting into independent design choices. Introduces c_skip/c_out/c_in preconditioning. Sets new CIFAR-10 FID 1.79. The paper that makes the "what should I implement and why" question answerable systematically.

**Notable omission in the draft scope:** The score-SDE paper (Song et al. 2021, #3 above) is not in the original draft's implied reading list - the draft mentions "the variational/score view" without citing the paper that formalized it. This omission should be corrected.

---

## 6. 2024-2026 developments that change how this should be taught

### V-prediction is now the default, not a stretch

Every major production diffusion model released since 2022 - Stable Diffusion 2.x (2022), Stable Diffusion 3 (2024), FLUX (2024) - uses v-prediction or the flow matching equivalent. The reason is straightforward: epsilon-prediction has NaN/instability issues at low noise levels that appear in fine-tuning and distillation but not always in initial training. Teaching epsilon-prediction as the primary target and v-prediction as stretch creates a mismatch with every real codebase a student will encounter.

### Flow matching has largely superseded discrete DDPM in production

FLUX (Black Forest Labs, 2024), Stable Diffusion 3 (Stability AI, 2024), and most video generation models (Sora-class systems, 2024-2025) use flow matching with linear interpolation paths rather than the DDPM/DDIM framework. Flow matching is simpler (straight-line ODE, no noise schedule, direct velocity regression), converges faster, and produces comparable or better quality. The DDPM framework is still the right pedagogical starting point, but teaching it without explicitly naming flow matching as "what the field moved to" leaves students puzzled when they look at any 2024+ model.

### EDM2 (Karras et al. CVPR 2024) refined the training dynamics story

EDM2 (arXiv:2312.02696) identified poor training dynamics in the ADM U-Net - activations, weights, and update magnitudes vary across layers in ways that hurt convergence. The fix (magnitude-preserving layers, weight normalization, adjusted initialization) is architecture-level and beyond A5 scope. But the paper's analysis is a good read for understanding why "just implement the equations from the paper" often produces worse results than published numbers.

### Consistency models (Song et al. 2023) and consistency training

Consistency models learn to map any noisy x_t directly to x_0 (jumping the entire trajectory in one step), achieving near-DDPM quality in one or two function evaluations. They can be trained from scratch or distilled from a pretrained diffusion model. As of 2025, latent consistency models are used in Stable Diffusion Turbo and similar fast-inference systems. This is beyond A5 scope but should be mentioned as a consequence of understanding the ODE trajectory - once you have a probability flow ODE, you can train a model to be self-consistent along it.

### Min-SNR weighting (Hang et al. ICCV 2023) is a free training speedup

Adding Min-SNR-gamma weighting to the loss (w_t = min(SNR_t, gamma) / SNR_t with gamma=5) accelerates convergence ~3.4x on ImageNet benchmarks with no architecture change and one extra line of code. Any student implementing the training loop should add this; it should be at minimum a footnote in the scope.

### DiT (Peebles & Xie, ICCV 2023) is the architecture backbone for A7

The U-Net is no longer the default backbone. DiT replaced it for ImageNet class-conditional generation and then for Stable Diffusion 3, FLUX, and video generation. Teaching the time-embedded U-Net in A5 is correct (it is the simplest architecture to understand and MNIST-scale fits on any GPU), but the scope should name DiT and note that A7 will replace the U-Net with a transformer - so the conditioning mechanism (adaLN-Zero) is worth previewing.

### Diffusion vs. flow matching unification is being formalized (2024-2025)

Several papers in 2025 have derived unified measure-theoretic frameworks showing DDPM, score-based models, and flow matching as special cases of a single object (e.g., "A Unified Measure-Theoretic View of Diffusion, Score-Based, and Flow Matching Generative Models," arXiv:2605.06829, 2025). This is primarily of theoretical interest for A5, but it confirms that teaching the SDE/score connection in A5 is the right foundation for understanding that A6 is a simplification rather than a replacement.

---

*Sources consulted: Ho et al. 2020 (arXiv:2006.11239), Song et al. 2020 DDIM (arXiv:2010.02502), Song et al. 2021 Score-SDE (arXiv:2011.13456), Nichol & Dhariwal 2021 (arXiv:2102.09672), Salimans & Ho 2022 (arXiv:2202.00512), Ho & Salimans 2022 CFG (arXiv:2207.12598), Karras et al. 2022 EDM (arXiv:2206.00364), Peebles & Xie 2023 DiT (ICCV 2023), Hang et al. 2023 Min-SNR (arXiv:2303.09556), Karras et al. 2024 EDM2 (arXiv:2312.02696), Lipman et al. 2022 Flow Matching (arXiv:2210.02747), Chi et al. 2023 Diffusion Policy. MIT 6.S184 IAP 2025 course structure also consulted.*
