# A7 — Latent Diffusion & a Tiny DiT: Validation Report

## 1. Key concepts a student must learn

### Why latent space

Pixel-space diffusion is expensive at high resolution because every denoising step runs over all pixel dimensions. LDM (Rombach et al., CVPR 2022) separates perceptual compression from generative learning: a pre-trained autoencoder compresses images into a spatially smaller, semantically dense latent, and the diffusion model operates entirely in that latent. The 4x spatial downsampling (f=4) already removes most imperceptible high-frequency detail; at f=8 or f=16 the latent is 64x smaller than the pixel grid, cutting compute by roughly that factor per diffusion step.

The conceptual point worth stressing: the autoencoder is trained independently with a reconstruction objective (pixel loss + perceptual loss + low-weight adversarial loss) and a KL penalty to keep the latent near a unit Gaussian. The generative model - diffusion or flow-matching - then only has to learn the semantic structure of that already-compressed distribution. This split makes the generative problem easier and cheaper.

### The autoencoder + generative prior split

This two-stage design is worth treating as a first-class concept, not just implementation scaffolding. The encoder E maps x to z = E(x). The diffusion or flow model learns p(z). At generation time, z_0 ~ p(z) is decoded by D to get the image. The student should understand what each stage is responsible for and why coupling them end-to-end is generally unnecessary and costly.

For the KL-regularized VAE: the KL term in the ELBO is weighted very lightly (typically beta = 1e-6 to 1e-4) so that the latent does not collapse to a pure Gaussian - it preserves enough spatial structure to be decoded back cleanly. This differs from a classical beta-VAE or a VQ-VAE; the latent is continuous and roughly Gaussian but not truly isotropic.

### DiT architecture

Peebles & Xie (ICCV 2023) replace the U-Net denoiser with a Vision Transformer operating on patchified latents. The pipeline is:

1. **Patchify**: divide the 2D latent z (shape H/f x W/f x C) into non-overlapping p x p patches and linearly project each patch to a token of dimension d. For a 32x32 latent with patch size 2, this gives 256 tokens.
2. **Positional embedding**: standard 2D sinusoidal or learned embeddings added to the token sequence.
3. **DiT blocks**: each block is a standard transformer block (pre-norm, self-attention, FFN) but with **adaLN-Zero** conditioning in place of the usual LayerNorm.
4. **Unpatchify**: final linear projection and reshape back to the latent shape.

The model predicts either the noise epsilon (DDPM target) or the velocity v (flow-matching target) for each patch-token.

### adaLN-Zero conditioning

This is the most implementation-specific concept in A7 and warrants careful treatment.

Standard LayerNorm computes `(x - mean) / std` then applies learned gamma and beta (scale and shift). Adaptive LayerNorm instead regresses gamma, beta, and an additional gating scalar alpha from a conditioning embedding `c = MLP(t_emb + class_emb)`, producing six scalars per block: `(gamma_1, beta_1, alpha_1)` for the self-attention sub-layer and `(gamma_2, beta_2, alpha_2)` for the FFN sub-layer.

The **Zero** part is initialization: the linear layer that outputs the six scalars is initialized to zero. This means at training start, alpha = 0 everywhere, so every DiT block is an identity function (the residual path passes through unchanged). This dramatically stabilizes early training, analogous to zero-initializing the last layer of each residual branch in ResNets. The paper showed this outperforms cross-attention conditioning and in-context conditioning, while being cheaper.

The student should implement this from scratch: a small MLP that takes `t_emb + class_emb` and outputs 6 x d_model scalars, applied inside each DiT block via modulated LayerNorm.

### Patchify / unpatchify of latents

Patchification converts a spatial tensor to a sequence of tokens, which is the interface between the spatial autoencoder output and the transformer. The student should write both directions: patchify as a reshape + linear projection, and unpatchify as the inverse linear projection + reshape. At test time, verifying that patchify -> unpatchify is a consistent round-trip (up to the linear projection) is a useful sanity check.

---

## 2. Mechanisms to implement from scratch, with verifiable tasks

### 2.1 KL-regularized VAE (small)

**What to build**: encoder (stack of strided conv blocks, outputs mu and log-sigma), reparameterization trick, decoder (transposed convs or upsample + conv), KL term in the loss.

**Tiny problem**: 4-channel 8x8 latent for 32x32 grayscale patches (or MNIST). Target latent channel count C=4 (matching SD-VAE style).

**Verifiable tasks**:
- Shape test: encoder(x).shape == (B, 2*C, H/f, W/f) where f=4; sampled z.shape == (B, C, H/f, W/f).
- `torch.autograd.gradcheck` on the reparameterization and KL term.
- Overfit a single batch (8 images): reconstruction loss < 0.01 after ~500 steps.

### 2.2 DiT block with adaLN-Zero

**What to build**: one DiT block - pre-norm self-attention + FFN with adaLN-Zero conditioning (6 scalars regressed from conditioning embedding, zero-initialized output projection).

**Verifiable tasks**:
- At initialization (before any training step): forward pass with conditioning = zeros returns output numerically equal to input (identity check, tolerance ~1e-6).
- Shape: input (B, N, d) -> output (B, N, d).
- `torch.autograd.gradcheck` on the block including the conditioning path.

### 2.3 Full tiny DiT (patchify + L blocks + unpatchify)

**What to build**: patchify linear projection, sinusoidal or learned positional embedding, stack of DiT blocks, unpatchify linear projection. Conditioning: timestep sinusoidal embedding + class label embedding, summed.

**Tiny problem**: 4x4 latent (= 16 tokens at patch_size=1, or 4 tokens at patch_size=2), 2 or 4 DiT blocks, d_model=64, 2-head attention. Class-conditional on toy 2-class dataset (or single-class overfit).

**Verifiable tasks**:
- Shape: input (B, C, H, W) latent + (B,) timestep + (B,) class label -> output (B, C, H, W) predicted noise or velocity, same shape.
- Overfit one batch: the DiT alone (no VAE, just noise-predict on fixed z) reaches near-zero MSE on a single latent within a few hundred steps using DDPM or flow-matching objective.
- Sampling: run the reverse process (DDPM or Euler ODE), decode through the frozen VAE decoder, visually check the output is coherent on the overfit sample.

### 2.4 End-to-end latent diffusion pipeline

**What to build**: tie 2.1 and 2.3 together - encode a small image dataset to latents with the trained VAE, train the DiT on those latents, sample latents and decode.

**Verifiable task**: overfit-one-batch end-to-end: the pipeline can reconstruct recognizable images from the single training batch within ~1000 steps.

---

## 3. Assessment of the draft scope

### What is right

The core scope is pedagogically sound. The three-way structure (VAE, DiT denoiser, latent-space generation) correctly identifies the architecture underlying most production image generators since 2022. The choice to swap the U-Net from A5/A6 for a DiT is the right architectural escalation. adaLN-Zero is the correct conditioning variant to teach - it is both the original DiT choice and the one that has propagated most widely. Patchify/unpatchify of latents is a real implementation task that forces students to understand the spatial-to-sequence interface.

### What is missing or under-emphasized

**Flow-matching as the forward process.** The draft says "run A5/A6 diffusion/flow in latent space." A5/A6 presumably cover both DDPM and flow-matching, but given that SD3 (2024), FLUX (2024), and essentially all new production models train with rectified flow rather than DDPM, the A7 implementation should explicitly use a flow-matching objective (linear interpolant, velocity prediction, Euler ODE sampler) rather than defaulting to DDPM. This is not a correction so much as a clarification: the draft allows it, but should require it.

**MM-DiT (double-stream joint attention) should at minimum be explained, even if not implemented.** The draft uses class-conditional DiT, which is pedagogically appropriate for the overfit task. However, SD3 (Esser et al., 2024) and FLUX (Black Forest Labs, 2024) use MM-DiT, where image tokens and text tokens are carried in separate streams with their own weight matrices and are only joined at the attention step. This architecture is now the standard for production text-to-image models. The course should include at least a conceptual explanation and a diagram, and ideally a minimal single-block MM-DiT implementation (two linear projections into a shared attention, split back out) as an extension exercise, even if the overfit task uses class-conditional DiT.

**REPA as a training trick deserves a mention.** Yu et al. (ICLR 2025 oral, arXiv 2410.06940) showed that adding an auxiliary loss aligning intermediate DiT activations with features from a frozen DINOv2 encoder speeds up SiT/DiT training by over 17x. This is now a standard technique in DiT training papers. A single paragraph explaining the mechanism - that diffusion models struggle to build clean representations under noisy inputs, and a semantic alignment loss shortcircuits that - is worth including in the reading note.

**The video-generation connection needs to be explicit about spacetime patchification.** Sora (OpenAI, 2024) and most subsequent video DiTs operate on spacetime latents: the VAE compresses spatially and temporally, and the DiT patchifies across (x, y, t). The patch embedding step in the video case includes a temporal dimension. The conceptual note should call this out directly so students see exactly what extends.

### What is outdated or mis-emphasized

**KL-VAE is still standard for diffusion models.** The draft's implicit assumption that KL-VAE is the latent encoder is correct. Discrete VQ tokenizers dominate autoregressive image generation (VQGAN, LlamaGen) but not diffusion-based models. SD3 and FLUX both use continuous KL-VAE latents (FLUX uses a 16-channel VAE at f=8). This distinction should be stated explicitly: VQ is for AR, KL-continuous is for diffusion/flow. The recent trend is toward higher channel-count VAEs (16 channels instead of 4) and deeper compression (DC-AE, used by SANA), but the KL objective and continuous latent are unchanged.

**Class-conditional DiT is not outdated, just limited.** Teaching it is fine for the mechanism. But students should know that production systems condition on text via cross-attention (PixArt-alpha style) or via MM-DiT joint attention (SD3/FLUX style). The class label is a stand-in for any conditioning signal; the adaLN-Zero mechanism generalizes directly to timestep + text pooling.

**The draft correctly identifies the video reading note connection.** No change needed there.

### Suggested reorder / restructuring

1. Motivate latent space (cost argument with numbers).
2. Build and verify the KL-VAE.
3. Encode a small dataset; visualize the latent distribution.
4. Implement adaLN-Zero DiT block (identity-at-init test first).
5. Implement full DiT; overfit one latent batch with flow-matching objective.
6. Run sampling; decode through VAE.
7. Reading note: MM-DiT (SD3/FLUX), REPA, video spacetime patching.

---

## 4. Dependencies and connections

**Depends on A1 (transformer block)**: the DiT block is a transformer block with an adaLN-Zero conditioning wrapper. Students who have implemented multi-head self-attention and FFN in A1 reuse that code directly; the only new mechanism is adaLN-Zero conditioning and the conditioning embedding (sinusoidal t_emb + class embedding).

**Depends on A5/A6 (DDPM, flow-matching)**: the forward process, noise schedule or interpolant, and loss function carry over unchanged. The only difference is that the denoiser input is now a latent z rather than a pixel x.

**The architecture behind current image systems**: SD3 (2024) and FLUX (2024) are direct descendants. Both are flow-matching DiTs operating in KL-VAE latent space, with MM-DiT conditioning instead of class-conditional adaLN. Students who implement A7 have built every component of these systems except the MM-DiT double stream and the text encoder.

**Basis for the video generation reading note**: Sora and subsequent video DiTs add only one conceptual extension - spacetime patchification. The VAE gains a temporal compression axis; the patch embedding becomes 3D (x, y, t patches); the transformer structure is otherwise identical. The course note should make this connection explicit with a diagram.

---

## 5. Must-read sources

1. **Rombach et al. (2022). "High-Resolution Image Synthesis with Latent Diffusion Models." CVPR 2022.** arXiv:2112.10752. The LDM paper introducing the KL-VAE + latent diffusion two-stage design. The key read for the perceptual compression / semantic compression argument and the compression factor ablation (f=4/8/16).

2. **Peebles & Xie (2023). "Scalable Diffusion Models with Transformers." ICCV 2023.** arXiv:2212.09748. The DiT paper. Read for patchify, adaLN-Zero vs alternatives, and the scaling law (FLOPs vs FID).

3. **Esser et al. (2024). "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis." ICML 2024.** arXiv:2403.03206. The SD3 paper introducing MM-DiT and rectified flow with perceptually-biased noise sampling. Read for the double-stream block design and why text tokens get separate weights.

4. **Black Forest Labs (2024). FLUX.1 technical report / model card.** github.com/black-forest-labs/flux. FLUX is a 12B parameter hybrid single-stream + double-stream MM-DiT trained with rectified flow on a 16-channel f=8 KL-VAE. Currently the strongest open-weights image model. No full paper; the model card and architecture description on GitHub and Hugging Face are the primary sources.

5. **Yu et al. (2024). "Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think." ICLR 2025 oral.** arXiv:2410.06940. REPA - aligning intermediate DiT activations with DINOv2 features, 17.5x training speedup. Directly relevant as a training technique.

6. **Ma et al. (2024). "SiT: Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers."** arXiv:2401.08740. Connects DiT architecture to flow-matching (stochastic interpolants) and shows flow-matching outperforms DDPM on DiT at the same architecture/FLOPs. The natural pairing with the DiT paper if A7 uses flow-matching as the forward process.

7. **Chen et al. (2023). "PixArt-alpha: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis." ICLR 2024.** arXiv:2310.00426. Optional but useful: shows how to add cross-attention text conditioning to DiT (the step between class-conditional DiT and MM-DiT).

**Omissions in the draft scope that should be flagged**: SD3 (Esser 2024) and FLUX (BFL 2024) are not named. These are the dominant production architectures as of 2026 and the course reading note should reference them. REPA is also unmentioned.

---

## 6. 2024-2026 developments that change how A7 should be taught

### Flow-matching has displaced DDPM for DiT training

SD3 (March 2024), FLUX (August 2024), and the subsequent wave of video generation models (CogVideo, Wan, HunyuanVideo) all train with rectified flow or continuous flow-matching, not DDPM. SiT (2024) demonstrated on the same DiT architecture that flow-matching achieves better FID than DDPM at equal training budget. Teaching A7 with DDPM as the diffusion objective is not wrong but is increasingly dated. The minimal change is to use a linear interpolant and velocity prediction target (x_t = (1-t)*x_0 + t*eps; v_t = eps - x_0) with an Euler ODE sampler - this is simpler to code than DDPM and more representative of current practice.

### MM-DiT double-stream conditioning is now the standard for text-to-image

Class-conditional adaLN-Zero is appropriate for the overfit task, but students should know the production architecture differs: SD3 and FLUX use double-stream blocks where image tokens and text tokens are processed with separate weight matrices and joined only at attention. The bidirectional flow of information between text and image tokens (vs one-way cross-attention in PixArt) is what enables SD3/FLUX's typographic capabilities. A conceptual note + diagram costs little and directly connects A7 to the current landscape.

### 16-channel VAE at higher compression is increasingly standard

The original SD VAE used 4 channels at f=8. FLUX uses 16 channels at f=8. SANA uses DC-AE (deep compression autoencoder) at f=32. For the toy implementation, 4 channels at f=4 remains appropriate. But the reading note should mention that production models have moved to higher channel counts (richer latent space) and that f=8 is still the dominant compression factor.

### REPA cuts training cost by 17.5x and is ICLR 2025 oral

Aligning intermediate DiT activations with frozen DINOv2 features is now routinely used in DiT training ablations. The mechanism is conceptually simple (auxiliary MSE loss between a projected DiT hidden state and a DINOv2 patch feature) and worth including as a training note, even if not implemented in the overfit exercise.

### VQ tokenizers are not replacing KL-VAE for diffusion

As of 2026, continuous KL-VAE remains the standard latent for diffusion and flow-based generators. Discrete VQ tokenizers are used in autoregressive image generation (LlamaGen, LWM, etc.) but not in the DiT/flow-matching family. The draft's implicit use of KL-VAE is correct; just state it explicitly.

### Video generation as a direct extension

Sora (February 2024) and subsequent open-weights video DiTs (CogVideoX, HunyuanVideo, Wan 2.1) all use the same architecture as A7 extended to spacetime: the VAE gains temporal compression, and the DiT patchification adds a time dimension. The course reading note on video generation should describe exactly this: patchify(t, x, y) -> token sequence -> DiT -> unpatchify, with the only addition being 3D RoPE or full 3D positional embeddings. Students who complete A7 are one conceptual step away from understanding these models.
