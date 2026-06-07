# A1 — The Transformer, from scratch: scope validation

*Researched June 2026. Sources verified against original papers and current repos.*

---

## 1. Key concepts a student must learn

### Conceptual core

**Attention as soft, content-addressable lookup.** The mechanism computes a weighted average over a value set, where the weights come from the similarity of a query against a set of keys. The learning insight is that the network can decide at every position which other positions to aggregate from, making the receptive field dynamic and input-dependent rather than fixed (as in convolutions).

**Scaled dot-product.** Dividing logits by `sqrt(d_k)` keeps the pre-softmax values in a regime where softmax has useful gradients. Without scaling, large `d_k` pushes dot products into regions where softmax saturates to near-one-hot, and gradients vanish. The student should verify this empirically by running without the scale.

**Multi-head attention as a parallel ensemble of subspace projections.** Splitting into h heads lets the model attend to different representation subspaces simultaneously. Each head projects Q, K, V into a lower-dimensional space, runs attention, and the outputs are concatenated and re-projected. This is not just a speed trick; it provides the diversity of routing patterns that single-head attention cannot express.

**Causal masking.** Adding `-inf` to future positions before softmax enforces that each position only attends to itself and past tokens. The mechanism is trivial to implement but essential to understand: it is what makes the decoder auto-regressive rather than bidirectional.

**Positional encoding.** Transformer attention is permutation-invariant by construction; position must be injected explicitly. The sinusoidal scheme (Vaswani et al., 2017) uses fixed sin/cos functions at different frequencies, giving a deterministic relative-distance signal without learned parameters. Learned absolute embeddings (used in GPT-2) are conceptually simpler. RoPE (Su et al., 2021) applies a rotation to Q and K such that their dot product depends only on relative displacement — this is the 2024-standard approach and qualitatively different from the earlier two.

**Residual + normalization block structure.** The pre-LN variant wraps each sub-layer as `x = x + sublayer(LN(x))`. This gives well-behaved gradients at initialization without needing warmup (Xiong et al., 2020) and is the current consensus. Post-LN (original paper) is less stable and requires warmup; it sometimes achieves slightly better final performance but is harder to train.

**Feed-forward sub-layer.** Two linear projections with a nonlinearity between them, typically expanding by 4x then contracting. Modern practice (LLaMA, Mistral, Qwen, etc.) replaces the GELU MLP with SwiGLU: `FFN(x) = (W1x * swish(W_g * x)) @ W2`, which is a gated design. The expansion ratio with SwiGLU is conventionally 8/3 rather than 4 to preserve parameter count.

### Mathematical core

The student should be able to derive and implement:

- `Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V`
- Multi-head: project to h subspaces, attend, concat, project back
- Sinusoidal encoding: `PE(pos,2i) = sin(pos/10000^(2i/d))`, `PE(pos,2i+1) = cos(...)`
- RoPE: rotate Q and K in pairs of dimensions by angle `pos * theta^(-2i/d)`, so `<R(q,m), R(k,n)> = f(q,k,m-n)`, encoding only relative offset
- Pre-LN block: `y = x + MHA(RMSNorm(x)); y = y + FFN(RMSNorm(y))`
- RMSNorm: `RMSNorm(x) = x / rms(x) * gamma`, where `rms(x) = sqrt(mean(x^2))`, no mean subtraction

---

## 2. Mechanisms worth implementing from scratch

### 2a. Scaled dot-product attention

**Verifiable task:** `(B, T, d_k)` Q/K/V inputs. Confirm output shape `(B, T, d_v)`. Run `torch.autograd.gradcheck` with `float64` inputs. Apply causal mask, verify that token i attends only to positions <= i by inspecting the attention weight matrix on a handcrafted input.

### 2b. Multi-head attention

**Verifiable task:** Wrap SDPA into MHA. Confirm the parameter count matches `4 * d_model^2` (or the appropriate formula for d_k != d_model). Pass `gradcheck`. Verify that setting h=1 reduces to single-head attention.

### 2c. Pre-LN transformer block (Attention + FFN)

**Verifiable task:** Single block, then stack N. Confirm residual paths are intact by checking that output equals input when all sub-layer weights are zeroed (they won't be exactly, but gradient magnitudes at init should be stable). Overfit one batch of random token sequences to near-zero cross-entropy loss.

### 2d. Decoder-only char-LM on tiny Shakespeare

**Verifiable task:** ~6 layers, ~6 heads, d_model=384. Train to BPC < 1.5 on the 1 MB Tiny Shakespeare dataset (Karpathy build-nanogpt is the reference). This takes ~15 minutes on a 4080 at this scale. The pass criterion is not the BPC number per se, but that generated text is recognizably English prose. Forbidden: `nn.MultiheadAttention`, `nn.Transformer*`, `F.scaled_dot_product_attention`.

### 2e. RoPE implementation (stretch → core)

**Verifiable task:** Replace learned absolute positional embedding with RoPE applied inside MHA. Pass `gradcheck`. Confirm that the char-LM loss curve is approximately the same as with learned embeddings (within a factor of 2 at equal steps). Check that rotating Q and K by the same angle cancels out in the dot product (relative invariance property).

### 2f. KV-cache for autoregressive decoding (stretch)

**Verifiable task:** Implement a simple list-of-tensors cache per layer. Verify that generating 100 tokens one-at-a-time with cache produces the identical logit sequence as a single forward pass over the full context. This is a correctness test, not a benchmark.

### 2g. SwiGLU FFN (stretch → core)

**Verifiable task:** Replace the standard GELU MLP with `SwiGLU`. Confirm parameter count matches (adjust expansion to 8/3). `gradcheck` the FFN block alone. Loss curve on tiny Shakespeare should be comparable.

---

## 3. Assessment of the draft scope

### What is right

The draft correctly identifies the foundational mechanics: SDPA, MHA, causal masking, pre-LN, AdamW+warmup, and a decoder-only char-LM. These remain the right backbone for a foundations module. The toy tasks (copy/sort) are a good addition for isolating correctness from dataset noise. The forbidden list (`nn.MultiheadAttention`, `nn.Transformer*`, `F.scaled_dot_product_attention`) is exactly right.

### What is mis-emphasized or outdated

**Sinusoidal + learned positional encodings as co-equal options.** Teaching both as comparable alternatives is reasonable for history, but for a 2026 course the student should understand that both are deprecated in production-scale practice. RoPE is the encoding used by LLaMA 2/3, Mistral, Gemma 2, Falcon, Qwen, OLMo, and most other open models since 2023. Sinusoidal and learned absolute embeddings can be covered briefly for contrast, but RoPE should be the one the student actually implements and carries forward.

**Pre-LN vs post-LN ablation as stretch.** This is a reasonable stretch experiment. However, the conceptual understanding of *why* pre-LN is more stable (gradient scale at initialization, Xiong et al. 2020) should be core content, not stretch.

**No mention of RMSNorm.** The draft says "pre-LN block" but does not specify the normalization function. LayerNorm (as in nanoGPT/GPT-2) is the easy choice, but RMSNorm has been the standard since LLaMA 1 (2023) and is a trivial two-line replacement. A 2026 course that builds on this transformer in later modules (ViT, BEVFormer, VLM) will need RMSNorm to match current codebases. It should be core, not ignored.

**No mention of SwiGLU.** Same argument. The GELU MLP is fine for teaching the structure of the FFN, but every major open model since PaLM and LLaMA uses SwiGLU. Teaching it here means later modules (DiT, VLA) do not need to introduce it. The implementation is a one-block exercise.

**RoPE, KV-cache, pre-LN vs post-LN ablation as stretch.** Given the course's stated goal of taking a practitioner from the ~2020 DETR era to 2026, RoPE and RMSNorm should be *core*, not stretch. KV-cache is a reasonable stretch since it is inference-only and not exercised in training.

**GQA/MQA absent.** The draft does not mention grouped-query attention or multi-query attention. These are used in LLaMA 2/3, Mistral 7B, Gemma, and many others and directly affect KV-cache size. For a course that will later cover BEVFormer and VLA inference, a brief conceptual treatment and toy implementation of GQA would round out the module. This is a stretch item, but it should appear in the outline at all.

### What to add

1. RMSNorm (core): Replace LayerNorm with RMSNorm in the pre-LN block. One extra line of theory, a few lines of code. The student carries this into every subsequent module.
2. SwiGLU (core or warm stretch): Replace the GELU MLP. Requires understanding gated activations; connects to the 8/3 expansion ratio used in LLaMA-family models.
3. RoPE (core, not stretch): The rotation math is a useful exercise in understanding positional inductive biases. It is the positional encoding the student will encounter in every later module.
4. GQA/MQA (stretch): Conceptual treatment + toy implementation. Demonstrates the KV-cache footprint tradeoff.

### What to cut or deprioritize

- Teaching sinusoidal and learned absolute encodings at equal depth is no longer the best use of time. Cover sinusoidal briefly as historical context, cover learned absolute embeddings in one sentence, then focus implementation time on RoPE.
- The copy/sort toy task is useful for verifying causal masking correctness before training the char-LM. Keep it, but frame it as a correctness check, not a separate training objective.

### Suggested reordering

1. SDPA + causal mask (gradcheck, copy-sort toy)
2. MHA (gradcheck, shape tests)
3. Pre-LN block with RMSNorm (overfit one batch)
4. SwiGLU FFN (replace GELU, overfit same batch)
5. Decoder-only char-LM with learned absolute PE (tiny Shakespeare)
6. Replace with RoPE (compare loss curves)
7. Stretch: KV-cache correctness test
8. Stretch: GQA/MQA toy
9. Stretch: Pre-LN vs post-LN ablation

---

## 4. Connections to other course topics

This module is the substrate for every subsequent topic in the course.

**ViT.** A vision transformer is structurally identical to the decoder-only LM implemented here, with three differences: the input sequence is a set of flattened image patches rather than token embeddings, the positional encoding is 2D rather than 1D (typically learned 2D embeddings or 2D RoPE), and there is no causal mask (bidirectional attention). The pre-LN block, MHA, and FFN carry over unchanged.

**CLIP.** Two transformer encoders (one text, one image ViT) trained with a contrastive objective. The architectural building blocks are exactly the pre-LN blocks built here.

**DiT (Diffusion Transformer).** A ViT backbone conditioned on a diffusion timestep and optional class label. Uses the same pre-LN blocks; the SwiGLU FFN knowledge is directly applicable.

**BEVFormer.** Uses deformable cross-attention between BEV queries and camera feature maps. The core QKV abstraction, causal/spatial masking, and multi-head logic are all directly from this module. Students who have implemented MHA from scratch will understand the deformable attention variant much more easily.

**VLM (Vision-Language Model).** A large transformer decoder that ingests both image patch tokens and text tokens. Identical architecture to the char-LM here, scaled up. RoPE, RMSNorm, SwiGLU, and GQA knowledge from this module all transfer directly.

**VLA (Vision-Language-Action).** Adds an action head or diffusion policy on top of a VLM. The transformer backbone is unchanged.

The transformer foundations module thus has the highest leverage of any module in the course. Every architectural choice made here (RMSNorm vs LayerNorm, RoPE vs absolute PE, SwiGLU vs GELU) will be encountered again in every subsequent module.

---

## 5. Must-read sources

1. **Vaswani et al., "Attention Is All You Need," NeurIPS 2017** (arxiv 1706.03762) — original transformer. Read the architecture section carefully; the encoder-decoder framing is less relevant for this module than SDPA and MHA.

2. **Xiong et al., "On Layer Normalization in the Transformer Architecture," ICML 2020** (arxiv 2002.04745) — the theoretical justification for pre-LN. Explains why post-LN requires warmup while pre-LN does not. Essential context for the pre-LN vs post-LN ablation.

3. **Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding," 2021, published Neurocomputing 2023** (arxiv 2104.09864) — original RoPE paper. The math is dense but the key result is compact: rotating Q and K by position-dependent angles makes their dot product a function of only their relative displacement.

4. **Zhang and Sennrich, "Root Mean Square Layer Normalization," NeurIPS 2019** (arxiv 1910.07467) — motivates dropping mean-centering from LayerNorm. Short and clear; the implementation is five lines.

5. **Shazeer, "GLU Variants Improve Transformer," 2020** (arxiv 2002.05202) — introduces SwiGLU alongside ReGLU and GEGLU. The empirical results showing consistent gains over vanilla GELU across architectures justify its adoption as a default.

6. **Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints," EMNLP 2023** (arxiv 2305.13245) — introduces grouped-query attention and shows it recovers most of MHA quality at MQA speed. Directly relevant to KV-cache discussions.

7. **Karpathy, "Let's build GPT: from scratch, in code, spelled out," YouTube 2023 + build-nanogpt repo** (github.com/karpathy/build-nanogpt) — not a paper but the most useful single reference for implementing this module. The git commit history is the intended learning path. Note that nanoGPT uses LayerNorm + GELU + absolute PE (GPT-2 style), so the student will need to upgrade to RMSNorm + SwiGLU + RoPE as explicit exercises.

**What the draft omits:** No mention of Zhang & Sennrich (RMSNorm), Shazeer 2020 (SwiGLU), or Ainslie et al. (GQA). These are important given the 2026 framing.

---

## 6. 2024-2026 developments that change how this should be taught

**The modern LLM stack is now a stable consensus.** By 2024-2025, the architectural choices for transformer-based language models crystallized: pre-norm with RMSNorm, RoPE, SwiGLU with 8/3 expansion, GQA, and bias-free layers. This is documented across 53+ models in a 2025 survey (Tan, "The Crystallization of Transformer Architectures (2017-2025)"). The original Vaswani et al. design is now a historical baseline, not a template to follow. A course teaching "the transformer" in 2026 should teach the current consensus stack, not the 2017 variant, while using the original paper for conceptual derivation.

**nanoGPT is GPT-2 style, not LLaMA style.** The Karpathy nanoGPT/build-nanogpt reference implementation uses LayerNorm (not RMSNorm), GELU (not SwiGLU), and learned absolute embeddings (not RoPE). This was accurate for the GPT-2 era. Students who use nanoGPT as a reference will naturally build a GPT-2-style model; the course should make explicit that additional exercises are needed to reach the LLaMA-family architecture that they will encounter in all subsequent modules.

**Flash attention context.** PyTorch 2.0+ includes `F.scaled_dot_product_attention` with optional FlashAttention dispatch. The course correctly forbids this in learner code. However, students should know what FlashAttention (Dao et al., 2022) does conceptually - it is an IO-aware tiling of the standard SDPA that avoids materializing the full attention matrix in HBM, gaining speed and memory proportional to sequence length. This is conceptual knowledge, not implementation work.

**RoPE extensions for long context.** YaRN, NTK-aware scaling, and LongRoPE extend RoPE to context lengths far beyond training. These are relevant for production but not for a foundations module. Mentioning them briefly as "this is why RoPE dominates" is sufficient.

**Karpathy's microgpt (Feb 2026).** A 200-line, no-dependency pure-Python implementation covering dataset, tokenizer, autograd engine, GPT-2-style network, Adam, and inference. Useful reference for showing how minimal the core mechanism is, though it is even more stripped-down than nanoGPT and lacks the modern stack (RMSNorm, RoPE, SwiGLU).

**KV-cache and GQA are now interview-level knowledge.** Any practitioner working on inference-heavy systems (BEVFormer, VLA deployment) in 2026 is expected to understand KV-cache mechanics and why GQA reduces its footprint. Treating these as purely optional stretches underserves the course audience.
