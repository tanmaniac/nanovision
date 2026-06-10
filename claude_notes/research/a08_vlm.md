# A8 — Vision-Language Models: research and scope validation

## 1. Key concepts a student must learn

### The visual-token interface

A VLM feeds visual information into a language model by converting image content into a sequence of tokens that sit in the same embedding space as text tokens. The student needs to understand what that actually means mechanically: the vision encoder (a ViT or CLIP backbone) produces a grid of patch embeddings; those embeddings must be projected — dimensionally and semantically — into the vocabulary/embedding space the LM expects. Once projected, the visual tokens are prepended (or interleaved) with the text tokens and passed through the LM's self-attention stack. The LM then produces text autoregressively, conditioning on both the visual tokens and any preceding text.

This interface has three degrees of freedom the student should reason about concretely:
- how many visual tokens enter the LM (dense: one per patch; compressed: fixed-length from a resampler; tiled: one-per-patch per crop)
- how positional information is communicated (2D RoPE, absolute patch position IDs, or none beyond sequence order)
- at what point in the LM the visual information is injected (prepend to context vs. cross-attention into every LM layer vs. early-fusion at tokenization time)

### Projector / connector design

Three families of connector exist in the literature:

**Linear or MLP projector (LLaVA / LLaVA-1.5 / LLaMA 3.2-Vision / Qwen2.5-VL).**
The vision encoder produces N patch embeddings (e.g., 576 for CLIP ViT-L/14 at 336 px). A linear layer or two-layer MLP with GELU maps each embedding from the encoder's dimension (e.g., 1024) to the LM's embedding dimension (e.g., 4096). All N tokens enter the LM context. Simple, fast, and — as LLaVA-1.5 showed in a controlled ablation — surprisingly competitive.

**Resampler / cross-attention pooling (Flamingo Perceiver Resampler / BLIP-2 Q-Former / original Qwen-VL).**
A fixed number of learnable query vectors (32–256) attend over the full patch sequence via cross-attention, producing a compact fixed-length representation regardless of image resolution. Flamingo then feeds these into gated cross-attention layers inserted between frozen LM layers rather than prepending to context. BLIP-2's Q-Former adds a shared self-attention stream between the queries and text, enabling contrastive + generative pre-training in two stages. The resampler compresses heavily and handles variable resolution naturally, but the cross-attention machinery is more complex and can lose spatial detail.

**Pixel-level / early fusion (Fuyu / Chameleon / Llama 4).**
No separate vision encoder. Image patches are linearly projected directly into the LM's first layer (Fuyu), or images are tokenized by a discrete VQ-VAE and all tokens — image and text — are fed into a single transformer trained from scratch on interleaved data (Chameleon, Llama 4). Maximally unified but requires far larger pretraining compute and data; not practical to demonstrate from scratch at course scale.

### How visual tokens enter the LM context

In the prepend / concat pattern (LLaVA family), projected visual tokens occupy positions 0..N-1 of the sequence and the instruction text follows at position N. The LM's causal self-attention attends over the full sequence; every text token attends to all visual tokens. No modification to the LM architecture is needed.

In the cross-attention pattern (Flamingo), the LM layers are interleaved with new gated cross-attention layers that accept the resampled visual tokens as keys and values. The LM's existing weights are frozen; only the new cross-attention layers (and the resampler) train. This makes Flamingo's LM weights strictly separate from the visual pathway.

The student should understand why both patterns are valid and their practical tradeoffs: prepend is simpler and trains faster but scales linearly with token count; cross-attention allows frozen LM reuse and decouples LM context length from vision granularity.

### Training stages: freeze / unfreeze curriculum

LLaVA's published training recipe is the canonical two-stage example:

**Stage 1 (alignment pre-training).** The vision encoder and LM are both frozen. Only the projector trains on a large image-caption dataset (LLaVA used 595K filtered CC3M pairs). The goal is to map visual embeddings into the LM's embedding space — alignment, not instruction following.

**Stage 2 (visual instruction tuning).** The projector and the LM are unfrozen (the vision encoder stays frozen in LLaVA-1.5; some later work unfreezes the top ViT blocks). The model trains on instruction-following data (LLaVA-1.5 used ~665K samples including academic VQA datasets reformatted as conversations).

The pedagogical point: the projector is the only component that must be initialized during stage 1. The LM's weights do not move. This makes stage 1 cheap — a few GPU-hours even at modest scale. Stage 2 is where grounding actually forms, but it builds on the alignment already learned in stage 1.

A later variant introduced in BLIP-2 splits pre-training further: (1) train the Q-Former with frozen encoder on contrastive + matching + captioning losses; (2) then connect to a frozen LLM and train on generative loss only. The student should understand why that extra stage exists: the Q-Former needs to learn to extract "what the LLM wants to know" rather than just copying encoder features.

---

## 2. Mechanisms to implement from scratch

### Mechanism 1: MLP projector + prepend VLM on a toy caption/VQA task

**What to build.** Reuse the ViT from A2/A4 (frozen, pretrained on CIFAR-10 or ImageNet-1k) and the decoder-only LM from A1. Add a two-layer MLP projector that maps vision encoder output dimension to LM embedding dimension. Prepend visual tokens to the text sequence. Train only the projector in stage 1 (caption objective on a small dataset), then unfreeze the LM in stage 2 (VQA objective).

**Toy dataset.** Build a minimal synthetic VQA set: 100–500 CIFAR-10 images with labels used as captions ("a photo of a dog") and three-choice questions ("what animal is in this image?"). At this scale the full pipeline fits in under 1GB of GPU memory and a single overfit-one-batch run takes seconds.

**Verifiable tasks.**
- Shape test: `visual_tokens = projector(encoder(img)); assert visual_tokens.shape == (B, N_patches, lm_embed_dim)`.
- Gradcheck on projector weights with a single forward/backward pass.
- Overfit-one-batch: loss reaches near-zero on a single (image, caption) pair within 100 steps; the LM generates the correct caption token-for-token.
- Ablation 1 (projector vs. identity): replace the trained MLP with a random linear projection; measure VQA accuracy drop on the toy set. Should drop substantially, demonstrating the projector carries grounding.
- Ablation 2 (stage 1 vs. no stage 1): skip alignment pre-training and go straight to stage 2; plot convergence curves side by side.

### Mechanism 2: fixed-length resampler connector (optional / advanced)

**What to build.** Implement a minimal Perceiver-style resampler: a `nn.Parameter` tensor of Q learned query vectors, a single cross-attention layer attending over the N patch embeddings, producing Q output tokens. Swap the MLP projector for this resampler and repeat the overfit test.

**Verifiable tasks.**
- Shape test: `resampled = resampler(patch_embeds); assert resampled.shape == (B, Q, lm_embed_dim)` regardless of input resolution.
- Confirm the token count entering the LM is Q, not N, even when N changes (test with two image resolutions).
- Compare final VQA accuracy vs. MLP projector on the toy set. At tiny scale, the MLP will likely match or exceed the resampler, which is itself a pedagogical data point.

---

## 3. Assessment of the draft scope

**What is right.** The core idea — frozen encoder + projector + small LM decoder, trained on toy VQA/caption, ablation on the projector — is the right minimal unit to teach. LLaVA-style architecture is still the dominant family in open-weight models as of 2026 (LLaMA 3.2-Vision, Qwen2.5-VL's patch merger, InternVL2, SmolVLM all use MLP-based connectors). The focus on the visual-token interface rather than pretraining a real LLM is the right pedagogical choice.

**What is missing or underemphasized.**

*Resampler / cross-attention variant.* The draft mentions it as context but does not build it. This is a gap. Flamingo (2022) and BLIP-2 (2023) are canonical works; the resampler is a genuinely different architectural pattern — fixed output length regardless of input resolution — and understanding why it exists (compute budget, handling variable-resolution images before AnyRes existed) is part of the history. An optional mechanism-2 implementation of a minimal 1-layer cross-attention resampler, even on the toy problem, would close this gap without adding much complexity.

*AnyRes / tiling.* LLaVA-NeXT (January 2024) introduced AnyRes: the image is split into sub-crops, each encoded independently, the patch token grids concatenated. This is now the standard technique in most production VLMs (LLaVA-OneVision, InternVL2, Qwen2.5-VL, SmolVLM). A short conceptual explanation with a shape-level exercise (show that N_tokens scales with number of crops) should be included, even if full implementation is out of scope.

*Two-stage training.* The draft mentions "train on a toy VQA/caption set" without specifying the stage-1 alignment / stage-2 instruction-tune split. This distinction is one of the most important engineering lessons from LLaVA and BLIP-2 and should be explicit.

*Visual token count as a first-class design knob.* The number of tokens the vision encoder injects — 64, 256, 576, 2880 — has a direct cost in LM context length and therefore in inference speed and memory. Students should compute this concretely: at 336 px with patch size 14, N = 576; with AnyRes 2x2 grid, N = 4 × 576 = 2304 patch tokens plus a 336px overview image (also 576 tokens) = 2880. The overview is a full downsized image, not a single token. This is a shape-level exercise, not extra code.

**What is outdated or mis-emphasized.**

The draft does not mention Fuyu or Chameleon. Both are important to cite as existence proofs of the early-fusion direction, even though neither is the implementation target. Llama 4 (April 2025) is the current high-profile open-weight example of early-fusion native multimodality, and students in a 2026 course should know it exists and understand why its architecture is fundamentally different from LLaVA's.

The draft's framing "frozen vision encoder (from A2/A4) + projector + tiny LM (from A1)" is correct for the implementation. However, it should note explicitly that in 2024-2025 practice, the vision encoder is often partially unfrozen during stage 2 (InternVL-series and Qwen2-VL both fine-tune the ViT), and that the full freeze is a simplifying choice for the course, not a universal best practice.

**What to add / reorder.**

1. Make the two-stage training curriculum explicit in the scope statement.
2. Add a brief conceptual section on AnyRes / tiling with a shape exercise.
3. Add optional mechanism-2: minimal 1-layer cross-attention resampler, same toy problem, comparison with MLP.
4. Add a "what we are not building" note: early fusion (Chameleon / Llama 4) — students should know it exists but implementing it from scratch would require pretraining a full multimodal model, which is out of scope.
5. Make the grounding ablation more concrete: measure accuracy drop when (a) projector is random, (b) stage 1 is skipped, (c) visual tokens are dropped entirely. These three data points together tell the story of where grounding lives.

**Is the frozen-encoder + MLP still representative?** Yes, as of 2026 the MLP projector is the most widely deployed connector in open-weight models. The resampler (Q-Former / Perceiver) is historically important and conceptually distinct, and warrants a brief implementation. Early fusion is the emerging direction but is not the implementation target at this scale.

---

## 4. Dependencies and downstream connections

**Depends on.**
- A1 (decoder-only LM): the course's tiny LM is reused directly as the language backbone. Students need to understand how to prepend an arbitrary token sequence to the LM context without modifying the LM's weight matrices.
- A2 (ViT): the vision encoder whose patch embeddings become visual tokens. Students need to know how to extract intermediate representations (usually the penultimate layer CLS token or the full patch grid) rather than just the classification head output.
- A4 (CLIP / contrastive vision encoders): CLIP's image encoder is the most common choice of frozen backbone in VLMs. Understanding why CLIP features are a good starting point for VLM alignment is part of the lesson.

**Feeds into.**
- A13 (Vision-Language-Action / VLA): a VLA is a VLM policy. The action token is appended to the output sequence and predicted autoregressively, exactly like a text token. The visual-token interface built in A8 is reused verbatim; the only addition is the action head and the robot observation / action data. A student who understands A8 deeply will find A13's architecture almost trivially familiar.

---

## 5. Must-read sources

1. **Flamingo: a Visual Language Model for Few-Shot Learning** — Alayrac et al., DeepMind, NeurIPS 2022. Introduced the Perceiver Resampler and gated cross-attention layers interleaved into a frozen LM; the first large-scale demonstration that visual and language backbones can be connected without joint pretraining.

2. **BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models** — Li et al., Salesforce, ICML 2023 (arXiv January 2023). Introduced the Q-Former two-stage training recipe and showed that a 188M-parameter bridging module could match an 80B Flamingo on zero-shot VQA.

3. **Visual Instruction Tuning (LLaVA)** — Liu et al., NeurIPS 2023 Oral (arXiv April 2023). The paper that showed GPT-4-generated instruction data + a simple linear projector + a 7B LLaMA derivative produces strong visual assistants at low training cost. The baseline for the course implementation.

4. **Improved Baselines with Visual Instruction Tuning (LLaVA-1.5)** — Liu et al., CVPR 2024 (arXiv October 2023). Replaced the linear projector with a two-layer MLP and added academic VQA data; achieved state-of-the-art across 11 benchmarks with 1.2M samples. The controlled ablation showing MLP > linear is the primary empirical evidence that projector choice matters.

5. **Qwen-VL: A Frontier Large Vision-Language Model with Versatile Abilities** — Bai et al., Alibaba, arXiv August 2023. Used a single-layer cross-attention VL-Adapter (not full Q-Former) to compress 1024 patch embeddings to 256 tokens, with 2D positional encodings. Representative of the compressed-token family and the transition toward high-resolution-aware connectors.

6. **Chameleon: Mixed-Modal Early-Fusion Foundation Models** — Chameleon Team, Meta, arXiv May 2024. Trained a 34B transformer from scratch on interleaved image-text tokens; no separate vision encoder, no projector. Important as the canonical recent existence proof of early fusion, which is where the field is heading at scale (also see Llama 4, April 2025).

7. **Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs** — Tong et al., NYU, NeurIPS 2024 Oral. Systematic benchmark of over 20 vision encoders and multiple connector architectures (linear, MLP, resampler, Spatial Vision Aggregator); introduced CV-Bench. The most thorough controlled comparison of connector designs available as of 2025 and a useful empirical reference for the ablation section of A8.

**Notable omission in the draft scope:** The draft's sources list was not specified, but any course reading list that omits Flamingo or BLIP-2 is missing the two papers that established the resampler paradigm and the two-stage training recipe. Both should be assigned alongside LLaVA.

---

## 6. 2024-2026 developments that change how this should be taught

**AnyRes / dynamic tiling is now the default.**
LLaVA-NeXT (January 2024) popularized splitting a high-resolution image into a grid of 336 px sub-crops plus a downsized overview image, encoding each independently, and concatenating the patch grids. This is now standard in LLaVA-OneVision, InternVL2, Qwen2.5-VL, SmolVLM, and others. A student who learns only the flat 336 px encoding will be confused by any 2024-era open-weight model. The concept should be explained even if full AnyRes is not implemented.

**Visual token count / compression became a central design axis.**
At 336 px with ViT patch 14, a single image = 576 tokens. With AnyRes 2x2 = 2880 tokens (4 crops of 576 + a 576-token overview). In a 4096-token context, that is more than half the budget. By 2024-2025, token compression (pruning, pooling, pixel-shuffle, patch merging) became an active research area. Students should understand the tension: more tokens = more detail = higher cost, and the connector design is the primary control surface. InternVL2's pixel-shuffle (4 patches → 1 token via channel expansion) and Qwen2.5-VL's patch merger (two-layer MLP on 2x2 patch groups) are both simple enough to show as code snippets.

**MLP projector empirically dominated resamplers in the 2023-2024 generation.**
LLaVA-1.5 was released in October 2023 and held strong benchmark positions through 2024. PaliGemma (Google, July 2024) used a linear connector and found it preferable to an MLP in ablations. LLaMA 3.2-Vision (Meta, September 2024) uses an MLP. Qwen2.5-VL (March 2025) uses a patch merger (MLP variant). The resampler is now more of an architectural curiosity than the mainstream choice for the prepend-based VLM family, though it remains used in cross-attention VLMs and in video models where the token count without compression is prohibitive.

**Early fusion is now a live architectural competition.**
Chameleon (May 2024), Emu3 (October 2024), and Llama 4 Scout/Maverick (April 2025) all demonstrate that a single transformer trained from scratch on interleaved image-text tokens can match or exceed the frozen-encoder + connector paradigm. Llama 4 Scout is a 17B-active-parameter MoE with a 10M-token context and native multimodality. This does not make the LLaVA-style architecture obsolete for teaching — it is still the right starting point for a course — but students need to understand that "frozen encoder + projector" is an architectural choice with trade-offs, not the only option.

**The vision encoder is increasingly trained jointly, not kept frozen.**
LLaVA-1.5 froze the CLIP encoder. By 2024, InternVL2 and Qwen2-VL both fine-tune the ViT during stage 2 or use a ViT that was pretrained on a larger, more multimodal distribution (SigLIP, InternViT). PaliGemma unfreezes the SigLIP encoder entirely. The "frozen encoder" simplification is pedagogically valid but should be stated as a simplification, not as the state of the art.

**SigLIP replaced CLIP as the preferred vision backbone.**
LLaVA-1.5 uses CLIP ViT-L/14. By 2024, most new models — PaliGemma, LLaMA 3.2-Vision, SmolVLM, Cambrian-1's multi-encoder stack — use SigLIP (sigmoid contrastive loss, no temperature normalization) as the vision backbone. The difference in feature quality is measurable. A course that uses CLIP as a frozen encoder is fine for pedagogical purposes, but should note that SigLIP is the current default.

**Grounding, OCR, and document understanding became first-class VLM tasks.**
In 2023 VLMs were evaluated mainly on VQAv2 and COCO caption. By 2025, leading models are benchmarked on OCR, chart understanding, document QA, and GUI navigation. This does not change the A8 implementation target — the toy VQA/caption task is the right starting point — but the discussion of what the projector carries should mention that spatial detail retention is critical for these tasks, which is part of why AnyRes and higher resolution encoders became important.
