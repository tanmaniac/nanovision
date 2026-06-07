# assignments/a01_transformer/ASSIGNMENT.md

```yaml
id: a01_transformer
title: The Transformer, from scratch (LLaMA-style)
module: 0
type: Core
estimated_learner_hours: 7
depends_on: [a00_harness]
builds_into_shared_lib:
  - nanovision.attention.scaled_dot_product_attention
  - nanovision.attention.MultiHeadAttention
  - nanovision.transformer.TransformerBlock
  - nanovision.transformer.TransformerEncoder
  - nanovision.transformer.TransformerDecoder
  - nanovision.transformer.build_causal_mask
  - nanovision.transformer.apply_rope
  - nanovision.transformer.SinusoidalPositionalEncoding
  - nanovision.transformer.LearnedPositionalEncoding
  - nanovision.primitives.RMSNorm
  - nanovision.primitives.SwiGLU
forbidden_imports:
  - nn.MultiheadAttention
  - nn.Transformer
  - nn.TransformerEncoder
  - nn.TransformerDecoder
  - torch.nn.functional.scaled_dot_product_attention
fits_12gb: true
external_data: none
```

## motivation
Build the transformer from scratch as a pure sequence model. It is the block
imported by A2 (ViT), A4 (CLIP text tower), A7 (DiT), A8 (VLM), A11.5c (BEVFormer
cross-attention), and A13 (VLA). We build the 2026 LLaMA-style stack (RMSNorm,
RoPE, SwiGLU) with the 2017 original selectable for contrast. Full treatment,
history, paper links, and forward connections are in the README.

## background
See the README for the worked equations and shapes. The holes implement:
scaled dot-product attention `softmax(Q K^T / sqrt(d) + M) V` (stable softmax,
additive 0/-inf mask); multi-head with self/cross/GQA (n_kv_heads < n_heads, one
KV head per group, repeat_interleave before attending); the additive causal mask;
RoPE `q' = q*cos + rotate_half(q)*sin`; RMSNorm `x / sqrt(mean(x^2)+eps) * weight`;
SwiGLU `(silu(W_g x) * (W_u x)) W_d` with inner width ~8/3 dim; the pre-norm block
`h = x + Attn(Norm(x)); y = h + FFN(Norm(h))`. Shapes: (B, S, dim) at boundaries,
(B, H, S, Dh) inside attention, attention weights (B, H, Sq, Sk).

## what_you_implement
- Scaled dot-product attention (stable softmax; returns weights for viz/tests).
- Multi-head attention with self-, cross-, and grouped-query support.
- The additive causal mask.
- RoPE (apply_rope) and RMSNorm and SwiGLU.
- The pre-norm transformer block (used by the provided encoder/decoder stacks).
- The provided char-LM assembly then overfits one batch as the integration check.
  Sinusoidal and learned absolute encodings are provided for the historical
  contrast and are not graded holes.

## tasks
- **Task 1 - scaled_dot_product_attention** (`starter/attention.py`): given q,k,v
  of shape (B,H,Sq,Dh)/(B,H,Sk,Dh) and an optional additive mask, return
  (out, attn) with out (B,H,Sq,Dh) and attn (B,H,Sq,Sk). Subtract the row max
  before exponentiating. Teaches: the core gather and stable softmax.
- **Task 2 - MultiHeadAttention.forward** (`starter/attention.py`): project x (and
  kv if cross-attention) to Q,K,V; split heads; repeat KV heads for GQA; call
  Task 1; merge heads; output projection. Teaches: head split/merge; self-vs-cross
  is just where K,V come from; GQA shares KV across query groups.
- **Task 3 - build_causal_mask** (`starter/transformer.py`): additive (S,S) mask,
  -inf above the diagonal. Teaches: why a decoder cannot see the future.
- **Task 4 - apply_rope** (`starter/transformer.py`): rotate q,k by position.
  Teaches: relative position as rotation; the modern positional scheme.
- **Task 5 - RMSNorm.forward** (`starter/primitives.py`): rescale by RMS times a
  learned gain. Teaches: the LLaMA-style norm and why mean subtraction is dropped.
- **Task 6 - SwiGLU.forward** (`starter/primitives.py`): gated SiLU FFN. Teaches:
  the gated feed-forward used across the modern stack.
- **Task 7 - TransformerBlock.forward** (`starter/transformer.py`): the two (or
  three, with cross-attention) pre-norm residual sub-layers. Teaches: residual +
  norm placement; the block is the unit reused everywhere.

## tests
Run in this order (also in the README):
1. `tests/test_shapes.py` - output shapes for Tasks 1-7 (shape).
2. `tests/test_gradcheck.py` - `check_gradients` at float64 on SDPA, MHA (incl.
   GQA), RMSNorm, SwiGLU (gradcheck).
3. `tests/test_attention_reference.py` - one-hot keys so attention selects a single
   value; exact expected output (reference-value).
4. `tests/test_causal.py` - causal attention zeroes the upper triangle and matches
   an explicit-mask reference (reference-value).
5. `tests/test_overfit.py` - the assembled char-LM overfits one batch to
   cross-entropy < 0.05 in 500 steps on CPU (overfit-one-batch).
6. `tests/test_forbidden_imports.py` - the solution and shared-lib modules contain
   no `nn.MultiheadAttention` / `nn.Transformer*` / `F.scaled_dot_product_attention`
   in actual code (string/comment mentions are allowed).

## provided_boilerplate
The encoder/decoder stacks, the absolute positional encodings (sinusoidal and
learned), the char-LM assembly (`charlm.py`), the copy/sort/char generators in
`nanovision.data.toy`, the `Trainer` and `set_seed` from A0, and the loss-curve
plotting. The learner writes only Tasks 1-7.

## compute_notes
Everything runs on CPU in seconds to a few minutes; no GPU needed. The overfit
test uses dim 64, 4 heads, depth 2, seq 32, batch 8, Adam lr 3e-3, 500 steps, and
reaches cross-entropy ~0.013. A flat loss curve points to a wrong mechanism (often
the mask or the residual placement), not a tuning problem.

## stretch_goals
1. KV-cache for autoregressive generation; measure the speedup.
2. Pre-norm vs post-norm: train both, plot the stability difference.
3. Swap RoPE for sinusoidal on the copy/sort task and compare.
4. Multi-query vs grouped-query vs full multi-head: parameter count and quality.

## further_reading
- Vaswani et al., "Attention Is All You Need" (2017) - the original.
- Su et al., "RoFormer" (2021) - rotary position embedding.
- Zhang & Sennrich, "Root Mean Square Layer Normalization" (2019).
- Shazeer, "GLU Variants Improve Transformer" (2020) - SwiGLU.
- Ainslie et al., "GQA" (2023) - grouped-query attention.

## solution_notes
`set_seed(0)` makes the overfit test deterministic; final cross-entropy ~0.013,
comfortably under the 0.05 threshold. The reference-value test uses one-hot keys so
the attended value is exact to 1e-6. gradcheck runs at float64 with dropout off.
RoPE requires an even head_dim. The forbidden-imports test strips comments and
docstrings via tokenize so the modules can name the forbidden symbols in prose.
