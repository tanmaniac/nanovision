# assignments/a01_transformer/ASSIGNMENT.md  (WORKED EXAMPLE)

This is a fully filled-in example of the template, so the builder has an
unambiguous reference for the level of detail expected in every other
`ASSIGNMENT.md`. Match this depth.

```yaml
id: a01_transformer
title: The Transformer, from scratch
module: 0
type: Core
estimated_learner_hours: 6
depends_on: [a00_harness]
builds_into_shared_lib:
  - nanovision.attention.scaled_dot_product_attention
  - nanovision.attention.MultiHeadAttention
  - nanovision.transformer.TransformerBlock
  - nanovision.transformer.TransformerEncoder
  - nanovision.transformer.TransformerDecoder
  - nanovision.transformer.SinusoidalPositionalEncoding
  - nanovision.transformer.LearnedPositionalEncoding
  - nanovision.transformer.apply_rope        # stretch only
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
The transformer is the substrate for everything that follows in this course —
ViT, CLIP, DiT, VLMs, BEVFormer, and VLA policies all reuse the exact attention
and block code written here. Building it first as a pure sequence model (before
any pixels) earns the "why attention" intuition cleanly: attention is a
content-based, permutation-equivariant gather over a set, and positional encoding
is what reintroduces order. An engineer who last worked in the CNN/DETR era knows
attention conceptually; this assignment makes the shapes and the gradient flow
concrete.

## background
Scaled dot-product attention, for queries `Q ∈ R^{n×d}`, keys `K ∈ R^{m×d}`,
values `V ∈ R^{m×d_v}`:

  Attention(Q,K,V) = softmax( Q Kᵀ / √d + M ) V

where `M` is an optional additive mask (0 / −∞) for causality or padding. Output
is `R^{n×d_v}`. Multi-head splits `d` into `h` heads of size `d/h`, runs attention
per head, concatenates, and projects. A pre-LN transformer block is:

  x = x + MHA(LN(x))
  x = x + MLP(LN(x))

Sinusoidal positional encoding for position `pos`, dimension `i`:

  PE(pos, 2i)   = sin(pos / 10000^{2i/d})
  PE(pos, 2i+1) = cos(pos / 10000^{2i/d})

All shapes are `(batch, seq, dim)` in this course; attention weights are
`(batch, heads, seq_q, seq_k)`.

## what_you_implement
- Scaled dot-product attention (with mask support; return weights for viz/tests).
- Multi-head attention supporting self- and cross-attention.
- Sinusoidal and learned positional encodings.
- A pre-LN transformer block, and encoder/decoder stacks.
- Causal masking for the decoder.
- Training: a decoder-only tiny char-LM and a copy/sort toy task.

## tasks
- **Task 1 — scaled_dot_product_attention** (file:
  `starter/attention.py`, symbol: `scaled_dot_product_attention`): given
  `q,k,v` of shape `(B,H,Sq,Dh)`/`(B,H,Sk,Dh)` and optional additive `mask`,
  return `(out, attn)` where `out` is `(B,H,Sq,Dh)` and `attn` is
  `(B,H,Sq,Sk)`. Implement the softmax(QKᵀ/√Dh + mask)V formula. Teaches: the
  core gather; numerical-stable softmax. NO `F.scaled_dot_product_attention`.
- **Task 2 — MultiHeadAttention.forward** (file: `starter/attention.py`,
  symbol: `MultiHeadAttention`): project `x` (and `kv` if cross-attention) to
  Q,K,V; reshape to heads; call Task 1; merge heads; output projection. Teaches:
  head splitting/merging; the self-vs-cross distinction is just where K,V come from.
- **Task 3 — causal mask** (file: `starter/transformer.py`, symbol:
  `build_causal_mask`): return an additive `(Sq,Sk)` mask with −inf above the
  diagonal. Teaches: why a decoder can't see the future.
- **Task 4 — SinusoidalPositionalEncoding** (file: `starter/transformer.py`):
  implement the sin/cos table; add to token embeddings. Teaches: order injection.
- **Task 5 — TransformerBlock.forward** (file: `starter/transformer.py`):
  the two pre-LN residual sub-layers. Teaches: residual + norm placement.
- **Task 6 — assemble + train** (file: `starter/train_charlm.py`): wire blocks
  into a decoder-only LM, train on the provided tiny corpus with the provided
  Trainer. Teaches: the full forward + the AdamW/warmup recipe (recipe provided).

## tests
Run in this order (also documented in README "How to verify"):
1. `tests/test_shapes.py` — asserts output shapes for Tasks 1–5 on table-driven
   cases (shape).
2. `tests/test_attention_gradcheck.py` — `nanovision.gradcheck.check_gradients`
   on SDPA and MHA at float64, tiny dims (gradcheck).
3. `tests/test_attention_reference.py` — SDPA on a hand-constructed input where
   the answer is known (e.g. one-hot keys ⇒ attn picks a single value);
   reference-value.
4. `tests/test_causal.py` — asserts attn weights are zero in the upper triangle
   (reference-value).
5. `tests/test_overfit_copy.py` — decoder-only model overfits the copy task to
   loss < 1e-2 in < 500 steps on CPU (overfit-one-batch).

## provided_boilerplate
Tiny char corpus + tokenizer; copy/sort task generators (in
`nanovision.data.toy`); the `Trainer` wiring and AdamW+warmup config; plotting
of the loss curve; the model-config dataclass. Learner writes only the
mechanism in Tasks 1–6.

## compute_notes
Everything here runs on CPU in seconds to minutes; no GPU needed. Tiny char-LM:
dim 128, 4 heads, 4 layers, seq 128 — trains to coherent toy text in a few
minutes on the 4080 (or ~10–20 min CPU). Healthy loss: copy task drops to near 0
fast; char-LM cross-entropy should fall well below the unigram-entropy baseline
(state the baseline in the README so progress is legible).

## stretch_goals
1. RoPE (`apply_rope`) and an ablation vs. sinusoidal on the copy task.
2. KV-cache for autoregressive generation; measure the speedup.
3. Pre-LN vs post-LN: train both, plot stability difference.
4. Multi-query / grouped-query attention.

## further_reading
- Vaswani et al., "Attention Is All You Need" (2017) — the original.
- Xiong et al., "On Layer Normalization in the Transformer Architecture" (2020) —
  why pre-LN trains more stably.
- Su et al., "RoFormer" (2021) — rotary position embedding (for the stretch).

## solution_notes
Seed 0 makes the overfit-copy test deterministic and reliable under 500 steps.
The reference-value test uses one-hot keys so the expected attended value is
exact (no tolerance fuzz beyond 1e-6). Watch the softmax stability: subtract the
row max before exponentiating. gradcheck must use float64 and dropout disabled.
```
