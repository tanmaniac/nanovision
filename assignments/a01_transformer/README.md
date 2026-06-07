# A1 — The Transformer, from scratch (LLaMA-style)

## Motivation
The transformer is the substrate for everything later in this course: ViT, CLIP,
DiT, VLMs, BEVFormer, and VLA policies all reuse the attention and block code you
write here. Building it first as a pure sequence model, before any pixels, earns
the "why attention" intuition cleanly: attention is a content-based,
permutation-equivariant gather over a set, and positional encoding is what
reintroduces order. You build the 2026 consensus stack (RMSNorm, RoPE, SwiGLU),
with the 2017 original (LayerNorm, absolute encodings, GELU MLP) kept as a
selectable historical contrast.

## Background
Scaled dot-product attention:

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d) + M) V

M is an optional additive mask (0 to keep, -inf to forbid). Multi-head splits the
model dim into h heads, attends per head, concatenates, and projects. Grouped-query
attention uses fewer KV heads than query heads (one KV head per group); a single KV
head is multi-query attention.

The block is pre-norm:

    h = x + Attn(Norm(x))
    y = h + FFN(Norm(h))

Defaults: RMSNorm (rescale by root-mean-square over the last axis times a learned
gain, no mean subtraction or bias), RoPE (rotate pairs of Q/K channels by an angle
proportional to position, so the dot product depends only on relative offset), and
SwiGLU (a gated SiLU feed-forward). Shapes are (batch, seq, dim) at boundaries,
(batch, heads, seq, head_dim) inside attention, and attention weights are
(batch, heads, seq_q, seq_k).

## What you'll implement
Seven holes: `scaled_dot_product_attention` and `MultiHeadAttention.forward` (in
`starter/attention.py`); `build_causal_mask`, `apply_rope`, and
`TransformerBlock.forward` (in `starter/transformer.py`); `RMSNorm.forward` and
`SwiGLU.forward` (in `starter/primitives.py`). The encoder/decoder stacks, the
absolute positional encodings, and the char-LM assembly are provided.

## Tasks
1. `scaled_dot_product_attention` — the stable-softmax gather; return weights too.
2. `MultiHeadAttention.forward` — head split/merge; self-, cross-, and GQA.
3. `build_causal_mask` — additive -inf above the diagonal.
4. `apply_rope` — rotary position embedding on q and k.
5. `RMSNorm.forward` — rescale by RMS times a learned gain.
6. `SwiGLU.forward` — gated SiLU feed-forward.
7. `TransformerBlock.forward` — the pre-norm residual sub-layers.

Each maps to a `raise NotImplementedError(...)` in `starter/` and to one test.

## How to verify
From the repo root with the `nanovision` env active, in this order:

    make test A=a01_transformer     # your starter (red until you fill the holes)

The tests run shapes → gradcheck → reference-value → causal → overfit →
forbidden-imports. To confirm the reference passes and render the figures:

    make verify A=a01_transformer   # reference solution (should be green)
    make viz    A=a01_transformer   # writes out/causal_attention.png, out/charlm_loss.png

The reference implementation is visible in `nanovision/attention.py`,
`nanovision/transformer.py`, and `nanovision/primitives.py`; read it if you get
stuck.

## Compute notes
CPU only, seconds to a few minutes. The overfit test uses dim 64, 4 heads, depth
2, seq 32, batch 8, Adam lr 3e-3, 500 steps, and reaches cross-entropy ~0.013. A
flat loss curve usually means a wrong mask or misplaced residual, not a tuning
issue. gradcheck runs at float64 with dropout off.

## Stretch goals
1. KV-cache for autoregressive generation; measure the speedup.
2. Pre-norm vs post-norm: train both and plot the stability difference.
3. Swap RoPE for sinusoidal on the copy/sort task and compare.
4. Multi-query vs grouped-query vs full multi-head: parameter count and quality.

## Further reading
- Vaswani et al., "Attention Is All You Need" (2017).
- Su et al., "RoFormer" (2021) — rotary position embedding.
- Zhang & Sennrich, "Root Mean Square Layer Normalization" (2019).
- Shazeer, "GLU Variants Improve Transformer" (2020) — SwiGLU.
- Ainslie et al., "GQA" (2023) — grouped-query attention.
