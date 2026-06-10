# A8 - vision-language model (LLaVA-style): build plan

Status: draft for expert review. Build subagent reads this whole file plus
`agent_build_guide.md` and mirrors the exemplar `assignments/a06_0_flow_matching` for
layout. A8 is assignment-LOCAL except it imports the shared ViT and transformer (already
shimmed - see "Reused, already in place").

## What this assignment teaches

A VLM connects a frozen image encoder to a language model by turning the image into a sequence
of tokens that live in the LM's embedding space. The student builds the LLaVA bridge: take the
ViT's per-patch features, project them with a 2-layer MLP into the LM embedding dimension,
prepend them to the text token embeddings, and run the decoder-only LM over the combined
sequence. The LM predicts the caption autoregressively, conditioning on the visual tokens. The
mechanism is the visual-token interface: how many tokens enter, where they are injected
(prepend vs cross-attention), and how the loss is masked so only text positions are supervised.

Secondary mechanisms: a Perceiver-style cross-attention resampler (the Flamingo/Q-Former
connector family, fixed output length regardless of patch count) as a contrast to the MLP
projector; the two-stage freeze curriculum (stage 1 trains only the projector, stage 2
unfreezes the LM); and an AnyRes token-count shape exercise. The grounding ablation shows the
visual path carries the grounding: remove or randomize the visual tokens and the caption loss
jumps.

## Reused, already in place (do NOT rebuild or re-shim)

- `from nanovision.vit import ViT`: the A2 ViT. `ViT.forward_features(img)` returns the per-patch
  grid `(B, n_patches, dim_v)` (CLS and register tokens already dropped). This is the frozen
  vision encoder. The shim and `forward_features` were added at the orchestrator level; they
  exist and are verified.
- `from nanovision.transformer import TransformerDecoder`: the A1 decoder-only LM. It operates
  in EMBEDDING SPACE: `forward(x, memory=None, mask=None)` takes `x = (B, L, dim)` embeddings and
  returns `(B, L, dim)`. It has NO token-embedding table and NO output head of its own - A8
  supplies those. It supports `cross_attn=True` + `memory` (used by the resampler variant's
  cross-attention path; for the prepend path A8 uses plain `cross_attn=False`).
- `from nanovision.attention import MultiHeadAttention`: `forward(x, kv=None, mask=None)`. With
  `kv` of a different sequence length it is cross-attention - the resampler uses this (queries
  as `x`, patch features as `kv`).
- `nanovision.data.toy.image_text_batch(batch, size, channels, n_classes, n_attrs, vocab_size,
  max_len, seed)`: returns `(images (B,C,16,16), token_ids (B,L))`. Token layout: 0 = pad,
  1..n_classes = class tokens, n_classes+1..n_classes+n_attrs = attribute tokens,
  vocab_size-1 = EOS. The caption for an image is `[class_tok, attr_tok, EOS, pad...]`. The
  class and attribute are functions of the image (color channel and blob position), so a model
  that ignores the visual tokens cannot predict them - this is what makes the grounding
  ablation valid. Reuse this; do NOT add new toy data.

## The caption objective and why it is a valid grounding probe

A8 trains the VLM to predict the caption tokens from the image (image captioning). Because the
caption's class and attribute tokens are determined by the image content, predicting them
requires reading the visual tokens. This is the same mechanism as VQA - a VQA prompt would
prepend a fixed question prefix before the answer tokens; the README explains that, but the
graded task is captioning to avoid inventing question vocabulary. The grounding ablation
(drop/randomize visual tokens -> loss jumps) is the empirical payoff.

Note on the frozen LM: the A1 decoder here is randomly initialized, not pretrained on language
(the course has no large text corpus). So "freeze the LM" in stage 1 means freezing a random
decoder while the projector and the output head learn to drive it. The lesson is the visual-token
interface and the freeze-curriculum mechanics, not exploiting a language prior the course does
not have. State this honestly in the README; do not imply the frozen LM carries linguistic
knowledge. (Open question 3 asks the expert whether to instead pretrain the tiny LM first.)

## Shapes (fix these numbers)

- Image: (B, 3, 16, 16). ViT: patch 4 -> grid 4x4 -> n_patches = 16, dim_v = 32.
- Visual features: `ViT.forward_features(img)` -> (B, 16, 32).
- LM embedding dim dim_l = 64, vocab_size V = 32, depth 2-4, n_heads 2-4.
- Projector: (B, 16, 32) -> (B, 16, 64). Visual token count N = 16.
- Text: token_ids (B, L) with L = max_len = 8; text embeds (B, 8, 64).
- Combined sequence: prepend visual -> (B, 16 + 8, 64) = (B, 24, 64).
- LM output -> head -> logits (B, 24, V). Loss only on the text-target positions.

## Files (mirror the exemplar layout)

### `projector.py` (holed; solution copy)

Hole:
- `MLPProjector.forward(self, feats)`: a 2-layer MLP, `Linear(dim_v, dim_l) -> GELU ->
  Linear(dim_l, dim_l)`, applied per patch token. (B, N, dim_v) -> (B, N, dim_l). This is the
  LLaVA-1.5 connector. `__init__` (the two Linears + GELU) is provided; only `forward` is the
  hole. State in the docstring that LLaVA-1.5 replaced LLaVA's single linear layer with this
  2-layer MLP and that the controlled ablation showed it helps.

### `resampler.py` (holed; solution copy)

Hole:
- `PerceiverResampler.forward(self, feats)`: Q learned query vectors (an `nn.Parameter`
  `(1, Q, dim_l)`, expanded to batch) cross-attend over the projected patch features via one
  `MultiHeadAttention(queries, kv=feats_proj)` layer, returning (B, Q, dim_l). A small input
  projection `Linear(dim_v, dim_l)` maps features to dim_l first (provided in `__init__`). The
  structure is: `out = norm(queries + attn(queries, kv=feats_proj))` - a single cross-attention
  with one residual+norm, so the gradcheck target is unambiguous. The output length is Q
  regardless of N. `__init__` (queries Parameter, input proj, the MHA, the norm) is provided;
  only `forward` is the hole. Docstring must state this is a deliberate single-layer miniature
  of a real Perceiver Resampler / Q-Former, which stacks several blocks each with query
  self-attention + an FFN, and that A8 still PREPENDS these Q tokens (it is not Flamingo's gated
  cross-attention injection).

### `vlm.py` (holed + provided)

Holes:
- `prepend_visual(visual_tokens, text_embeds)`: concatenate along the sequence axis,
  (B, N, d) + (B, L, d) -> (B, N+L, d). One line, but it is the interface the whole assignment
  is about - keep it a named hole with a docstring on token ordering (visual first, then text).
- `vlm_loss(logits, token_ids, n_visual)`: next-token cross-entropy supervised ONLY on the text
  positions. The combined sequence is `[visual_0..visual_{N-1}, text_0..text_{L-1}]`. Standard
  teacher forcing predicts position i+1 from positions <= i. The targets are the text tokens
  `text_1..text_{L-1}` (and the model must also predict text_0 from the last visual token).
  Build a label tensor of length N+L by: fill the whole (B, N+L) with -100, write the true
  tokens into the text slice `labels[:, N:N+L] = token_ids`, then re-mask pads WITHIN that slice
  `labels[:, N:N+L][token_ids == 0] = -100`. Do NOT index `labels[token_ids == 0]` directly -
  labels has length N+L and token_ids has length L, so the boolean mask would mismatch. Then
  `F.cross_entropy(logits[:, :-1].reshape(-1, V), labels[:, 1:].reshape(-1), ignore_index=-100)`.
  Spell the shift out explicitly in the docstring: the last visual position predicts text_0
  (the class token), text_k predicts text_{k+1}, pads ignored - no BOS/separator is used because
  generation also starts from an empty text side, so train and inference agree.

Provided (identical both copies):
- `VLM.__init__(cfg)`: builds the frozen ViT (`nanovision.vit.ViT`, `requires_grad_(False)`),
  the `MLPProjector` (or resampler if `cfg.connector == "resampler"`), a token-embedding table
  `nn.Embedding(V, dim_l)`, the `TransformerDecoder` (cross_attn=False), a final `nn.LayerNorm`
  and output head `nn.Linear(dim_l, V)`. A `causal` mask is built for the combined length via
  `nanovision.transformer.build_causal_mask`.
- `VLM.forward(img, token_ids)`: `feats = vit.forward_features(img)`; `vis = projector(feats)`;
  `txt = embed(token_ids)`; `seq = prepend_visual(vis, txt)`; `h = decoder(seq, mask=causal)`;
  `logits = head(norm(h))`. Return logits (B, N+L, V) and n_visual. (forward is provided; it
  calls the holed prepend_visual.)
- `VLM.set_stage(stage)`: stage 1 -> only the projector (and the output head + token embedding)
  train, ViT and decoder frozen; stage 2 -> decoder also unfrozen, ViT still frozen. Implement
  by toggling `requires_grad`. Provided.
- `VLM.generate(img, max_new_tokens, ...)`: greedy autoregressive decode starting from an empty
  (or BOS-less) text sequence - prepend visual tokens, then sample/argmax one token at a time,
  appending to the text side. Provided.

### `anyres.py` (provided only; the shape exercise)

- `anyres_token_count(n_patches_per_crop, grid, with_overview=True)`: returns
  `grid*grid*n_patches_per_crop (+ n_patches_per_crop for the overview)`. The LLaVA-NeXT AnyRes
  arithmetic. Provided with a docstring giving the 336px example: a 2x2 grid is 4 crops at 576
  tokens each (2304) plus a 336px overview image also at 576 tokens, total 2880. The overview is
  a full image (576 tokens), NOT a single token.
- `tile_image(img, grid)`: split (B,C,H,W) into grid*grid non-overlapping crops, return
  (B, grid*grid, C, H/grid, W/grid). A reshape exercise for the README's tiling figure.
  Provided.

### `config.py` (provided)

`@dataclass VLMConfig`: img_size 16, patch 4, in_chans 3, vit_dim 32, vit_depth 2, vit_heads 2,
n_registers 0 (keep the ViT simple for the encoder), dim_l 64, vocab_size 32, lm_depth 2,
lm_heads 4, n_classes 4, n_attrs 4, max_len 8, connector "mlp" (or "resampler"), n_queries 8
(resampler Q). Note: instantiate the ViT with num_classes set to anything (unused via
forward_features) and pool any valid value.

### `viz.py` (provided)

Train the VLM briefly (stage 1 then stage 2) on a small image_text_batch, show a few images
with their generated captions, and render the grounding ablation as a bar chart (caption loss:
full model vs visual-tokens-zeroed vs projector-randomized). Save to `out/`. Not graded.

### `conftest.py`, `__init__.py`, `solution/`

Mirror the exemplar conftest (adjust the docstring file list). `solution/` holds the three
holed files: `projector.py`, `resampler.py`, `vlm.py`, plus `__init__.py`.

## Tests (env python, CPU, seconds each; mirror the exemplar test style)

1. `test_shapes.py`: `MLPProjector` (B,16,32)->(B,16,64); `VLM.forward` logits (B, 24, V) and
   n_visual == 16; `PerceiverResampler` (B,16,32)->(B,Q,64).
2. `test_resampler_token_count.py`: the resampler output length is Q for two different patch
   counts N (e.g. feed (B,16,32) and (B,9,32)); confirms fixed-length compression independent of
   resolution.
3. `test_prepend.py`: `prepend_visual` yields length N+L, visual tokens occupy positions 0..N-1
   (assert the first N rows equal the visual input), text follows. And `vlm_loss` ignores the
   visual + pad positions: construct logits/labels where only the text span should count, verify
   the loss equals a hand-computed cross-entropy over just those positions (and that changing a
   visual-span logit does not change the loss).
4. `test_gradcheck.py`: float64 gradcheck on `MLPProjector.forward` and on
   `PerceiverResampler.forward` (small dims). Follow the exemplar's gradcheck setup.
5. `test_overfit_caption.py`: overfit ONE image_text_batch (e.g. 4 pairs) in stage 2; the caption
   cross-entropy falls to near zero within a bounded step count (~300), and `VLM.generate`
   reproduces each caption's class+attr+EOS tokens exactly (greedy). Report the measured final
   loss; set the threshold from it per the no-thrash rule.
6. `test_stages.py`: after `set_stage(1)`, only projector (+ head + token embedding) params have
   `requires_grad=True` and the ViT and decoder are frozen; after `set_stage(2)` the decoder is
   trainable and the ViT is still frozen. Also: one optimizer step in stage 1 changes projector
   params but leaves decoder params unchanged (assert with a before/after clone).
7. `test_grounding_ablation.py`: take an overfit model from test 5's regime (train briefly in
   the test, a few hundred steps), then compare caption loss with the true visual tokens vs with
   the visual tokens zeroed (or the projector re-initialized to random). Assert
   `loss_ablated > loss_full` by a wide margin (the visual path carries the grounding). Keep the
   training short and the assertion a ratio, not an absolute. Harden against the autoregressive
   leak the expert flagged: pick a seed/batch where at least two rows share a CLASS but differ in
   ATTRIBUTE (assert this property on the drawn captions in the test setup), so the image is
   required to disambiguate and the attribute cannot be inferred from the class token alone. The
   class position is always image-grounded (predicted from the last visual token), so also report
   the per-position loss split (class position vs attribute position) for both full and ablated -
   the class-position gap is the seed-robust grounding signal. Measure and report all numbers.
8. `test_anyres.py`: `anyres_token_count(576, grid=2, with_overview=True)` == 2880 (4*576 patch
   tokens + 576 overview); `with_overview=False` == 2304. Assert the exact LLaVA-NeXT arithmetic.
   `tile_image` round-trips shapes: (B,3,16,16) -> (B,4,3,8,8) for grid 2.
9. `test_forbidden_imports.py`: one static scan over the holed files + solution + the relevant
   shims, mirroring the exemplar. Forbid importing a ready-made VLM/processor (transformers,
   timm VLM heads); forbid bare imports of the owned shared files (must come via nanovision.*).

Solution mode all green; default mode fails only at the holes (NotImplementedError), except
`test_forbidden_imports` (static scan, passes both modes). Run with
`/home/tanmay/miniconda3/envs/nanovision/bin/python`.

## README (comprehensive lecture notes, per the lecture-notes skill, real LaTeX)

Fixed section order. Cover:
- The visual-token interface: vision encoder -> patch grid -> projector into the LM embedding
  space -> prepend to text -> autoregressive LM. The three degrees of freedom (how many tokens,
  where injected, how positions are communicated).
- Connector families: MLP projector (LLaVA / LLaVA-1.5, and the MLP patch-merger of Qwen2.5-VL -
  the implemented family: one token per patch or per merged patch group, all enter the context),
  the resampler / cross-attention pooling (Flamingo Perceiver Resampler, BLIP-2 Q-Former, the
  original Qwen-VL's single-layer cross-attention adapter - fixed Q tokens, variable resolution),
  and early fusion (Fuyu, Chameleon, Llama 4 - no separate encoder, image patches or VQ tokens
  into one transformer). Keep Qwen-VL (2023, resampler family) distinct from Qwen2.5-VL (2025,
  MLP patch-merger family). Explain the trade-offs, do not implement early fusion.
- How visual tokens enter: prepend/concat (LLaVA, no LM change) vs gated cross-attention into
  frozen LM layers (Flamingo). The course implements prepend; the resampler variant shows the
  cross-attention pooling machinery but STILL prepends its Q resampled tokens - it changes how
  many tokens and how they are pooled, not the injection point. Do not conflate "resampler" with
  Flamingo's gated-cross-attention injection.
- How positions are communicated: A8 relies on the reused decoder's sequence-order RoPE (applied
  inside attention), with no separate visual position IDs - visual tokens get positions 0..N-1
  and text follows at N.., so train and generation positions agree. List 2D RoPE and absolute
  patch-position IDs as alternatives real VLMs use that A8 does not implement.
- The two-stage freeze curriculum: stage 1 aligns the projector with the frozen encoder and LM;
  stage 2 unfreezes the LM for instruction following. State the honest caveats: (a) the course's
  LM is randomly initialized, so the lesson is the interface and curriculum mechanics, not a
  language prior; (b) the token-embedding table and output head must train in stage 1 here (real
  LLaVA freezes the LM's embedding/unembedding and moves only the projector - call this a
  course-scale deviation, since with a random decoder there would otherwise be no fixed
  embedding geometry to map visual features into); (c) report stage-1 vs stage-2 final loss
  separately so the reader sees stage 1 is partial alignment and stage 2 is where the caption
  actually fits - do not imply stage 1 alone produces good captions.
- AnyRes / tiling (LLaVA-NeXT) as the shape exercise: split into sub-crops, encode each,
  concatenate the grids; visual token count scales with crop count and becomes a first-class
  cost knob (576 -> 2880 at 2x2: 4 crops of 576 + a 576-token overview). Pixel-shuffle
  (InternVL2) and patch-merger (Qwen2.5-VL) as token-compression responses, mentioned not
  implemented.
- Where this goes: the VLA capstone is a VLM policy - the action token is appended and predicted
  autoregressively exactly like a text token; the visual-token interface built here is reused
  verbatim. Also note SigLIP replacing CLIP as the default backbone and the 2024+ trend of
  unfreezing the ViT (a simplification the course does not follow).

Verify every arXiv id by fetching `https://arxiv.org/abs/<id>` before citing: Flamingo
2204.14198, BLIP-2 2301.12597, LLaVA 2304.08485, LLaVA-1.5 2310.03744, Qwen-VL 2308.12966,
Chameleon 2405.09818, Cambrian-1 2406.16860. (Re-verify each title; do not trust this list
blind.) Run the mandatory context-less style review on the README before finishing.

## ASSIGNMENT.md

Concise builder contract in `TEMPLATE.md` format: the holes (MLPProjector.forward,
PerceiverResampler.forward, prepend_visual, vlm_loss), what is provided, the verify command,
the measured thresholds. Do not echo the README prose.

## Decisions resolved by the expert review (build to these)

The expert verified the core mechanism (prepend + causal mask + masked next-token CE shift) is
correct as designed. Resolutions, now folded into the spec above:

1. Captioning is an adequate VQA stand-in and gives a sound grounding ablation, with the test
   hardened (test 7) against the autoregressive attribute leak: pin a seed where a class maps to
   multiple attributes and report the per-position loss split. No question-prefix needed.
2. The loss shift is correct (expert traced it numerically). Last visual position predicts the
   class token, text_k predicts text_{k+1}, pads ignored. No BOS/separator - generation starts
   from an empty text side so train and inference agree. Build labels by fill-(-100) then write
   the text slice then re-mask pads (see vlm_loss; avoids the length-mismatch indexing bug).
3. Keep the LM randomly initialized; do NOT add an LM pretraining stage (3-token captions carry
   no language prior worth freezing). Reframe stage 1 as visual-to-LM-embedding alignment + freeze
   mechanics, and state the course-scale deviations honestly: the token embedding + output head
   train in stage 1 here (real LLaVA freezes them), and report stage-1 vs stage-2 loss separately.
4. One cross-attention layer with Q learned queries is a faithful minimal Perceiver/Q-Former
   core; MLP-beats-resampler on the toy is the right data point (mirrors LLaVA-1.5). README states
   real resamplers stack multiple blocks (query self-attn + FFN) and that A8's resampler still
   prepends.
5. AnyRes stays a shape exercise (token-count arithmetic + tile_image reshape). Optional viz
   extra: encode the 4 tiles through the frozen ViT to show the 4*16=64 token concatenation on
   the 16x16 toy - cheap, not required, no new training.

Corrected numbers: AnyRes 2x2 = 2880 tokens (4 crops of 576 + a 576-token overview), NOT 2305 -
the overview is a full image, not one token. Qwen-VL (2023) is resampler-family; Qwen2.5-VL
(2025) is MLP-patch-merger-family - keep them distinct.
