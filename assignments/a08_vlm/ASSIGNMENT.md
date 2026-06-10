# assignments/a08_vlm/ASSIGNMENT.md

```yaml
id: a08_vlm
title: Vision-language model (LLaVA-style)
module: 3
type: Core
estimated_learner_hours: 8
depends_on: [a00_harness, a01_transformer, a02_vit]
builds_into_shared_lib: []
forbidden_imports:
  - import transformers
  - from transformers
  - import timm
  - from timm
  - import vit
  - from vit import
  - import transformer
  - from transformer import
  - import attention
  - from attention import
fits_12gb: true
external_data: "none (synthetic image-caption pairs from nanovision.data.toy)"
```

## motivation
A VLM connects a frozen image encoder to a language model by turning the image into a
sequence of tokens that live in the LM's embedding space. This assignment builds the LLaVA bridge: take
the ViT's per-patch features, project them with a 2-layer MLP into the LM embedding dim,
prepend them to the text token embeddings, and run a decoder-only LM over the combined
sequence to predict the caption autoregressively. The three degrees of freedom are how many
visual tokens enter, where they are injected (prepend vs cross-attention), and how positions
are communicated. The Perceiver resampler is the contrast connector (fixed token count from a
single cross-attention). The grounding ablation shows the visual path carries the caption:
zero the visual tokens and the loss jumps ~2500x. See the README for the math and history.

## background
See the README. The frozen ViT's `forward_features` returns the per-patch grid (B, 16, 32).
The projector maps (B, 16, 32) -> (B, 16, 64). prepend_visual concatenates visual then text
into (B, 24, 64). The decoder-only LM (RoPE, sequence order) runs over it under a causal
mask, and the head produces logits (B, 24, V). The loss is masked next-token cross-entropy
over the text positions only: the last visual position predicts the class token, text_k
predicts text_{k+1}, pads ignored, no BOS. The LM is randomly initialized (no language
prior); stage 1 trains the projector, embedding, and head with the LM frozen, stage 2 also
unfreezes the LM.

## what_you_implement
- `MLPProjector.forward` (`projector.py`): Linear -> GELU -> Linear per patch token,
  (B, N, dim_v) -> (B, N, dim_l). The LLaVA-1.5 connector.
- `PerceiverResampler.forward` (`resampler.py`): Q learned queries cross-attend over the
  projected patch features, out = norm(queries + attn(queries, kv=feats_proj)),
  (B, N, dim_v) -> (B, Q, dim_l). A single-layer Perceiver/Q-Former miniature.
- `prepend_visual` (`vlm.py`): concatenate visual then text, (B,N,d)+(B,L,d)->(B,N+L,d).
- `vlm_loss` (`vlm.py`): masked next-token cross-entropy over the text positions only.

The frozen ViT wiring, the token embedding, the decoder LM, the output head, the stage
toggling, generate, anyres, config, and viz are provided.

## tasks
1. `MLPProjector.forward` (`projector.py`): `fc2(act(fc1(feats)))`.
2. `PerceiverResampler.forward` (`resampler.py`): project feats to dim_l, expand the query
   parameter to the batch, then `norm(queries + attn(queries, kv=feats_proj))`.
3. `prepend_visual` (`vlm.py`): `torch.cat([visual_tokens, text_embeds], dim=1)`.
4. `vlm_loss` (`vlm.py`): fill (B, N+L) labels with -100, write `labels[:, N:N+L]=token_ids`,
   re-mask pads `labels[:, N:N+L][token_ids==0]=-100`, then
   `cross_entropy(logits[:,:-1], labels[:,1:], ignore_index=-100)`.

## tests
Run with `NANOVISION_IMPL=solution python -m pytest assignments/a08_vlm/tests`.
1. `tests/test_shapes.py` - projector (B,16,32)->(B,16,64); VLM logits (B,24,V), n_visual 16;
   resampler (B,16,32)->(B,Q,64).
2. `tests/test_resampler_token_count.py` - resampler output length is Q for N in {16, 9}.
3. `tests/test_prepend.py` - prepend ordering (visual first); vlm_loss equals a hand-computed
   CE over only the text targets and ignores visual-span logits.
4. `tests/test_gradcheck.py` - float64 gradcheck of the projector and the resampler.
5. `tests/test_overfit_caption.py` - stage 2 overfits 4 pairs to caption loss < 0.05 in 300
   steps and greedy generate reproduces [class, attr, EOS] exactly.
6. `tests/test_stages.py` - stage 1 freezes ViT + decoder, trains projector/embedding/head;
   stage 2 unfreezes the decoder, ViT stays frozen; a stage-1 step moves the projector but
   not the decoder.
7. `tests/test_grounding_ablation.py` - after a short train, zeroing the visual tokens raises
   the caption loss > 10x; the class-position loss raises > 10x. The batch is pinned so two
   rows share a class but differ in attribute.
8. `tests/test_anyres.py` - `anyres_token_count(576, 2, True)==2880`, `(576, 2, False)==2304`;
   `tile_image` (B,3,16,16)->(B,4,3,8,8) for grid 2.
9. `tests/test_forbidden_imports.py` - no transformers/timm; the owned shared modules (vit,
   transformer, attention) come via `nanovision.*`, not bare. Passes with the holes in place.

## provided_boilerplate
`vlm.py` `VLM` (frozen ViT, connector, token embedding, decoder, head, set_stage, generate).
`anyres.py` `anyres_token_count` and `tile_image`. `config.py` `VLMConfig`.
`nanovision.data.toy.image_text_batch` (image-caption pairs). `viz.py` trains stage 1 then
stage 2, prints generated captions, and renders the grounding-ablation bar chart. The ViT,
the decoder-only LM, and multi-head attention come from `nanovision.vit`,
`nanovision.transformer`, and `nanovision.attention`.

## compute_notes
All tests on CPU in seconds. The overfit and grounding tests train 4 fixed pairs for 300
Adam steps. Measured (seed 0): overfit stage-2 caption loss ~7e-4 and generate reproduces all
4 captions exactly; grounding loss_full ~7e-4 vs loss_zeroed ~1.8 (ratio ~2500x), per
position class 0.0009 -> 3.0 and attribute 0.0007 -> 2.4 when the visual tokens are zeroed.
Stage 1 final loss ~0.002, stage 2 ~1e-5. MLP connector ~7e-4 vs resampler ~9e-4 at 300
steps (MLP slightly beats the single-layer resampler, the LLaVA-1.5 data point). Thresholds
are set loose (caption loss < 0.05, ablation ratio > 10) per the no-thrash rule.

## solution_notes
The combined sequence is [visual_0..visual_{N-1}, text_0..text_{L-1}] under one causal mask
of length N+L. The loss shift supervises text positions only: the last visual position
(index N-1) predicts text_0, text_k predicts text_{k+1}, pads ignored, no BOS - generation
also starts from an empty text side so train and inference agree. Build labels by
fill-(-100) then write the text slice then re-mask pads; indexing `labels[token_ids==0]`
directly would mismatch lengths (labels is N+L, token_ids is L). The resampler is a single
cross-attention with one residual+norm, so its output length is Q for any N and the gradcheck
target is unambiguous; it still prepends its Q tokens (not Flamingo's gated injection). Stage
1 here trains the token embedding and output head as well as the projector, a course-scale
deviation from real LLaVA (which freezes embedding/unembedding) because the random decoder
has no fixed embedding geometry to align into. AnyRes 2x2 = 2880 (4 crops of 576 + a
576-token overview, not one token). A8 adds no new shared nanovision symbol.
