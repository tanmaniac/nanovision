# assignments/a035_video/ASSIGNMENT.md

```yaml
id: a035_video
title: Video and temporal modeling - video MAE with tube masking
module: 1
type: Core
estimated_learner_hours: 5
depends_on: [a00_harness, a01_transformer, a02_vit, a03_ssl]
builds_into_shared_lib:
  - nanovision.transformer.TubeletEmbedding
forbidden_imports:
  - nn.MultiheadAttention
  - nn.TransformerEncoder
  - nn.TransformerEncoderLayer
  - nn.Transformer
  - nn.LayerNorm
  - torch.nn.functional.scaled_dot_product_attention
  - import timm
  - import transformers
  - torchvision.models
fits_12gb: true
external_data: "none (synthetic toy clips)"
```

## motivation
Extend the image MAE (A3) to video. The two genuinely new pieces are the tubelet
embedding (a 3D strided conv that tokenizes a clip, ViViT) and tube masking (mask whole
spatial columns across all frames, VideoMAE), the rest is the A1 encoder and A3's
asymmetric MAE unchanged. Tube masking exists because video is temporally redundant: a
per-token random mask leaks through time and degenerates to copy-the-neighbor. The
README covers the redundancy argument, the 90-95% VideoMAE ratio, and the joint-vs-
factorized attention cost.

## background
See the README. Shapes: a clip is (B, C, T, H, W); with spatial patch p and temporal
tubelet t the grid is T' = T/t by S' = (H/p)*(W/p), for N = T'*S' tokens. The tiny
config is T=6, t=2 (T'=3), 16x16 frames, p=4 (S'=16), so N=48.

Tubelet embed: Conv3d(C->dim, kernel=(t,p,p), stride=(t,p,p)) then flatten(2).transpose,
temporal-outermost (idx = t'*S' + s). Tube masking: one per-sample spatial keep set of
n_keep_spatial = round((1-r)*S'), applied to every temporal step, with explicit
ids_keep/ids_drop/ids_shuffle and ids_restore = argsort(ids_shuffle). Loss: A3's masked
MSE with p*p -> t*p*p, per-tubelet-normalized, on masked tubelets only.

## what_you_implement
- The tubelet embedding (the one shared symbol, exposed as
  nanovision.transformer.TubeletEmbedding).
- Tube masking (the structured space-time mask).
- The masked-tubelet reconstruction loss.

The tubelet helpers, the video ViT encoder, the mask-token reassembly, and the VideoMAE
module wiring are provided.

## tasks
- **Task 1 - TubeletEmbedding.forward** (`tubelet.py`, exposed as
  `nanovision.transformer.TubeletEmbedding`): apply self.proj (Conv3d kernel/stride
  (t,p,p)), then flatten(2).transpose(1,2) to (B, N, dim), temporal-outermost. Teaches
  that a time axis is one more strided conv dimension (ViViT tubelet embedding).
- **Task 2 - tube_masking** (`video_mae.py`): from x (B, N, D) with N = t_prime*S',
  keep n_keep_spatial = round((1-mask_ratio)*S') spatial positions per sample and apply
  the SAME pattern to every temporal step. Build ids_keep/ids_drop (idx = t*S'+s),
  ids_shuffle = cat(keep, drop), ids_restore = argsort(ids_shuffle); gather x_kept; build
  mask (B, N) in original order (1 = masked). Teaches why video needs tube masking, not
  A3's per-token masking.
- **Task 3 - video_mae_loss** (`video_mae.py`): pred, target (B, N, t*p*p*C), mask
  (B, N) with 1 on masked tubelets. Per-tubelet MSE (mean over the pixel dim), averaged
  over masked tubelets only. Teaches that the video loss is A3's loss on a bigger patch.

## tests
Run in this order (also in the README):
1. `tests/test_shapes.py` - TubeletEmbedding (2,3,6,16,16)->(2,48,64); tube_masking keeps
   n_keep=6, mask (2,48) sums to 42; VideoMAE forward pred (2,48,96) (shape).
2. `tests/test_gradcheck.py` - float64 gradcheck of TubeletEmbedding w.r.t. the conv
   weight and of the encode->decode->loss pipeline w.r.t. it (gradcheck).
3. `tests/test_tube_masking.py` - the mask is identical across all temporal steps (tube
   property), exact keep count, unshuffle restores visible tubelets, deterministic under
   a seed (reference-value).
4. `tests/test_tubelet_equivalence.py` - the Conv3d embed equals unfold-into-tubes times
   the reshaped conv weight (reference-value).
5. `tests/test_overfit.py` - the VideoMAE memorizes one fixed toy-clip batch (fixed tube
   mask) to masked-tubelet MSE < 0.05 (overfit-one-batch).
6. `tests/test_forbidden_imports.py` - the top-level files, the solution, and the
   nanovision.transformer shim use no prebuilt attention/transformer module, fused SDPA,
   nn.LayerNorm, timm, transformers, or torchvision.models. Conv3d is allowed. Passes
   with the holes in place too.

## provided_boilerplate
`backbone.py` (identical at the top level and in solution/): the tubelet helpers
(tubeletify / untubeletify / per_tubelet_normalize, the video analog of A3's patchify
helpers) and the VideoViTEncoder (TubeletEmbedding + a learned spatiotemporal PE + the
A1 TransformerEncoder, joint space-time attention). In `video_mae.py` the mask-token
reassembly (`_append_mask_tokens`, the same contract as A3 Task 2) and the `VideoMAE`
module are provided. `config.py` holds all hyperparameters. The learner writes only the
three mechanism bodies. `nanovision.data.toy.video_batch` provides the toy clips
(independently moving Gaussian blobs).

## compute_notes
CPU, synthetic seeded clips, no download. N=48 tokens, encoder dim 64 depth 4, decoder
dim 48 depth 2, batch 8, Adam lr 2e-3, 800 steps with a fixed tube mask, reaching
masked-tubelet MSE ~7e-3 against 0.05. Per-tubelet normalization puts the untrained
baseline near MSE 1.0. Fits 12GB trivially; the gating signal is correctness, not scale.

## solution_notes
Token-order convention is load-bearing: TubeletEmbedding flattens temporal-outermost
(idx = t'*S' + s), and tubeletify, the spatiotemporal PE, and tube_masking all assume
it; a transpose silently breaks the tube test. tube_masking must NOT reuse A3's
argsort(noise) per-token body; it builds explicit [keep; drop] indices so ids_restore
inverts the tube ordering and the provided append/unshuffle still works. The mask is
sampled per clip (per-sample keep set). The overfit test holds the tube mask fixed by
re-seeding torch before each forward, so it is a clean memorization signal; final MSE
~7e-3 (threshold 0.05). The toy clips use independently moving blobs (not a single
translating shape) so the masked tubes are not recoverable by copying one visible
column, which would let a per-token mask succeed and defeat the point. mask_ratio is
0.875 (keep 2 of 16 spatial columns), below VideoMAE's 90-95% (the tube structure
quantizes the ratio to 1 - k/16); the README discloses this. The encoder uses joint
space-time attention, O(N^2); the README names TimeSformer/ViViT factorization as the
real-scale alternative. Conv3d is the taught mechanism and is allowed by the forbidden
scan.
