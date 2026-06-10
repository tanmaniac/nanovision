# A3.5 build plan - video / temporal modeling

Status: planned. Core, compact. Deps: A2 (ViT, patch embed), A3 (MAE masking and
loss). Bridges to A11.5c (BEVFormer temporal attention) and A12 (world models).

## Scope (from BUILD_ORDER)

Extend MAE to spatiotemporal video: a tubelet embedding (3D patch embed), tube
masking (the VideoMAE recipe), and a tiny video ViT that reconstructs masked
tubelets and overfits a single toy-video batch. The taught contrast is image MAE
(A3) vs video MAE: what changes when the input gains a time axis, and why tube
masking (mask a spatial location across all frames) is the right masking for video
instead of independent per-token masking.

## New shared symbol

`nanovision.transformer.TubeletEmbedding` (ARCHITECTURE.md section 3 already lists
it). Under the current layout this means:

- It is owned by `assignments/a03_5_video/tubelet.py` (the student writes it there).
- `nanovision/transformer.py` (currently a pure A1 shim) grows one more line that
  loads `TubeletEmbedding` from a035 via `load("a03_5_video", "tubelet")`, while the
  rest of the transformer symbols keep loading from a01. The shim sources different
  symbols from different owners; that is fine.
- ARCHITECTURE.md section 3 needs no signature change (already documented). The
  shim's `__all__` and the ownership note get TubeletEmbedding added.

Everything else in A3.5 is assignment-local (video MAE module, tube masking, the
provided video ViT backbone), imported by bare name like A3's mae/dino/backbone.

## Holes the student writes

Three, sized like A3's:

1. `TubeletEmbedding.forward` (`tubelet.py`, shared). A `Conv3d` with kernel and
   stride both `(t, p, p)` over a video `(B, C, T, H, W)`, flattened to tubelet
   tokens `(B, N, D)` with `N = (T/t) * (H/p) * (W/p)`. The temporal analog of A2's
   patch embed: a non-overlapping strided 3D conv is one shared linear map per
   space-time tubelet (ViViT, arXiv:2103.15691). Teaches that adding a time axis is
   one more strided dimension, not a new mechanism.
   FLATTEN ORDER (pin this, the tube masking depends on it): the Conv3d output is
   `(B, D, T', H', W')`; flatten temporal-outermost so token index
   `idx = t*S' + s` with `S' = H'*W'`, `s = h*W' + w`. State this convention once in
   the README so the student does not transpose it and silently break the tube test.
2. `tube_masking` (`video_mae.py`, local). Per sample, draw one spatial keep set
   `keep_s` of `n_keep_spatial = round((1-r)*S')` positions out of `S'`, and apply
   the SAME spatial keep/drop pattern to every temporal step (a "tube"). This is NOT
   A3's `argsort(noise)` per-token shuffle; the visible set is structured, so the
   student builds the indices explicitly so the provided append/unshuffle still
   works:
       ids_keep   = [t*S' + s for t in range(T') for s in keep_s]   # (B, n_keep)
       ids_drop   = [t*S' + s for t in range(T') for s in drop_s]
       ids_shuffle = cat(ids_keep, ids_drop, dim=1)                 # (B, N)
       ids_restore = argsort(ids_shuffle, dim=1)
   Gather the first `n_keep = T'*n_keep_spatial` tokens into `x_kept`, build `mask`
   `(B, N)` in original token order (1 = masked), return `(x_kept, mask,
   ids_restore)`. The tube is sampled per clip (per-sample `keep_s`), matching
   VideoMAE. Teaches why video needs tube masking: a per-token random mask leaks
   through time because an adjacent frame shows the same content, so the task
   collapses to copy-the-neighbor; masking a spatial column across all frames forces
   genuine spatiotemporal inference (VideoMAE's central point, arXiv:2203.12602).
3. `video_mae_loss` (`video_mae.py`, local). The same masked-patch MSE as A3's
   `mae_loss` with `p*p` replaced by `t*p*p`: per-tubelet MSE over targets
   `(B, N, t*p*p*C)`, averaged over masked tubelets only, against per-tubelet-
   normalized pixels. Implemented locally (not imported from a03, since local files
   do not cross assignments). The README should say plainly it is A3's loss on a
   bigger patch, so the learner sees it is the same mechanism, not a new loss.

The mask-token reassembly (append mask tokens, unshuffle by `ids_restore`) is
provided in the video MAE module wiring; because hole 2 builds `ids_restore` as the
argsort of the explicit `[keep; drop]` index list, A3's append/unshuffle machinery
works unchanged. The student does not re-implement A3 Task 2; the new conceptual
content is the tubelet embed and the structured tube mask.

## Provided (no holes)

- `backbone.py`: a tiny video ViT encoder = TubeletEmbedding + a learned
  spatiotemporal positional embedding + the A1 `TransformerEncoder`
  (`norm="layer", ffn="mlp", pos="none"`, classic ViT block), plus
  `tubeletify`/`untubeletify`/`per_tubelet_normalize` helpers (the video analog of
  A3's patchify helpers). Identical at the top level and in `solution/`.
- `VideoMAE` module (in `video_mae.py`): chains tubelet embed + PE, tube masking,
  encoder on visible tubelets, project to decoder dim, append mask tokens + decoder
  PE, light decoder, linear predictor to `(B, N, t*p*p*C)`, loss on masked tubelets.
  Provided plumbing; the student fills the three bodies it calls.
- `nanovision/data/toy.py`: add `video_batch(...)` (provided). A seeded batch of `B`
  clips on a noiseless background, exactly reconstructable but NOT a single global
  low-rank trajectory: 2-3 independently moving gaussian blobs with different
  per-clip velocities (optionally a per-frame intensity change). A single translating
  square is reconstructable from any visible tube by linear extrapolation, which a
  per-token mask would also solve and which would defeat the point of tube masking;
  multiple independent motions make the masked tubes genuinely require both space and
  time. Returns `(B, C, T, H, W)`.
- `config.py`: tiny sizes. Proposed: `T=6, t=2` (3 temporal steps), `img=16, p=4`
  (4x4 spatial grid, `S'=16`), `C=3`, so `N = 3*16 = 48` tubelets; enc dim 64 depth
  4, dec dim 48 depth 2, overfit batch 8, Adam lr 2e-3, ~800 steps. All CPU, fits
  12GB trivially. Masking: the tube structure quantizes the ratio to `1 - k/16`.
  VideoMAE uses 90-95% (arXiv:2203.12602); we use `mask_ratio = 0.875` (keep 2 of 16
  spatial columns -> `n_keep = 3*2 = 6` tubelets, 42 masked), high enough to be in
  the paper's spirit while a depth-4 toy still overfits one clip. The README discloses
  the paper's 90-95% explicitly and that 0.875 is the toy compromise.

## Tests (mirror A3's structure and the verify-before-train order)

1. `test_shapes.py`: TubeletEmbedding `(2,3,6,16,16) -> (2,48,64)`; tube_masking on
   `(2,48,D)` at r=0.875 keeps `n_keep = 3*2 = 6` tubelets (2 spatial cols x 3
   steps), `mask (2,48)` with 42 ones; VideoMAE forward gives pred `(2,48,96)`
   (`t*p*p*C = 2*4*4*3 = 96`).
2. `test_gradcheck.py`: float64 gradcheck of TubeletEmbedding w.r.t. its conv weight,
   and of the VideoMAE encode->decode->loss pipeline.
3. `test_tube_masking.py` (the centerpiece, reference-value): the kept/dropped
   spatial pattern is IDENTICAL across all temporal steps (the tube property: reshape
   `mask` to `(B, T', S')` and assert every temporal slice equals slice 0), the keep
   count is exactly `n_keep`, unshuffle restores order, deterministic under a fixed
   seed. This is the test that distinguishes tube masking from A3's per-token masking
   and is the assignment's conceptual gate.
4. `test_tubelet_equivalence.py` (reference-value): the Conv3d tubelet embed equals
   unfold-into-`(B, N, C*t*p*p)`-tubes times the conv weight reshaped to
   `(C*t*p*p, D)`, the 3D analog of A2's patch-equivalence test.
5. `test_overfit.py` (overfit-one-batch): VideoMAE memorizes one fixed toy-video
   batch (fixed tube mask) to masked-tubelet MSE < 0.05.
6. `test_forbidden_imports.py`: top-level files + solution use no prebuilt
   attention/transformer module, fused SDPA, nn.LayerNorm, timm, transformers, or
   any prebuilt video model in actual code. Conv3d is allowed (it is the mechanism).

All tests use synthetic seeded tensors, no download, CPU. Each test has a known
reachable pass condition (the overfit target is the toy clip, exactly representable),
following the lesson from A3: cheap tests with a reachable bar, no threshold hunting.

## Layout (new convention)

```
assignments/a03_5_video/
  __init__.py
  conftest.py            # standard: _here then _impl on sys.path
  config.py
  tubelet.py             # SHARED: TubeletEmbedding (student) -> nanovision.transformer
  video_mae.py           # LOCAL: tube_masking, video_mae_loss, VideoMAE module
  backbone.py            # provided video ViT + tubelet helpers (identical in solution/)
  viz.py                 # renders a masked-vs-reconstructed clip filmstrip to out/
  README.md              # lecture notes (verified citations, Mermaid)
  ASSIGNMENT.md          # builder contract
  tests/                 # the six tests above
  solution/
    __init__.py
    tubelet.py  video_mae.py  backbone.py   # filled references
```

## Shared-lib + docs touch-ups

- `nanovision/transformer.py`: add `_v = load("a03_5_video", "tubelet");
  TubeletEmbedding = _v.TubeletEmbedding` and add it to `__all__`. Keep the a01
  loads as they are. Guard: a035 must exist before A3.5; until then the shim must
  not eagerly import a035 (it will once A3.5 lands, which is now).
- `nanovision/data/toy.py`: add `video_batch`.
- ARCHITECTURE.md section 3: add TubeletEmbedding to the transformer ownership note.
- BUILD_CHECKLIST.md: tick A3.5 and the TubeletEmbedding shared-lib line when green.

## Disclosures the README must make (from expert review)

These are simplifications the toy makes; stating them is what keeps the learner from
internalizing the toy as the real recipe.

- Masking ratio: the standard video ratio is 90-95% (VideoMAE), far above image
  MAE's 75%, because temporal correlation makes video much more redundant. The toy
  uses 0.875 only so a depth-4 model can memorize one clip; say so.
- Attention: the assignment uses JOINT space-time self-attention (the A1 encoder over
  all `N = T'*S'` tubelet tokens), which is `O(N^2)`. At real video scale this is why
  video transformers were expensive, and why TimeSformer (arXiv:2102.05095) uses
  divided space-time attention and ViViT (arXiv:2103.15691) uses factorized
  encoder/attention variants. Joint attention is fine for `N=48`; name the factorized
  alternative and the cost it removes.
- Tubelet embed alternative: per-frame 2D patch embedding (`t=1`) is the simpler
  option; ViViT's "central frame initialisation" bootstraps the 3D conv from a
  pretrained 2D conv by placing the 2D kernel on the center temporal slice and zeroing
  the rest (so at init the tubelet embed equals the center frame's patch embed). One
  line in the README.
- The overfit test verifies the mechanism plumbing, not that the representation is
  good (same honesty caveat A3 makes for its linear probe). Tube masking's
  representation benefit shows only at scale; the toy demonstrates the mechanism.
- Target is normalized pixels (VideoMAE), not features; latent/feature targets are
  the V-JEPA/I-JEPA variant and belong in a stretch goal.

## Papers to cite (verify each arXiv id before citing, per the lecture-notes skill)

- VideoMAE (Tong et al., 2022, arXiv:2203.12602) - tube masking, the recipe built.
- VideoMAE V2 (Wang et al., 2023, arXiv:2303.16727) - dual masking, scaling.
- ViViT (Arnab et al., 2021, arXiv:2103.15691) - tubelet embedding, factorized
  space-time attention.
- TimeSformer (Bertasius et al., 2021, arXiv:2102.05095) - divided space-time
  attention, the attention-factorization contrast.
- MAE (He et al., 2022, arXiv:2111.06377) - the image parent (already in A3).

## Build order within A3.5

1. config.py + toy.video_batch + backbone.py (provided pieces, get shapes flowing).
2. solution/tubelet.py, solution/video_mae.py (reference); verify the VideoMAE
   overfits a toy batch at solution level before holing.
3. Hole the top-level tubelet.py / video_mae.py; wire nanovision.transformer shim.
4. Tests; confirm solution green and holed fails cleanly at the NotImplementedError.
5. viz.py (filmstrip), README lecture notes, ASSIGNMENT.md.
6. Run make verify A=a03_5_video green; update docs/checklist; commit.
