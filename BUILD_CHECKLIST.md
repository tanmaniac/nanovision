# BUILD_CHECKLIST.md - progress tracker for the builder

Check each box only when the assignment's `solution/` makes all of its `tests/`
pass AND its `README.md` + `ASSIGNMENT.md` + `viz.py` are complete. Build
top-to-bottom (see `BUILD_ORDER.md`). 22 built assignments.

## Shared library lands incrementally - verify these contracts as you go
- [x] `nanovision.primitives` (A0; RMSNorm/SwiGLU added in A1; ConvNeXt block in A2)
- [x] `nanovision.trainer` / `gradcheck` / `determinism` / `data` (A0)
- [x] `nanovision.attention` + `nanovision.transformer` (A1; RoPE/RMSNorm/SwiGLU/GQA)
- [x] `nanovision.transformer.TubeletEmbedding` (A3.5)
- [x] `nanovision.quantize` (A6.5)
- [x] `nanovision.vit` (A2 ViT exposed; `ViT.forward_features` patch grid added for A8)
- [x] `nanovision.volume` (A9: volume_render + ray generation; reused by A10, A11.5d)
- [x] `nanovision.geometry` + `nanovision.data.nuscenes_mini` (A11.5a)
- [x] `nanovision.geometry` pointmap/depth utils (A10.5)

## Phase 0 - Foundation
- [x] A0  harness & primitives        [Core]   deps: none
- [x] A1  transformer from scratch    [Core]   deps: A0   (LLaMA-style core)

## Phase 1 - Visual representations
- [x] A2  vision transformers         [Core]   deps: A0,A1   (+register tokens, ConvNeXt)
- [x] A3  self-supervised (MAE+DINO)  [Core]   deps: A1,A2   (+iBOT)
- [x] A3.5 video / temporal modeling  [Core]   deps: A2,A3   (video MAE, tube masking)
- [x] A4  CLIP & open-vocab           [Core]   deps: A1,A2   (SigLIP primary)

## Phase 2 - Generative
- [x] A5  diffusion (DDPM/DDIM)       [Core]   deps: A0,A1   (v-prediction)
- [x] A6  flow matching / rect. flow  [Core]   deps: A5      (+OT coupling)
- [x] A6.5 VQ tokenizer               [Core]   deps: A1,A2
- [x] A7  latent diffusion + DiT      [Core]   deps: A1,A5,A6,A6.5  (flow-matching DiT)

## Phase 3 - Multimodal & 3D
- [x] A8  VLM (LLaVA-style)           [Core]   deps: A1,A2,A4
- [x] A9  NeRF (prequel)              [Core]   deps: A0
- [x] A10 Gaussian splatting          [Core]   deps: A9
- [x] A10.5 geometry foundation models[Mixed]  deps: A1,A2,A9
- [x] A11 detection & segmentation    [Mixed]  deps: A1,A2

## Phase 3.5 - Autonomous-driving perception (nuScenes-mini)
- [x] A11.5a camera geometry & BEV    [Core]   deps: A0       (owns dataset "step zero")
- [ ] A11.5b lift-splat-shoot         [Core]   deps: A11.5a,A2
- [ ] A11.5c BEVFormer attention      [Core]   deps: A11.5a,A1,A11.5b,A3.5
- [ ] A11.5d 3D occupancy             [Core]   deps: A11.5a,A9,A11.5b/c  (render-supervised)
- [ ] A11.5e prediction → planning    [Mixed]  deps: A11.5b/c,A11.5d

## Phase 4 - Action & dynamics
- [ ] A12 world models (RSSM/Dreamer) [Core]   deps: A0,A1,A3.5
- [ ] A13 VLA capstone                [Core]   deps: A5/A6,A8,(A12)

## Reading-only notes (Markdown, not built; in notes/)
- [ ] note: video generation (flow matching in spatiotemporal VAE latent)
- [ ] note: VLA data engines / scaling (Open X-Embodiment, DROID, FAST)
- [ ] note: efficient attention (FlashAttention / IO-aware)
- [ ] note: state-space backbones (Vision Mamba / SSMs)
- [ ] note: MM-DiT and REPA training techniques
- [ ] note: video world models (Genie 2, DIAMOND, DreamerV4)

## Definition of done (whole course)
- [ ] `pytest` green across all assignments' solutions, run from the repo root with
      no install (`make verify-all`)
- [ ] every assignment's top-level files fail cleanly at each TODO with a contract message
- [ ] every README has the fixed section order (ARCHITECTURE.md §5)
- [ ] every assignment has a runnable `viz.py`
- [ ] all `forbidden_imports` grep-tests pass
- [ ] cross-assignment imports resolve against the shared-lib contract
