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
- [x] `nanovision.lift_splat` (A11.5b: depth lift + frustum + cumsum splat; reused by A11.5d)
- [x] `nanovision.bevformer` (A11.5c: SCA + TSA + dense BEV grid; reused by A11.5d/e)

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
- [x] A11.5b lift-splat-shoot         [Core]   deps: A11.5a,A2
- [x] A11.5c BEVFormer attention      [Core]   deps: A11.5a,A1,A11.5b,A3.5
- [x] A11.5d 3D occupancy             [Core]   deps: A11.5a,A9,A11.5b/c  (render-supervised)
- [x] A11.5e prediction → planning    [Mixed]  deps: A11.5b/c,A11.5d  (leaf, no shim)

## Phase 4 - Action & dynamics
- [x] A12 world models (DreamerV3)    [Core]   deps: A0,A1,A3.5  (cartpole-from-pixels, dyn-backprop actor)
- [x] A13 VLA capstone                [Core]   deps: A5/A6,A8,(A12)  (flow-matching action head, leaf)

## Phase 5 - Classical SLAM / localization (C++; plan: claude_notes/a14_classical_slam_plan.md)
- [x] A14.0 Lie groups SO(3)/SE(3)    [Core]   deps: A11.5a (conventions)  (C++/Eigen/pybind11; Rerun viz; 19 tests)
- [ ] A14.1 KF / EKF / UKF            [Core]   deps: A14.0
- [ ] A14.2 EKF-SLAM                  [Core]   deps: A14.1
- [ ] A14.3 multi-view geometry       [Core]   deps: A14.0,A11.5a
- [ ] A14.4 ICP registration          [Core]   deps: A14.0  (Open3D oracle)
- [ ] A14.5 factor graph / BA         [Core]   deps: A14.0,A14.3  (g2o/GTSAM oracle)
- [ ] note: IMU pre-integration / VIO, calibration, place recognition, classical<->learned bridge

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
