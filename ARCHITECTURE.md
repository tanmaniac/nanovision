# nanovision - Architecture & Course Specification

A from-scratch, implement-the-mechanism course covering computer vision from the
transformer era through 2026 (ViT, SSL, CLIP, diffusion, flow matching, latent
diffusion/DiT, VLMs, NeRF, Gaussian splatting, modern detection/segmentation,
autonomous-driving BEV/occupancy, world models, and VLA).

This document is the authoritative spec for the build. It is written to be handed
to a Claude Code instance (or any engineer) to implement. Read this file first,
then `BUILD_ORDER.md`, then the per-assignment specs in `assignments/`.

---

## 1. Learner profile & design philosophy

The learner is an experienced perception/robotics engineer (stereo VO, SLAM,
multi-sensor EKF fusion, multi-camera SfM, BEV theory) who last implemented vision
systems around the EfficientDet/DETR era (~2020) and wants to internalize what
happened since, by **building the core mechanisms**.

Design principles, in priority order:

1. **Implement the mechanism, not the factory.** For each topic, the learner writes
   the architectural primitive from scratch (attention, the diffusion noise
   schedule, the LSS lift, the splatting rasterizer forward, the BEV projection),
   on a *tiny problem where correctness is verifiable*. Training at real scale is
   never the goal.
2. **Verify before you train.** Every core mechanism ships with a test that proves
   the forward pass is correct (shape tests, `torch.autograd.gradcheck` on a
   double-precision tiny instance, and/or a reference-value comparison) BEFORE any
   training loop is run. The canonical correctness signal is **overfit a single
   batch to ~zero loss**.
3. **PyTorch autograd throughout.** No hand-derived backward passes. The learner
   implements `forward()` from low-level ops (`nn.Linear`, tensor ops) and relies
   on autograd. They are explicitly forbidden from importing the high-level
   shortcut for the thing being taught (see "Anti-cheat imports" per assignment,
   e.g. no `nn.MultiheadAttention` in the attention assignment, no `diffusers`).
4. **Compute ceiling: a single 12GB laptop GPU (RTX 4080).** Every "train it for
   real" target must fit in 12GB with small models, small images, small batches.
   Where a topic cannot fit (video generation, real BEV training), the assignment
   says so explicitly and targets correctness + single-/few-batch overfitting
   instead, so a flat loss curve is not mistaken for a bug.
5. **Each assignment is self-contained but reuses earlier primitives.** The
   transformer block built in A1 is imported by A2, A8, A11.5c, A13, etc. No
   re-implementation of solved pieces; explicit imports across assignments.

---

## 2. Repository layout

```
nanovision/
├── README.md                      # course overview, how to use (no install step)
├── pyproject.toml                 # dependency list + pytest config; NOT installed
├── environment.yml                # conda env (CUDA 12.x, PyTorch)
├── ARCHITECTURE.md                # this file
├── BUILD_ORDER.md                 # dependency-ordered build plan
├── nanovision/                    # SHARED library: shims + provided infra
│   ├── __init__.py                # light; does not eagerly import submodules
│   ├── _student.py                # loader: imports student symbols from assignments/
│   ├── primitives.py              # SHIM -> a00 (LayerNorm/gelu/MLP), a01 (RMSNorm/SwiGLU), a02 (ConvNeXtBlock)
│   ├── attention.py               # SHIM -> a01 (SDPA / MHSA / cross-attn, reused widely)
│   ├── transformer.py             # SHIM -> a01 (blocks, enc/dec, RoPE/RMSNorm/SwiGLU)
│   ├── quantize.py                # SHIM -> a06.5 (VQ codebook + straight-through)
│   ├── trainer.py                 # SHIM -> a00 (generic training loop)
│   ├── gradcheck.py               # provided: gradcheck + shape-test helpers (A0)
│   ├── determinism.py             # provided: seeding, deterministic flags (A0)
│   ├── data/                      # toy datasets + nuScenes-mini loader
│   │   ├── toy.py                 # synthetic shapes, copy/sort, tiny char LM (A0)
│   │   ├── images.py              # CIFAR / MNIST wrappers (A0)
│   │   └── nuscenes_mini.py       # nuScenes-mini loader + calib utils (A11.5a)
│   ├── geometry.py                # SHIM -> a11.5a (pinhole proj, SE3, CameraRig, ipm_to_bev)
│   └── viz.py                     # provided: attention maps, BEV grids, splat renders (A0)
├── assignments/                   # 22 built assignments; see BUILD_ORDER.md
│   ├── a00_harness/               # a09_nerf/
│   ├── a01_transformer/           # a10_gaussian_splatting/
│   ├── a02_vit/                   # a105_geometry_foundation/
│   ├── a03_ssl/                   # a11_detection_segmentation/
│   ├── a035_video/                # a115a_camera_geometry_bev/
│   ├── a04_clip/                  # a115b_lift_splat_shoot/
│   ├── a05_diffusion/             # a115c_bevformer/
│   ├── a06_flow_matching/         # a115d_occupancy/
│   ├── a065_vq_tokenizer/         # a115e_pred_planning/
│   ├── a07_latent_dit/            # a12_world_models/
│   ├── a08_vlm/                   # a13_vla/
│   └── ...                        #
├── notes/                         # reading-only markdown notes (not built)
└── tests/                         # cross-cutting / integration tests
```

### Per-assignment directory structure (uniform)

```
assignments/aXX_name/
├── README.md          # the "handout": background, tasks, deliverables, hints
├── __init__.py        # makes the dir importable (the nanovision shims load from it)
├── conftest.py        # puts the assignment dir (or solution/) on sys.path per NANOVISION_IMPL
├── <module>.py        # the code the STUDENT edits, with `raise NotImplementedError` holes
│   └── ...            #   (primitives.py, attention.py, vit.py, mae.py, ...)
├── config.py          # per-assignment hyperparameters (where applicable)
├── solution/          # reference implementation (always visible; see below)
│   ├── __init__.py
│   └── <module>.py    # the filled-in answer key for each top-level <module>.py
├── tests/             # pytest: shape tests, gradcheck, overfit-one-batch
├── viz.py             # required: renders the result to out/ (eyeball check)
├── notebooks/         # pretrained-weights probes only (where applicable)
└── ASSIGNMENT.md      # machine-readable spec for the builder (see TEMPLATE.md)
```

**The student edits the top-level files; solution/ is the answer key.** The code
the learner writes lives at the top level of the assignment dir (`primitives.py`,
`vit.py`, ...), with the pedagogically essential lines replaced by
`raise NotImplementedError("...")` plus a docstring describing the contract
(input/output shapes, what to implement, which formula). `solution/` holds the
filled-in copy of each of those files. It sits in plain sight: there is no gating;
correctness rests on the learner's self-discipline, and the learner is expected to
read `solution/` when stuck.

**The student's work becomes the shared library.** Symbols that later assignments
reuse (LayerNorm, attention, the Trainer, geometry, ...) are NOT written in the
`nanovision/` package. They are written by the student in the owning assignment, and
`nanovision/<module>.py` is a thin shim that imports them back from there through
`nanovision/_student.py`. So once the learner implements `attention.py` in A1, every
later assignment that does `from nanovision.attention import MultiHeadAttention` gets
the learner's own code. See section 3 for the ownership map.

**The impl switch.** Tests never hard-code which copy to import. The loader and each
conftest read the `NANOVISION_IMPL` environment variable: unset (the default) means
the student's top-level files; `solution` means the `solution/` answer key. `make
test A=aXX` runs the student's code (expected to fail cleanly until filled); `make
verify A=aXX` runs `solution/` (must be green); `make viz A=aXX` runs `viz.py`
against `solution/`. A shared symbol's owning file is imported ONLY through
`nanovision.*` (never by bare name); an assignment-local file (vit.py, mae.py) is
imported ONLY by bare name (never through nanovision). That split keeps each file a
single module identity.

**Local vs shared.** A file is shared (owned, exposed through `nanovision/`) when a
later assignment imports it: primitives, attention, transformer, the Trainer,
geometry. A file is local when it is glue for one assignment only (the ViT/DiT/UNet
model definitions, training scripts, the LSS/BEVFormer heads, the SSL backbone); it
lives at the top level and in `solution/`, imported by bare name, and never touches
`nanovision/`.

---

## 3. Shared library contract (`nanovision/`)

The shared package is built incrementally: A0 establishes it, later assignments add
modules. Every public symbol below is a stable import path that downstream
assignments depend on, and the builder MUST honor these signatures so
cross-assignment imports do not break.

The symbols are not defined in `nanovision/` itself. Each shared module there is a
shim that loads its symbols from the assignment that owns them (via
`nanovision/_student.py`, keyed on `NANOVISION_IMPL`). Ownership: `primitives`
splits across A0 (`LayerNorm`, `gelu`, `MLP`), A1 (`RMSNorm`, `SwiGLU`), and A2
(`ConvNeXtBlock`); `attention`/`transformer` are A1; `trainer` is A0; `geometry` is
A11.5a; `quantize` is A6.5. An owning file imports its own dependencies through
`nanovision.*` too (or, to avoid a self-cycle, through the loader directly), so the
`NANOVISION_IMPL` switch stays consistent down the chain.

### `nanovision.primitives` (A0; ConvNeXt block A2)
- `class LayerNorm(nn.Module)` - `(dim, eps=1e-5)`, implements normalization from
  mean/var ops (no `nn.LayerNorm`).
- `class RMSNorm(nn.Module)` - `(dim, eps=1e-6)`, the LLaMA-style norm (A1).
- `def gelu(x: Tensor) -> Tensor` - exact (erf) GELU.
- `class MLP(nn.Module)` - `(dim, hidden, dropout=0.0, act=gelu)`, two linear layers.
- `class SwiGLU(nn.Module)` - `(dim, hidden)`, the gated SiLU FFN used in A1's
  LLaMA-style block and downstream (A7 DiT, etc.).
- `class ConvNeXtBlock(nn.Module)` - `(dim)`, depthwise conv + pointwise + LN
  (built in A2 for the conv-vs-transformer comparison).

### `nanovision.attention` (A1)
- `def scaled_dot_product_attention(q, k, v, mask=None) -> (out, attn)` - from
  scratch; returns attention weights too (for viz/tests). No `F.sdpa`.
- `class MultiHeadAttention(nn.Module)` - `(dim, n_heads, causal=False,
  n_kv_heads=None)`; supports self- and cross-attention via optional `kv` argument
  in `forward(x, kv=None, mask=None)`; `n_kv_heads < n_heads` gives GQA/MQA.

### `nanovision.transformer` (A1; tubelet embed A3.5)
- `class TransformerBlock(nn.Module)` - pre-norm by default; `(dim, n_heads,
  mlp_ratio, causal, cross_attn=False, norm="rms", ffn="swiglu", pos="rope")`.
  The LLaMA-style configuration (RMSNorm + RoPE + SwiGLU) is the default/core;
  LayerNorm + absolute PE + GELU-MLP are selectable for the historical contrast.
- `class TransformerEncoder` / `class TransformerDecoder` - stacks.
- `def build_causal_mask(seq_len) -> Tensor` - additive −inf upper-triangular mask.
- `class SinusoidalPositionalEncoding`, `class LearnedPositionalEncoding` -
  absolute schemes (historical contrast).
- `def apply_rope(q, k, ...)` - rotary position embedding; core in A1.
- `class TubeletEmbedding(nn.Module)` - spatiotemporal patch embed for video (A3.5).

### `nanovision.trainer` (A0)
- `class Trainer` - `(model, optimizer, loss_fn, device, log_dir)`, methods
  `fit(train_loader, val_loader=None, max_steps=...)`, `overfit_one_batch(batch,
  steps=...)`. Logs loss to console + a CSV the notebooks can plot. No external
  experiment-tracker dependency.

### `nanovision.gradcheck` (A0)
- `def check_gradients(module, example_inputs, eps=1e-6) -> bool` - wraps
  `torch.autograd.gradcheck` with double precision and a readable failure message.
- `def assert_shapes(fn, cases)` - table-driven shape testing helper.

### `nanovision.geometry` (A11.5a; pointmap utils A10.5)
- `def project_points(pts_cam, K) -> px` - pinhole projection.
- `def unproject(px, depth, K) -> pts_cam` - inverse.
- SE(3) primitives: `make_transform(R, t)`, `apply_transform(T, pts)`,
  `invert_transform(T)`, `compose_transforms(*Ts)` - reused verbatim by every AV
  assignment.
- `class CameraRig` - holds per-camera K (intrinsics) and T (extrinsics, SE3),
  with `world_to_cam`, `cam_to_world`, `world_to_pixel` for a multi-cam set.
- `def ipm_to_bev(images, rig, bev_grid, ground_z=0.0)` - inverse perspective
  mapping onto a ground plane.
- `def reproject_pointmap(pmap, K, T) -> px` and pointmap helpers (A10.5).

### `nanovision.quantize` (A6.5)
- `class VectorQuantizer(nn.Module)` - `(codebook_size, dim, beta=0.25)`; forward
  returns `(quantized, indices, vq_loss)` with the straight-through estimator and
  the commitment loss. No prebuilt VQ layer.

### `nanovision.data.nuscenes_mini` (A11.5a)
- `class NuScenesMini(Dataset)` - wraps `nuscenes-devkit`, exposes per-sample:
  6 camera images, per-camera K & extrinsics, lidar points, ego pose, and
  (where used) BEV-rasterized map / box annotations. Downsamples images to a
  configurable size (default ~400×224) to fit 12GB.

> Builder note: the nuScenes loader is the single biggest external dependency.
> A11.5a's README documents the account/license click-through and the
> `v1.0-mini` (~4GB) download as "step zero". The loader must degrade gracefully
> (clear error message) if the dataset path is unset.

---

## 4. Environment & dependencies

- **No install step.** `nanovision` is not pip-installed; the repo runs from its
  root. pytest puts the root on `sys.path` via `pythonpath = ["."]` in
  `pyproject.toml`, and scripts run as modules from the root (`python -m
  assignments.aXX.viz`). `pyproject.toml` documents the dependency set only.
  Importing `nanovision` from outside the repo is a non-goal.
- Python 3.11, PyTorch 2.x with CUDA 12.x (must run on a single 12GB GPU and also
  on CPU for the tiny gradcheck tests).
- Core deps: `torch`, `torchvision`, `numpy`, `einops` (allowed - it's notation,
  not a shortcut for the taught mechanism), `matplotlib`, `pytest`, `tqdm`.
- Assignment-specific (declared as extras in `pyproject.toml`):
  - `a04_clip`, `a08_vlm`: `open_clip_torch` and/or `transformers` (ONLY for
    loading pretrained weights in the probe notebooks - never for the from-scratch
    parts).
  - `a02_vit`, `a03_ssl`, `a035_video`: `timm` (pretrained DINOv2 / video probes).
  - `a105_geometry_foundation`: `transformers` / model-hub access for the
    DepthAnything / Marigold / VGGT survey probes (probe notebooks only).
  - `a115*`: `nuscenes-devkit`, `pyquaternion`, `shapely`.
- **Anti-cheat principle:** pretrained-weight libraries are import-allowed ONLY in
  clearly-marked probe notebooks, NEVER in assignment mechanism code (the top-level
  files or `solution/`). Each `ASSIGNMENT.md` lists explicit `forbidden_imports` for
  its mechanism code, and a test greps both the student top-level files and the
  solution to enforce it.

---

## 5. House style for assignments (applies to every aXX)

- **README.md is comprehensive lecture notes, not a terse handout.** It is the
  primary thing the learner reads to understand the topic, written for an
  experienced engineer returning to the field. Fixed section order: Motivation →
  Background → What you'll implement → Tasks → How to verify → Compute notes →
  Stretch goals → Further reading. Depth requirements:
  - **Motivation** must teach, not gesture. Explain the pre-existing landscape and
    its limitation, what the originating paper(s) changed and *why that was
    significant at the time* (the specific problem it solved, the result that made
    people notice), the core technical idea in plain terms, and concretely how this
    mechanism feeds later assignments (name them and say what they reuse). Cite the
    key papers inline with links (arXiv abs URLs) at the point they are relevant -
    do not save all citations for Further reading. Several substantial paragraphs,
    not three sentences. Do not write filler like "this is the foundation" or "X
    builds on it" without saying what specifically and why it matters.
  - **Background** is the concise math: the key equations the learner implements,
    every shape stated, derivations only where they aid implementation.
  - **How to verify** lists the test commands in run order (shapes → gradcheck →
    reference-value → overfit-one-batch → optional real run).
  - **Compute notes** state what fits in 12GB, default sizes, expected runtime, and
    what a healthy loss curve looks like (so a flat/slow curve is not misread).
  - **Further reading** is 3-6 papers with one-line annotations and links.
  - **Figures and diagrams** carry the structural explanation. Use inline Mermaid
    blocks (```mermaid) for architecture, data flow, and tensor-shape diagrams
    (GitHub renders them natively, no binary assets); link a specific figure in the
    originating paper (ar5iv HTML `https://ar5iv.org/abs/<id>` or the arXiv page,
    verified to resolve) when a published figure is the clearest reference; and
    commit a generated PNG under the assignment's `assets/` only for quantitative
    plots that need real numbers (noise schedules, attention maps), produced by a
    small reproducible script. Prefer Mermaid for anything structural. Do not
    hotlink or invent figures you have not confirmed exist.
- **README.md vs ASSIGNMENT.md are different documents with different jobs.** The
  README is the learner-facing lecture notes above. ASSIGNMENT.md is the concise,
  machine-readable builder contract: the YAML frontmatter, `what_you_implement`
  bullets, the per-task contracts (file, symbol, exact in/out shapes, the
  formula/algorithm, the 1:1 test), `provided_boilerplate`, terse `compute_notes`,
  and builder-only `solution_notes` (seeds, reference values and how they were
  obtained, numerical gotchas). ASSIGNMENT.md must NOT re-narrate the README's
  prose motivation/background; keep its `motivation`/`background` to a few lines or
  a pointer to the README. The README explains; ASSIGNMENT.md specifies.
- **Difficulty calibration:** each TODO is sized so the learner writes the
  *interesting* lines (the actual mechanism) and is given the boilerplate
  (data loading, arg parsing, the training-loop skeleton via `nanovision.Trainer`).
  Aim: 30–120 minutes of focused work per core TODO, not days of plumbing.
- **Tests are the contract.** Every task has a corresponding test. Test order
  encodes the intended workflow: a learner runs shape tests first, then gradcheck,
  then overfit-one-batch, then (optionally) a real run.
- **No silent magic numbers.** Hyperparameters live in a small `config.py` or
  YAML per assignment with comments explaining each choice and its 12GB
  implications.

---

## 6. Course structure (22 assignments, 6 modules)

See `BUILD_ORDER.md` for the dependency-ordered build plan and per-assignment
one-line scope, and `docs/curriculum_review.md` for the validation that shaped it.
The modules:

- **Module 0 - Foundations:** A0 harness, A1 transformer-from-scratch
  (LLaMA-style: RMSNorm/RoPE/SwiGLU core).
- **Module 1 - Visual representations:** A2 ViT (+ register tokens, ConvNeXt),
  A3 SSL (MAE+DINO+iBOT), A3.5 video/temporal, A4 CLIP (SigLIP).
- **Module 2 - Generative:** A5 diffusion (DDPM/DDIM, v-pred), A6 flow matching/
  rectified flow, A6.5 VQ tokenizer, A7 latent diffusion + flow-matching DiT.
- **Module 3 - Multimodal & 3D:** A8 VLM, A9 NeRF, A10 Gaussian splatting,
  A10.5 geometry foundation models, A11 detection/segmentation.
- **Module 3.5 - Autonomous-driving perception:** A11.5a camera geometry & BEV,
  A11.5b Lift-Splat-Shoot, A11.5c BEVFormer attention, A11.5d 3D occupancy,
  A11.5e prediction → planning.
- **Module 4 - Action & dynamics:** A12 world models (RSSM/Dreamer),
  A13 VLA capstone.

Reading-only notes (not built, in `notes/`): video generation; VLA data engines/
scaling; efficient attention (FlashAttention); state-space backbones (Vision
Mamba); MM-DiT/REPA; video world models.

Core vs. survey is specified per assignment in `BUILD_ORDER.md`.
