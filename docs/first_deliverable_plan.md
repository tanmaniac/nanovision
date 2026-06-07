# First deliverable plan: scaffold + A0, A1, A11.5a

This plan covers the first deliverable the spec asks for: the repo scaffold plus
three fully built and verified assignments (A0 harness, A1 transformer, A11.5a
camera geometry & BEV), after which we pause for you to calibrate house style and
difficulty before the remaining 16.

It exists so you can understand and sign off on the teaching method before I
write course code. Nothing below is built yet.

## Calibration decisions (from you)

- Reference solution is always present next to the starter; correctness rests on
  self-discipline, not gating. No git branch tricks, no hidden/encoded solution.
- Verification is pytest-primary, and each assignment also ships a required viz
  artifact (a script or notebook that renders the result for you to eyeball).
- Real-run target is a local RTX 4080 with 12GB. CPU is the fallback for the tiny
  gradcheck/shape tests only.

## How you will work an assignment (the interface)

The unit you interact with is one assignment directory. Your loop:

1. Read `assignments/aXX_name/README.md`. Section order is fixed by ARCHITECTURE.md
   §5: Motivation, Background (the math), What you'll implement, Tasks, How to
   verify, Compute notes, Stretch goals, Further reading.
2. Open `assignments/aXX_name/starter/` and fill the TODO holes. Each hole is a
   `raise NotImplementedError(...)` with a docstring stating the contract: input
   and output shapes, and the formula or algorithm to implement. Each hole maps
   1:1 to a numbered task in the README and 1:1 to a test.
3. Run the tests, in the order the README lists them. Tests are the contract and
   your main signal. The order encodes the workflow: shapes, then gradcheck, then
   reference values, then overfit-one-batch, then an optional real run where it
   fits 12GB.
4. Run the viz artifact and eyeball the rendered result (attention map, samples,
   BEV grid, reconstructed view, depth distribution).
5. If stuck, diff your `starter/` file against the adjacent `solution/` file.

### Commands

A top-level `Makefile` gives you a small, memorizable surface. `aXX` is the
assignment id, e.g. `a01_transformer`.

- `make test A=aXX` runs your starter against that assignment's tests. Red until
  you fill the holes; this is what you watch go green task by task.
- `make verify A=aXX` runs the reference solution against the same tests. Use it
  to confirm the tests are passable and the environment is sane.
- `make viz A=aXX` runs the assignment's viz artifact and writes images under
  `assignments/aXX/out/`.
- `make test-all` / `make verify-all` run across every built assignment (the
  whole-course green bar from BUILD_CHECKLIST.md).

Under the hood these are `pytest` invocations scoped to one assignment's
`tests/`. The starter-vs-solution switch is an environment variable
(`NANOVISION_IMPL=starter|solution`) that the test files read to import from the
right directory, so the exact same test file proves both. That is also how the
forbidden-imports grep test runs against `solution/`.

### A concrete walk-through of one task (the teaching method in miniature)

Take A1, Task 1, scaled dot-product attention, so you can see the full loop before
I commit to it. The handout's Background gives the formula
`softmax(Q Kᵀ / √d + M) V` and states every shape. The starter file is:

```python
def scaled_dot_product_attention(q, k, v, mask=None):
    """Compute scaled dot-product attention.

    Args:
      q: (B, H, Sq, Dh)
      k: (B, H, Sk, Dh)
      v: (B, H, Sk, Dh)
      mask: optional additive mask broadcastable to (B, H, Sq, Sk); 0 keep, -inf block.
    Returns:
      out:  (B, H, Sq, Dh)
      attn: (B, H, Sq, Sk)  softmax weights, returned for viz and tests.

    Implement softmax(q @ k^T / sqrt(Dh) + mask) @ v.
    Subtract the row max before exponentiating for numerical stability.
    Do NOT use torch.nn.functional.scaled_dot_product_attention.
    """
    raise NotImplementedError("A1 Task 1: implement scaled_dot_product_attention")
```

You write the body. Then, in order:

- `make test A=a01_transformer` first surfaces `test_shapes.py`: feeds table-driven
  shapes, asserts `out` and `attn` come back at `(B,H,Sq,Dh)` and `(B,H,Sq,Sk)`.
- `test_attention_gradcheck.py`: `nanovision.gradcheck.check_gradients` at float64
  on tiny dims, dropout off. Proves the forward is correct to gradient precision
  without you ever writing a backward pass.
- `test_attention_reference.py`: one-hot keys so attention must pick a single
  value; the expected output is exact to 1e-6, no tolerance fuzz.
- `test_causal.py`: asserts the upper triangle of `attn` is zero once the causal
  mask is in.
- `test_overfit_copy.py`: a decoder-only model overfits the copy task to
  loss < 1e-2 in < 500 steps on CPU.

Then `make viz A=a01_transformer` renders an attention heatmap so you see the
diagonal/causal structure rather than only trusting the assertion. That full
sequence (handout, fill one hole, watch ordered tests go green, eyeball the viz)
is the teaching method, repeated 19 times at rising difficulty. If this loop feels
right after A0/A1/A11.5a, the rest follows the same mold.

## Repo scaffold to build first

```
nanovision/
├── pyproject.toml          # installable package `nanovision`, deps + extras
├── environment.yml         # conda env, CUDA 12.x + PyTorch 2.x, Python 3.11
├── Makefile                # test / verify / viz / *-all targets above
├── conftest.py             # NANOVISION_IMPL switch, shared fixtures, seeding
├── nanovision/             # shared library; lands incrementally
│   ├── __init__.py
│   ├── primitives.py       # A0: LayerNorm, gelu, MLP
│   ├── trainer.py          # A0: Trainer.fit / overfit_one_batch + CSV logging
│   ├── gradcheck.py        # A0: check_gradients, assert_shapes
│   ├── determinism.py      # A0: seeding + deterministic flags
│   ├── viz.py              # A0: stubs, filled per topic
│   ├── data/
│   │   ├── toy.py          # A0: synthetic shapes, copy/sort, tiny char corpus
│   │   ├── images.py       # A0: CIFAR / MNIST wrappers
│   │   └── nuscenes_mini.py# A11.5a
│   ├── attention.py        # A1
│   ├── transformer.py      # A1
│   └── geometry.py         # A11.5a
└── assignments/
    ├── a00_harness/
    ├── a01_transformer/
    └── a115a_camera_geometry_bev/
```

Each assignment directory carries `README.md`, `ASSIGNMENT.md`, `starter/`,
`solution/`, `tests/`, and (per your viz choice) `notebooks/` or a `viz.py`.

Dependency rule from BUILD_ORDER.md still holds: build strictly A0, then A1, then
A11.5a, because the shared-library contract in ARCHITECTURE.md §3 is what later
assignments import. I will honor those signatures exactly.

## A0 — harness & primitives [Core]

Builds the substrate every later assignment imports.

- `nanovision.primitives`: `LayerNorm(dim, eps=1e-5)` from mean/var ops (no
  `nn.LayerNorm`), exact erf `gelu`, `MLP(dim, hidden, dropout, act)`.
- `nanovision.trainer.Trainer(model, optimizer, loss_fn, device, log_dir)` with
  `fit(train_loader, val_loader=None, max_steps=...)` and
  `overfit_one_batch(batch, steps=...)`, logging loss to console and a CSV the viz
  reads. No external tracker.
- `nanovision.gradcheck.check_gradients(module, example_inputs, eps=1e-6)` wrapping
  `torch.autograd.gradcheck` at double precision with a readable failure message;
  `assert_shapes(fn, cases)` for table-driven shape tests.
- `nanovision.determinism`: seeding and deterministic flags.
- `nanovision.data.toy` and `nanovision.data.images`; `nanovision.viz` stubs.

Learner tasks (the holes): LayerNorm forward, gelu, MLP forward, and the
`overfit_one_batch` step. Verify: gradcheck passes on LayerNorm and MLP; Trainer
overfits a toy linear-regression batch to ~0. Viz: the loss curve dropping to ~0.
Fits 12GB trivially; runs on CPU in seconds. External data: none.

## A1 — the transformer, from scratch [Core]

Its `ASSIGNMENT.md` already exists as the worked example; I build to it verbatim.

- `nanovision.attention`: `scaled_dot_product_attention` (returns weights too),
  `MultiHeadAttention(dim, n_heads, causal)` supporting self- and cross-attention.
- `nanovision.transformer`: `TransformerBlock` (pre-LN), `TransformerEncoder`,
  `TransformerDecoder`, `SinusoidalPositionalEncoding`, `LearnedPositionalEncoding`,
  `build_causal_mask`, and `apply_rope` (stretch only).

Tasks 1-6 exactly as the example lists them: SDPA, MHA, causal mask, sinusoidal
PE, the pre-LN block, then assemble and train a decoder-only char-LM plus a
copy/sort toy. Forbidden imports: `nn.MultiheadAttention`, `nn.Transformer*`,
`F.scaled_dot_product_attention`, enforced by grep over `solution/`. Verify in the
order shown above. Viz: attention heatmaps and the loss curve against the stated
unigram-entropy baseline. Fits 12GB; mostly CPU-runnable. External data: none.

## A11.5a — camera geometry & the BEV transform [Core]

The unfamiliar autonomous-driving plumbing, built early on purpose. Owns dataset
"step zero".

- `nanovision.geometry`: `project_points(pts_cam, K)`, `unproject(px, depth, K)`,
  `CameraRig` holding per-camera K and SE3 extrinsics with `world_to_cam`,
  `cam_to_world`, `world_to_pixel` for the 6-camera rig, and
  `ipm_to_bev(images, rig, bev_grid, ground_z=0.0)`.
- `nanovision.data.nuscenes_mini.NuScenesMini`: wraps `nuscenes-devkit`, exposes
  per sample the 6 camera images, per-camera K and extrinsics, lidar points, ego
  pose, and BEV map / box annotations where used; downsamples images (default
  ~400x224) to fit 12GB; degrades with a clear error if the dataset path is unset.

Tasks (the holes): pinhole project and unproject; the intrinsic/extrinsic chain in
`CameraRig`; ground-plane IPM into a BEV grid. Verify: lidar points projected into
all 6 cameras land on the right objects; multi-cam images warp into one BEV grid;
flat-ground breakage is visible and explained, not silently wrong. This is mostly
geometry, so tests lean on reference-value and a calibration-sanity check rather
than overfitting. Viz: the 6-camera lidar-overlay panel and the stitched BEV.
External data: nuScenes v1.0-mini (~4GB, account + license click-through), which
the README documents as the first step.

## Resolved spec questions

1. nuScenes-mini access: cold start. A11.5a's README treats account creation, the
   license click-through, the ~4GB `v1.0-mini` download, and `nuscenes-devkit`
   install as step zero. Every A11.5a test that needs the data skips cleanly (with
   a clear reason) when the dataset path is unset, so the rest of the suite stays
   green without it.
2. Viz artifact form: a runnable headless `viz.py` per assignment that writes
   images to `assignments/aXX/out/`, plus a notebook only where interactive
   probing of pretrained weights is the point (A2 DINOv2, A4 CLIP).
3. Git: initialize now. Scaffold and each assignment land as reviewable commits.
   Commit messages carry no `Co-Authored-By` trailer (your global rule).

## Proposed sequence after sign-off

1. Initialize git and lay down the scaffold (pyproject, environment.yml, Makefile,
   conftest, empty package modules, the three assignment dirs).
2. Build A0 end to end: shared modules, starter holes, solution, tests, README,
   ASSIGNMENT.md, viz. Confirm `make verify A=a00_harness` is green and
   `make test A=a00_harness` fails cleanly at each hole.
3. Build A1 to the existing worked ASSIGNMENT.md, same green/red confirmation.
4. Build A11.5a, with the dataset-absent skip behavior verified.
5. Pause. You work through A0/A1 yourself and tell me whether difficulty, hole
   sizing, and the test/viz rhythm are right before I produce the remaining 16.
