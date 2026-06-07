# nanovision — spec package (read me first)

This package is a **specification to be built**, not the built course. Hand it to
a Claude Code instance (or an engineer) to implement the `nanovision` repository.

## What this is

A 22-assignment, implement-the-mechanism course taking an experienced
perception/robotics engineer from the transformer era through 2026 CV: ViT, SSL,
video/temporal modeling, CLIP, diffusion, flow matching, VQ tokenizers, latent
diffusion/DiT, VLMs, NeRF, Gaussian splatting, geometry foundation models, modern
detection/segmentation, autonomous-driving BEV/occupancy/prediction, world models,
and VLA. The subject list was validated against 2026 sources (see
`docs/curriculum_review.md`); the build plan is `docs/revised_curriculum.md`.

## Read in this order

1. **ARCHITECTURE.md** — design philosophy, repo layout, the shared-library
   contract (stable import paths every assignment depends on), environment, and
   house style. This is authoritative; when in doubt it wins.
2. **BUILD_ORDER.md** — the dependency-ordered build plan, what each assignment
   implements, depends on, and its Core/Survey/Mixed designation. Build strictly
   in this order.
3. **TEMPLATE.md** — the per-assignment `ASSIGNMENT.md` format you fill in.
4. **EXAMPLE_a01_transformer.md** — a fully worked `ASSIGNMENT.md` showing the
   required level of detail. Match this depth for every assignment.

## The build contract (non-negotiable)

- **PyTorch autograd throughout.** No hand-derived backward passes. Learner
  implements `forward()`; correctness is proven by `gradcheck`, shape tests,
  reference values, and overfit-one-batch.
- **Honor the shared-library signatures** in ARCHITECTURE.md §3 exactly, or
  cross-assignment imports break.
- **Enforce `forbidden_imports`** per assignment with a grep test over
  `solution/` — the whole point is the learner builds the mechanism, not imports
  it.
- **12GB ceiling.** Every "real run" must fit an RTX 4080. Where it can't, the
  assignment is overfit-only and says so, with the reason, so a flat loss curve
  isn't misread as a bug.
- **Each assignment ships:** `README.md` (handout), `starter/` (TODO holes),
  `solution/` (passes all tests), `tests/` (one test per task, ordered), and any
  `notebooks/` (pretrained-weight probes only — never mechanism code).
- **`solution/` must make `tests/` pass; `starter/` must fail cleanly** at each
  unfilled TODO with a contract-describing message.

## Suggested first deliverable

Per BUILD_ORDER.md cadence: build and fully verify **A0, A1, and A11.5a** first
(the harness, the from-scratch transformer, and the nuScenes camera-geometry/BEV
foundation), then pause for the learner to calibrate house style and difficulty
before producing the remaining 16.

## What still needs authoring

Only A1's `ASSIGNMENT.md` is written (as the worked example). The builder should
author each remaining `ASSIGNMENT.md` from TEMPLATE.md at the EXAMPLE's depth
*before* writing that assignment's code, and may surface questions back to the
learner where a spec field is genuinely underdetermined.
