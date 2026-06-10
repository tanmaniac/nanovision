# Assignment layout restructure

Goal: make the code the student writes live at the top level of each assignment
directory, and make `nanovision/` expose those student implementations at stable
import paths by importing them from the assignment tree. The student's own work
becomes the library that later assignments build on. All student edits are scoped
to `assignments/aXX/`; `nanovision/` is a thin aggregation layer.

This replaces the previous `starter/` + `solution/` + `NANOVISION_IMPL` import
switch and the "canonical code in nanovision/, solution re-exports it" rule.

Non-goal: importing `nanovision` from outside this repo. This is a learning tool,
not a distributable package; everything runs from the repo root. No editable
install, no packaging accommodations for external use.

## Motivation (the two nits)

1. The canonical implementation of a shared symbol (for example `ConvNeXtBlock`)
   lived in `nanovision/primitives.py`, and `solution/primitives.py` only
   re-exported it. The code the student wrote in `starter/` was read by the test
   and then discarded - it never became the thing later assignments imported.
2. The editable code was buried in `starter/` while tests sat at the assignment
   root, and an env var switched which directory was on `sys.path`. That split
   existed for build-time verification, not for the learner.

## Target layout

```
nanovision/
  _student.py            # loader: repo-root on sys.path + NANOVISION_IMPL switch
  primitives.py          # link shim: re-exposes A0/A1/A2 symbols from assignments
  attention.py           # link shim: re-exposes A1 symbols
  transformer.py         # link shim: re-exposes A1 symbols
  geometry.py            # link shim: re-exposes A11.5a symbols
  trainer.py             # link shim: re-exposes A0 Trainer
  gradcheck.py           # native (provided infra, no student holes)
  determinism.py         # native
  data/                  # native (toy, images, nuscenes_mini are provided)
  viz.py                 # native

assignments/
  __init__.py            # makes the tree a package
  a02_vit/
    __init__.py
    convnext.py          # SHARED symbol the student writes (ConvNeXtBlock). Imported only via nanovision.
    vit.py               # assignment-local model the student writes. Imported bare by tests.
    config.py
    conftest.py          # local-code path switch (top-level vs solution/)
    tests/               # import shared symbols from nanovision; local symbols bare
    solution/
      convnext.py        # reference for convnext.py
      vit.py             # reference for vit.py
    viz.py  README.md  ASSIGNMENT.md
```

The student edits `assignments/a02_vit/convnext.py` and `assignments/a02_vit/vit.py`.
`nanovision.primitives.ConvNeXtBlock` is `assignments.a02_vit.convnext.ConvNeXtBlock`.
`solution/` is a read-only answer key, used only by the build-time switch.

## No editable install

`nanovision` is not pip-installed. The editable install's PEP 660 finder exposed
only the `nanovision` package and hid the repo root, which is why `assignments`
was not importable. Without the install, both `nanovision/` and `assignments/`
are plain packages at the repo root, importable once the repo root is on
`sys.path`. We put it there two ways:

- Tests: `pyproject.toml` gets `[tool.pytest.ini_options] pythonpath = ["."]`
  (pytest prepends the repo root for the session). `pip install -e .` is removed;
  `pyproject.toml` stays only as dependency documentation.
- Scripts (viz.py, training scripts): run as modules from the repo root, for
  example `python -m assignments.a02_vit.viz`. The Makefile viz target uses `-m`.

Confirm `import nanovision` from a clean environment after removing the install:
`pip uninstall nanovision`, then `python -m pytest` from the repo root.

## The loader (`nanovision/_student.py`)

```python
import importlib, os

def load(assignment: str, module: str):
    """Import a student-written module from an assignment dir.

    Default: assignments.<assignment>.<module> (the top-level file the student edits).
    NANOVISION_IMPL=solution: assignments.<assignment>.solution.<module> (reference).
    Repo root is on sys.path via pytest pythonpath (tests) or `-m` from root (scripts).
    """
    sub = "solution." if os.environ.get("NANOVISION_IMPL") == "solution" else ""
    return importlib.import_module(f"assignments.{assignment}.{sub}{module}")
```

Each link-shim module in `nanovision/` then does, for example in
`nanovision/attention.py`:

```python
from nanovision._student import load
_m = load("a01_transformer", "attention")
scaled_dot_product_attention = _m.scaled_dot_product_attention
MultiHeadAttention = _m.MultiHeadAttention
```

`nanovision/__init__.py` stays light (no eager submodule imports) so the shims load
lazily and no import cycle forms.

## Ownership map (nanovision symbol -> owning assignment file)

- `primitives`: `LayerNorm, gelu, MLP` -> a00_harness/primitives.py;
  `RMSNorm, SwiGLU` -> a01_transformer/primitives.py;
  `ConvNeXtBlock` -> a02_vit/convnext.py.
- `trainer.Trainer` -> a00_harness/trainer.py (the `step` hole; rest provided).
- `attention`: `scaled_dot_product_attention, MultiHeadAttention` -> a01_transformer/attention.py.
- `transformer`: `TransformerBlock, TransformerEncoder, TransformerDecoder,
  build_causal_mask, apply_rope, _RoPEAttention, SinusoidalPositionalEncoding,
  LearnedPositionalEncoding` -> a01_transformer/transformer.py.
- `geometry`: `project_points, unproject, make_transform, apply_transform,
  invert_transform, compose_transforms, CameraRig, BEVGrid, ipm_to_bev`
  -> a11_5a_camera_geometry_bev/geometry.py.
- Native (no student holes, stay in nanovision/): `gradcheck, determinism,
  data.toy, data.images, data.nuscenes_mini, viz`.

A module is owned by an assignment when it contains a hole. Pure provided infra
stays native in `nanovision/`.

## conftest for assignment-local code

Local model files (vit.py, mae.py, ...) are imported by tests with a bare name.
conftest puts the assignment top-level dir on `sys.path` by default, or `solution/`
when `NANOVISION_IMPL=solution`:

```python
import os, sys
from pathlib import Path
_here = Path(__file__).parent
_impl = _here / "solution" if os.environ.get("NANOVISION_IMPL") == "solution" else _here
if str(_impl) not in sys.path:
    sys.path.insert(0, str(_impl))
```

Rule to avoid dual module identity: a shared symbol's owning file (e.g.
a02_vit/convnext.py) is imported ONLY through `nanovision.*`, never bare. Local
model files are imported ONLY bare, never via nanovision. Tests follow this split.

## Per-assignment migration

For each of A0, A1, A2, A11.5a:
1. Move the real implementation out of `nanovision/<mod>.py` into the owning
   assignment's `solution/<file>.py` (the reference).
2. Create the holed top-level `assignments/aXX/<file>.py` (what the student edits),
   with full `__init__`/scaffolding and `raise NotImplementedError("AX Task N: ...")`
   bodies for the holes. Provided helper files (charlm.py, train_cifar.py, the SSL
   backbone) move to the top level too, identical in solution/.
3. Replace `nanovision/<mod>.py` with the link shim.
4. Delete the old `starter/` dir. Keep `solution/` as the reference copy of every
   top-level file the assignment ships.
5. Add `__init__.py` to the assignment dir.

A3 (in flight) adds no shared symbols, so its migration is only: move `starter/*`
to the top level, keep `solution/`, add `__init__.py`, update conftest. No shim.

Add `assignments/__init__.py` once.

## Verification (both modes must stay green)

Per assignment, from repo root:
- default (holed): `pytest assignments/aXX -q` -> task tests fail cleanly at the
  NotImplementedError; forbidden-imports passes.
- reference: `NANOVISION_IMPL=solution pytest assignments/aXX -q` -> all pass.
- cross-assignment: with `NANOVISION_IMPL=solution`, a later assignment's tests
  resolve `nanovision.*` to the reference and pass. With default, a later
  assignment depends on the owning assignment's top-level code being implemented.

`make verify-all` runs one process per assignment with `NANOVISION_IMPL=solution`
so the whole reference suite is checked green. `make test-all` runs default
(holed) and is expected to fail at the holes.

Also confirm `python -m pytest` and `python -m assignments.a02_vit.viz` both work
from the repo root with no editable install present.

## Docs to update after the code change

- ARCHITECTURE.md section 4 (environment): drop the `pip install -e .` step;
  state that the repo runs from its root with `pythonpath` for tests and `-m` for
  scripts, and that `pyproject.toml` is dependency documentation only.
- ARCHITECTURE.md section 2 (repo layout) and section 3 (shared-lib contract: the
  symbols are unchanged, but note they are now sourced from the assignment dirs via
  the loader) and section 5 (drop the starter/solution framing; the editable code is
  top-level, solution/ is the reference).
- TEMPLATE.md: the file-layout description.
- .claude/skills/lecture-notes: no change (it only writes README/ASSIGNMENT).
- BUILD_ORDER.md / BUILD_CHECKLIST.md: layout references if any.
- Memory: record the new layout convention.

## Sequencing

1. Let the A3 build finish on the current convention; verify and commit it as-is.
2. Do the restructure across A0, A1, A2, A11.5a, A3 in one pass (touches nanovision/,
   so it must not overlap a running A3 build/test).
3. Re-verify both modes green, update docs, commit.
