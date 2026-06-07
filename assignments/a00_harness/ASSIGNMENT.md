# assignments/a00_harness/ASSIGNMENT.md

```yaml
id: a00_harness
title: Harness & primitives
module: 0
type: Core
estimated_learner_hours: 2
depends_on: []
builds_into_shared_lib:
  - nanovision.primitives.gelu
  - nanovision.primitives.LayerNorm
  - nanovision.primitives.MLP
  - nanovision.trainer.Trainer
  - nanovision.gradcheck.check_gradients
  - nanovision.gradcheck.assert_shapes
  - nanovision.determinism.set_seed
  - nanovision.data.toy            # linreg, copy/sort, tiny char corpus
  - nanovision.data.images         # MNIST / CIFAR wrappers
  - nanovision.viz.plot_loss_curve
forbidden_imports:
  - torch.nn.LayerNorm
  - torch.nn.GELU
fits_12gb: true
external_data: none
```

## motivation
Every later assignment imports this harness: the primitives (LayerNorm, gelu,
MLP) that transformer blocks are built from, the `Trainer` that runs the
overfit-one-batch correctness check, and the `check_gradients` helper that proves
a hand-written `forward()` is correct without a hand-written backward. Building it
first makes the course's verification workflow concrete on the simplest possible
mechanisms before any of the interesting ones arrive.

## background
LayerNorm over the last dimension:

  y = (x − mean) / sqrt(var + eps) · weight + bias

with mean and biased var taken over the last axis. Exact GELU:

  GELU(x) = x · 0.5 · (1 + erf(x / √2))

The optimization step is the standard rhythm: zero grads, forward, loss,
backward, optimizer step. `overfit_one_batch` repeats that step on one fixed batch
and expects the loss to fall to ~0 for a model with enough capacity — the
canonical signal that a forward pass and its gradients are wired correctly.

## what_you_implement
- `gelu` (exact erf form).
- `LayerNorm.forward` from mean/var ops.
- `MLP.forward` (Linear → act → dropout → Linear).
- `Trainer.step` (the one optimization step).

## tasks
- **Task 1 — gelu** (file: `starter/primitives.py`, symbol: `gelu`): return the
  exact erf GELU of `x`; same shape in and out. Teaches: the activation as a smooth
  gate, and that `gelu(0) == 0`.
- **Task 2 — LayerNorm.forward** (file: `starter/primitives.py`, symbol:
  `LayerNorm`): normalize over the last dim, then scale+shift by the learned
  `weight`/`bias`. Teaches: normalization built from primitive ops, not `nn.LayerNorm`.
- **Task 3 — MLP.forward** (file: `starter/primitives.py`, symbol: `MLP`): the
  two-layer feed-forward used inside every transformer block. Teaches: the block's
  position-wise FFN.
- **Task 4 — Trainer.step** (file: `starter/trainer.py`, symbol: `Trainer.step`):
  one optimization step returning the scalar loss. Teaches: the zero-grad / forward
  / backward / step rhythm that `overfit_one_batch` and `fit` drive.

## tests
Run in this order (see README "How to verify"):
1. `tests/test_shapes.py` — output shapes for Tasks 1-3, plus the LayerNorm
   output statistics (zero mean, unit std over the last dim). (shape)
2. `tests/test_gradcheck.py` — `check_gradients` at float64 on LayerNorm and MLP,
   and a smoke test of the `assert_shapes` helper. (gradcheck)
3. `tests/test_overfit.py` — `Trainer.overfit_one_batch` drives a noiseless
   linear-regression batch to MSE < 1e-4 in 500 steps. (overfit-one-batch)

## provided_boilerplate
`nanovision.gradcheck` (check_gradients, assert_shapes), `nanovision.determinism`,
`nanovision.data.toy` (linreg/copy/sort/char), `nanovision.data.images`, and the
non-stepped parts of `Trainer` (logging, `fit`, `overfit_one_batch`). The learner
writes only Tasks 1-4.

## compute_notes
Everything runs on CPU in seconds. No GPU needed. The overfit test uses Adam
lr=0.1 for 500 steps; a correct step drives MSE from O(1) to <1e-4 on the
noiseless batch. A flat curve means the step is wrong (e.g. missing
`zero_grad`/`backward`/`step`), not a tuning problem.

## stretch_goals
1. Add `RMSNorm` to `nanovision.primitives` (A1 will need it) and gradcheck it.
2. Make `Trainer.fit` log a validation loss when a `val_loader` is given.
3. Plot a linreg-vs-iterations animation from the CSV log.

## further_reading
- Ba et al., "Layer Normalization" (2016).
- Hendrycks & Gimpel, "Gaussian Error Linear Units (GELUs)" (2016).

## solution_notes
`set_seed(0)` + Adam lr=0.1 makes the overfit test reliably reach <1e-4 well under
500 steps; the noiseless linreg batch is exactly representable by `Linear(8, 1)`.
gradcheck casts to float64 and calls `eval()` so dropout (p=0 here) is inert.
