# A0 — Harness & primitives

## Motivation
This is the substrate every later assignment imports: the primitives that
transformer blocks are built from (LayerNorm, gelu, MLP), the `Trainer` that runs
the overfit-one-batch correctness check, and `check_gradients`, which proves a
hand-written `forward()` is correct without you ever writing a backward pass. You
build it first so the verification workflow is concrete on trivial mechanisms
before the interesting ones arrive.

## Background
LayerNorm over the last dimension:

    y = (x - mean) / sqrt(var + eps) * weight + bias

with mean and biased variance over the last axis. Exact GELU:

    GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))

The optimization step is the standard rhythm: zero grads, forward, compute loss,
backward, optimizer step. `overfit_one_batch` repeats that on one fixed batch; a
correct model drives the loss to ~0.

## What you'll implement
`gelu`, `LayerNorm.forward`, `MLP.forward` (in `starter/primitives.py`), and
`Trainer.step` (in `starter/trainer.py`). Four small holes.

## Tasks
1. `gelu` — exact erf GELU.
2. `LayerNorm.forward` — normalize over the last dim, then scale+shift.
3. `MLP.forward` — Linear → act → dropout → Linear.
4. `Trainer.step` — one optimization step returning the scalar loss.

Each maps to a `raise NotImplementedError(...)` in `starter/` and to one test.

## How to verify
Run from the repo root with the `nanovision` env active, in this order:

    make test A=a00_harness      # your starter (red until you fill the holes)

The tests run shapes → gradcheck → overfit. To confirm the reference passes and
to render the loss curve:

    make verify A=a00_harness    # reference solution (should be green)
    make viz    A=a00_harness    # writes out/loss_curve.png

The reference implementation is visible in `nanovision/primitives.py` and
`nanovision/trainer.py`; read it if you get stuck.

## Compute notes
CPU only, seconds to run. The overfit test uses Adam lr=0.1 for 500 steps on a
noiseless linear-regression batch and expects MSE < 1e-4. A flat loss curve means
the step is wired wrong (missing zero_grad/backward/step), not a tuning issue.

## Stretch goals
1. Add `RMSNorm` to `nanovision.primitives` (A1 needs it) and gradcheck it.
2. Have `Trainer.fit` log validation loss when given a `val_loader`.
3. Animate the linreg fit from the CSV log.

## Further reading
- Ba et al., "Layer Normalization" (2016).
- Hendrycks & Gimpel, "Gaussian Error Linear Units (GELUs)" (2016).
