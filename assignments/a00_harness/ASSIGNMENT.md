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
The course harness plus the three primitives every transformer block is built
from. See the README for why correctness is verified before training and what
gradcheck and overfit-one-batch actually prove.

## background
LayerNorm over the last dim, mean and biased var (`unbiased=False`):

  y = (x − mean) / sqrt(var + eps) · weight + bias

Exact GELU: `GELU(x) = x · 0.5 · (1 + erf(x / √2))`. MLP:
`fc2(drop(act(fc1(x))))`. Step rhythm: `train`, `zero_grad`, forward, loss,
`backward`, `step`, return `loss.item()`. Shapes in the per-task contracts below.

## what_you_implement
- `gelu` (exact erf form).
- `LayerNorm.forward` from mean/var ops.
- `MLP.forward` (Linear → act → dropout → Linear).
- `Trainer.step` (the one optimization step).

## tasks
- **Task 1 - gelu** (file: `primitives.py`, symbol: `gelu`):
  `x: (...,)` → `(...,)` same shape. Formula `x * 0.5 * (1 + torch.erf(x /
  math.sqrt(2)))`. Test: `test_shapes.py::test_gelu_shape_and_zero`. Teaches the
  activation as a smooth gate, and `gelu(0) == 0`.
- **Task 2 - LayerNorm.forward** (file: `primitives.py`, symbol:
  `LayerNorm`): `x: (..., dim)` → `(..., dim)`. `mean = x.mean(-1, keepdim=True)`,
  `var = x.var(-1, keepdim=True, unbiased=False)`, then
  `(x - mean) / sqrt(var + eps) * weight + bias` with `weight`, `bias` of shape
  `(dim,)`, `eps=1e-5`. Test: `test_shapes.py::test_layernorm_shape_and_stats` and
  `test_gradcheck.py::test_layernorm_gradcheck`. Teaches normalization from
  primitive ops, not `nn.LayerNorm`.
- **Task 3 - MLP.forward** (file: `primitives.py`, symbol: `MLP`):
  `x: (..., dim)` → `(..., dim)`, inner width `hidden`. Return
  `fc2(drop(act(fc1(x))))`. Test: `test_shapes.py::test_mlp_shape` and
  `test_gradcheck.py::test_mlp_gradcheck`. Teaches the block's position-wise FFN.
- **Task 4 - Trainer.step** (file: `trainer.py`, symbol: `Trainer.step`):
  `batch = (inputs, targets)` → scalar `float`. Move batch to device,
  `model.train()`, `optimizer.zero_grad()`, `pred = model(inputs)`,
  `loss = loss_fn(pred, targets)`, `loss.backward()`, `optimizer.step()`,
  `return loss.item()`. Test: `test_overfit.py::test_overfit_linreg`. Teaches the
  zero-grad / forward / backward / step rhythm that `overfit_one_batch` and `fit`
  drive.

## tests
Run in shapes → gradcheck → overfit order (see README "How to verify").
1. `tests/test_shapes.py` - output shapes for Tasks 1-3, LayerNorm output stats
   (zero mean, unit std over last dim), `gelu(0) == 0`. (shape)
2. `tests/test_gradcheck.py` - `check_gradients` at float64 on LayerNorm and MLP,
   plus an `assert_shapes` smoke test. (gradcheck)
3. `tests/test_overfit.py` - `Trainer.overfit_one_batch` drives a noiseless
   linreg batch to MSE < 1e-4 in 500 steps. (overfit-one-batch)

## provided_boilerplate
`nanovision.gradcheck` (check_gradients, assert_shapes), `nanovision.determinism`
(set_seed), `nanovision.data.toy` (linreg/copy/sort/char), `nanovision.data.images`,
`nanovision.viz.plot_loss_curve`, and the non-stepped parts of `Trainer` (logging,
`overfit_one_batch`, `fit`). The learner writes only Tasks 1-4.

## compute_notes
CPU, seconds; no GPU. Overfit test: Adam lr=0.1, 500 steps, noiseless
`linreg_batch(n=64, d=8)`, exactly fit by `Linear(8, 1)`; final MSE < 1e-4. A flat
curve from step one = step wiring wrong (`zero_grad`/`backward`/`step`); drop-then-
stall = model/loss, not optimizer.

## stretch_goals
1. Add `RMSNorm` to `nanovision.primitives` (A1 needs it) and gradcheck it.
2. Make `Trainer.fit` log a validation loss when a `val_loader` is given.
3. Animate the linreg fit from the CSV log.

## further_reading
- Ba, Kiros & Hinton, "Layer Normalization" (2016), arXiv:1607.06450.
- Hendrycks & Gimpel, "Gaussian Error Linear Units (GELUs)" (2016),
  arXiv:1606.08415.
- Kingma & Ba, "Adam: A Method for Stochastic Optimization" (2014),
  arXiv:1412.6980.

## solution_notes
`set_seed(0)` + Adam lr=0.1 makes the overfit test reach <1e-4 well under 500 steps;
verified reference run starts at MSE ≈ 14.5 and ends ≈ 2e-13 at step 500. The
noiseless linreg batch is exactly representable by `Linear(8, 1)`. `check_gradients`
casts module and float inputs to float64 and calls `eval()` so dropout (p=0 here) is
inert; float64 is what makes the finite-difference comparison tight enough to catch
bugs. LayerNorm uses biased variance (`unbiased=False`).
