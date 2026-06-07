"""Gradient- and shape-checking helpers.

These prove a `forward()` is correct without the learner ever writing a backward
pass: `check_gradients` runs `torch.autograd.gradcheck` at double precision on a
tiny instance, comparing autograd against numerical finite differences.
"""

from typing import Sequence

import torch
from torch import Tensor, nn


def check_gradients(
    module: nn.Module,
    example_inputs,
    eps: float = 1e-6,
    atol: float = 1e-4,
    rtol: float = 1e-3,
) -> bool:
    """Run gradcheck on `module` at float64 with the given inputs.

    The module and any floating-point inputs are cast to double; dropout is
    disabled via `eval()`. If the module returns a tuple (e.g. attention returns
    `(out, attn)`), gradcheck is run on the first element. Returns True or raises
    AssertionError with a readable message.
    """
    module = module.double().eval()

    if isinstance(example_inputs, Tensor):
        example_inputs = (example_inputs,)
    inputs = tuple(
        x.double().detach().requires_grad_(True)
        if torch.is_tensor(x) and torch.is_floating_point(x)
        else x
        for x in example_inputs
    )

    def func(*ins):
        out = module(*ins)
        if isinstance(out, (tuple, list)):
            return out[0]
        return out

    try:
        torch.autograd.gradcheck(func, inputs, eps=eps, atol=atol, rtol=rtol)
    except Exception as e:  # noqa: BLE001 — surface a readable message
        raise AssertionError(f"gradcheck failed for {type(module).__name__}: {e}")
    return True


def assert_shapes(fn, cases: Sequence[dict]) -> None:
    """Table-driven shape testing.

    Each case is a dict with optional `args`/`kwargs` and a required `expected`
    shape (a tuple for a single output, or a tuple-of-tuples when `fn` returns
    multiple tensors).
    """
    for i, case in enumerate(cases):
        args = case.get("args", ())
        kwargs = case.get("kwargs", {})
        expected = case["expected"]
        out = fn(*args, **kwargs)
        if isinstance(out, (tuple, list)):
            for j, (o, e) in enumerate(zip(out, expected)):
                assert tuple(o.shape) == tuple(e), (
                    f"case {i} output {j}: got {tuple(o.shape)}, expected {tuple(e)}"
                )
        else:
            assert tuple(out.shape) == tuple(expected), (
                f"case {i}: got {tuple(out.shape)}, expected {tuple(expected)}"
            )
