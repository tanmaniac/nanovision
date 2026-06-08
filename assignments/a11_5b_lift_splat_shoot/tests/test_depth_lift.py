"""Mechanism A: the depth + context heads and the outer-product lift.

Pins the output shapes, the one-hot-depth behavior (the volume collapses to the context at the
selected bin), and float64-gradcheck of the outer-product map.
"""

import torch

from nanovision.lift_splat import DepthLift


def test_shapes():
    m = DepthLift(c_in=4, D=4, C_ctx=8)
    feat = torch.randn(1, 4, 2, 2)
    logits, context = m(feat)
    assert logits.shape == (1, 4, 2, 2)
    assert context.shape == (1, 8, 2, 2)
    volume = m.lift(feat)
    assert volume.shape == (1, 4, 8, 2, 2)


def test_one_hot_depth_picks_context():
    # Force the depth softmax to ~one-hot at bin d=2 by writing the conv bias directly
    # (a 1x1 conv with zero weight outputs its bias everywhere).
    m = DepthLift(c_in=4, D=4, C_ctx=8)
    with torch.no_grad():
        m.depth_head.weight.zero_()
        m.depth_head.bias.zero_()
        m.depth_head.bias[2] = 50.0          # softmax concentrates at bin 2
    feat = torch.randn(1, 4, 2, 2)
    _, context = m(feat)
    volume = m.lift(feat)                     # (1, 4, 8, 2, 2)
    # All bins except 2 are ~zero; bin 2 equals the context.
    for d in range(4):
        if d == 2:
            assert torch.allclose(volume[:, d], context, atol=1e-4)
        else:
            assert volume[:, d].abs().max() < 1e-4


def test_lift_gradcheck():
    # Gradcheck the pure outer-product map (alpha, context) -> volume, the differentiable core.
    torch.manual_seed(0)
    D, C, Hf, Wf = 3, 2, 2, 2
    alpha = torch.rand(1, D, Hf, Wf, dtype=torch.float64, requires_grad=True)
    context = torch.rand(1, C, Hf, Wf, dtype=torch.float64, requires_grad=True)

    def outer(a, c):
        return a[:, :, None] * c[:, None]

    assert torch.autograd.gradcheck(outer, (alpha, context))
