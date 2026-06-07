"""Task 2: SE(3) primitives. Run after projection."""

import math

import torch

from nanovision.gradcheck import check_gradients
from geometry import (
    apply_transform,
    compose_transforms,
    invert_transform,
    make_transform,
)


def _rot_z(theta):
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_make_transform_shape_and_blocks():
    R = _rot_z(0.3)
    t = torch.tensor([1.0, 2.0, 3.0])
    T = make_transform(R, t)
    assert T.shape == (4, 4)
    assert torch.allclose(T[:3, :3], R)
    assert torch.allclose(T[:3, 3], t)
    assert torch.allclose(T[3], torch.tensor([0.0, 0.0, 0.0, 1.0]))


def test_apply_transform_reference():
    # 90-degree rotation about z, then translate by (1, 0, 0).
    T = make_transform(_rot_z(math.pi / 2), torch.tensor([1.0, 0.0, 0.0]))
    p = torch.tensor([[1.0, 0.0, 0.0]])
    out = apply_transform(T, p)
    # (1,0,0) rotates to (0,1,0), then +1 in x -> (1, 1, 0).
    assert torch.allclose(out[0], torch.tensor([1.0, 1.0, 0.0]), atol=1e-6)


def test_invert_is_inverse():
    T = make_transform(_rot_z(0.7), torch.tensor([3.0, -1.0, 2.0]))
    I = compose_transforms(invert_transform(T), T)
    assert torch.allclose(I, torch.eye(4), atol=1e-6)
    I2 = compose_transforms(T, invert_transform(T))
    assert torch.allclose(I2, torch.eye(4), atol=1e-6)


def test_compose_associativity_and_chain():
    A = make_transform(_rot_z(0.2), torch.tensor([1.0, 0.0, 0.0]))
    B = make_transform(_rot_z(-0.5), torch.tensor([0.0, 2.0, 0.0]))
    C = make_transform(_rot_z(1.1), torch.tensor([0.0, 0.0, 3.0]))
    left = compose_transforms(compose_transforms(A, B), C)
    right = compose_transforms(A, compose_transforms(B, C))
    assert torch.allclose(left, right, atol=1e-6)
    # Composed transform equals applying C, then B, then A to a point.
    p = torch.tensor([[1.0, 1.0, 1.0]])
    chained = apply_transform(compose_transforms(A, B, C), p)
    stepwise = apply_transform(A, apply_transform(B, apply_transform(C, p)))
    assert torch.allclose(chained, stepwise, atol=1e-6)


def test_apply_transform_gradcheck():
    T = make_transform(_rot_z(0.4), torch.tensor([1.0, 2.0, 3.0])).double()

    class App(torch.nn.Module):
        def forward(self, pts):
            return apply_transform(T, pts)

    pts = torch.randn(6, 3, dtype=torch.double)
    assert check_gradients(App(), (pts,))
