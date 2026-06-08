"""Exact, training-free checks of the discretized volume renderer.

These pin the compositing math (alpha, exclusive transmittance, weights) without any
training run, so they are the graded correctness gate alongside the overfit test.
"""

import torch

from nanovision.volume import volume_render


def test_opaque_sample_returns_its_color():
    # One opaque sample (large finite sigma) in front, the rest transparent: the rendered
    # color is that sample's color.
    R, N = 3, 5
    sigmas = torch.zeros(R, N)
    sigmas[:, 1] = 1e3
    colors = torch.rand(R, N, 3)
    deltas = torch.ones(R, N)
    color, weights = volume_render(sigmas, colors, deltas)
    assert torch.allclose(color, colors[:, 1], atol=1e-4)


def test_all_zero_density_is_background():
    R, N = 4, 6
    sigmas = torch.zeros(R, N)
    colors = torch.rand(R, N, 3)
    deltas = torch.ones(R, N)
    color, weights = volume_render(sigmas, colors, deltas)
    assert torch.allclose(color, torch.zeros(R, 3), atol=1e-6)
    assert torch.allclose(weights.sum(dim=1), torch.zeros(R), atol=1e-6)


def test_all_zero_density_white_background():
    R, N = 2, 4
    sigmas = torch.zeros(R, N)
    colors = torch.rand(R, N, 3)
    deltas = torch.ones(R, N)
    color, _ = volume_render(sigmas, colors, deltas, white_background=True)
    assert torch.allclose(color, torch.ones(R, 3), atol=1e-6)


def test_weights_nonneg_and_bounded():
    torch.manual_seed(0)
    R, N = 8, 16
    sigmas = torch.rand(R, N) * 5.0
    colors = torch.rand(R, N, 3)
    deltas = torch.rand(R, N) + 0.1
    _, weights = volume_render(sigmas, colors, deltas)
    assert (weights >= 0).all()
    assert (weights.sum(dim=1) <= 1.0 + 1e-6).all()


def test_first_weight_equals_alpha0():
    # T_0 = 1 exactly, so the first weight equals alpha_0 = 1 - exp(-sigma_0 delta_0).
    torch.manual_seed(1)
    R, N = 5, 7
    sigmas = torch.rand(R, N) * 3.0
    colors = torch.rand(R, N, 3)
    deltas = torch.rand(R, N) + 0.2
    _, weights = volume_render(sigmas, colors, deltas)
    alpha0 = 1.0 - torch.exp(-sigmas[:, 0] * deltas[:, 0])
    assert torch.allclose(weights[:, 0], alpha0, atol=1e-6)


def test_telescoping_identity():
    # sum_i w_i = 1 - prod_i (1 - alpha_i), exactly. Pins the whole transmittance structure.
    torch.manual_seed(2)
    R, N = 6, 10
    sigmas = torch.rand(R, N) * 4.0
    colors = torch.rand(R, N, 3)
    deltas = torch.rand(R, N) + 0.1
    _, weights = volume_render(sigmas, colors, deltas)
    alpha = 1.0 - torch.exp(-sigmas * deltas)
    rhs = 1.0 - torch.prod(1.0 - alpha, dim=1)
    assert torch.allclose(weights.sum(dim=1), rhs, atol=1e-6)


def test_large_sigma_saturates_weight():
    R, N = 3, 8
    sigmas = torch.zeros(R, N)
    sigmas[:, 2] = 1e3
    colors = torch.rand(R, N, 3)
    deltas = torch.ones(R, N)
    _, weights = volume_render(sigmas, colors, deltas)
    assert torch.allclose(weights.sum(dim=1), torch.ones(R), atol=1e-4)


def test_gradcheck_sigmas_and_colors():
    R, N = 3, 4
    sigmas = (torch.rand(R, N, dtype=torch.float64) * 2.0).requires_grad_(True)
    colors = torch.rand(R, N, 3, dtype=torch.float64, requires_grad=True)
    deltas = (torch.rand(R, N, dtype=torch.float64) + 0.2)
    assert torch.autograd.gradcheck(
        lambda s, c: volume_render(s, c, deltas)[0], (sigmas, colors)
    )
