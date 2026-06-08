"""Float64 gradcheck of the path, the CFM loss, and the Euler sampler."""

import torch

from flow import cfm_loss, score_from_velocity
from path import linear_path
from sampling import euler_sample


class LinearModel:
    """A fixed linear velocity field x -> x @ W (ignores t), for gradchecking."""

    def __init__(self, W):
        self.W = W

    def __call__(self, x, t):
        return x @ self.W


def test_linear_path_gradcheck():
    x0 = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
    x1 = torch.randn(4, 2, dtype=torch.float64)
    t = torch.rand(4, dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda a: linear_path(a, x1, t), (x0,))


def test_score_from_velocity_gradcheck():
    v = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
    x_t = torch.randn(4, 2, dtype=torch.float64)
    t = 0.05 + 0.85 * torch.rand(4, dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda a: score_from_velocity(a, x_t, t), (v,))


def test_cfm_loss_gradcheck():
    W = torch.randn(2, 2, dtype=torch.float64)
    model = LinearModel(W)
    x0 = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
    x1 = torch.randn(4, 2, dtype=torch.float64)
    t = 0.1 + 0.8 * torch.rand(4, dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda a: cfm_loss(model, a, x1, t), (x0,))


def test_euler_gradcheck():
    W = 0.1 * torch.randn(2, 2, dtype=torch.float64)
    model = LinearModel(W)
    x0 = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a: euler_sample(model, a, 4), (x0,))
