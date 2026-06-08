"""Minibatch OT coupling: the optimal swap, a valid permutation, lower total cost."""

import torch

from coupling import ot_coupling


def test_swap_case():
    x0 = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    x1 = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    x1r, perm = ot_coupling(x0, x1)
    # the optimal pairing swaps x1 so each x0 maps to the colocated x1.
    assert torch.allclose(x1r, x0)
    assert perm.tolist() == [1, 0]


def test_valid_permutation_and_lower_cost():
    g = torch.Generator().manual_seed(0)
    x0 = torch.randn(32, 2, generator=g)
    x1 = torch.randn(32, 2, generator=g)
    x1r, perm = ot_coupling(x0, x1)
    assert sorted(perm.tolist()) == list(range(32))            # a bijection
    ot_cost = ((x0 - x1r) ** 2).sum()
    identity_cost = ((x0 - x1) ** 2).sum()
    assert ot_cost <= identity_cost + 1e-6
