"""The CFM interpolant endpoints and the t-independent velocity target (exact, training-free)."""

import torch

from flow import cfm_target


def test_interpolant_endpoints():
    g = torch.Generator().manual_seed(0)
    a = torch.randn(5, 4, 2, generator=g)
    z0 = torch.randn(5, 4, 2, generator=g)
    z_at_0, _ = cfm_target(a, z0, torch.zeros(5, 1, 1))
    z_at_1, _ = cfm_target(a, z0, torch.ones(5, 1, 1))
    z_at_half, _ = cfm_target(a, z0, torch.full((5, 1, 1), 0.5))
    assert torch.allclose(z_at_0, z0, atol=1e-6)            # t=0 is the noise
    assert torch.allclose(z_at_1, a, atol=1e-6)             # t=1 is the action
    assert torch.allclose(z_at_half, 0.5 * (z0 + a), atol=1e-6)


def test_velocity_is_displacement_and_t_independent():
    g = torch.Generator().manual_seed(1)
    a = torch.randn(6, 4, 2, generator=g)
    z0 = torch.randn(6, 4, 2, generator=g)
    # The target v = a - z0 must be exact and identical at every t (constant along the path).
    ts = [torch.zeros(6, 1, 1), torch.rand(6, 1, 1, generator=g), torch.ones(6, 1, 1)]
    refs = [cfm_target(a, z0, t)[1] for t in ts]
    for v in refs:
        assert torch.allclose(v, a - z0, atol=1e-6)
    assert torch.allclose(refs[0], refs[1], atol=1e-6)
    assert torch.allclose(refs[1], refs[2], atol=1e-6)


def test_shapes():
    a = torch.randn(3, 4, 2)
    z0 = torch.randn(3, 4, 2)
    z_t, v = cfm_target(a, z0, torch.rand(3, 1, 1))
    assert z_t.shape == (3, 4, 2)
    assert v.shape == (3, 4, 2)
