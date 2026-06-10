"""Exact, training-free check of the DreamerV3 lambda-return recursion."""

import torch

from actor_critic import compute_lambda_returns


def _manual(rewards, values, conts, gamma, lam):
    """Reference: backward recursion R_t = r_t + g*c_t*((1-lam)*V_{t+1} + lam*R_{t+1}), R_H = V_H."""
    H = rewards.shape[1]
    out = [None] * H
    R_next = values[:, H].clone()
    for t in reversed(range(H)):
        R_next = rewards[:, t] + gamma * conts[:, t] * (
            (1 - lam) * values[:, t + 1] + lam * R_next
        )
        out[t] = R_next.clone()
    return torch.stack(out, 1)


def test_matches_hand_computed_three_step():
    rewards = torch.tensor([[1.0, 0.0, 2.0]])
    values = torch.tensor([[0.5, 1.0, 1.5, 3.0]])  # V_0..V_3, V_3 is the bootstrap
    conts = torch.tensor([[1.0, 1.0, 1.0]])
    gamma, lam = 0.9, 0.8
    got = compute_lambda_returns(rewards, values, conts, gamma, lam)

    # Closed form by hand.
    R3 = 3.0
    R2 = 2.0 + 0.9 * ((1 - 0.8) * 3.0 + 0.8 * R3)
    R1 = 0.0 + 0.9 * ((1 - 0.8) * 1.5 + 0.8 * R2)
    R0 = 1.0 + 0.9 * ((1 - 0.8) * 1.0 + 0.8 * R1)
    expected = torch.tensor([[R0, R1, R2]])
    assert torch.allclose(got, expected, atol=1e-6), (got, expected)


def test_continuation_zero_kills_bootstrap_and_tail():
    rewards = torch.tensor([[1.0, 0.0, 2.0]])
    values = torch.tensor([[0.5, 1.0, 1.5, 3.0]])
    conts = torch.tensor([[1.0, 0.0, 1.0]])  # episode ends at t=1
    gamma, lam = 0.9, 0.8
    got = compute_lambda_returns(rewards, values, conts, gamma, lam)
    # At t=1, c=0 zeros the discounted future, so R_1 = r_1 = 0.
    assert torch.allclose(got[:, 1], torch.tensor([0.0]), atol=1e-6)
    # R_0 then only sees R_1 = 0 in the lambda mix.
    R1 = 0.0
    R0 = 1.0 + 0.9 * ((1 - 0.8) * 1.0 + 0.8 * R1)
    assert torch.allclose(got[:, 0], torch.tensor([R0]), atol=1e-6)


def test_matches_reference_on_random_batch():
    torch.manual_seed(0)
    B, H = 4, 7
    rewards = torch.randn(B, H)
    values = torch.randn(B, H + 1)
    conts = (torch.rand(B, H) > 0.2).float()
    gamma, lam = 0.997, 0.95
    got = compute_lambda_returns(rewards, values, conts, gamma, lam)
    ref = _manual(rewards, values, conts, gamma, lam)
    assert torch.allclose(got, ref, atol=1e-6), (got - ref).abs().max()
