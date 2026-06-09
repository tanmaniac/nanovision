"""Double-precision gradcheck on roi_align_bev and on both wta_loss paths."""

import torch

from predict import roi_align_bev, wta_loss


def test_roi_align_gradcheck():
    # Centers at FRACTIONAL interior locations: bilinear sampling is non-differentiable exactly
    # at integer grid nodes and at the border, so sample strictly between cells, away from edges.
    torch.manual_seed(0)
    C, nx, ny = 3, 6, 7
    bev = torch.randn(C, nx, ny, dtype=torch.float64, requires_grad=True)
    centers = torch.tensor([[2.3, 3.6], [3.7, 2.4]], dtype=torch.float64)
    assert torch.autograd.gradcheck(
        lambda b: roi_align_bev(b, centers, out_size=3, radius=1.0), (bev,)
    )


def test_wta_loss_hard_gradcheck():
    # One mode is strictly the minFDE winner (its endpoint is closest), off the tie boundary,
    # so the detached argmin is stable under perturbation.
    torch.manual_seed(0)
    B, K, T = 2, 4, 5
    gt = torch.randn(B, T, 2, dtype=torch.float64)
    trajs = torch.randn(B, K, T, 2, dtype=torch.float64, requires_grad=True)
    scores = torch.randn(B, K, dtype=torch.float64, requires_grad=True)
    # Plant a clear winner per sample: mode 0 endpoint near gt, the rest far away.
    with torch.no_grad():
        trajs[:, 0, -1, :] = gt[:, -1, :] + 0.05
        for k in range(1, K):
            trajs[:, k, -1, :] = gt[:, -1, :] + 5.0 + k
    assert torch.autograd.gradcheck(
        lambda tr, sc: wta_loss(tr, sc, gt, temperature=None), (trajs, scores)
    )


def test_wta_loss_soft_gradcheck():
    torch.manual_seed(1)
    B, K, T = 2, 4, 5
    gt = torch.randn(B, T, 2, dtype=torch.float64)
    trajs = torch.randn(B, K, T, 2, dtype=torch.float64, requires_grad=True)
    scores = torch.randn(B, K, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda tr, sc: wta_loss(tr, sc, gt, temperature=1.5), (trajs, scores)
    )
