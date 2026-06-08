"""The confidence-weighted, jointly-scale-normalized pointmap loss.

Checks: gradcheck; joint scale invariance (scaling BOTH pred maps and BOTH gt maps by the
same factor leaves the loss unchanged); the shared-scale guard (rescaling ONLY view 2 changes
the loss, proving the scale is joint and not per-map); and that the -alpha*log(C) term gives a
finite optimal confidence for a fixed residual.
"""

import torch

from loss import pointmap_loss, normalize_scale


def _rand_case(B=2, h=4, w=4, dtype=torch.float64, seed=0):
    g = torch.Generator().manual_seed(seed)
    p1 = torch.randn(B, h, w, 3, generator=g, dtype=dtype)
    p2 = torch.randn(B, h, w, 3, generator=g, dtype=dtype)
    g1 = torch.randn(B, h, w, 3, generator=g, dtype=dtype)
    g2 = torch.randn(B, h, w, 3, generator=g, dtype=dtype)
    c1 = 1.0 + torch.rand(B, h, w, generator=g, dtype=dtype)
    c2 = 1.0 + torch.rand(B, h, w, generator=g, dtype=dtype)
    v1 = torch.rand(B, h, w, generator=g) > 0.3
    v2 = torch.rand(B, h, w, generator=g) > 0.3
    return p1, p2, g1, g2, c1, c2, v1, v2


def test_loss_gradcheck():
    p1, p2, g1, g2, c1, c2, v1, v2 = _rand_case()
    p1 = p1.requires_grad_(True)
    p2 = p2.requires_grad_(True)
    c1 = c1.requires_grad_(True)
    c2 = c2.requires_grad_(True)

    def f(a, b, ca, cb):
        return pointmap_loss(a, b, g1, g2, ca, cb, v1, v2, alpha=0.2)

    assert torch.autograd.gradcheck(f, (p1, p2, c1, c2))


def test_joint_scale_invariance():
    p1, p2, g1, g2, c1, c2, v1, v2 = _rand_case(seed=1)
    base = pointmap_loss(p1, p2, g1, g2, c1, c2, v1, v2)
    s = 3.7
    scaled = pointmap_loss(s * p1, s * p2, g1, g2, c1, c2, v1, v2)
    assert torch.allclose(base, scaled, atol=1e-9), (base.item(), scaled.item())
    # Scaling GT jointly is also absorbed.
    scaled_gt = pointmap_loss(p1, p2, 2.1 * g1, 2.1 * g2, c1, c2, v1, v2)
    assert torch.allclose(base, scaled_gt, atol=1e-9)


def test_shared_scale_guard():
    # Rescaling ONLY view 2 (not view 1) must change the loss: the scale is computed jointly
    # over both maps, so it cannot absorb a view-2-only rescale.
    p1, p2, g1, g2, c1, c2, v1, v2 = _rand_case(seed=2)
    base = pointmap_loss(p1, p2, g1, g2, c1, c2, v1, v2)
    only_v2 = pointmap_loss(p1, 2.5 * p2, g1, g2, c1, c2, v1, v2)
    assert not torch.allclose(base, only_v2, atol=1e-4), (base.item(), only_v2.item())


def test_normalize_scale_is_joint():
    # normalize_scale over both stacked maps equals the mean norm of all valid points.
    B, h, w = 2, 4, 4
    pts = torch.randn(B, 2, h, w, 3, dtype=torch.float64)
    valid = torch.rand(B, 2, h, w) > 0.2
    z = normalize_scale(pts, valid)
    assert z.shape == (B,)
    norms = pts.norm(dim=-1)
    for b in range(B):
        m = valid[b]
        expect = norms[b][m].mean()
        assert torch.allclose(z[b], expect, atol=1e-9)


def test_confidence_finite_optimum():
    # For a fixed residual ell, the per-pixel cost C*ell - alpha*log(C) is minimized at a
    # finite C = alpha/ell when alpha/ell > 1, not at C -> 1 or C -> inf. Build a case with a
    # known normalized residual and sweep the confidence on the single valid pixel.
    #
    # Construct unit-norm points so the joint scale z = zbar = 1, making the normalized
    # residual exactly the raw difference. Map 2 matches its GT (zero residual); map 1 has a
    # small residual ell = 0.1, so alpha/ell = 0.2/0.1 = 2 > 1 and the optimum is interior.
    alpha = 0.2
    ell = 0.1
    h = w = 1
    # All points unit norm: pred1=(1,0,0), gt1=(cos t, sin t, 0) with chord ell to pred1.
    import math
    t = 2.0 * math.asin(ell / 2.0)  # chord length between two unit vectors = 2 sin(t/2)
    pred = torch.tensor([[[[1.0, 0.0, 0.0]]]])
    gt = torch.tensor([[[[math.cos(t), math.sin(t), 0.0]]]])
    pred2 = torch.tensor([[[[0.0, 1.0, 0.0]]]])
    gt2 = torch.tensor([[[[0.0, 1.0, 0.0]]]])  # map 2 matches exactly (unit norm, anchors scale)
    v = torch.ones(1, h, w, dtype=torch.bool)

    # Sanity: residual on map 1 is ell after the (unit) joint normalization.
    losses = []
    Cs = torch.linspace(1.01, 8.0, 400)
    for C in Cs:
        c1 = torch.full((1, h, w), float(C))
        c2 = torch.ones(1, h, w)
        losses.append(pointmap_loss(pred, pred2, gt, gt2, c1, c2, v, v, alpha=alpha).item())
    losses = torch.tensor(losses)
    arg = int(losses.argmin())
    assert 0 < arg < len(Cs) - 1, f"optimum at boundary (arg={arg})"
    # The interior optimum should sit near C = alpha/ell = 2 (the confidence on the residual
    # pixel; the matched pixel's C is held at 1).
    assert abs(Cs[arg].item() - alpha / ell) < 0.3, Cs[arg].item()
