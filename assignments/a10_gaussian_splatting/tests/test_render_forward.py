"""Training-free exact checks of the rasterizer: a centered blob, value range, front-to-back
ordering, and the all-transparent case equal to the background.
"""

import torch

from render import splat_render

H = W = 16


def _iso_cov(n, s=2.0, dtype=torch.float32):
    cov = torch.zeros(n, 2, 2, dtype=dtype)
    cov[:, 0, 0] = s
    cov[:, 1, 1] = s
    return cov


def test_centered_blob_brightest_at_center():
    center = torch.tensor([[(W - 1) / 2.0, (H - 1) / 2.0]])
    img = splat_render(center, _iso_cov(1, s=3.0), torch.ones(1, 3),
                       torch.ones(1), torch.zeros(1), H, W, bg=0.0)
    gray = img.mean(-1)
    flat = gray.argmax()
    cy, cx = divmod(flat.item(), W)
    assert abs(cx - (W - 1) / 2.0) <= 1.0 and abs(cy - (H - 1) / 2.0) <= 1.0
    # Falls off with radius: center brighter than a corner.
    assert gray[H // 2, W // 2] > gray[0, 0]


def test_output_shape_and_range():
    g = torch.Generator().manual_seed(0)
    n = 5
    means = torch.rand(n, 2, generator=g) * W
    img = splat_render(means, _iso_cov(n), torch.rand(n, 3, generator=g),
                       torch.rand(n, generator=g), torch.rand(n, generator=g), H, W, bg=0.3)
    assert img.shape == (H, W, 3)
    assert img.min() >= 0.0 and img.max() <= 1.0


def test_front_gaussian_dominates_overlap():
    # Two opaque Gaussians at the same pixel, different colors and depths.
    center = torch.tensor([[(W - 1) / 2.0, (H - 1) / 2.0]]).repeat(2, 1)
    cov = _iso_cov(2, s=3.0)
    colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # red, blue
    op = torch.full((2,), 0.99)
    px = (H // 2, W // 2)

    near_red = splat_render(center, cov, colors, op, torch.tensor([1.0, 2.0]), H, W)
    assert near_red[px][0] > near_red[px][2], "near red should dominate"

    near_blue = splat_render(center, cov, colors, op, torch.tensor([2.0, 1.0]), H, W)
    assert near_blue[px][2] > near_blue[px][0], "swapped depth: near blue should dominate"


def test_zero_opacity_equals_background():
    g = torch.Generator().manual_seed(2)
    n = 4
    means = torch.rand(n, 2, generator=g) * W
    img = splat_render(means, _iso_cov(n), torch.rand(n, 3, generator=g),
                       torch.zeros(n), torch.rand(n, generator=g), H, W, bg=0.42)
    assert torch.allclose(img, torch.full_like(img, 0.42), atol=1e-6)
