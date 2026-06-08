"""End-to-end fit of N Gaussians to one 16x16 toy view, the whole forward in seconds.

Fits N=64 Gaussians to a single posed image from the NeRF toy scene with Adam and an L1
loss (no D-SSIM: the structural-similarity term is degenerate on a 16x16 image). N=64
against 256 pixels forces the Gaussians to place and shape themselves, but is small enough
that a projection or sort bug shows up as a non-decreasing loss. The L1 floor was measured
at build time: from a ~0.10-0.14 start it reaches ~0.005-0.007 in 400 steps, so the
threshold is 0.02, comfortably above the floor and well below the start. PSNR >= 20 dB and
the full multi-view fit are a GPU viz demo, not this CPU test.
"""

import torch

from gaussian import GaussianModel
from project import project_gaussians
from render import splat_render

from nanovision.data import toy
from nanovision.geometry import invert_transform

THRESHOLD = 0.02


def test_overfit_single_view():
    torch.manual_seed(0)
    images, poses, K, _, _ = toy.nerf_synthetic_scene(n_views=6, H=16, W=16)
    target = images[0]
    w2c = invert_transform(poses[0])
    H = W = 16

    model = GaussianModel.random_init(64, spread=1.0, init_scale=0.2, seed=0)
    opt = torch.optim.Adam([
        {"params": [model.means], "lr": 1e-2},
        {"params": [model.log_scales], "lr": 1e-2},
        {"params": [model.quats], "lr": 1e-3},
        {"params": [model.opacity_logits], "lr": 5e-2},
        {"params": [model.color_logits], "lr": 2e-2},
    ])

    first = None
    for _ in range(400):
        means2d, cov2d, depths = project_gaussians(model, K, w2c)
        img = splat_render(means2d, cov2d, model.colors, model.opacities, depths, H, W, bg=1.0)
        loss = (img - target).abs().mean()
        if first is None:
            first = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()

    final = loss.item()
    assert final < THRESHOLD, f"L1 {final:.4f} did not reach {THRESHOLD} (start {first:.4f})"
    assert final < 0.5 * first, f"L1 barely moved: {first:.4f} -> {final:.4f}"
