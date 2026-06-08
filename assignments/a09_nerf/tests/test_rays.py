"""Ray generation: center/corner directions, sample shapes, deltas, points on the ray."""

import torch

from nanovision.volume import (
    deltas_from_z,
    sample_along_rays,
    stratified_sample_rays,
)


def _intrinsic(H, W, f):
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    return torch.tensor([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])


def test_center_ray_is_plus_z_identity_pose():
    H = W = 17  # odd so the exact center pixel exists
    f = 20.0
    K = _intrinsic(H, W, f)
    c2w = torch.eye(4)
    rays_o, rays_d, z_vals = stratified_sample_rays(
        H, W, K, c2w, 2.0, 6.0, 8, perturb=False
    )
    center = (H // 2) * W + (W // 2)
    assert torch.allclose(rays_d[center], torch.tensor([0.0, 0.0, 1.0]), atol=1e-5)
    assert torch.allclose(rays_o[center], torch.zeros(3), atol=1e-6)


def test_corner_ray_tangent():
    H = W = 16
    f = 16.0
    K = _intrinsic(H, W, f)
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    c2w = torch.eye(4)
    rays_o, rays_d, _ = stratified_sample_rays(H, W, K, c2w, 2.0, 6.0, 8, perturb=False)
    # Pixel (u=0, v=0) is the first row-major ray. Camera-frame direction before
    # normalization is ((0-cx)/f, (0-cy)/f, 1); the tangents x/z and y/z survive
    # normalization since the pose is identity.
    d = rays_d[0]
    assert torch.allclose(d[0] / d[2], torch.tensor((0.0 - cx) / f), atol=1e-5)
    assert torch.allclose(d[1] / d[2], torch.tensor((0.0 - cy) / f), atol=1e-5)


def test_directions_are_unit_length():
    H = W = 16
    K = _intrinsic(H, W, 16.0)
    c2w = torch.eye(4)
    _, rays_d, _ = stratified_sample_rays(H, W, K, c2w, 2.0, 6.0, 8, perturb=False)
    assert torch.allclose(rays_d.norm(dim=-1), torch.ones(H * W), atol=1e-5)


def test_z_vals_in_bounds():
    H = W = 8
    K = _intrinsic(H, W, 8.0)
    c2w = torch.eye(4)
    g = torch.Generator().manual_seed(0)
    near, far = 2.0, 6.0
    _, _, z_vals = stratified_sample_rays(H, W, K, c2w, near, far, 12, perturb=True, generator=g)
    assert z_vals.shape == (H * W, 12)
    assert (z_vals >= near).all() and (z_vals <= far).all()
    # Sorted within each ray (uniform bins, one jitter per bin).
    assert (z_vals[:, 1:] >= z_vals[:, :-1]).all()


def test_sample_along_rays_on_the_ray():
    R, N = 5, 4
    rays_o = torch.randn(R, 3)
    rays_d = torch.randn(R, 3)
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)
    z_vals = torch.rand(R, N).sort(dim=-1).values + 1.0
    pts = sample_along_rays(rays_o, rays_d, z_vals)
    assert pts.shape == (R, N, 3)
    expected = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[:, :, None]
    assert torch.allclose(pts, expected, atol=1e-6)


def test_deltas_last_is_large():
    z_vals = torch.tensor([[1.0, 2.0, 4.0, 7.0]])
    d = deltas_from_z(z_vals)
    assert torch.allclose(d[:, :-1], torch.tensor([[1.0, 2.0, 3.0]]))
    assert d[0, -1] >= 1e9
