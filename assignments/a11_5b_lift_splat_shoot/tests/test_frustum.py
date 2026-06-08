"""Mechanism B: back-project the feature-cell frustum into ego-frame 3D points.

The round-trip error is exactly 0 (the unproject/project pair are inverses), so the assertions
are tight. The signs are confirmed against the ego frame (x forward, y left, z up) and the
OpenCV camera frame (x right, y down, z forward).
"""

import torch

from nanovision.geometry import CameraRig
from nanovision.lift_splat import frustum_points


def _origin_forward_E():
    # A forward camera AT THE EGO ORIGIN: cam x = ego -y, cam y = ego -z, cam z = ego +x.
    # det(R) = +1, so this is a proper rotation. The center ray points along ego +x (forward).
    # Build E as a raw 4x4 (no make_transform) so the test scaffolding has no geometry-hole dep.
    R = torch.tensor([[0.0, -1.0, 0.0],
                      [0.0, 0.0, -1.0],
                      [1.0, 0.0, 0.0]])
    E = torch.eye(4)
    E[:3, :3] = R
    return E, R


def _K(img=32, f=16.0):
    c = (img - 1) / 2.0
    return torch.tensor([[f, 0.0, c], [0.0, f, c], [0.0, 0.0, 1.0]])


def test_center_pixel_maps_to_forward_axis():
    E, _ = _origin_forward_E()
    K = _K()
    cx = cy = K[0, 2].item()
    # A single feature cell at the principal point (the center ray).
    pixel_xy = torch.tensor([[[cx, cy]]])      # (1, 1, 2)
    depths = torch.tensor([1.0, 2.0, 3.0, 4.0])
    pts = frustum_points(pixel_xy, depths, K, E)  # (D, 1, 1, 3)
    for d in range(depths.numel()):
        p = pts[d, 0, 0]
        # Center pixel at depth d -> ego (d, 0, 0) for the origin camera.
        assert abs(p[0].item() - depths[d].item()) < 1e-4
        assert abs(p[1].item()) < 1e-4
        assert abs(p[2].item()) < 1e-4


def test_right_pixel_maps_to_negative_y():
    # A pixel right of center (u > cx) sees content to the ego right, which is ego -y.
    E, _ = _origin_forward_E()
    K = _K()
    cx = cy = K[0, 2].item()
    pixel_xy = torch.tensor([[[cx + 8.0, cy]]])  # right of center
    depths = torch.tensor([4.0])
    pts = frustum_points(pixel_xy, depths, K, E)
    assert pts[0, 0, 0, 1].item() < -1e-3        # ego y < 0 (to the right)


def test_roundtrip_to_pixels():
    # Project the produced ego points back with the same K, E and recover the pixel centers.
    E, _ = _origin_forward_E()
    K = _K()
    Hf = Wf = 4
    stride = 8
    vs = (torch.arange(Hf, dtype=torch.float32) + 0.5) * stride
    us = (torch.arange(Wf, dtype=torch.float32) + 0.5) * stride
    gv, gu = torch.meshgrid(vs, us, indexing="ij")
    pixel_xy = torch.stack([gu, gv], dim=-1)     # (Hf, Wf, 2)
    depths = torch.tensor([2.0, 5.0])
    pts = frustum_points(pixel_xy, depths, K, E)  # (D, Hf, Wf, 3)

    rig = CameraRig({"c": K}, {"c": E})
    for d in range(depths.numel()):
        ego = pts[d].reshape(-1, 3)
        px, valid = rig.world_to_pixel("c", ego)
        assert valid.all()
        recovered = px.reshape(Hf, Wf, 2)
        assert torch.allclose(recovered, pixel_xy, atol=1e-3)
