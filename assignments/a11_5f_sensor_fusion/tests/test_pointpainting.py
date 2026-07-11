"""PointPainting geometry: the painted score equals the seg score at the projected pixel.

Implementation-independent checks built from the shared geometry primitives:
- an in-frame point is painted with exactly the seg score at its projected pixel;
- a behind-camera point (camera-frame z <= 0) is painted with the zero (background) vector, and
  its coordinates are left untouched;
- an in-front point that projects outside the image is also painted zero.
"""

import torch

from config import FusionConfig
from fusion import paint_points

from nanovision.data import toy
from nanovision.geometry import apply_transform, project_points


def _expected_pixel(point_ego, K, E):
    p_cam = apply_transform(E, point_ego[None])          # (1, 3) camera frame
    px = project_points(p_cam, K)[0]                     # (u, v)
    u = int(px[0].round().item())
    v = int(px[1].round().item())
    return u, v, float(p_cam[0, 2])


def test_in_frame_point_gets_pixel_score():
    torch.manual_seed(0)
    cfg = FusionConfig()
    scene = toy.bev_fusion_scene(seed=0)
    K, E, seg = scene["K"], scene["E"], scene["seg_scores"]
    H, W = cfg.img, cfg.img

    # A point on the ground plane, well in front of the forward camera.
    point = torch.tensor([4.0, 0.5, 0.5])
    u, v, z = _expected_pixel(point, K, E)
    assert z > 0 and 0 <= u < W and 0 <= v < H, "test point must be in frame"

    painted = paint_points(point[None], seg, K, E, (H, W))
    assert painted.shape == (1, 3 + cfg.n_classes)
    assert torch.allclose(painted[0, :3], point)
    assert torch.allclose(painted[0, 3:], seg[:, v, u], atol=1e-5)


def test_behind_camera_point_gets_zero():
    cfg = FusionConfig()
    scene = toy.bev_fusion_scene(seed=0)
    K, E, seg = scene["K"], scene["E"], scene["seg_scores"]

    # Ego x < 0 is behind the forward camera (camera-frame z = ego x <= 0).
    point = torch.tensor([-3.0, 0.0, 0.5])
    _, _, z = _expected_pixel(point, K, E)
    assert z <= 0, "test point must be behind the camera"

    painted = paint_points(point[None], seg, K, E, (cfg.img, cfg.img))
    assert torch.allclose(painted[0, :3], point)
    assert torch.count_nonzero(painted[0, 3:]) == 0


def test_out_of_image_point_gets_zero():
    cfg = FusionConfig()
    scene = toy.bev_fusion_scene(seed=0)
    K, E, seg = scene["K"], scene["E"], scene["seg_scores"]

    # In front of the camera but far to the side, so it projects outside the image.
    point = torch.tensor([2.0, 20.0, 0.5])
    u, v, z = _expected_pixel(point, K, E)
    assert z > 0 and not (0 <= u < cfg.img and 0 <= v < cfg.img)

    painted = paint_points(point[None], seg, K, E, (cfg.img, cfg.img))
    assert torch.count_nonzero(painted[0, 3:]) == 0
