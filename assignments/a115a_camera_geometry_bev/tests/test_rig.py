"""Task 3: CameraRig.world_to_pixel on a synthetic 4-camera rig."""

import torch

from synthetic import make_synthetic_rig


def test_front_point_visible_only_in_front():
    rig = make_synthetic_rig(w=400, h=224, fx=400.0)
    p = torch.tensor([[10.0, 0.0, 0.0]])  # 10 m ahead (+x)
    visible = {name: bool(rig.world_to_pixel(name, p)[1][0]) for name in rig.names}
    assert visible == {"front": True, "left": False, "back": False, "right": False}
    # On the front camera's optical axis -> projects to the image center.
    px, _ = rig.world_to_pixel("front", p)
    assert torch.allclose(px[0], torch.tensor([200.0, 112.0]), atol=1e-3)


def test_left_and_back_points():
    rig = make_synthetic_rig()
    left_pt = torch.tensor([[0.0, 10.0, 0.0]])
    back_pt = torch.tensor([[-10.0, 0.0, 0.0]])
    assert bool(rig.world_to_pixel("left", left_pt)[1][0])
    assert not bool(rig.world_to_pixel("front", left_pt)[1][0])
    assert bool(rig.world_to_pixel("back", back_pt)[1][0])
    assert not bool(rig.world_to_pixel("front", back_pt)[1][0])


def test_cube_projects_into_expected_cameras():
    rig = make_synthetic_rig()
    # A small cube placed 8 m ahead of the vehicle.
    base = torch.tensor([8.0, 0.0, 0.0])
    offs = torch.tensor(
        [[dx, dy, dz] for dx in (-0.5, 0.5) for dy in (-0.5, 0.5) for dz in (-0.5, 0.5)]
    )
    cube = base + offs
    px, valid = rig.world_to_pixel("front", cube)
    assert valid.all(), "all cube corners should be in the front camera"
    # No cube corner should be visible in the back camera.
    _, valid_back = rig.world_to_pixel("back", cube)
    assert not valid_back.any()


def test_cam_to_world_inverts_world_to_cam():
    rig = make_synthetic_rig()
    pts = torch.tensor([[8.0, 1.0, 0.5], [3.0, -2.0, 1.0]])
    cam = rig.world_to_cam("front", pts)
    back = rig.cam_to_world("front", cam)
    assert torch.allclose(back, pts, atol=1e-5)
