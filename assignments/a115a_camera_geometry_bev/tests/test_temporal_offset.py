"""The lidar/camera temporal-offset exercise, on synthetic poses.

This uses only the Task 2 SE(3) primitives. The four-step lidar-to-camera chain
is:
    lidar sensor -> ego@lidar_time -> global -> ego@cam_time -> camera.
The naive shortcut reuses ego@lidar_time for the camera step, ignoring that the
vehicle moved between the lidar sweep and the image. At highway speed the ~50 ms
offset is ~1.5 m, large enough to misalign projected points.
"""

import torch

from geometry import (
    apply_transform,
    compose_transforms,
    invert_transform,
    make_transform,
)


def _chain(lidar_to_ego, ego_pose_lidar, ego_pose_cam, cam_extr):
    """Full four-step lidar-to-camera transform (cam_extr is ego-to-camera)."""
    return compose_transforms(
        cam_extr,
        invert_transform(ego_pose_cam),
        ego_pose_lidar,
        lidar_to_ego,
    )


def test_temporal_correction_shifts_by_ego_motion():
    R = torch.eye(3)
    lidar_to_ego = make_transform(R, torch.zeros(3))
    cam_extr = make_transform(R, torch.zeros(3))  # camera == ego frame
    # Ego at origin at lidar time, moved +1.5 m forward by camera time.
    ego_pose_lidar = make_transform(R, torch.tensor([0.0, 0.0, 0.0]))
    ego_pose_cam = make_transform(R, torch.tensor([1.5, 0.0, 0.0]))

    p_lidar = torch.tensor([[10.0, 0.0, 0.0]])  # a static point, 10 m ahead

    T_correct = _chain(lidar_to_ego, ego_pose_lidar, ego_pose_cam, cam_extr)
    # Naive: reuse the lidar-time ego pose for the camera step.
    T_naive = _chain(lidar_to_ego, ego_pose_lidar, ego_pose_lidar, cam_extr)

    pc = apply_transform(T_correct, p_lidar)
    pn = apply_transform(T_naive, p_lidar)

    # The vehicle drove 1.5 m toward the point, so the correct camera-frame
    # depth is 1.5 m smaller than the naive one.
    assert torch.allclose(pc[0], torch.tensor([8.5, 0.0, 0.0]), atol=1e-5)
    assert torch.allclose(pn[0], torch.tensor([10.0, 0.0, 0.0]), atol=1e-5)
    assert torch.allclose((pn - pc)[0], torch.tensor([1.5, 0.0, 0.0]), atol=1e-5)


def test_no_ego_motion_makes_naive_and_correct_agree():
    R = torch.eye(3)
    lidar_to_ego = make_transform(R, torch.zeros(3))
    cam_extr = make_transform(R, torch.zeros(3))
    ego = make_transform(R, torch.tensor([5.0, 1.0, 0.0]))  # same pose both times
    p_lidar = torch.tensor([[10.0, 2.0, 0.0]])
    T_correct = _chain(lidar_to_ego, ego, ego, cam_extr)
    T_naive = _chain(lidar_to_ego, ego, ego, cam_extr)
    assert torch.allclose(
        apply_transform(T_correct, p_lidar), apply_transform(T_naive, p_lidar)
    )
