"""Task 1: pinhole project / unproject. Run first."""

import torch

from nanovision.gradcheck import check_gradients
from geometry import project_points, unproject


def _K():
    return torch.tensor([[500.0, 0.0, 200.0], [0.0, 500.0, 112.0], [0.0, 0.0, 1.0]])


def test_project_reference_value():
    K = _K()
    # A point at (1, 2, 4) projects to (fx*1/4 + cx, fy*2/4 + cy) = (325, 362).
    pts = torch.tensor([[1.0, 2.0, 4.0]])
    px = project_points(pts, K)
    assert px.shape == (1, 2)
    assert torch.allclose(px[0], torch.tensor([325.0, 362.0]), atol=1e-4)


def test_project_center_on_axis():
    K = _K()
    # A point on the +z axis lands on the principal point for any depth.
    px = project_points(torch.tensor([[0.0, 0.0, 7.0]]), K)
    assert torch.allclose(px[0], torch.tensor([200.0, 112.0]), atol=1e-4)


def test_roundtrip_unproject_project():
    K = _K()
    torch.manual_seed(0)
    pts = torch.randn(32, 3)
    pts[:, 2] = pts[:, 2].abs() + 1.0  # in front of the camera
    px = project_points(pts, K)
    rt = unproject(px, pts[:, 2], K)
    assert torch.allclose(rt, pts, atol=1e-4)


def test_project_gradcheck():
    K = _K().double()

    class Proj(torch.nn.Module):
        def forward(self, pts):
            return project_points(pts, K)

    pts = torch.randn(5, 3, dtype=torch.double)
    pts[:, 2] = pts[:, 2].abs() + 2.0
    assert check_gradients(Proj(), (pts,))
