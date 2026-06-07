"""Task 4: flat-ground IPM into the BEV grid, and its documented breakage."""

import torch

from geometry import BEVGrid, CameraRig, ipm_to_bev
from synthetic import make_ground_camera


def _rig_and_grid():
    K, extr = make_ground_camera(height=1.5, pitch_deg=-15.0, w=400, h=224, fx=300.0)
    rig = CameraRig({"cam": K}, {"cam": extr}, {"cam": (400, 224)})
    grid = BEVGrid(x_min=0.0, x_max=40.0, y_min=-20.0, y_max=20.0, resolution=0.5)
    return rig, grid


def test_ground_marker_lands_in_expected_cell():
    rig, grid = _rig_and_grid()
    # Drive the test from a chosen cell center: project its ground point to a
    # pixel, paint a marker there, then warp back and confirm that cell lights up.
    ti, tj = 16, 44  # ego x ~ 8.25 m, y ~ 2.25 m
    gp = grid.cell_centers()[ti, tj]
    pt = torch.tensor([[gp[0], gp[1], 0.0]])
    px, valid = rig.world_to_pixel("cam", pt)
    assert bool(valid[0])
    u, v = int(round(px[0, 0].item())), int(round(px[0, 1].item()))
    img = torch.zeros(1, 224, 400)
    img[0, v, u] = 1.0
    bev = ipm_to_bev({"cam": img}, rig, grid, ground_z=0.0)
    assert bev.shape == (1, grid.nx, grid.ny)
    peak = bev[0].flatten().argmax()
    pi, pj = int(peak // grid.ny), int(peak % grid.ny)
    assert (pi, pj) == (ti, tj), f"expected peak at {(ti, tj)}, got {(pi, pj)}"
    assert bev[0, pi, pj] > 0.3


def test_flat_ground_breaks_for_elevated_points():
    # The flat-ground homography maps an above-ground feature to a BEV cell
    # beyond its true footprint. Show it: a feature at height h projects to the
    # same pixel as a ground point that is farther from the camera.
    rig, grid = _rig_and_grid()
    footprint = torch.tensor([[8.0, 0.0, 0.0]])  # true ground location
    elevated = torch.tensor([[8.0, 0.0, 1.7]])  # a head, 1.7 m up, same footprint
    px_elev, _ = rig.world_to_pixel("cam", elevated)
    # Where on the ground does that pixel land? Find the ground point whose
    # projection matches the elevated pixel by scanning forward distance.
    xs = torch.linspace(8.0, 40.0, 2000)
    ground = torch.stack([xs, torch.zeros_like(xs), torch.zeros_like(xs)], dim=-1)
    px_ground, _ = rig.world_to_pixel("cam", ground)
    j = (px_ground[:, 1] - px_elev[0, 1]).abs().argmin()
    apparent_x = xs[j].item()
    # The head is painted well beyond its 8 m footprint, away from the camera.
    assert apparent_x > footprint[0, 0].item() + 2.0, (
        f"elevated point should map past its footprint; got apparent_x={apparent_x:.1f}"
    )
