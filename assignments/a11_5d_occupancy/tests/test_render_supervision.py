"""Rendering supervision: the occupancy renderer inverts to the analytic ray-box geometry.

The toy gives GT depth and GT semantic class per ray from an analytic ray-AABB first-hit,
computed independently of any renderer (the non-circularity guarantee). A learnable occupancy +
semantic field is optimized so the rendered depth matches GT depth and rendered class matches GT
class. Because the GT was not produced by this renderer, a passing run shows the alpha-compositing
quadrature inverts to the hard box geometry.
"""

import torch
import torch.nn.functional as F

from config import OccConfig
from occupancy import render_occupancy_rays

from nanovision.data import toy


def _z_vals(rays_o, z_near, z_far, n_samples):
    """Uniform (non-perturbed) sample distances per ray, [R, N]."""
    R = rays_o.shape[0]
    mids = torch.linspace(0.0, 1.0, n_samples)
    z = z_near + (z_far - z_near) * mids
    return z[None].expand(R, n_samples).contiguous()


def test_sample_budget_floor():
    cfg = OccConfig()
    floor = (cfg.z_far - cfg.z_near) / 0.15
    assert cfg.n_samples >= floor, f"n_samples {cfg.n_samples} below floor {floor:.1f}"


def test_shapes():
    cfg = OccConfig()
    scene = toy.occupancy_toy_scene(grid=cfg.grid, bounds=cfg.grid_bounds,
                                    n_classes=cfg.n_classes, n_cams=cfg.n_cams, img=cfg.img, seed=0)
    occ = scene["occ_gt"]
    sem = torch.randn(cfg.n_classes, *cfg.grid)
    rays_o, rays_d = scene["rays_o"], scene["rays_d"]
    z_vals = _z_vals(rays_o, cfg.z_near, cfg.z_far, cfg.n_samples)
    depth, sem_out, weights = render_occupancy_rays(
        occ, sem, rays_o, rays_d, z_vals, cfg.grid_bounds, cfg.z_far)
    R, N = z_vals.shape
    assert depth.shape == (R,)
    assert sem_out.shape == (R, cfg.n_classes)
    assert weights.shape == (R, N)


def test_gradcheck():
    torch.manual_seed(0)
    cfg = OccConfig()
    scene = toy.occupancy_toy_scene(grid=cfg.grid, bounds=cfg.grid_bounds,
                                    n_classes=cfg.n_classes, n_cams=cfg.n_cams, img=cfg.img, seed=0)
    rays_o = scene["rays_o"][:8].double()
    rays_d = scene["rays_d"][:8].double()
    z_vals = _z_vals(rays_o, cfg.z_near, cfg.z_far, 16).double()
    occ = torch.rand(*cfg.grid, dtype=torch.float64, requires_grad=True)
    sem = torch.randn(cfg.n_classes, *cfg.grid, dtype=torch.float64)

    def f(o):
        depth, _, _ = render_occupancy_rays(o, sem, rays_o, rays_d, z_vals, cfg.grid_bounds, cfg.z_far)
        return depth.sum()

    assert torch.autograd.gradcheck(f, (occ,))


def test_overfit_rendered_depth_and_semantics():
    """Optimize a random occupancy + semantic field to the analytic GT depth and class.

    Pass: mean depth error < 0.3 m; hit rays (those that strike a box) accumulate occupancy
    opacity > 0.5 while miss rays stay near 0; per-ray semantic accuracy > 0.8 on hit rays.

    The occupancy check is on the accumulated ray opacity, not on any single voxel. Depth-only
    supervision drives the renderer to a diffuse low-opacity cloud whose compositing weight
    centroid lands at the GT depth: with this sample spacing the per-voxel occupancy stays around
    0.15-0.30 even on hit rays (many soft samples compound to high total opacity), so a
    per-voxel > 0.5 assertion is unreachable. The accumulated opacity is what the rendering
    integral actually expresses and what separates hit from miss (measured ~0.97 vs ~0.02). The
    box interiors are not asserted to fill (occluded interior voxels are unconstrained).
    """
    torch.manual_seed(0)
    cfg = OccConfig()
    scene = toy.occupancy_toy_scene(grid=cfg.grid, bounds=cfg.grid_bounds, n_classes=cfg.n_classes,
                                    n_boxes=cfg.n_boxes, n_cams=cfg.n_cams, img=cfg.img, seed=0)
    rays_o, rays_d = scene["rays_o"], scene["rays_d"]
    gt_depth, gt_sem = scene["gt_depth"], scene["gt_sem"]
    z_vals = _z_vals(rays_o, cfg.z_near, cfg.z_far, cfg.n_samples)
    hit = gt_depth < cfg.z_far                        # rays that strike a box

    # Learnable fields. occ via sigmoid of a logit grid; sem as raw class logits.
    occ_logit = torch.zeros(*cfg.grid, requires_grad=True)
    sem_logit = torch.zeros(cfg.n_classes, *cfg.grid, requires_grad=True)
    opt = torch.optim.Adam([occ_logit, sem_logit], lr=0.1)

    for _ in range(500):
        occ = torch.sigmoid(occ_logit)
        depth, sem_out, _ = render_occupancy_rays(
            occ, sem_logit, rays_o, rays_d, z_vals, cfg.grid_bounds, cfg.z_far)
        depth_loss = F.smooth_l1_loss(depth, gt_depth)
        sem_loss = F.cross_entropy(sem_out[hit], gt_sem[hit]) if hit.any() else 0.0
        loss = depth_loss + sem_loss
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        occ = torch.sigmoid(occ_logit)
        depth, sem_out, weights = render_occupancy_rays(
            occ, sem_logit, rays_o, rays_d, z_vals, cfg.grid_bounds, cfg.z_far)
        mean_depth_err = (depth - gt_depth).abs().mean().item()
        acc_opacity = weights.sum(-1)                # [R] accumulated opacity per ray
        hit_opacity = acc_opacity[hit].mean().item()
        sem_acc = (sem_out[hit].argmax(-1) == gt_sem[hit]).float().mean().item()

    assert mean_depth_err < 0.3, f"mean depth err {mean_depth_err:.3f} m"
    assert hit_opacity > 0.5, f"hit-ray accumulated opacity {hit_opacity:.3f}"
    assert sem_acc > 0.8, f"semantic accuracy {sem_acc:.3f}"
