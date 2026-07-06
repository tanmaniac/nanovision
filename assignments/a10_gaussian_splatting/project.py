"""EWA projection: a 3D Gaussian projects to a 2D Gaussian in image space.

The forward rasterizer needs each 3D Gaussian's footprint in pixels. A 3D Gaussian does
not project to an exact 2D Gaussian under perspective (the projection is nonlinear), so
the EWA (elliptical weighted average) splatting of Zwicker et al. linearizes the
projection by its Jacobian at the Gaussian's mean and pushes the covariance through that
linear map.

Convention: OpenCV camera frame, +z forward, matching nanovision.geometry. The pinhole
projection is the same project_points the camera-geometry assignment uses, so the Jacobian
below matches d(project_points)/d(x,y,z) at the mean by construction.

See the "The EWA projection" section of the README for the projection.

Assignment-local. Import bare.
"""

import torch
from torch import Tensor

from nanovision.geometry import apply_transform, project_points


def perspective_jacobian(means_cam: Tensor, K: Tensor) -> Tensor:
    """The 2x3 Jacobian of the pinhole projection at each camera-space mean.

    This is d(project_points)/d(pts_cam) evaluated at each camera-space mean. See the
    "The EWA projection" section of the README.

    Args:
        means_cam: (N, 3) Gaussian means in the camera frame (+z forward).
        K: (3, 3) intrinsic matrix.

    Returns:
        (N, 2, 3) Jacobians.
    """
    raise NotImplementedError("build the 2x3 perspective Jacobian per Gaussian")


def project_cov_to_2d(cov3d: Tensor, W: Tensor, J: Tensor, *, dilation: float = 0.3) -> Tensor:
    """Project 3D covariances to 2D screen-space covariances (the EWA step).

    W is the world-to-camera ROTATION (3x3) and J is the 2x3 perspective Jacobian at the
    camera-space mean. The translation of the world-to-camera transform shifts the mean,
    not the covariance, so only the rotation enters here.

    The dilation adds `dilation` (px^2) to the two diagonal entries of the 2x2 result. It
    keeps the covariance invertible and guarantees each Gaussian covers at least about a
    pixel, so the closed-form 2x2 inverse used by the rasterizer is safe (the determinant
    is bounded away from zero). The original 3D Gaussian splatting uses dilation ~= 0.3.

    See the "The EWA projection" section of the README.

    Args:
        cov3d: (N, 3, 3) world-space covariances.
        W: (3, 3) world-to-camera rotation (the 3x3 block of invert_transform(c2w)).
        J: (N, 2, 3) perspective Jacobians.
        dilation: px^2 added to the 2x2 diagonal.

    Returns:
        (N, 2, 2) screen-space covariances.
    """
    raise NotImplementedError("project the 3D covariance into 2D image space")


def project_gaussians(model, K: Tensor, w2c: Tensor, *, dilation: float = 0.3):
    """Project a GaussianModel into one camera. Provided wiring around the two holes.

    Transforms means to camera space, projects to 2D pixel means and depths with
    project_points, builds the 3D covariances, and runs the EWA step.

    Args:
        model: a GaussianModel.
        K: (3, 3) intrinsic.
        w2c: (4, 4) world-to-camera transform (invert_transform of the c2w pose).
        dilation: px^2 covariance dilation passed to project_cov_to_2d.

    Returns:
        means2d: (N, 2) projected pixel centers.
        cov2d: (N, 2, 2) screen-space covariances.
        depths: (N,) camera-space z of each Gaussian (the depth sort key).
    """
    means_cam = apply_transform(w2c, model.means)        # (N, 3)
    means2d = project_points(means_cam, K)               # (N, 2)
    depths = means_cam[:, 2]                              # (N,)
    cov3d = model.covariance_3d()                         # (N, 3, 3)
    W = w2c[:3, :3]                                       # world-to-camera rotation
    J = perspective_jacobian(means_cam, K)               # (N, 2, 3)
    cov2d = project_cov_to_2d(cov3d, W, J, dilation=dilation)
    return means2d, cov2d, depths
