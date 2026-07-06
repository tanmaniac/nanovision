"""The 3D Gaussian representation: parameters, the quaternion rotation, and the covariance.

A scene is a cloud of 3D Gaussians. Each Gaussian has a center mean in world space, an
anisotropic shape (a 3x3 covariance), an opacity, and a color. The covariance is stored
in a factored form, scale plus rotation, so it stays a valid (symmetric positive
semi-definite) covariance for every parameter value and is differentiable everywhere the
quaternion is nonzero. Storing log-scales keeps the scales positive under unconstrained
gradient descent.

See the "The covariance factorization" section of the README for the factored form.

Assignment-local: A10's renderer is not reused downstream, so nothing here is exported
through a nanovision shim. Import these names bare.
"""

import torch
from torch import Tensor


def quat_to_rotmat(q: Tensor) -> Tensor:
    """Convert quaternions to rotation matrices.

    The quaternion is (w, x, y, z), real part first, and may have any nonzero magnitude.
    See the "The covariance factorization" section of the README.

    Args:
        q: (N, 4) quaternions, any nonzero magnitude.

    Returns:
        (N, 3, 3) rotation matrices.
    """
    raise NotImplementedError("normalize q, then build the 3x3 rotation matrix")


def build_covariance_3d(quats: Tensor, log_scales: Tensor) -> Tensor:
    """Build per-Gaussian 3D covariances from quaternions and log-scales.

    The factorization keeps the covariance symmetric positive semi-definite for any
    parameter value and differentiable, the standard pure-PyTorch pattern (no
    eigendecomposition, no constrained optimization). See the "The covariance
    factorization" section of the README.

    Args:
        quats: (N, 4) quaternions.
        log_scales: (N, 3) log of the per-axis standard deviations.

    Returns:
        (N, 3, 3) covariance matrices.
    """
    raise NotImplementedError("build the 3D covariance from quats and log_scales")


class GaussianModel(torch.nn.Module):
    """The optimizable parameters of a 3D Gaussian cloud.

    Holds the raw (pre-activation) parameters. Opacity and color are stored as logits and
    squashed with sigmoid so they stay in [0, 1] under unconstrained gradient descent:

        opacities = sigmoid(opacity_logits)   in [0, 1]
        colors    = sigmoid(color_logits)     in [0, 1]^3

    Storing an RGB color and squashing with sigmoid is a toy stand-in for a
    view-independent color, the simplest case where appearance does not depend on viewing
    direction. This is NOT degree-0 spherical harmonics as an equality: true degree-0 SH
    stores one coefficient per channel scaled by the constant basis function
    Y_0 = 1 / (2 * sqrt(pi)); the sigmoid is just a convenient [0, 1] map for the toy.
    View-dependent color (degree-1 and higher SH, which captures specular highlights that
    change with the camera) is README context only and not implemented here.

    Parameters:
        means: (N, 3) Gaussian centers in world space.
        log_scales: (N, 3) log per-axis standard deviations.
        quats: (N, 4) rotation quaternions (w, x, y, z).
        opacity_logits: (N,) pre-sigmoid opacity.
        color_logits: (N, 3) pre-sigmoid RGB.
    """

    def __init__(self, means: Tensor, log_scales: Tensor, quats: Tensor,
                 opacity_logits: Tensor, color_logits: Tensor):
        super().__init__()
        self.means = torch.nn.Parameter(means)
        self.log_scales = torch.nn.Parameter(log_scales)
        self.quats = torch.nn.Parameter(quats)
        self.opacity_logits = torch.nn.Parameter(opacity_logits)
        self.color_logits = torch.nn.Parameter(color_logits)

    @property
    def opacities(self) -> Tensor:
        """(N,) opacities in [0, 1]."""
        return torch.sigmoid(self.opacity_logits)

    @property
    def colors(self) -> Tensor:
        """(N, 3) colors in [0, 1]."""
        return torch.sigmoid(self.color_logits)

    def covariance_3d(self) -> Tensor:
        """(N, 3, 3) world-space covariances from the current quats and log_scales."""
        return build_covariance_3d(self.quats, self.log_scales)

    @classmethod
    def random_init(cls, n: int, *, spread: float = 1.0, init_scale: float = 0.1,
                    seed: int = 0, device=None, dtype=torch.float32) -> "GaussianModel":
        """Initialize N Gaussians in a cube centered on the world origin.

        The toy scene (nerf_synthetic_scene) puts its object at the world origin and rings
        the cameras around it, so the Gaussians start uniform in a cube of half-width
        `spread` centered at the origin, where the object actually is. Scales start
        isotropic at `init_scale`, rotations at identity (quaternion (1, 0, 0, 0)),
        opacities mid-range (logit 0 -> 0.5), colors gray (logit 0 -> 0.5). Used by the viz
        fit and the overfit test as a starting point.
        """
        g = torch.Generator(device="cpu").manual_seed(seed)
        means = (torch.rand(n, 3, generator=g, dtype=dtype) - 0.5) * 2.0 * spread
        log_scales = torch.full((n, 3), float(torch.log(torch.tensor(init_scale))), dtype=dtype)
        quats = torch.zeros(n, 4, dtype=dtype)
        quats[:, 0] = 1.0
        opacity_logits = torch.zeros(n, dtype=dtype)
        color_logits = torch.zeros(n, 3, dtype=dtype)
        model = cls(means, log_scales, quats, opacity_logits, color_logits)
        if device is not None:
            model = model.to(device)
        return model
