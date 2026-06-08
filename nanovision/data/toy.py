"""Synthetic datasets: linear regression (A0), copy/sort and a tiny char corpus
(used by A1). These are provided boilerplate, never the taught mechanism.
"""

import torch
from torch import Tensor


def linreg_batch(n: int = 64, d: int = 8, noise: float = 0.0, seed: int = 0,
                 device: str = "cpu") -> tuple[Tensor, Tensor]:
    """A single linear-regression batch with a fixed ground-truth (w, b).

    Returns (X, y) with X of shape (n, d) and y of shape (n, 1). With noise=0 the
    batch is exactly fit by a Linear(d, 1), so the overfit test should reach ~0.
    """
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g)
    w = torch.randn(d, 1, generator=g)
    b = torch.randn(1, generator=g)
    y = X @ w + b
    if noise:
        y = y + noise * torch.randn(n, 1, generator=g)
    return X.to(device), y.to(device)


def copy_task(batch: int = 32, seq: int = 10, vocab: int = 16, seed: int = 0,
              device: str = "cpu") -> tuple[Tensor, Tensor]:
    """Copy task: target equals input. Token 0 is reserved (pad/bos)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(1, vocab, (batch, seq), generator=g)
    return x.to(device), x.clone().to(device)


def sort_task(batch: int = 32, seq: int = 10, vocab: int = 16, seed: int = 0,
              device: str = "cpu") -> tuple[Tensor, Tensor]:
    """Sort task: target is the sorted input sequence."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(1, vocab, (batch, seq), generator=g)
    y, _ = torch.sort(x, dim=-1)
    return x.to(device), y.to(device)


TINY_CORPUS = (
    "the quick brown fox jumps over the lazy dog. "
    "a transformer attends to a set of tokens and gathers by content. "
    "positional encoding reintroduces order into the permutation invariant gather. "
)


class CharTokenizer:
    """Minimal character-level tokenizer over a fixed corpus."""

    def __init__(self, text: str = TINY_CORPUS):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for c, i in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)


def char_lm_batch(seq: int = 64, batch: int = 16, text: str = TINY_CORPUS,
                  seed: int = 0, device: str = "cpu"):
    """Next-character LM batch: (x, y) where y is x shifted by one position."""
    tok = CharTokenizer(text)
    ids = torch.tensor(tok.encode(text * (1 + (seq * batch) // len(text))))
    g = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, len(ids) - seq - 1, (batch,), generator=g)
    x = torch.stack([ids[s:s + seq] for s in starts])
    y = torch.stack([ids[s + 1:s + seq + 1] for s in starts])
    return x.to(device), y.to(device), tok


def video_batch(batch: int = 8, n_frames: int = 6, size: int = 16, channels: int = 3,
                n_blobs: int = 3, seed: int = 0, device: str = "cpu") -> Tensor:
    """A batch of tiny synthetic videos: independently moving Gaussian blobs (A3.5).

    Returns (B, C, T, H, W). Each clip has `n_blobs` Gaussian blobs, each with its own
    random start position, constant velocity, channel, width, and amplitude, on a zero
    background. The clip is smooth and fully determined by those parameters, so a model
    can overfit one batch to near-zero reconstruction error.

    The blobs move INDEPENDENTLY with different velocities, so a clip is not a single
    global low-rank trajectory. That matters for the video MAE: a masked spatiotemporal
    tube cannot be recovered by copying one visible spatial column, which is what makes
    tube masking a meaningful task here rather than a copy-the-neighbor shortcut.
    """
    g = torch.Generator().manual_seed(seed)
    vid = torch.zeros(batch, channels, n_frames, size, size)
    ys = torch.arange(size).float().view(size, 1)
    xs = torch.arange(size).float().view(1, size)
    # Bound the per-frame velocity so a blob stays mostly in frame across the clip.
    vmax = size / (2.0 * n_frames)
    for b in range(batch):
        for _ in range(n_blobs):
            x0 = torch.rand(1, generator=g).item() * (size - 1)
            y0 = torch.rand(1, generator=g).item() * (size - 1)
            vx = (torch.rand(1, generator=g).item() - 0.5) * 2.0 * vmax
            vy = (torch.rand(1, generator=g).item() - 0.5) * 2.0 * vmax
            c = int(torch.randint(0, channels, (1,), generator=g).item())
            sigma = 1.5 + torch.rand(1, generator=g).item()
            amp = 0.6 + 0.4 * torch.rand(1, generator=g).item()
            for t in range(n_frames):
                cx = x0 + vx * t
                cy = y0 + vy * t
                blob = amp * torch.exp(-(((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma ** 2)))
                vid[b, c, t] += blob
    return vid.clamp(0.0, 1.0).to(device)


def image_text_batch(batch: int = 8, size: int = 16, channels: int = 3, n_classes: int = 4,
                     n_attrs: int = 4, vocab_size: int = 32, max_len: int = 8, seed: int = 0,
                     device: str = "cpu") -> tuple[Tensor, Tensor]:
    """A batch of paired (image, caption) examples for contrastive learning (A4).

    Returns (images (B, C, H, W), token_ids (B, L)). Each pair draws a latent class c
    and a nuisance attribute a, with the (c, a) combinations chosen distinct within the
    batch. The class sets the blob's channel (color) and the attribute sets its
    position; the caption is [class_token(c), attr_token(a), EOS, pad...]. Because
    n_classes < batch, classes repeat across pairs, so two same-class pairs are in-batch
    "negatives" yet semantically close (the false-negative pathology of in-batch
    contrastive learning). EOS is the largest token id (vocab_size - 1), so the CLIP
    argmax-of-token-ids EOS-pooling trick lands on it.

    Token id layout: 0 = pad, 1..n_classes = class tokens, n_classes+1..n_classes+n_attrs
    = attribute tokens, vocab_size-1 = EOS.
    """
    g = torch.Generator().manual_seed(seed)
    eos = vocab_size - 1
    assert n_classes + n_attrs + 1 < vocab_size, "vocab too small for the token layout"
    assert batch <= n_classes * n_attrs, "need at least `batch` distinct (class, attr) combos"

    # Distinct (class, attr) combinations, classes repeating since n_classes < batch.
    combos = [(c, a) for c in range(n_classes) for a in range(n_attrs)]
    perm = torch.randperm(len(combos), generator=g)[:batch]
    pairs = [combos[i] for i in perm.tolist()]

    images = torch.zeros(batch, channels, size, size)
    tokens = torch.zeros(batch, max_len, dtype=torch.long)
    ys = torch.arange(size).float().view(size, 1)
    xs = torch.arange(size).float().view(1, size)
    margin = size / 6.0
    for i, (c, a) in enumerate(pairs):
        ch = c % channels
        # Attribute -> a position on a small grid of cell centers.
        side = max(1, int(round(n_attrs ** 0.5)))
        ax, ay = a % side, a // side
        cx = margin + (size - 2 * margin) * (ax / max(1, side - 1) if side > 1 else 0.5)
        cy = margin + (size - 2 * margin) * (ay / max(1, side - 1) if side > 1 else 0.5)
        sigma = 1.5 + 0.5 * (c / max(1, n_classes - 1))
        blob = torch.exp(-(((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma ** 2)))
        images[i, ch] += blob
        tokens[i, 0] = 1 + c                  # class token
        tokens[i, 1] = 1 + n_classes + a      # attribute token
        tokens[i, 2] = eos                    # EOS (largest id); rest stay pad (0)
    return images.clamp(0.0, 1.0).to(device), tokens.to(device)


def diffusion_image_batch(n: int = 16, num_classes: int = 3, size: int = 16,
                          channels: int = 1, seed: int = 0,
                          device: str = "cpu") -> tuple[Tensor, Tensor]:
    """A batch of tiny shape images with class labels, for diffusion (A5).

    Returns (images (B, C, H, W) in [-1, 1], labels (B,) in 0..num_classes-1). Each class
    is a distinct shape - 0: filled disk, 1: axis-aligned square, 2: plus/cross - drawn at
    a randomly jittered center and size. The class label does NOT pin down position or
    size, so the wide intra-class spread gives classifier-free guidance a real
    fidelity/diversity trade-off to show: high guidance sharpens the canonical shape and
    collapses the position/size variation. Deterministic per seed, so overfitting one
    batch to near-zero is exact. Data is scaled to [-1, 1] (background -1, shape +1), the
    range diffusion models assume.
    """
    g = torch.Generator().manual_seed(seed)
    imgs = torch.full((n, channels, size, size), -1.0)
    labels = torch.randint(0, num_classes, (n,), generator=g)
    ys = torch.arange(size).float().view(size, 1)
    xs = torch.arange(size).float().view(1, size)
    for i in range(n):
        cls = int(labels[i])
        # Half-size (radius) jittered wide; center jittered so the shape stays in frame.
        r = 2.5 + torch.rand(1, generator=g).item() * 3.5     # 2.5 .. 6.0
        cx = r + torch.rand(1, generator=g).item() * (size - 2 * r)
        cy = r + torch.rand(1, generator=g).item() * (size - 2 * r)
        dx = (xs - cx).abs()
        dy = (ys - cy).abs()
        if cls == 0:                                          # filled disk
            mask = ((xs - cx) ** 2 + (ys - cy) ** 2) <= r ** 2
        elif cls == 1:                                        # axis-aligned square
            mask = (dx <= r) & (dy <= r)
        else:                                                 # plus / cross
            arm = r / 2.5
            mask = ((dx <= arm) & (dy <= r)) | ((dy <= arm) & (dx <= r))
        imgs[i, 0][mask] = 1.0
    return imgs.to(device), labels.to(device)


def eight_gaussians(n: int = 256, scale: float = 4.0, std: float = 0.25,
                    generator: torch.Generator | None = None, device: str = "cpu") -> Tensor:
    """A 2D mixture of eight Gaussians on a ring, for flow matching (A6).

    Returns (n, 2). Eight well-separated modes at the corners and edge-midpoints of a
    square ring; the clear mode structure makes optimal-transport coupling's trajectory
    straightening legible at batch size ~256.
    """
    g = generator or torch.Generator().manual_seed(0)
    centers = scale * torch.tensor([
        [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0],
        [0.7071, 0.7071], [0.7071, -0.7071], [-0.7071, 0.7071], [-0.7071, -0.7071],
    ])
    idx = torch.randint(0, 8, (n,), generator=g)
    return (centers[idx] + std * torch.randn(n, 2, generator=g)).to(device)


def two_moons(n: int = 256, noise: float = 0.1,
              generator: torch.Generator | None = None, device: str = "cpu") -> Tensor:
    """The two-moons 2D distribution, for flow matching (A6).

    Returns (n, 2). Two interleaving half-circle manifolds; a non-Gaussian-manifold target
    that complements the eight-Gaussians mixture.
    """
    g = generator or torch.Generator().manual_seed(0)
    n0 = n // 2
    n1 = n - n0
    t0 = torch.rand(n0, generator=g) * torch.pi
    upper = torch.stack([torch.cos(t0), torch.sin(t0)], dim=1)
    t1 = torch.rand(n1, generator=g) * torch.pi
    lower = torch.stack([1.0 - torch.cos(t1), 1.0 - torch.sin(t1) - 0.5], dim=1)
    pts = torch.cat([upper, lower], dim=0)
    pts = pts + noise * torch.randn(n, 2, generator=g)
    return (pts * 2.0).to(device)        # scale up to a few units across


def nerf_synthetic_scene(
    n_views: int = 6,
    H: int = 16,
    W: int = 16,
    radius: float = 1.0,
    sphere_sigma: float = 8.0,
    cam_dist: float = 4.0,
    focal: float | None = None,
    bg: float = 1.0,
    seed: int = 0,
    device: str = "cpu",
) -> tuple[Tensor, Tensor, Tensor, float, float]:
    """A tiny posed-image scene of one colored solid sphere, for NeRF (A9).

    The object is a solid sphere of the given radius centered at the world origin,
    constant interior density `sphere_sigma`, and a smooth position-dependent color.
    Cameras sit on a horizontal ring at distance `cam_dist`, each looking at the
    origin in the OpenCV convention (camera +z points into the scene, +x right,
    +y down). This matches nanovision.geometry, not the original NeRF's OpenGL -z.

    Ground truth is rendered by the closed-form ray-sphere chord, NOT by
    volume_render, so a learner whose discretized renderer reproduces these pixels
    has shown the quadrature converges to the analytic Beer-Lambert integral rather
    than to its own renderer. For a ray that enters the sphere over an interior
    chord of length l, the exact transmittance through constant density is
    exp(-sphere_sigma * l), giving alpha = 1 - exp(-sphere_sigma * l) and a
    composited pixel alpha * c_sphere + (1 - alpha) * bg. A ray that misses the
    sphere keeps the background. The hard sphere silhouette is the sharp feature
    the spectral-bias ablation needs (a raw-coordinate MLP blurs it).

    Args:
        n_views: number of cameras on the ring (the last one is the held-out view).
        H, W: image height and width in pixels.
        radius: sphere radius in world units.
        sphere_sigma: constant interior volume density.
        cam_dist: camera distance from the origin.
        focal: pinhole focal length in pixels; defaults to W (a ~53 degree FOV).
        bg: background gray level in [0, 1], applied to all three channels.
        seed: unused placeholder for API symmetry (the scene is deterministic).
        device: target device for the returned tensors.

    Returns:
        images: (n_views, H, W, 3) ground-truth pixels in [0, 1].
        poses: (n_views, 4, 4) camera-to-world transforms (OpenCV +z forward).
        K: (3, 3) shared pinhole intrinsic.
        near: float, a conservative near distance along the ray.
        far: float, a conservative far distance along the ray.
    """
    del seed  # deterministic per the fixed ring; kept for a uniform toy API
    f = float(focal) if focal is not None else float(W)
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    K = torch.tensor([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])

    # near/far bracket the sphere along every ray: cam_dist +- radius with margin.
    near = cam_dist - radius - 0.5
    far = cam_dist + radius + 0.5

    # A smooth color over the sphere surface, so the held-out view tests color too.
    def _sphere_color(p: Tensor) -> Tensor:
        # p: (..., 3) the entry point on the sphere; map normalized coords to RGB.
        d = p / radius
        r = 0.5 + 0.5 * d[..., 0]
        gch = 0.5 + 0.5 * d[..., 1]
        b = 0.5 + 0.5 * d[..., 2]
        return torch.stack([r, gch, b], dim=-1).clamp(0.0, 1.0)

    poses = []
    images = []
    for i in range(n_views):
        ang = 2.0 * torch.pi * i / n_views
        cam_pos = torch.tensor([cam_dist * torch.cos(torch.tensor(ang)),
                                0.0,
                                cam_dist * torch.sin(torch.tensor(ang))])
        # Look-at in OpenCV: forward = (target - eye) normalized is camera +z.
        forward = -cam_pos / cam_pos.norm()
        world_up = torch.tensor([0.0, -1.0, 0.0])  # OpenCV +y is down
        right = torch.cross(forward, world_up, dim=0)
        right = right / right.norm()
        down = torch.cross(forward, right, dim=0)
        R = torch.stack([right, down, forward], dim=1)  # columns are cam axes in world
        c2w = torch.eye(4)
        c2w[:3, :3] = R
        c2w[:3, 3] = cam_pos
        poses.append(c2w)

        # Camera-frame ray directions for every pixel, then rotate to world.
        vs, us = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                                torch.arange(W, dtype=torch.float32), indexing="ij")
        dirs_cam = torch.stack([(us - cx) / f, (vs - cy) / f, torch.ones_like(us)], dim=-1)
        dirs_world = dirs_cam @ R.T
        dirs_world = dirs_world / dirs_world.norm(dim=-1, keepdim=True)
        o = cam_pos  # (3,)

        # Closed-form ray-sphere chord: |o + t d|^2 = radius^2, d unit so a = 1.
        b_coef = 2.0 * (dirs_world * o).sum(-1)
        c_coef = (o * o).sum() - radius * radius
        disc = b_coef * b_coef - 4.0 * c_coef
        hit = disc > 0.0
        sqrt_disc = torch.sqrt(disc.clamp(min=0.0))
        t0 = (-b_coef - sqrt_disc) / 2.0
        t1 = (-b_coef + sqrt_disc) / 2.0
        chord = (t1 - t0).clamp(min=0.0) * hit.float()
        alpha = 1.0 - torch.exp(-sphere_sigma * chord)

        entry = o + t0[..., None] * dirs_world  # first intersection point
        c_sphere = _sphere_color(entry)
        img = alpha[..., None] * c_sphere + (1.0 - alpha[..., None]) * bg
        img = torch.where(hit[..., None], img, torch.full_like(img, bg))
        images.append(img.clamp(0.0, 1.0))

    images = torch.stack(images, dim=0).to(device)
    poses = torch.stack(poses, dim=0).to(device)
    return images, poses, K.to(device), float(near), float(far)
