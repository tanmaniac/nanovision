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


def detection_batch(
    batch: int = 4,
    size: int = 32,
    num_classes: int = 3,
    max_objects: int = 2,
    seed: int = 0,
    device: str = "cpu",
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """A tiny detection toy: solid colored squares on a black background (A11).

    Each image holds 1 to max_objects axis-aligned squares. A square's color is one of
    num_classes fixed RGB colors, and the color is its class id. The box is the square's
    exact extent, returned in normalized cxcywh in [0, 1] (center x, center y, width,
    height, each divided by the image size). The number of objects varies per image, so
    boxes and labels are padded to max_objects and a boolean mask marks the valid rows.

    Returns:
        images (batch, 3, size, size) in [0, 1].
        boxes  (batch, max_objects, 4) cxcywh in [0, 1]; padded rows are zero.
        labels (batch, max_objects) long class ids in [0, num_classes); padded rows zero.
        mask   (batch, max_objects) bool; True where the (box, label) row is a real object.

    Deterministic per seed. Squares are placed without overlap so each pixel has one
    unambiguous class, and side lengths are a few pixels so the box is several patches wide.
    """
    g = torch.Generator().manual_seed(seed)
    # Fixed class colors: red, green, blue (extend by cycling if num_classes > 3).
    palette = torch.tensor([
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0],
    ])
    colors = palette[torch.arange(num_classes) % palette.shape[0]]

    images = torch.zeros(batch, 3, size, size)
    boxes = torch.zeros(batch, max_objects, 4)
    labels = torch.zeros(batch, max_objects, dtype=torch.long)
    mask = torch.zeros(batch, max_objects, dtype=torch.bool)

    side_min, side_max = max(4, size // 8), max(6, size // 4)
    for b in range(batch):
        n_obj = int(torch.randint(1, max_objects + 1, (1,), generator=g))
        placed: list[tuple[int, int, int, int]] = []  # (x0, y0, x1, y1) in pixels
        for j in range(n_obj):
            for _ in range(50):  # rejection-sample a non-overlapping placement
                side = int(torch.randint(side_min, side_max + 1, (1,), generator=g))
                x0 = int(torch.randint(0, size - side + 1, (1,), generator=g))
                y0 = int(torch.randint(0, size - side + 1, (1,), generator=g))
                x1, y1 = x0 + side, y0 + side
                if all(x1 <= px0 or x0 >= px1 or y1 <= py0 or y0 >= py1
                       for (px0, py0, px1, py1) in placed):
                    break
            placed.append((x0, y0, x1, y1))
            cls = int(torch.randint(0, num_classes, (1,), generator=g))
            images[b, :, y0:y1, x0:x1] = colors[cls][:, None, None]
            cx = (x0 + x1) / 2.0 / size
            cy = (y0 + y1) / 2.0 / size
            w = side / size
            h = side / size
            boxes[b, j] = torch.tensor([cx, cy, w, h])
            labels[b, j] = cls
            mask[b, j] = True

    return images.to(device), boxes.to(device), labels.to(device), mask.to(device)


def bev_toy_scene(
    n_vehicles: int = 3,
    bev_x: tuple[float, float] = (0.0, 8.0),
    bev_y: tuple[float, float] = (-8.0, 8.0),
    res: float = 1.0,
    img: int = 32,
    stride: int = 4,
    d_min: float = 1.0,
    d_max: float = 9.0,
    d_step: float = 1.0,
    focal: float | None = None,
    cam_height: float = 1.5,
    vehicle_z: float = 0.75,
    seed: int = 0,
    device: str = "cpu",
) -> dict:
    """A tiny single-camera BEV scene for Lift-Splat-Shoot (A11.5b).

    One forward camera at ego (0, 0, cam_height) looking along +x renders a few "vehicles"
    as colored blobs at the pixels where their 3-D centroids project. The ground-truth BEV
    occupancy marks each vehicle's ego cell. The image -> BEV mapping is geometrically exact
    (a blob sits at its vehicle's projected centroid, depth = ego forward distance), so an LSS
    pipeline with correct frustum geometry can overfit it while one that ignores the geometry
    cannot route a blob to the right pillar.

    Conventions match nanovision.geometry: ego frame x forward, y left, z up; camera frame
    OpenCV x right, y down, z forward. The extrinsic E is T_cam_ego (ego -> camera). For a
    forward camera, camera-frame depth equals the ego forward coordinate x exactly, so vehicles
    are placed with x in [d_min, d_max) to stay reachable by the depth bins. Vehicle cells are
    rejection-sampled to be both in-frame and depth-reachable; a ValueError is raised if
    n_vehicles cannot be placed (the focal is too narrow).

    Args:
        n_vehicles: number of vehicles to place.
        bev_x, bev_y: ego BEV extent (meters). bev_x forward extent must lie within the depth
            range [d_min, d_max] - the frustum cannot reach past the deepest bin.
        res: BEV cell size (meters), square cells. Grid is (nx, ny) with nx along x, ny along y.
        img: square image side in pixels.
        stride: backbone downsample; the feature grid is (img // stride) per side.
        d_min, d_max, d_step: depth-bin spec; centers are arange(d_min, d_max, d_step).
        focal: pinhole focal in pixels; defaults to img / 2 (a ~90 degree horizontal FOV).
        cam_height: camera height above the ego origin (meters).
        vehicle_z: vehicle centroid height (meters).
        seed: RNG seed (deterministic placement).
        device: target device for returned tensors.

    Returns:
        dict with:
            image: (1, 3, img, img) in [0, 1].
            K: (3, 3) pinhole intrinsic.
            E: (4, 4) T_cam_ego extrinsic (ego -> camera).
            bev_gt: (nx, ny) float in {0, 1}, vehicle occupancy on the BEV grid.
            depth_bin_labels: (Hf, Wf) long, nearest depth bin at each vehicle's feature cell.
            depth_mask: (Hf, Wf) bool, True at the labeled cells.
            vehicles: (n_vehicles, 2) ego (x, y) of each vehicle (for viz).
            bins: (D,) depth-bin centers.
    """
    g = torch.Generator().manual_seed(seed)
    f = float(focal) if focal is not None else img / 2.0
    cx = cy = (img - 1) / 2.0
    K = torch.tensor([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])

    # E = T_cam_ego for a forward camera: cam x = ego -y, cam y = ego -z, cam z = ego +x.
    R_ec = torch.tensor([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    t_ec = -R_ec @ torch.tensor([0.0, 0.0, cam_height])
    E = torch.eye(4)
    E[:3, :3] = R_ec
    E[:3, 3] = t_ec

    bins = torch.arange(d_min, d_max, d_step)  # (D,) bin centers
    nx = int(round((bev_x[1] - bev_x[0]) / res))
    ny = int(round((bev_y[1] - bev_y[0]) / res))
    Hf = Wf = img // stride

    def project(x: float, y: float, z: float):
        # ego (x, y, z) -> camera frame -> pixel (u, v) and depth.
        p_ego = torch.tensor([x, y, z])
        p_cam = R_ec @ p_ego + t_ec
        depth = float(p_cam[2])
        u = float(f * p_cam[0] / p_cam[2] + cx)
        v = float(f * p_cam[1] / p_cam[2] + cy)
        return u, v, depth

    # Place one vehicle per depth band so depths (and forward pillars) are distinct: stratify x
    # into n_vehicles equal bands over the reachable range, then rejection-sample an in-frame y.
    x_lo = max(bev_x[0], float(d_min))
    x_hi = min(bev_x[1], float(d_max))
    vehicles: list[tuple[float, float]] = []
    cells: set[tuple[int, int]] = set()
    for k in range(n_vehicles):
        band_lo = x_lo + (x_hi - x_lo) * k / n_vehicles
        band_hi = x_lo + (x_hi - x_lo) * (k + 1) / n_vehicles
        for _ in range(2000):
            x = band_lo + float(torch.rand(1, generator=g)) * (band_hi - band_lo)
            y = bev_y[0] + float(torch.rand(1, generator=g)) * (bev_y[1] - bev_y[0])
            u, v, depth = project(x, y, vehicle_z)
            if not (0.0 <= u < img and 0.0 <= v < img and d_min <= depth <= d_max):
                continue
            ix = int((x - bev_x[0]) / res)
            iy = int((y - bev_y[0]) / res)
            if not (0 <= ix < nx and 0 <= iy < ny) or (ix, iy) in cells:
                continue
            cells.add((ix, iy))
            vehicles.append((x, y))
            break
    if len(vehicles) < n_vehicles:
        raise ValueError(
            f"could only place {len(vehicles)}/{n_vehicles} vehicles in frame; "
            "widen the focal (smaller f) or the BEV extent"
        )

    image = torch.full((3, img, img), 0.3)
    color = torch.tensor([1.0, 0.6, 0.2])  # one vehicle class
    bev_gt = torch.zeros(nx, ny)
    depth_bin_labels = torch.zeros(Hf, Wf, dtype=torch.long)
    depth_mask = torch.zeros(Hf, Wf, dtype=torch.bool)
    vs, us = torch.meshgrid(
        torch.arange(img, dtype=torch.float32),
        torch.arange(img, dtype=torch.float32),
        indexing="ij",
    )
    for (x, y) in vehicles:
        u, v, depth = project(x, y, vehicle_z)
        sigma = float(min(max(8.0 / depth, 1.0), 6.0))  # closer vehicles are larger
        blob = torch.exp(-((us - u) ** 2 + (vs - v) ** 2) / (2.0 * sigma * sigma))
        image = torch.maximum(image, color[:, None, None] * blob[None])
        ix = int((x - bev_x[0]) / res)
        iy = int((y - bev_y[0]) / res)
        for di in (-1, 0):  # a 2-cell forward footprint, clamped to the grid
            jx = min(max(ix + di, 0), nx - 1)
            bev_gt[jx, iy] = 1.0
        fi = min(int(v // stride), Hf - 1)
        fj = min(int(u // stride), Wf - 1)
        depth_bin_labels[fi, fj] = int(torch.argmin((bins - depth).abs()))
        depth_mask[fi, fj] = True

    out = {
        "image": image.clamp(0.0, 1.0)[None].to(device),
        "K": K.to(device),
        "E": E.to(device),
        "bev_gt": bev_gt.to(device),
        "depth_bin_labels": depth_bin_labels.to(device),
        "depth_mask": depth_mask.to(device),
        "vehicles": torch.tensor(vehicles).to(device),
        "bins": bins.to(device),
    }
    return out


def bev_multicam_scene(
    n_cams: int = 4,
    n_vehicles: int = 4,
    bev: tuple[float, float] = (-8.0, 8.0),
    res: float = 1.0,
    img: int = 32,
    stride: int = 4,
    n_frames: int = 1,
    ego_step: float = 1.0,
    focal: float | None = None,
    occlude_moving: bool = False,
    cam_height: float = 1.5,
    vehicle_z: float = 0.75,
    seed: int = 0,
    device: str = "cpu",
) -> dict:
    """A tiny multi-camera ring BEV scene for BEVFormer-style attention (A11.5c).

    n_cams cameras at ego (0, 0, cam_height) with yaw uniformly over 360 degrees (front, left,
    back, right for n_cams=4), OpenCV convention. Each "vehicle" is rendered as a colored blob in
    whichever cameras see its centroid. The centered BEV occupancy ground truth marks each
    vehicle's ego cell. BEV cells reaching back into image space (the query-pull view transform)
    can recover the occupancy because the projection geometry is exact.

    A 4-camera cardinal ring does not produce cells seen by two cameras at once (the FOV
    boundaries meet along the diagonals), so every BEV cell is single-view or unseen regardless of
    focal; the multi-view averaging in spatial cross-attention is exercised by its unit test, not
    by this toy's geometry. The default focal (img / 2, ~90 degree FOV) covers almost the whole
    grid (only the far corners are unseen) while keeping many single-view cells for the hit-mask
    test.

    Conventions match nanovision.geometry: ego x forward, y left, z up; camera OpenCV x right,
    y down, z forward; extrinsic E = T_cam_ego. A camera at ego yaw a has axes z_cam=(cos a,
    sin a, 0), x_cam=(sin a, -cos a, 0), y_cam=(0, 0, -1); at a=0 this is the forward camera of
    bev_toy_scene. The default focal (img) gives a ~53 degree FOV so adjacent cameras leave gaps
    and some BEV cells are single-view (the spatial-cross-attention hit-mask test needs that).

    Temporal use (n_frames=2): the ego moves forward ego_step meters per frame; vehicles are fixed
    in the world, so their ego coordinates shift backward by ego_step each frame. ego_deltas
    carries the per-frame SE(2) ego motion (forward, lateral, yaw). With occlude_moving=True, one
    vehicle is rendered into every frame EXCEPT the last (current) frame, but stays in the BEV
    ground truth of every frame; occluded_cells marks its current-frame cells, the region the
    temporal-self-attention test scores (recoverable from warped history, not from the current
    image alone).

    Args:
        n_cams: number of ring cameras.
        n_vehicles: vehicles to place (including the moving/occluded one when occlude_moving).
        bev: ego BEV extent (meters), same for x and y (centered grid).
        res: BEV cell size (meters).
        img: square image side (pixels).
        stride: backbone downsample (feature grid is img // stride per side).
        n_frames: number of frames (1 for the single-frame tests, 2 for the temporal test).
        ego_step: forward ego motion per frame (meters), used when n_frames > 1.
        focal: pinhole focal (pixels); defaults to img / 2 (~90 deg FOV).
        occlude_moving: render one vehicle into all frames but the last (temporal test).
        cam_height, vehicle_z: camera and vehicle-centroid heights (meters).
        seed: RNG seed.
        device: target device.

    Returns:
        dict with:
            images: (n_frames, n_cams, 3, img, img) in [0, 1].
            K: (3, 3) shared intrinsic.
            E: (n_cams, 4, 4) per-camera T_cam_ego (frame-independent; ego motion moves vehicles).
            bev_gt: (n_frames, nx, ny) vehicle occupancy on the centered BEV grid.
            ego_deltas: (n_frames, 3) SE(2) ego motion (forward, lateral, yaw); frame 0 is zero.
            vehicles: (n_frames, n_vehicles, 2) ego (x, y) per frame.
            occluded_cells: (k, 2) long current-frame cells of the moving vehicle, or an empty
                tensor when occlude_moving is False.
    """
    g = torch.Generator().manual_seed(seed)
    f = float(focal) if focal is not None else img / 2.0
    cx = cy = (img - 1) / 2.0
    K = torch.tensor([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
    n = int(round((bev[1] - bev[0]) / res))  # nx == ny (centered square grid)

    # Per-camera extrinsics E = T_cam_ego for the ring.
    Es, R_ecs, t_ecs = [], [], []
    for k in range(n_cams):
        a = 2.0 * torch.pi * k / n_cams
        ca, sa = float(torch.cos(torch.tensor(a))), float(torch.sin(torch.tensor(a)))
        x_cam = torch.tensor([sa, -ca, 0.0])
        y_cam = torch.tensor([0.0, 0.0, -1.0])
        z_cam = torch.tensor([ca, sa, 0.0])
        R_ce = torch.stack([x_cam, y_cam, z_cam], dim=1)  # columns are cam axes in ego
        R_ec = R_ce.T
        t_ec = -R_ec @ torch.tensor([0.0, 0.0, cam_height])
        E = torch.eye(4)
        E[:3, :3] = R_ec
        E[:3, 3] = t_ec
        Es.append(E)
        R_ecs.append(R_ec)
        t_ecs.append(t_ec)

    def project(cam: int, x: float, y: float, z: float):
        p_cam = R_ecs[cam] @ torch.tensor([x, y, z]) + t_ecs[cam]
        if float(p_cam[2]) <= 0:
            return None
        u = float(f * p_cam[0] / p_cam[2] + cx)
        v = float(f * p_cam[1] / p_cam[2] + cy)
        if 0.0 <= u < img and 0.0 <= v < img:
            return u, v
        return None

    def cell_of(x: float, y: float):
        ix = int((x - bev[0]) / res)
        iy = int((y - bev[0]) / res)
        if 0 <= ix < n and 0 <= iy < n:
            return ix, iy
        return None

    # Place vehicles in the WORLD frame (= frame-0 ego frame): inside the grid and seen by >=1
    # camera at every frame's ego pose. The last vehicle is the moving/occluded one.
    world: list[tuple[float, float]] = []
    margin = ego_step * (n_frames - 1) + 1.0  # keep vehicles in-grid across all frames
    for _ in range(4000):
        if len(world) == n_vehicles:
            break
        wx = bev[0] + margin + float(torch.rand(1, generator=g)) * (bev[1] - bev[0] - 2 * margin)
        wy = bev[0] + margin + float(torch.rand(1, generator=g)) * (bev[1] - bev[0] - 2 * margin)
        # Visible from some camera at every frame's ego pose?
        ok = True
        for fr in range(n_frames):
            ex, ey = wx - fr * ego_step, wy  # ego coords at frame fr (pure forward ego motion)
            if cell_of(ex, ey) is None or all(
                project(c, ex, ey, vehicle_z) is None for c in range(n_cams)
            ):
                ok = False
                break
        if not ok:
            continue
        if any(abs(wx - ox) < res and abs(wy - oy) < res for ox, oy in world):
            continue
        world.append((wx, wy))
    if len(world) < n_vehicles:
        raise ValueError(
            f"placed only {len(world)}/{n_vehicles} vehicles; widen focal or shrink ego_step"
        )

    moving_idx = n_vehicles - 1 if occlude_moving else -1
    color = torch.tensor([1.0, 0.6, 0.2])
    images = torch.full((n_frames, n_cams, 3, img, img), 0.3)
    bev_gt = torch.zeros(n_frames, n, n)
    veh_ego = torch.zeros(n_frames, n_vehicles, 2)
    ego_deltas = torch.zeros(n_frames, 3)
    vs, us = torch.meshgrid(
        torch.arange(img, dtype=torch.float32),
        torch.arange(img, dtype=torch.float32),
        indexing="ij",
    )
    for fr in range(n_frames):
        if fr > 0:
            ego_deltas[fr] = torch.tensor([ego_step, 0.0, 0.0])
        for vi, (wx, wy) in enumerate(world):
            ex, ey = wx - fr * ego_step, wy
            veh_ego[fr, vi] = torch.tensor([ex, ey])
            cell = cell_of(ex, ey)
            if cell is not None:
                bev_gt[fr, cell[0], cell[1]] = 1.0  # physically present even if occluded
            # Occluded vehicle is not rendered into the LAST (current) frame's images.
            if vi == moving_idx and fr == n_frames - 1:
                continue
            for c in range(n_cams):
                p = project(c, ex, ey, vehicle_z)
                if p is None:
                    continue
                u, v = p
                depth = float((R_ecs[c] @ torch.tensor([ex, ey, vehicle_z]) + t_ecs[c])[2])
                sigma = float(min(max(8.0 / depth, 1.0), 6.0))
                blob = torch.exp(-((us - u) ** 2 + (vs - v) ** 2) / (2.0 * sigma * sigma))
                images[fr, c] = torch.maximum(images[fr, c], color[:, None, None] * blob[None])

    occluded_cells = torch.zeros(0, 2, dtype=torch.long)
    if occlude_moving:
        wx, wy = world[moving_idx]
        ex, ey = wx - (n_frames - 1) * ego_step, wy
        cell = cell_of(ex, ey)
        if cell is not None:
            occluded_cells = torch.tensor([[cell[0], cell[1]]], dtype=torch.long)

    return {
        "images": images.clamp(0.0, 1.0).to(device),
        "K": K.to(device),
        "E": torch.stack(Es).to(device),
        "bev_gt": bev_gt.to(device),
        "ego_deltas": ego_deltas.to(device),
        "vehicles": veh_ego.to(device),
        "occluded_cells": occluded_cells.to(device),
    }
