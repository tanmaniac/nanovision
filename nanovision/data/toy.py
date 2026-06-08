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
