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
