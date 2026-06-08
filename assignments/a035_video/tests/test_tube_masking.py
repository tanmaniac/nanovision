"""The centerpiece: tube masking masks whole spatiotemporal tubes.

The defining property of tube masking (vs A3's per-token masking) is that the
kept/dropped SPATIAL pattern is identical across every temporal step. A per-token
mask would leak through time; the tube forces the model to infer masked content from
other space-time locations, not from the same patch in an adjacent frame.
"""

import torch

from config import VideoSSLConfig
from nanovision.determinism import set_seed
from video_mae import tube_masking

cfg = VideoSSLConfig()


def _grid():
    t_prime = cfg.n_frames // cfg.tubelet_t
    s = (cfg.img_size // cfg.patch) ** 2
    return t_prime, s, t_prime * s


def test_keep_count():
    t_prime, s, n = _grid()
    set_seed(0)
    x = torch.randn(3, n, 8)
    x_kept, mask, _ = tube_masking(x, t_prime, cfg.mask_ratio)
    n_keep = t_prime * round((1 - cfg.mask_ratio) * s)
    assert x_kept.shape[1] == n_keep
    for b in range(3):
        assert int(mask[b].sum().item()) == n - n_keep


def test_tube_property():
    # The mask, reshaped to (B, T', S'), must be identical across all temporal steps.
    t_prime, s, n = _grid()
    set_seed(1)
    x = torch.randn(4, n, 8)
    _, mask, _ = tube_masking(x, t_prime, cfg.mask_ratio)
    m = mask.reshape(4, t_prime, s)
    assert torch.equal(m, m[:, :1, :].expand(-1, t_prime, -1))


def test_unshuffle_restores_visible():
    # Concatenating [visible; placeholders] and gathering by ids_restore must put the
    # visible tubelets back at their original (mask==0) positions.
    t_prime, s, n = _grid()
    set_seed(2)
    d = 8
    x = torch.randn(2, n, d)
    x_kept, mask, ids_restore = tube_masking(x, t_prime, cfg.mask_ratio)
    n_keep = x_kept.shape[1]
    placeholders = torch.zeros(2, n - n_keep, d)
    full = torch.cat([x_kept, placeholders], dim=1)
    restored = torch.gather(full, 1, ids_restore.unsqueeze(-1).expand(-1, -1, d))
    keep = (mask == 0).unsqueeze(-1).expand(-1, -1, d)
    assert torch.allclose(restored[keep], x[keep], atol=1e-6)


def test_deterministic_under_seed():
    t_prime, s, n = _grid()
    x = torch.randn(2, n, 8)
    set_seed(7)
    _, mask_a, ids_a = tube_masking(x, t_prime, cfg.mask_ratio)
    set_seed(7)
    _, mask_b, ids_b = tube_masking(x, t_prime, cfg.mask_ratio)
    assert torch.equal(mask_a, mask_b)
    assert torch.equal(ids_a, ids_b)
