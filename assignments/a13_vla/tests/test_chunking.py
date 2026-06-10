"""chunk_actions round-trips under the specified de-chunk rule, and receding-horizon indices.

Overlapping windows are redundant, so the inverse must be defined: the full first chunk, then the
last action of each subsequent chunk. That reconstructs the original (B, T, 2) sequence exactly.
Exact, training-free.
"""

import torch

from bc import chunk_actions, de_chunk, receding_horizon_indices


def test_dechunk_roundtrip():
    g = torch.Generator().manual_seed(0)
    for T, H in [(10, 4), (8, 1), (6, 6), (12, 3)]:
        actions = torch.randn(5, T, 2, generator=g)
        chunks = chunk_actions(actions, H)
        recon = de_chunk(chunks)
        assert recon.shape == actions.shape, (T, H)
        assert torch.allclose(recon, actions, atol=1e-6), (T, H)


def test_receding_horizon_covers_sequence():
    # Starts are 0, H, 2H, ..., last clamped to T-H, no duplicates, and every step in [0, T) is
    # covered by some chunk [start, start+H).
    for T, H in [(10, 4), (8, 1), (12, 3), (7, 4)]:
        starts = receding_horizon_indices(T, H)
        assert starts[0] == 0
        assert all(0 <= s <= T - H for s in starts)
        assert len(starts) == len(set(starts))
        covered = set()
        for s in starts:
            covered.update(range(s, s + H))
        assert covered == set(range(T)), (T, H, starts)
