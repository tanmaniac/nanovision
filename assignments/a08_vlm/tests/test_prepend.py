"""prepend_visual ordering and vlm_loss masking.

prepend_visual puts visual tokens at positions 0..N-1 and text after. vlm_loss supervises
only the text positions: a hand-computed cross-entropy over just the text targets must match
the function, and perturbing a visual-span logit must not change the loss.
"""

import torch
import torch.nn.functional as F

from vlm import prepend_visual, vlm_loss


def test_prepend_ordering():
    B, N, L, d = 4, 16, 8, 64
    vis = torch.randn(B, N, d)
    txt = torch.randn(B, L, d)
    seq = prepend_visual(vis, txt)
    assert seq.shape == (B, N + L, d)
    assert torch.allclose(seq[:, :N], vis)        # visual first
    assert torch.allclose(seq[:, N:], txt)        # text after


def test_loss_matches_handcomputed_text_span():
    torch.manual_seed(0)
    B, N, L, V = 2, 3, 4, 7
    logits = torch.randn(B, N + L, V)
    # captions: [class, attr, EOS, pad]; pad id 0 must be ignored.
    token_ids = torch.tensor([[2, 5, 6, 0], [1, 4, 6, 0]])

    loss = vlm_loss(logits, token_ids, N)

    # Hand-compute: the prediction at position i targets the token at i+1 in the combined
    # sequence. Combined targets are [-100]*N then the text tokens, pad re-masked to -100.
    labels = torch.full((B, N + L), -100)
    labels[:, N:N + L] = token_ids
    labels[:, N:N + L][token_ids == 0] = -100
    ref = F.cross_entropy(logits[:, :-1].reshape(-1, V), labels[:, 1:].reshape(-1),
                          ignore_index=-100)
    assert torch.allclose(loss, ref)

    # Manually: supervised pairs are (pos N-1 -> text_0), (pos N -> text_1), (pos N+1 -> text_2).
    # text_3 is pad so it is never a target; pos N+2 predicting text_3 is ignored.
    rows = []
    for b in range(B):
        for src, tgt_pos in [(N - 1, 0), (N, 1), (N + 1, 2)]:
            rows.append(F.cross_entropy(logits[b, src:src + 1], token_ids[b, tgt_pos:tgt_pos + 1]))
    manual = torch.stack(rows).mean()
    assert torch.allclose(loss, manual, atol=1e-6)


def test_visual_span_logits_do_not_affect_loss():
    torch.manual_seed(1)
    B, N, L, V = 2, 3, 4, 7
    logits = torch.randn(B, N + L, V)
    token_ids = torch.tensor([[2, 5, 6, 0], [1, 4, 6, 0]])
    base = vlm_loss(logits, token_ids, N)

    # Positions 0..N-2 only ever predict visual or the (ignored) -100 visual targets.
    perturbed = logits.clone()
    perturbed[:, :N - 1] += 100.0
    assert torch.allclose(vlm_loss(perturbed, token_ids, N), base)
