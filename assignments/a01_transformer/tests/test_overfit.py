"""Task 8: the assembled decoder-only char-LM overfits one batch. Run last.

This is the end-to-end integration signal: with every mechanism correct, the
full forward + training loop drives the cross-entropy on a single fixed batch to
near zero. In starter mode it fails because the underlying holes raise.
"""

import torch
import torch.nn.functional as F

from charlm import CharLM

from nanovision.data.toy import char_lm_batch
from nanovision.determinism import set_seed
from nanovision.trainer import Trainer


def test_charlm_overfits_one_batch():
    set_seed(0)
    x, y, tok = char_lm_batch(seq=32, batch=8, seed=0)
    model = CharLM(tok.vocab_size, dim=64, n_heads=4, depth=2)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    vocab = tok.vocab_size

    def loss_fn(logits, target):
        return F.cross_entropy(logits.reshape(-1, vocab), target.reshape(-1))

    tr = Trainer(model, opt, loss_fn, device="cpu")
    losses = tr.overfit_one_batch((x, y), steps=500)
    assert losses[-1] < 0.05, f"char-LM should memorize the batch; final loss {losses[-1]}"
    assert losses[-1] < losses[0]
