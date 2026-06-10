"""A1 viz: a causal attention heatmap and the char-LM overfit loss curve.

Run with: make viz A=a01_transformer  (uses the reference solution).
"""

import sys
from pathlib import Path

from nanovision.viz import SHOW, finish, plt, plot_loss_curve  # noqa: E402  (sets the matplotlib backend)
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

# Make the assignment's CharLM importable when run as a script (no pytest conftest).
sys.path.insert(0, str(Path(__file__).parent / "solution"))

from charlm import CharLM  # noqa: E402

from nanovision.attention import scaled_dot_product_attention  # noqa: E402
from nanovision.data.toy import char_lm_batch  # noqa: E402
from nanovision.determinism import set_seed  # noqa: E402
from nanovision.trainer import Trainer  # noqa: E402
from nanovision.transformer import build_causal_mask  # noqa: E402


def attention_heatmap(out_path):
    set_seed(0)
    s, dh = 8, 16
    q, k, v = (torch.randn(1, 1, s, dh) for _ in range(3))
    mask = build_causal_mask(s).view(1, 1, s, s)
    _, attn = scaled_dot_product_attention(q, k, v, mask)
    fig, ax = plt.subplots(figsize=(4.2, 4))
    im = ax.imshow(attn[0, 0].detach().numpy(), cmap="viridis")
    ax.set_title("causal attention weights (lower triangular)")
    ax.set_xlabel("key position")
    ax.set_ylabel("query position")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    finish(out_path)


def overfit_curve(out_path):
    set_seed(0)
    x, y, tok = char_lm_batch(seq=32, batch=8, seed=0)
    model = CharLM(tok.vocab_size, dim=64, n_heads=4, depth=2)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    vocab = tok.vocab_size
    tr = Trainer(model, opt,
                 lambda logits, t: F.cross_entropy(logits.reshape(-1, vocab), t.reshape(-1)))
    losses = tr.overfit_one_batch((x, y), steps=500)
    plot_loss_curve(losses, out_path, title="A1 char-LM overfit (cross-entropy)")
    return losses[-1]


def main():
    out = Path(__file__).parent / "out"
    out.mkdir(exist_ok=True)
    attention_heatmap(out / "causal_attention.png")
    final = overfit_curve(out / "charlm_loss.png")
    print(f"final char-LM loss {final:.3e} - wrote {out}/causal_attention.png, {out}/charlm_loss.png")


if __name__ == "__main__":
    main()
    if SHOW:
        plt.show()
