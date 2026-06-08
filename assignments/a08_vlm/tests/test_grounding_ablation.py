"""The visual path carries the grounding: zero the visual tokens and the caption loss jumps.

Train a small model briefly, then compare the caption loss with the true visual tokens vs
with the visual tokens zeroed. The drawn batch is pinned so at least two rows share a CLASS
but differ in ATTRIBUTE, so the attribute cannot be read off the class token alone and the
image is required to disambiguate it. We report the per-position loss split (class position
vs attribute position) for full and ablated; the class position is always image-grounded
(predicted from the last visual token), so its full-vs-ablated gap is the seed-robust signal.

Measured (seed 0, 300 steps): loss_full ~7e-4, loss_ablated ~1.8 (ratio ~2500x). Per
position, class (0.0009 -> 3.0) and attribute (0.0007 -> 2.4) both jump when the visual
tokens are zeroed.
"""

import torch
import torch.nn.functional as F

from config import VLMConfig
from vlm import VLM, prepend_visual, vlm_loss

from nanovision.transformer import build_causal_mask
from nanovision.data import toy


def _logits(model, img, tok, zero_visual):
    feats = model.vit.forward_features(img)
    vis = model.connector(feats)
    if zero_visual:
        vis = torch.zeros_like(vis)
    seq = prepend_visual(vis, model.embed(tok))
    mask = build_causal_mask(seq.shape[1]).to(seq.device)
    h = model.decoder(seq, mask=mask)
    return model.head(model.norm(h)), vis.shape[1]


def _per_position(logits, tok, n_visual):
    # class token = tok[:, 0], predicted at combined position n_visual - 1.
    # attr token  = tok[:, 1], predicted at combined position n_visual.
    cls = F.cross_entropy(logits[:, n_visual - 1], tok[:, 0])
    attr = F.cross_entropy(logits[:, n_visual], tok[:, 1])
    return cls.item(), attr.item()


def test_grounding_ablation():
    torch.manual_seed(0)
    cfg = VLMConfig()
    model = VLM(cfg)
    img, tok = toy.image_text_batch(
        batch=4, size=cfg.img_size, channels=cfg.in_chans, n_classes=cfg.n_classes,
        n_attrs=cfg.n_attrs, vocab_size=cfg.vocab_size, max_len=cfg.max_len, seed=0,
    )

    # Require at least two rows with the same class but different attribute.
    classes = tok[:, 0]
    attrs = tok[:, 1]
    shared = False
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            if classes[i] == classes[j] and attrs[i] != attrs[j]:
                shared = True
    assert shared, "test batch must have two rows sharing a class but differing in attribute"

    model.set_stage(2)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-3)
    for _ in range(300):
        opt.zero_grad()
        logits, n_visual = model(img, tok)
        vlm_loss(logits, tok, n_visual).backward()
        opt.step()

    full_logits, n_visual = _logits(model, img, tok, zero_visual=False)
    abl_logits, _ = _logits(model, img, tok, zero_visual=True)
    loss_full = vlm_loss(full_logits, tok, n_visual).item()
    loss_abl = vlm_loss(abl_logits, tok, n_visual).item()

    cls_full, attr_full = _per_position(full_logits, tok, n_visual)
    cls_abl, attr_abl = _per_position(abl_logits, tok, n_visual)

    print(f"loss full {loss_full:.4f}  ablated {loss_abl:.4f}  ratio {loss_abl / loss_full:.0f}")
    print(f"class position  full {cls_full:.4f}  ablated {cls_abl:.4f}")
    print(f"attr position   full {attr_full:.4f}  ablated {attr_abl:.4f}")

    # Zeroing the visual tokens makes the caption far harder to predict.
    assert loss_abl > 10 * loss_full
    # The class position is image-grounded by construction; its gap is the robust signal.
    assert cls_abl > 10 * cls_full
