"""Render the video MAE: a tube-masked clip and its reconstruction filmstrip.

Run from the repo root: `python -m assignments.a03_5_video.viz` (or `make viz
A=a03_5_video`). Overfits one toy-clip batch, then renders, for one clip, three rows
across the T frames: the original, the tube-masked input (masked tubelets blanked),
and the reconstruction. Also writes the overfit loss curve. The masked columns are the
same across all frames, which is the tube the model has to infer.
"""

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

_here = Path(__file__).parent
_impl = _here / "solution" if os.environ.get("NANOVISION_IMPL") == "solution" else _here
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_impl))

from backbone import per_tubelet_normalize, tubeletify, untubeletify  # noqa: E402
from config import VideoSSLConfig  # noqa: E402
from video_mae import VideoMAE  # noqa: E402

from nanovision.data.toy import video_batch  # noqa: E402
from nanovision.determinism import set_seed  # noqa: E402


def main() -> None:
    set_seed(0)
    cfg = VideoSSLConfig()
    out = _here / "out"
    out.mkdir(exist_ok=True)

    vid = video_batch(batch=cfg.overfit_batch, n_frames=cfg.n_frames, size=cfg.img_size,
                      channels=cfg.in_chans, n_blobs=cfg.n_blobs, seed=0)
    model = VideoMAE(
        img_size=cfg.img_size, patch=cfg.patch, tubelet_t=cfg.tubelet_t,
        n_frames=cfg.n_frames, in_chans=cfg.in_chans, enc_dim=cfg.enc_dim,
        enc_depth=cfg.enc_depth, enc_heads=cfg.enc_heads, dec_dim=cfg.dec_dim,
        dec_depth=cfg.dec_depth, dec_heads=cfg.dec_heads, mlp_ratio=cfg.mlp_ratio,
        mask_ratio=cfg.mask_ratio,
    )
    opt = torch.optim.Adam(model.parameters(), lr=cfg.mae_lr)

    losses = []
    for _ in range(cfg.mae_steps):
        torch.manual_seed(1234)  # fixed tube mask
        loss, _, _ = model(vid)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    # One reconstruction with the same fixed mask.
    torch.manual_seed(1234)
    with torch.no_grad():
        _, pred, mask = model(vid)

    grid = cfg.img_size // cfg.patch
    # The model predicts per-tubelet-NORMALIZED pixels. Denormalize with the original
    # per-tubelet mean/std (standard MAE viz), and paste visible tubelets straight from
    # the input so only the masked tubes show the model's guess.
    tub_target = tubeletify(vid, cfg.tubelet_t, cfg.patch)        # (B, N, dim) raw pixels
    mean = tub_target.mean(dim=-1, keepdim=True)
    std = torch.sqrt(tub_target.var(dim=-1, keepdim=True, unbiased=False) + 1e-6)
    pred_pix = pred * std + mean
    keep = (mask == 0).unsqueeze(-1)
    recon_tub = torch.where(keep, tub_target, pred_pix)
    recon = untubeletify(recon_tub, cfg.tubelet_t, cfg.patch, cfg.in_chans, model.t_prime, grid)

    # Masked input: blank the masked tubelets in pixel space.
    tubelet_dim = cfg.tubelet_t * cfg.patch * cfg.patch * cfg.in_chans
    mask_pix = untubeletify(mask.unsqueeze(-1).expand(-1, -1, tubelet_dim),
                            cfg.tubelet_t, cfg.patch, cfg.in_chans, model.t_prime, grid)
    masked_input = vid * (1.0 - mask_pix)

    def to_img(frame):  # (C, H, W) -> (H, W, C) clamped for imshow
        return frame.permute(1, 2, 0).clamp(0, 1).numpy()

    b = 0
    T = cfg.n_frames
    rows = [("original", vid[b]), ("tube-masked", masked_input[b]), ("reconstruction", recon[b])]
    fig, axes = plt.subplots(3, T, figsize=(1.4 * T, 4.4))
    for r, (label, clip) in enumerate(rows):
        for t in range(T):
            ax = axes[r, t]
            ax.imshow(to_img(clip[:, t]))
            ax.set_xticks([]); ax.set_yticks([])
            if t == 0:
                ax.set_ylabel(label, fontsize=9)
            if r == 0:
                ax.set_title(f"t={t}", fontsize=8)
    fig.suptitle(f"video MAE, mask_ratio={cfg.mask_ratio} (tube masking)", fontsize=10)
    fig.tight_layout()
    film = out / "video_mae_reconstruction.png"
    fig.savefig(film, dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(losses)
    ax.set_yscale("log")
    ax.set_xlabel("step"); ax.set_ylabel("masked-tubelet MSE")
    ax.set_title("video MAE overfit-one-batch")
    fig.tight_layout()
    curve = out / "video_mae_loss.png"
    fig.savefig(curve, dpi=120)
    plt.close(fig)

    print(f"final masked-tubelet MSE {losses[-1]:.3e} - wrote {film}, {curve}")


if __name__ == "__main__":
    main()
