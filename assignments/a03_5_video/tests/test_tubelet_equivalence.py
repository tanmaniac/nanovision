"""The tubelet embed equals an unfold-into-tubes times a reshaped conv weight.

The 3D analog of A2's patch-equivalence test: a non-overlapping strided Conv3d is one
shared linear map applied per space-time tubelet, so its output equals tubeletify
(unfold each tubelet into a (C*t*p*p) vector) times the conv weight reshaped to
(C*t*p*p, dim), plus the bias.
"""

import torch

from backbone import tubeletify
from config import VideoSSLConfig
from nanovision.transformer import TubeletEmbedding

cfg = VideoSSLConfig()


def test_tubelet_equals_strided_conv():
    torch.manual_seed(0)
    vid = torch.randn(2, cfg.in_chans, cfg.n_frames, cfg.img_size, cfg.img_size)
    te = TubeletEmbedding(cfg.in_chans, cfg.enc_dim, cfg.patch, cfg.tubelet_t)
    out = te(vid)                                       # (B, N, dim)

    weight = te.proj.weight.reshape(cfg.enc_dim, -1)    # (dim, C*t*p*p)
    tubelets = tubeletify(vid, cfg.tubelet_t, cfg.patch)  # (B, N, C*t*p*p)
    ref = tubelets @ weight.T + te.proj.bias            # (B, N, dim)
    assert torch.allclose(out, ref, atol=1e-5)
