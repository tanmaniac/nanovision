"""Float64 gradcheck: the tubelet embed and the full video MAE pipeline.

Runs after shapes. Checks that gradients flow through TubeletEmbedding to its conv
weight, and through the tube-mask encode -> decode -> loss pipeline to that weight.
"""

import torch
import torch.nn.functional as F
from torch import nn

from nanovision.gradcheck import check_gradients
from nanovision.transformer import TubeletEmbedding
from video_mae import VideoMAE


class _TubeletWrtWeight(nn.Module):
    """forward(weight) -> the tubelet-embed output, as a function of the conv weight."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        # Tiny: T=4,t=2 -> T'=2; img 8, p 4 -> 2x2; N = 2*4 = 8.
        self.te = TubeletEmbedding(2, 6, 4, 2).double()
        self.vid = torch.randn(2, 2, 4, 8, 8, dtype=torch.double)

    def forward(self, weight):
        out = F.conv3d(self.vid, weight, self.te.proj.bias, stride=(2, 4, 4))
        return out.flatten(2).transpose(1, 2)


def test_tubelet_gradcheck():
    mod = _TubeletWrtWeight().double()
    weight = mod.te.proj.weight.detach().double()
    assert check_gradients(mod, (weight,))


class _VideoMAELossWrtWeight(nn.Module):
    """forward(weight) -> masked-tubelet loss as a function of the tubelet conv weight.

    Drives the tubelet conv functionally from the passed-in leaf so the loss gradient
    flows back to it; the tube mask is fixed by a seed for a deterministic graph.
    """

    def __init__(self):
        super().__init__()
        self.mae = VideoMAE(img_size=8, patch=4, tubelet_t=2, n_frames=4, in_chans=2,
                            enc_dim=8, enc_depth=1, enc_heads=2, dec_dim=8, dec_depth=1,
                            dec_heads=2, mask_ratio=0.5)
        torch.manual_seed(0)
        self.vid = torch.randn(2, 2, 4, 8, 8, dtype=torch.double)

    def forward(self, weight):
        from backbone import per_tubelet_normalize, tubeletify
        from video_mae import tube_masking, video_mae_loss
        enc = self.mae.encoder
        torch.manual_seed(0)  # fix the tube mask
        feat = F.conv3d(self.vid, weight, enc.tubelet.proj.bias, stride=(2, 4, 4))
        tokens = feat.flatten(2).transpose(1, 2) + enc.pos_embed
        x_kept, mask, ids_restore = tube_masking(tokens, enc.t_prime, self.mae.mask_ratio)
        x_enc = enc.forward_tokens(x_kept)
        pred = self.mae.forward_decoder(x_enc, ids_restore)
        target = per_tubelet_normalize(tubeletify(self.vid, self.mae.tubelet_t, self.mae.patch))
        return video_mae_loss(pred, target, mask)


def test_video_mae_pipeline_gradcheck():
    mod = _VideoMAELossWrtWeight().double()
    weight = mod.mae.encoder.tubelet.proj.weight.detach().double()
    assert check_gradients(mod, (weight,))
