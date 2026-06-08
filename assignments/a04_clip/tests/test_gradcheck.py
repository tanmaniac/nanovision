"""Float64 gradcheck of both losses w.r.t. their features and learnable parameters."""

import torch
from torch import nn

from losses import clip_loss, siglip_loss
from nanovision.gradcheck import check_gradients


class _ClipLoss(nn.Module):
    def forward(self, img, txt, logit_scale):
        return clip_loss(img, txt, logit_scale)


def test_clip_loss_gradcheck():
    torch.manual_seed(0)
    img = torch.randn(4, 8, dtype=torch.double)
    txt = torch.randn(4, 8, dtype=torch.double)
    logit_scale = torch.tensor(2.0, dtype=torch.double)
    assert check_gradients(_ClipLoss(), (img, txt, logit_scale))


class _SiglipLoss(nn.Module):
    def forward(self, img, txt, logit_scale, bias):
        return siglip_loss(img, txt, logit_scale, bias)


def test_siglip_loss_gradcheck():
    torch.manual_seed(0)
    img = torch.randn(4, 8, dtype=torch.double)
    txt = torch.randn(4, 8, dtype=torch.double)
    logit_scale = torch.tensor(2.0, dtype=torch.double)
    bias = torch.tensor(-1.0, dtype=torch.double)
    assert check_gradients(_SiglipLoss(), (img, txt, logit_scale, bias))
