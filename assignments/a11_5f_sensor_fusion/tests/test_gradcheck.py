"""Double-precision gradchecks for the two trainable fusion blocks.

BEVFuser (channel-concat + conv) and TransFuserBlock (self-attention over concatenated tokens)
are the pieces that carry a task gradient, so they must be differentiable wrt their inputs. Both
run at float64 on tiny dimensions.
"""

import torch

from fusion import BEVFuser
from transfuser import TransFuserBlock


def test_bev_fuser_gradcheck():
    torch.manual_seed(0)
    fuser = BEVFuser(cam_channels=2, lidar_channels=3, hidden=4, out_channels=2).double().eval()
    cam = torch.randn(2, 3, 3, dtype=torch.float64, requires_grad=True)
    lidar = torch.randn(3, 3, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a, b: fuser(a, b), (cam, lidar))


def test_transfuser_gradcheck():
    torch.manual_seed(0)
    block = TransFuserBlock(dim=4, n_heads=1).double().eval()
    cam_tokens = torch.randn(1, 3, 4, dtype=torch.float64, requires_grad=True)
    lidar_tokens = torch.randn(1, 2, 4, dtype=torch.float64, requires_grad=True)

    def both(a, b):
        cam_out, lidar_out = block(a, b)
        return torch.cat([cam_out.reshape(-1), lidar_out.reshape(-1)])

    assert torch.autograd.gradcheck(both, (cam_tokens, lidar_tokens))
