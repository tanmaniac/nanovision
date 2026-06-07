"""Tasks 1-4 gradients (float64 gradcheck) and the teacher no-grad contract.

Runs after shapes. Checks that:
  - gradients flow through the MAE encode -> decode -> loss pipeline to an encoder
    parameter (the patch-embed conv weight);
  - gradients flow through dino_loss to the student logits;
  - the teacher parameters have requires_grad False and a student-loss backward
    leaves their grads None.
"""

import torch
import torch.nn.functional as F
from torch import nn

from backbone import DINOModel
from config import SSLConfig
from dino import build_student_teacher, dino_loss, dino_step
from mae import MAE
from nanovision.determinism import set_seed
from nanovision.gradcheck import check_gradients


class _MAELossWrtEncoderWeight(nn.Module):
    """Wrap MAE so forward(weight) returns the masked-patch loss as a function of
    one encoder parameter (the patch-embed conv weight). gradcheck then verifies
    the loss gradient w.r.t. that encoder parameter against finite differences.
    """

    def __init__(self):
        super().__init__()
        # Tiny MAE; fixed masking via a fixed seed inside forward for a stable graph.
        self.mae = MAE(img_size=8, patch=4, in_chans=2, enc_dim=8, enc_depth=1,
                       enc_heads=2, dec_dim=8, dec_depth=1, dec_heads=2,
                       mask_ratio=0.5)
        torch.manual_seed(0)
        self.img = torch.randn(2, 2, 8, 8, dtype=torch.double)

    def forward(self, weight):
        # Drive the patch-embed conv functionally from the passed-in leaf `weight`
        # so the loss gradient flows back to it (assigning a plain tensor onto the
        # conv would detach the graph). The rest of the MAE pipeline is unchanged.
        enc = self.mae.encoder
        torch.manual_seed(0)  # fix the random masking so the function is deterministic
        from mae import append_mask_tokens, random_masking
        from backbone import patchify, per_patch_normalize
        feat = F.conv2d(self.img, weight, enc.proj.bias, stride=enc.patch)
        tokens = feat.flatten(2).transpose(1, 2) + enc.pos_embed
        x_kept, mask, ids_restore = random_masking(tokens, self.mae.mask_ratio)
        x_enc = enc.forward_tokens(x_kept)
        pred = self.mae.forward_decoder(x_enc, ids_restore)
        target = per_patch_normalize(patchify(self.img, self.mae.patch))
        from mae import mae_loss
        return mae_loss(pred, target, mask)


def test_mae_pipeline_gradcheck():
    mod = _MAELossWrtEncoderWeight().double()
    weight = mod.mae.encoder.proj.weight.detach().double()
    assert check_gradients(mod, (weight,))


class _DINOLossWrtStudent(nn.Module):
    """forward(student_logits) -> dino_loss, with the teacher logits and center
    fixed, so gradcheck verifies the loss gradient w.r.t. the student output.
    """

    def __init__(self, B=2, K=8, n_crops=3):
        super().__init__()
        torch.manual_seed(0)
        self.teacher_out = [torch.randn(B, K, dtype=torch.double) for _ in range(2)]
        self.center = torch.zeros(1, K, dtype=torch.double)
        self.n_crops = n_crops

    def forward(self, student_flat):
        # student_flat is (n_crops, B, K); split into a list of crops.
        student_out = [student_flat[i] for i in range(self.n_crops)]
        return dino_loss(student_out, self.teacher_out, self.center,
                         student_temp=0.1, teacher_temp=0.04)


def test_dino_loss_gradcheck():
    mod = _DINOLossWrtStudent().double()
    student_flat = torch.randn(3, 2, 8, dtype=torch.double)
    assert check_gradients(mod, (student_flat,))


def test_teacher_has_no_grad():
    cfg = SSLConfig()
    _, teacher = build_student_teacher(cfg)
    assert all(not p.requires_grad for p in teacher.parameters())


def test_no_grad_reaches_teacher_after_backward():
    set_seed(0)
    cfg = SSLConfig()
    student, teacher = build_student_teacher(cfg)
    center = torch.zeros(1, cfg.out_dim)
    opt = torch.optim.Adam(student.parameters(), lr=cfg.dino_lr)
    img = torch.randn(cfg.overfit_batch, 3, 32, 32)
    dino_step(student, teacher, center, img, cfg, opt)
    # The student backward must not have written any grad into the teacher.
    assert all(p.grad is None for p in teacher.parameters())
