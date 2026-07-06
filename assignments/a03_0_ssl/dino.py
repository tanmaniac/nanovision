"""A3 - DINO. Fill the three holes (Tasks 4-6), then run the tests.

The taught mechanisms are `dino_loss`, the pair `ema_update` / `update_center`,
and `teacher_entropy`. The student/teacher construction and the training-step
wiring used by the overfit and collapse tests are given. The reference lives in
`solution/dino.py` (read it if you get stuck).

Shapes: each crop produces (B, K) prototype logits; the center buffer is (1, K) or
(K,). The student is trained; the teacher is an EMA copy with requires_grad False.
"""

import copy

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from backbone import DINOModel, multi_crop


def dino_loss(student_out: list[Tensor], teacher_out: list[Tensor], center: Tensor,
              student_temp: float, teacher_temp: float) -> Tensor:
    """Cross-view self-distillation loss with centering + sharpening (Task 4).

    student_out: a list of (B, K) logits, one per student crop (all crops).
    teacher_out: a list of (B, K) logits, one per teacher crop (global crops only).
    center: (1, K) running mean subtracted from teacher logits.

    The teacher branch is centered and sharpened and carries a stop-gradient; the
    student branch is not. Cross-entropy is accumulated over every (teacher global
    crop, student crop) pair EXCEPT the matched same-index pair (a crop is not
    distilled against itself), and averaged over the counted pairs. Returns a scalar.

    See the DINO section of the README.
    """
    raise NotImplementedError("A3 Task 4: implement dino_loss")


@torch.no_grad()
def ema_update(student: nn.Module, teacher: nn.Module, momentum: float) -> None:
    """Update teacher params and buffers toward the student (Task 5a).

    Move every teacher parameter and buffer a momentum-weighted step toward the
    matching student tensor. Float buffers are mixed the same way as parameters;
    integer buffers are copied outright. The teacher is never touched by backprop;
    this EMA is the only thing that moves it, and higher momentum means a slower,
    more stable target. The function is already decorated @torch.no_grad().

    See the DINO section of the README.
    """
    raise NotImplementedError("A3 Task 5a: implement ema_update")


@torch.no_grad()
def update_center(center: Tensor, teacher_out: Tensor, center_momentum: float) -> Tensor:
    """EMA update of the centering buffer toward the batch mean (Task 5b).

    teacher_out: (M, K) stacked teacher logits over the batch (and crops). Move the
    center a momentum-weighted step toward the mean teacher logit over the M rows,
    keeping center's shape. This running mean is what centering subtracts before the
    teacher softmax; updating it outside the graph keeps it off the autograd path.

    See the DINO section of the README.
    """
    raise NotImplementedError("A3 Task 5b: implement update_center")


def teacher_entropy(teacher_out: Tensor, center: Tensor, teacher_temp: float) -> Tensor:
    """Mean entropy of the centered + sharpened teacher distribution (Task 6).

    teacher_out: (M, K) teacher logits. Form the same centered + sharpened teacher
    distribution used in the loss, and return the mean of its entropy over the M
    rows. This is the instrument the collapse test reads: near 0 means single-
    prototype collapse, near log K means uniform collapse, mid-range means healthy
    training.

    See the DINO section of the README.
    """
    raise NotImplementedError("A3 Task 6: implement teacher_entropy")


def build_student_teacher(cfg) -> tuple[nn.Module, nn.Module]:
    """Two DINOModel instances; the teacher starts as a frozen copy of the student.

    The teacher has requires_grad False on every parameter, so a student-loss
    backward never reaches it. It is moved only by ema_update.
    """
    student = DINOModel(
        img_size=cfg.img_size, patch=cfg.patch, in_chans=cfg.in_chans,
        dim=cfg.dino_dim, depth=cfg.dino_depth, n_heads=cfg.dino_heads,
        out_dim=cfg.out_dim, head_hidden=cfg.head_hidden,
    )
    teacher = copy.deepcopy(student)
    for p in teacher.parameters():
        p.requires_grad_(False)
    return student, teacher


def dino_step(student: nn.Module, teacher: nn.Module, center: Tensor, img: Tensor,
              cfg, optimizer, use_centering: bool = True,
              teacher_temp: float | None = None) -> tuple[float, Tensor, Tensor]:
    """One DINO optimization step (provided wiring for the overfit/collapse tests).

    Builds multi-crop views, runs the teacher on global crops (no grad) and the
    student on all crops, computes dino_loss, backprops into the student, EMA-updates
    the teacher, and updates the center. Returns (loss_value, new_center,
    teacher_global_logits). With use_centering=False the center stays at 0 (the
    no-centering ablation); teacher_temp overrides the sharpening temperature (the
    no-sharpening ablation uses a high value).
    """
    if teacher_temp is None:
        teacher_temp = cfg.teacher_temp

    globals_, locals_ = multi_crop(
        img, cfg.n_global, cfg.n_local, cfg.global_size, cfg.local_size
    )
    student_crops = globals_ + locals_

    with torch.no_grad():
        teacher_out = [teacher(g) for g in globals_]            # global crops only
    student_out = [student(c) for c in student_crops]

    loss = dino_loss(student_out, teacher_out, center, cfg.student_temp, teacher_temp)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    ema_update(student, teacher, cfg.ema_momentum)
    teacher_cat = torch.cat([t.detach() for t in teacher_out], dim=0)
    if use_centering:
        center = update_center(center, teacher_cat, cfg.center_momentum)
    return loss.item(), center, teacher_cat
