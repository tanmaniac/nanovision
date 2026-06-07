"""DINO distillation loss falls when the student is trained to match a frozen
teacher target. Runs after the MAE tests.

This isolates the dino_loss wiring (Task 4) from the EMA and centering dynamics. The
teacher is frozen and its outputs are captured once, so the target does not move and
the student can reduce the cross-view cross-entropy toward it. The moving-teacher
collapse dynamics (centering and sharpening) are exercised separately in
test_dino_collapse.

The loss does not reach zero: the head's unit-norm features and bounded cosine
logits at student_temp can only approach, not match, the sharp teacher_temp target,
and the student's own prototypes move as it trains. So the criterion is a clear
relative drop, not convergence to ~0. Two smooth (low-frequency) views are used so
they share signal; on pure noise the cross-view target is unlearnable.
"""

import torch
import torch.nn.functional as F

from config import SSLConfig
from dino import build_student_teacher, dino_loss
from nanovision.determinism import set_seed


def test_dino_loss_decreases():
    set_seed(0)
    cfg = SSLConfig()
    student, teacher = build_student_teacher(cfg)
    center = torch.zeros(1, cfg.out_dim)
    opt = torch.optim.Adam(student.parameters(), lr=cfg.dino_lr)

    low = torch.randn(cfg.overfit_batch, 3, 8, 8)
    img = F.interpolate(low, size=(32, 32), mode="bicubic", align_corners=False)
    views = [img, img + 0.02 * torch.randn_like(img)]
    with torch.no_grad():
        teacher_out = [teacher(v).detach() for v in views]  # fixed target, no center update

    losses = []
    for _ in range(cfg.dino_steps):
        student_out = [student(v) for v in views]
        loss = dino_loss(student_out, teacher_out, center, cfg.student_temp, cfg.teacher_temp)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    initial = sum(losses[:5]) / 5
    final = sum(losses[-5:]) / 5
    assert final < 0.85 * initial, (
        f"DINO loss should fall toward the frozen teacher: initial {initial:.3f}, final {final:.3f}"
    )
