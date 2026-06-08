"""The collapse test (centerpiece): centering and sharpening each prevent a
distinct teacher-distribution collapse. Runs after the overfit tests.

Three short DINO variants on one synthetic batch, reading teacher_entropy along the
way:
  (a) full DINO          - entropy stays mid-range (neither ~0 nor ~log K).
  (b) no centering       - entropy collapses toward 0 (single prototype).
  (c) no sharpening      - entropy rises toward log K (uniform).

Assert the end-state entropies are ordered collapse < full < uniform with margins.
K is kept small (out_dim in config) so log K and runtimes are modest.
"""

import math

import torch

from config import SSLConfig
from dino import build_student_teacher, dino_step, teacher_entropy
from nanovision.determinism import set_seed


def _run_variant(cfg, img, use_centering, teacher_temp, steps):
    set_seed(0)
    student, teacher = build_student_teacher(cfg)
    center = torch.zeros(1, cfg.out_dim)
    opt = torch.optim.Adam(student.parameters(), lr=cfg.dino_lr)
    entropies = []
    for _ in range(steps):
        _, center, teacher_cat = dino_step(
            student, teacher, center, img, cfg, opt,
            use_centering=use_centering, teacher_temp=teacher_temp,
        )
        # Read the instrument with the same center/temp the teacher distribution used.
        read_center = center if use_centering else torch.zeros_like(center)
        entropies.append(teacher_entropy(teacher_cat, read_center, teacher_temp).item())
    return entropies


def test_collapse_ordering():
    cfg = SSLConfig()
    cfg.ema_momentum = cfg.collapse_momentum  # fast-tracking teacher so collapse manifests
    set_seed(0)
    img = torch.randn(cfg.overfit_batch, 3, 32, 32)
    steps = cfg.dino_steps
    log_k = math.log(cfg.out_dim)

    full = _run_variant(cfg, img, use_centering=True, teacher_temp=cfg.teacher_temp, steps=steps)
    no_center = _run_variant(cfg, img, use_centering=False, teacher_temp=cfg.teacher_temp, steps=steps)
    no_sharp = _run_variant(cfg, img, use_centering=True, teacher_temp=1.0, steps=steps)

    e_full = sum(full[-10:]) / 10
    e_collapse = sum(no_center[-10:]) / 10
    e_uniform = sum(no_sharp[-10:]) / 10

    # No-centering drives single-prototype collapse: entropy near 0.
    assert e_collapse < 0.5, f"no-centering should collapse toward 0, got {e_collapse:.3f}"
    # No-sharpening drives uniform output: entropy near log K.
    assert e_uniform > 0.9 * log_k, f"no-sharpening should approach log K={log_k:.2f}, got {e_uniform:.3f}"
    # Full DINO sits in between with margins.
    assert e_collapse + 0.3 < e_full < e_uniform - 0.3, (
        f"ordering collapse<full<uniform failed: {e_collapse:.3f}, {e_full:.3f}, {e_uniform:.3f}"
    )
