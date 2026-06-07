"""Task 5 EMA teacher update and centering buffer update. Runs after masking.

ema_update mixes teacher params toward the student by momentum; update_center moves
the center toward the batch mean.
"""

import copy

import torch

from backbone import DINOModel
from config import SSLConfig
from dino import ema_update, update_center
from nanovision.determinism import set_seed


def test_ema_update_mixes_params():
    set_seed(0)
    cfg = SSLConfig()
    student = DINOModel(img_size=32, patch=4, dim=cfg.dino_dim, depth=cfg.dino_depth,
                        n_heads=cfg.dino_heads, out_dim=cfg.out_dim,
                        head_hidden=cfg.head_hidden)
    teacher = copy.deepcopy(student)
    for p in teacher.parameters():
        p.requires_grad_(False)

    # Perturb the student so its params differ from the teacher.
    with torch.no_grad():
        for p in student.parameters():
            p.add_(torch.randn_like(p))

    old_teacher = [p.detach().clone() for p in teacher.parameters()]
    student_params = [p.detach().clone() for p in student.parameters()]
    m = 0.9
    ema_update(student, teacher, m)

    for t_new, t_old, s in zip(teacher.parameters(), old_teacher, student_params):
        expected = m * t_old + (1 - m) * s
        assert torch.allclose(t_new, expected, atol=1e-6)


def test_update_center_moves_toward_mean():
    K = 8
    center = torch.zeros(1, K)
    teacher_out = torch.randn(20, K) + 3.0   # mean clearly away from zero
    cm = 0.9
    new_center = update_center(center, teacher_out, cm)
    expected = cm * center + (1 - cm) * teacher_out.mean(dim=0, keepdim=True)
    assert torch.allclose(new_center, expected, atol=1e-6)
    # One step moves a fraction (1 - cm) of the way toward the batch mean.
    assert torch.all(new_center.abs() > center.abs())
