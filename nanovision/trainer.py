"""Training loop, sourced from A0 where the student builds the optimization step.

Loaded from assignments/a00_harness/trainer.py (or solution/ under
NANOVISION_IMPL=solution) through nanovision/_student.py. Import as
`from nanovision.trainer import Trainer`.
"""

from nanovision._student import load

_m = load("a00_harness", "trainer")
Trainer = _m.Trainer

__all__ = ["Trainer"]
