"""A small generic training loop with CSV logging.

No external experiment tracker. `overfit_one_batch` is the canonical correctness
signal used across the course: a correct model drives a single batch to ~0 loss.
"""

import csv
import os
from typing import Optional

import torch


class Trainer:
    """Wraps a model, optimizer, and loss into fit / overfit_one_batch loops.

    A batch is `(inputs, targets)`; the loss is `loss_fn(model(inputs), targets)`.
    Losses are appended to `self.history` and, if `log_dir` is set, to a CSV the
    viz scripts read.
    """

    def __init__(self, model, optimizer, loss_fn, device: str = "cpu", log_dir: Optional[str] = None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.log_dir = log_dir
        self.history: list[tuple[int, float]] = []
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            self._csv = os.path.join(log_dir, "loss.csv")
            with open(self._csv, "w", newline="") as f:
                csv.writer(f).writerow(["step", "loss"])

    def _to_device(self, batch):
        return tuple(x.to(self.device) if torch.is_tensor(x) else x for x in batch)

    def step(self, batch) -> float:
        """One optimization step: zero grads, forward, loss, backward, step.

        Returns the scalar loss value.
        """
        inputs, targets = self._to_device(batch)
        self.model.train()
        self.optimizer.zero_grad()
        pred = self.model(inputs)
        loss = self.loss_fn(pred, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _log(self, step: int, loss: float) -> None:
        self.history.append((step, loss))
        if self.log_dir:
            with open(self._csv, "a", newline="") as f:
                csv.writer(f).writerow([step, loss])

    def overfit_one_batch(self, batch, steps: int = 500, log_every: int = 50) -> list[float]:
        """Repeatedly train on a single batch; return the per-step loss list."""
        losses = []
        for s in range(steps):
            loss = self.step(batch)
            losses.append(loss)
            if s % log_every == 0 or s == steps - 1:
                self._log(s, loss)
        return losses

    def fit(self, train_loader, val_loader=None, max_steps: int = 1000, log_every: int = 50):
        """Train over `train_loader` for up to `max_steps` optimization steps."""
        step = 0
        while step < max_steps:
            for batch in train_loader:
                loss = self.step(batch)
                if step % log_every == 0:
                    self._log(step, loss)
                step += 1
                if step >= max_steps:
                    break
        return self.history
