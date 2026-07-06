"""A0 - fill the one hole in `Trainer.step`.

The reference implementation lives in this assignment's `solution/trainer.py`. Everything except
the optimization step is given; implement the step rhythm yourself.
"""

import csv
import os
from typing import Optional

import torch


class Trainer:
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
        """Run one optimization step on batch = (inputs, targets) and return the
        scalar loss for that step.

        See the optimization step section of the README.
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
        losses = []
        for s in range(steps):
            loss = self.step(batch)
            losses.append(loss)
            if s % log_every == 0 or s == steps - 1:
                self._log(s, loss)
        return losses

    def fit(self, train_loader, val_loader=None, max_steps: int = 1000, log_every: int = 50):
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
