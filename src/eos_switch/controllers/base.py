"""Controller interface: a controller owns the active optimizer over training.

The training loop is controller-agnostic: each step it asks `begin_step` for
the optimizer to use (the controller may switch right there and record a
SwitchEvent), then reports the result via `end_step` / `end_epoch`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn


@dataclass
class SwitchEvent:
    step: int
    epoch: float
    from_name: str | None
    to_name: str
    from_lr: float | None
    to_lr: float
    reason: str

    def as_dict(self) -> dict:
        return asdict(self)


class Controller(ABC):
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.switch_events: list[SwitchEvent] = []
        self.model: nn.Module | None = None
        self.steps_per_epoch = 0
        self.total_epochs = 0

    def setup(self, model: nn.Module, steps_per_epoch: int, total_epochs: int) -> None:
        self.model = model
        self.steps_per_epoch = steps_per_epoch
        self.total_epochs = total_epochs
        self._build()

    @abstractmethod
    def _build(self) -> None:
        """Construct initial optimizer(s); called once from setup()."""

    @abstractmethod
    def begin_step(self, step: int) -> torch.optim.Optimizer:
        """Return the optimizer for this step (switching if needed)."""

    def end_step(self, step: int, loss: float) -> None:  # noqa: B027
        pass

    def end_epoch(self, epoch: int, val_acc: float) -> None:  # noqa: B027
        pass

    @property
    @abstractmethod
    def active_name(self) -> str:
        """Name of the currently active optimizer (for logging)."""

    @property
    @abstractmethod
    def active_optimizer(self) -> torch.optim.Optimizer:
        pass

    @property
    def active_lr(self) -> float:
        return float(self.active_optimizer.param_groups[0]["lr"])

    def _record_switch(self, event: SwitchEvent) -> None:
        self.switch_events.append(event)

    def epoch_of(self, step: int) -> float:
        return step / max(self.steps_per_epoch, 1)
