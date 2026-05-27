from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

from .config import CalibrationConfig


@dataclass(slots=True)
class TensorStats:
    """Summary statistics for an observed tensor."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    min_value: float
    max_value: float
    mean_abs: float


@dataclass(slots=True)
class CalibrationResult:
    """Collection of calibration observations and metadata."""

    stats: list[TensorStats] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class CalibrationRunner:
    """Placeholder entry point for future activation calibration passes."""

    def __init__(self, config: CalibrationConfig) -> None:
        self.config = config

    def summarize_tensor(self, name: str, tensor: Tensor) -> TensorStats:
        detached = tensor.detach()
        return TensorStats(
            name=name,
            shape=tuple(detached.shape),
            dtype=str(detached.dtype),
            min_value=float(detached.min().item()),
            max_value=float(detached.max().item()),
            mean_abs=float(detached.abs().mean().item()),
        )

    def calibrate_module(self, module: nn.Module) -> CalibrationResult:
        return CalibrationResult(
            metadata={
                "module": module.__class__.__name__,
                "status": "not_implemented",
                "num_samples": self.config.num_samples,
            }
        )
