from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import torch
from torch import Tensor, nn

from .config import CalibrationConfig
from .quantization import get_symmetric_scale


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


@dataclass(slots=True)
class CalibrationState:
    """EMA min/max state for tensor-parallel partition outputs."""

    min_vals: Tensor
    max_vals: Tensor
    initialized: bool = False


class EMAMinMaxCalibrator:
    """Track EMA min/max statistics for TP partial outputs during calibration."""

    def __init__(
        self,
        num_partitions: int,
        feature_dim: int,
        gamma: float = 0.01,
        device: Optional[torch.device | str] = None,
    ) -> None:
        if num_partitions <= 0:
            raise ValueError(f"num_partitions must be positive, got {num_partitions}.")
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}.")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {gamma}.")

        resolved_device = torch.device(device) if device is not None else None
        self.gamma = gamma
        self.num_partitions = num_partitions
        self.feature_dim = feature_dim
        self.state = CalibrationState(
            min_vals=torch.zeros(num_partitions, feature_dim, device=resolved_device),
            max_vals=torch.zeros(num_partitions, feature_dim, device=resolved_device),
            initialized=False,
        )

    @property
    def min_vals(self) -> Tensor:
        return self.state.min_vals

    @property
    def max_vals(self) -> Tensor:
        return self.state.max_vals

    @property
    def initialized(self) -> bool:
        return self.state.initialized

    def _reduce_partition(self, partial_output: Tensor) -> tuple[Tensor, Tensor]:
        if partial_output.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected final dimension {self.feature_dim}, got {partial_output.shape[-1]}."
            )
        if partial_output.ndim < 2:
            raise ValueError(
                f"Expected partial output shape [..., E] with at least 2 dims, got {tuple(partial_output.shape)}."
            )

        reduce_dims = tuple(range(partial_output.ndim - 1))
        return partial_output.amin(dim=reduce_dims), partial_output.amax(dim=reduce_dims)

    def update(self, partial_outputs: Sequence[Tensor]) -> None:
        if len(partial_outputs) != self.num_partitions:
            raise ValueError(
                f"Expected {self.num_partitions} partial outputs, got {len(partial_outputs)}."
            )

        current_mins: list[Tensor] = []
        current_maxs: list[Tensor] = []
        for partial_output in partial_outputs:
            min_vals, max_vals = self._reduce_partition(partial_output)
            current_mins.append(min_vals.to(device=self.min_vals.device, dtype=self.min_vals.dtype))
            current_maxs.append(max_vals.to(device=self.max_vals.device, dtype=self.max_vals.dtype))

        batch_min_vals = torch.stack(current_mins, dim=0)
        batch_max_vals = torch.stack(current_maxs, dim=0)

        if not self.initialized:
            self.state.min_vals = batch_min_vals
            self.state.max_vals = batch_max_vals
            self.state.initialized = True
            return

        gamma = self.gamma
        self.state.min_vals = (1.0 - gamma) * self.min_vals + gamma * batch_min_vals
        self.state.max_vals = (1.0 - gamma) * self.max_vals + gamma * batch_max_vals

    def ranges(self) -> Tensor:
        return 2.0 * torch.maximum(self.min_vals.abs(), self.max_vals.abs())

    def aggregated_ranges(self) -> Tensor:
        return self.ranges().sum(dim=0)

    def topk_features(self, k: int) -> Tensor:
        if k <= 0:
            return torch.empty(0, dtype=torch.long, device=self.min_vals.device)

        clamped_k = min(k, self.feature_dim)
        return torch.topk(self.aggregated_ranges(), k=clamped_k).indices

    def scales_per_partition(self, num_bits: int = 4, eps: float = 1e-8) -> Tensor:
        scales = [
            get_symmetric_scale(self.min_vals[i], self.max_vals[i], num_bits=num_bits, eps=eps)
            for i in range(self.num_partitions)
        ]
        return torch.stack(scales, dim=0)

    def state_dict(self) -> dict[str, Any]:
        return {
            "min_vals": self.min_vals.clone(),
            "max_vals": self.max_vals.clone(),
            "initialized": self.initialized,
            "gamma": self.gamma,
            "num_partitions": self.num_partitions,
            "feature_dim": self.feature_dim,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> EMAMinMaxCalibrator:
        calibrator = cls(
            num_partitions=int(state["num_partitions"]),
            feature_dim=int(state["feature_dim"]),
            gamma=float(state["gamma"]),
            device=state["min_vals"].device,
        )
        calibrator.state = CalibrationState(
            min_vals=state["min_vals"].clone(),
            max_vals=state["max_vals"].clone(),
            initialized=bool(state["initialized"]),
        )
        return calibrator


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
