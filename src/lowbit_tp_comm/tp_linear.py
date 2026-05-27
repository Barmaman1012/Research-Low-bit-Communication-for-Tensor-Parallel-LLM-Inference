from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .config import TensorParallelConfig
from .hooks import CommunicationHook, HookContext, NoOpCommunicationHook


@dataclass(slots=True)
class TensorParallelShardSpec:
    """Describes a virtual tensor-parallel partition for a linear layer."""

    in_features: int
    out_features: int
    tp_degree: int
    shard_dimension: str


class SimulatedTPLinear(nn.Module):
    """Single-process stand-in for a tensor-parallel linear layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        tp_config: TensorParallelConfig | None = None,
        layer_name: str | None = None,
        comm_hook: CommunicationHook | None = None,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.tp_config = tp_config or TensorParallelConfig()
        self.layer_name = layer_name or "tp_linear"
        self.comm_hook: CommunicationHook = comm_hook or NoOpCommunicationHook()
        self.shard_spec = TensorParallelShardSpec(
            in_features=in_features,
            out_features=out_features,
            tp_degree=self.tp_config.tp_degree,
            shard_dimension=self.tp_config.shard_dimension,
        )

    def set_comm_hook(self, hook: CommunicationHook) -> None:
        self.comm_hook = hook

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = self.linear(inputs)
        context = HookContext(
            layer_name=self.layer_name,
            collective="all-gather",
            metadata={"tp_degree": self.tp_config.tp_degree},
        )
        return self.comm_hook(outputs, context)
