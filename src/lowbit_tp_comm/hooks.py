from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from torch import Tensor


@dataclass(slots=True)
class HookContext:
    """Context passed to simulated communication hooks."""

    layer_name: str | None = None
    collective: str = "all-gather"
    metadata: dict[str, Any] = field(default_factory=dict)


class CommunicationHook(Protocol):
    """Protocol for communication transforms applied during simulation."""

    def __call__(self, tensor: Tensor, context: HookContext) -> Tensor:
        ...


class NoOpCommunicationHook:
    """Default hook that leaves tensors unchanged."""

    def __call__(self, tensor: Tensor, context: HookContext) -> Tensor:
        _ = context
        return tensor
