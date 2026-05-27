from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor


@dataclass(slots=True)
class QuantizedTensor:
    """Container for future quantized communication payloads."""

    values: Tensor
    scale: Tensor | None
    zero_point: Tensor | None
    num_bits: int
    metadata: dict[str, Any] = field(default_factory=dict)


class IdentityQuantizer:
    """Placeholder quantizer that preserves tensors exactly."""

    def __init__(self, num_bits: int = 16) -> None:
        self.num_bits = num_bits

    def quantize(self, tensor: Tensor) -> QuantizedTensor:
        return QuantizedTensor(
            values=tensor.detach().clone(),
            scale=None,
            zero_point=None,
            num_bits=self.num_bits,
            metadata={"scheme": "identity"},
        )

    def dequantize(self, qtensor: QuantizedTensor) -> Tensor:
        return qtensor.values.detach().clone()
