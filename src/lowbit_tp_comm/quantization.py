from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
from torch import Tensor


@dataclass(slots=True)
class QuantizedTensor:
    """Container for simulated quantized communication payloads."""

    values: Tensor
    scale: Tensor | None
    zero_point: Tensor | None
    num_bits: int
    metadata: dict[str, Any] = field(default_factory=dict)


def _validate_num_bits(num_bits: int) -> None:
    if num_bits != 4:
        raise ValueError(f"Only signed Int4 simulation is supported, got num_bits={num_bits}.")


def get_symmetric_scale(
    min_vals: Tensor,
    max_vals: Tensor,
    num_bits: int = 4,
    eps: float = 1e-8,
) -> Tensor:
    """Compute per-feature symmetric scales for signed Int4 quantization."""

    _validate_num_bits(num_bits)
    if min_vals.shape != max_vals.shape:
        raise ValueError(
            f"min_vals and max_vals must have the same shape, got {min_vals.shape} and {max_vals.shape}."
        )
    if min_vals.ndim != 1:
        raise ValueError(f"Expected per-feature vectors with shape [E], got ndim={min_vals.ndim}.")

    qmax = (2 ** (num_bits - 1)) - 1
    symmetric_abs = torch.maximum(min_vals.abs(), max_vals.abs())
    return torch.clamp(symmetric_abs / float(qmax), min=eps)


def quantize_symmetric(x: Tensor, scale: Tensor, num_bits: int = 4) -> Tensor:
    """Simulate signed Int4 quantization and store results in int8.

    Note:
        This is a numerical simulation only. PyTorch has no native packed Int4
        dtype here, so each quantized value still occupies one int8 element.
        Real communication savings would require bit-packing two 4-bit values
        into one byte before transport.
    """

    _validate_num_bits(num_bits)
    qmin = -(2 ** (num_bits - 1))
    qmax = (2 ** (num_bits - 1)) - 1

    q = torch.round(x / scale)
    q = torch.clamp(q, min=qmin, max=qmax)
    return q.to(torch.int8)


def dequantize_symmetric(
    q: Tensor,
    scale: Tensor,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Dequantize a simulated signed Int4 tensor using per-feature scales."""

    return (q.to(torch.float32) * scale).to(dtype)


def hybrid_quant_dequant(
    x: Tensor,
    scale: Tensor,
    bf16_feature_indices: Tensor | Sequence[int] | None = None,
    num_bits: int = 4,
    output_dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Quantize/dequantize all features, then restore selected features exactly.

    The selected feature set simulates channels that remain in higher precision.
    As above, this does not perform true Int4 packing yet, so communication
    compression is modeled numerically rather than by reduced byte count.
    """

    q = quantize_symmetric(x, scale, num_bits=num_bits)
    reconstructed = dequantize_symmetric(q, scale, dtype=output_dtype)

    if bf16_feature_indices is None:
        return reconstructed

    if isinstance(bf16_feature_indices, Tensor):
        indices = bf16_feature_indices.to(device=x.device, dtype=torch.long)
    else:
        indices = torch.tensor(list(bf16_feature_indices), device=x.device, dtype=torch.long)

    if indices.numel() == 0:
        return reconstructed

    reconstructed[..., indices] = x[..., indices].to(output_dtype)
    return reconstructed


class IdentityQuantizer:
    """Compatibility utility that preserves tensors exactly."""

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
