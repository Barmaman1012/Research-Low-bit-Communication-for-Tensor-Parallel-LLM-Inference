from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
from torch import Tensor


# Kept here with the two-tier quantization implementation so callers have one
# canonical spelling for the experimental mode.  It deliberately has no Int8
# counterpart: selected channels are restored with the normal BF16 + Int4 path.
THRESHOLD_BF16_MODE = "threshold_bf16"
RANGE_THRESHOLD_BF16_MODE = "range_threshold_bf16"
MATCHED_LOW_RANGE_BF16_MODE = "matched_low_range_bf16"

@dataclass(slots=True)
class QuantizedTensor:
    """Container for simulated quantized communication payloads."""

    values: Tensor
    scale: Tensor | None
    zero_point: Tensor | None
    num_bits: int
    metadata: dict[str, Any] = field(default_factory=dict)


def _validate_num_bits(num_bits: int) -> None:
    if not 2 <= num_bits <= 8:
        raise ValueError(f"Only signed low-bit simulation with 2 <= num_bits <= 8 is supported, got {num_bits}.")


def get_qmin_qmax(num_bits: int) -> tuple[int, int]:
    _validate_num_bits(num_bits)
    return -(2 ** (num_bits - 1)), (2 ** (num_bits - 1)) - 1


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

    _, qmax = get_qmin_qmax(num_bits)
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

    qmin, qmax = get_qmin_qmax(num_bits)

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


def multi_tier_quant_dequant(
    x: Tensor,
    int4_scale: Tensor,
    int8_scale: Tensor,
    bf16_feature_indices: Tensor | Sequence[int] | None = None,
    int8_feature_indices: Tensor | Sequence[int] | None = None,
    output_dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Simulate disjoint BF16, Int8, and Int4 communicated feature tiers."""

    feature_dim = x.shape[-1]

    def normalize(values: Tensor | Sequence[int] | None) -> Tensor:
        if values is None:
            result = torch.empty(0, dtype=torch.long, device=x.device)
        elif isinstance(values, Tensor):
            result = values.to(device=x.device, dtype=torch.long)
        else:
            result = torch.tensor(list(values), dtype=torch.long, device=x.device)
        if result.numel() and (result.min() < 0 or result.max() >= feature_dim):
            raise ValueError("Feature indices are out of range.")
        if result.numel() != result.unique().numel():
            raise ValueError("Feature indices must not contain duplicates.")
        return result

    bf16_indices = normalize(bf16_feature_indices)
    int8_indices = normalize(int8_feature_indices)
    if bf16_indices.numel() and int8_indices.numel() and torch.isin(bf16_indices, int8_indices).any():
        raise ValueError("BF16 and Int8 feature indices must be disjoint.")
    reconstructed = dequantize_symmetric(quantize_symmetric(x, int4_scale, num_bits=4), int4_scale, dtype=output_dtype)
    if int8_indices.numel():
        int8 = dequantize_symmetric(quantize_symmetric(x, int8_scale, num_bits=8), int8_scale, dtype=output_dtype)
        reconstructed[..., int8_indices] = int8[..., int8_indices]
    if bf16_indices.numel():
        reconstructed[..., bf16_indices] = x[..., bf16_indices].to(output_dtype)
    return reconstructed


def quantization_error_stats(
    original: Tensor,
    reconstructed: Tensor,
    q: Tensor | None = None,
    qmin: int = -8,
    qmax: int = 7,
    selected_indices: Tensor | Sequence[int] | None = None,
) -> dict[str, float]:
    """Summarize reconstruction and saturation behavior for quantized tensors."""

    if original.shape != reconstructed.shape:
        raise ValueError(
            f"original and reconstructed must have the same shape, got {original.shape} and {reconstructed.shape}."
        )

    original_fp = original.to(torch.float32)
    reconstructed_fp = reconstructed.to(torch.float32)
    error = reconstructed_fp - original_fp
    abs_error = error.abs()
    mse = (error ** 2).mean()
    original_std = float(original_fp.std(unbiased=False).item())

    stats: dict[str, float] = {
        "mean_abs_error": float(abs_error.mean().item()),
        "max_abs_error": float(abs_error.max().item()),
        "rmse": float(torch.sqrt(mse).item()),
        "mean_signed_error": float(error.mean().item()),
        "relative_rmse": float(torch.sqrt(mse).item()) / max(original_std, 1e-12),
    }

    if q is not None:
        q_fp = q.to(torch.float32)
        stats["quantized_min"] = float(q_fp.min().item())
        stats["quantized_max"] = float(q_fp.max().item())
        stats["saturation_low_rate"] = float((q == qmin).to(torch.float32).mean().item())
        stats["saturation_high_rate"] = float((q == qmax).to(torch.float32).mean().item())

    if selected_indices is not None:
        if isinstance(selected_indices, Tensor):
            indices = selected_indices.to(dtype=torch.long, device=original.device)
        else:
            indices = torch.tensor(list(selected_indices), dtype=torch.long, device=original.device)

        feature_dim = original.shape[-1]
        mask = torch.zeros(feature_dim, dtype=torch.bool, device=original.device)
        if indices.numel() > 0:
            mask[indices] = True

        selected_error = abs_error[..., mask]
        non_selected_error = abs_error[..., ~mask]
        stats["selected_fraction"] = float(mask.to(torch.float32).mean().item())
        stats["selected_mean_abs_error"] = float(selected_error.mean().item()) if selected_error.numel() > 0 else 0.0
        stats["selected_max_abs_error"] = float(selected_error.max().item()) if selected_error.numel() > 0 else 0.0
        stats["non_selected_mean_abs_error"] = (
            float(non_selected_error.mean().item()) if non_selected_error.numel() > 0 else 0.0
        )
        stats["non_selected_max_abs_error"] = (
            float(non_selected_error.max().item()) if non_selected_error.numel() > 0 else 0.0
        )

    return stats


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
