"""Explicit dtype routing and validation for replication runs."""

from __future__ import annotations

from collections import Counter
from typing import Any

import torch
from torch import nn


DTYPE_CHOICES = ("auto", "float32", "float16", "bfloat16")
_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def resolve_dtype(name: str) -> torch.dtype | None:
    """Resolve a CLI dtype name; ``auto`` intentionally leaves HF to decide."""

    if name == "auto":
        return None
    try:
        return _DTYPE_MAP[name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported dtype {name!r}; expected one of: {', '.join(DTYPE_CHOICES)}."
        ) from exc


def ensure_dtype_supported(dtype: torch.dtype | None, device: torch.device) -> None:
    """Fail loudly for CUDA BF16 requests that cannot execute natively."""

    if dtype is torch.bfloat16 and device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("BF16 was requested for CUDA, but CUDA is not available.")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "BF16 was requested on CUDA, but this GPU does not support native BF16 execution. "
                "Choose --dtype float16/float32 or use a BF16-capable GPU."
            )


def model_load_kwargs(dtype_name: str) -> dict[str, Any]:
    """Return the installed Transformers 5.x ``dtype`` argument explicitly.

    Transformers 5 exposes ``dtype`` and retains ``torch_dtype`` only for
    backwards compatibility.  Passing ``dtype`` preserves requested precision
    while ``auto`` retains checkpoint/config selection.
    """

    dtype = resolve_dtype(dtype_name)
    return {"dtype": "auto" if dtype is None else dtype}


def floating_parameter_dtypes(model: nn.Module) -> list[torch.dtype]:
    return sorted(
        {parameter.dtype for parameter in model.parameters() if parameter.is_floating_point()},
        key=str,
    )


def model_dtype_metadata(model: nn.Module) -> dict[str, Any]:
    parameters = [parameter for parameter in model.parameters() if parameter.is_floating_point()]
    buffers = [buffer for buffer in model.buffers() if buffer.is_floating_point()]
    parameter_counts = Counter(str(parameter.dtype) for parameter in parameters)
    buffer_counts = Counter(str(buffer.dtype) for buffer in buffers)
    first = parameters[0] if parameters else None
    return {
        "actual_model_device": str(first.device) if first is not None else None,
        "model_parameter_dtype": str(first.dtype) if first is not None else None,
        "model_parameter_dtypes": dict(sorted(parameter_counts.items())),
        "model_buffer_dtypes": dict(sorted(buffer_counts.items())),
    }


def validate_model_dtype(model: nn.Module, requested_dtype: torch.dtype | None) -> None:
    """Ensure an explicit request was not silently converted by model loading."""

    if requested_dtype is None:
        return
    actual = floating_parameter_dtypes(model)
    if not actual:
        raise ValueError("Loaded model has no floating-point parameters to validate.")
    if actual != [requested_dtype]:
        rendered = ", ".join(str(dtype) for dtype in actual)
        raise ValueError(
            f"Requested model dtype {requested_dtype}, but loaded floating parameters use: {rendered}."
        )


def validate_module_devices_and_dtypes(model: nn.Module, device: torch.device, expected_dtype: torch.dtype | None) -> None:
    """Check replacement/model state before inference; FP32 scales are allowed."""

    for name, parameter in model.named_parameters():
        if parameter.device != device:
            raise ValueError(f"Parameter {name!r} is on {parameter.device}, expected {device}.")
        if expected_dtype is not None and parameter.is_floating_point() and parameter.dtype != expected_dtype:
            raise ValueError(
                f"Parameter {name!r} has dtype {parameter.dtype}, expected {expected_dtype}."
            )
    for name, buffer in model.named_buffers():
        if buffer.device != device:
            raise ValueError(f"Buffer {name!r} is on {buffer.device}, expected {device}.")
