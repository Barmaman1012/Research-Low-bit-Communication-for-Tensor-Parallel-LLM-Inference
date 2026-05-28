from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, Sequence

import torch
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

from .calibration import EMAMinMaxCalibrator
from .tp_linear import (
    HybridQuantizedRowParallelConv1D,
    HybridQuantizedRowParallelLinear,
    make_random_bf16_indices,
)

try:
    from transformers.pytorch_utils import Conv1D
except ImportError:  # pragma: no cover
    Conv1D = None


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


def list_candidate_sync_modules(
    model: nn.Module,
    patterns: Sequence[str] | None = None,
    target_style: Literal["auto", "gpt2", "llama"] = "auto",
) -> list[tuple[str, nn.Module]]:
    """Find modules whose names look like TP synchronization points."""

    search_patterns = tuple(patterns or ["c_proj", "out_proj", "o_proj", "down_proj"])
    matches: list[tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if not name:
            continue
        if _matches_target_module(name, search_patterns, target_style):
            matches.append((name, module))
    return matches


def _matches_target_module(
    module_name: str,
    patterns: Sequence[str],
    target_style: Literal["auto", "gpt2", "llama"],
) -> bool:
    if target_style not in {"auto", "gpt2", "llama"}:
        raise ValueError(f"Unsupported target_style: {target_style}.")

    is_gpt2_sync = "attn.c_proj" in module_name or "mlp.c_proj" in module_name
    is_llama_sync = (
        module_name.endswith("o_proj")
        or module_name.endswith("down_proj")
        or ".o_proj" in module_name
        or ".down_proj" in module_name
    )

    if target_style == "gpt2":
        return is_gpt2_sync
    if target_style == "llama":
        return is_llama_sync

    if is_gpt2_sync or is_llama_sync:
        return True
    if "out_proj" in module_name and any(pattern == "out_proj" for pattern in patterns):
        return True
    return any(pattern in module_name for pattern in patterns if pattern not in {"c_proj", "o_proj", "down_proj"})


class ActivationCapture:
    """Register forward hooks and store detached CPU outputs by module name."""

    def __init__(self, model: nn.Module, module_names: Sequence[str]) -> None:
        self.module_names = list(module_names)
        self.outputs: dict[str, list[Tensor]] = defaultdict(list)
        self._handles: list[RemovableHandle] = []

        wanted = set(self.module_names)
        found = set()
        for name, module in model.named_modules():
            if name in wanted:
                self._handles.append(module.register_forward_hook(self._make_hook(name)))
                found.add(name)

        missing = sorted(wanted - found)
        if missing:
            self.remove()
            raise ValueError(f"Unknown module names for activation capture: {missing}")

    def _make_hook(self, module_name: str):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            if not isinstance(tensor, torch.Tensor):
                return
            self.outputs[module_name].append(tensor.detach().cpu())

        return hook

    def clear(self) -> None:
        self.outputs.clear()

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def get_outputs(self, module_name: str) -> list[Tensor]:
        return list(self.outputs.get(module_name, []))


def _resolve_child_module(parent: nn.Module, child_name: str) -> nn.Module:
    if child_name.isdigit():
        return parent[int(child_name)]  # type: ignore[index]
    return getattr(parent, child_name)


def _set_child_module(parent: nn.Module, child_name: str, replacement: nn.Module) -> None:
    if child_name.isdigit():
        parent[int(child_name)] = replacement  # type: ignore[index]
    else:
        setattr(parent, child_name, replacement)


def _resolve_module(model: nn.Module, module_name: str) -> nn.Module:
    current: nn.Module = model
    for part in module_name.split("."):
        current = _resolve_child_module(current, part)
    return current


def replace_modules_by_name(model: nn.Module, replacements: dict[str, nn.Module]) -> None:
    """Replace modules inside a model by their fully-qualified names."""

    for module_name, replacement in replacements.items():
        if not module_name:
            raise ValueError("Cannot replace the root module.")
        parent_name, child_name = module_name.rsplit(".", maxsplit=1) if "." in module_name else ("", module_name)
        parent_module = model if not parent_name else _resolve_module(model, parent_name)
        _set_child_module(parent_module, child_name, replacement)


def build_hybrid_replacements_from_calibration(
    model: nn.Module,
    calibration: dict,
    mode: str,
    num_partitions: int,
    num_bits: int = 4,
    seed: int = 0,
) -> dict[str, nn.Module]:
    """Construct simulated row-parallel module replacements from calibration results."""

    if mode == "full":
        return {}
    if mode not in {"int4", "selected_bf16", "random_bf16"}:
        raise ValueError(f"Unsupported mode: {mode}.")

    replacements: dict[str, nn.Module] = {}
    named_modules = dict(model.named_modules())
    for module_name, module_payload in calibration["modules"].items():
        if module_name not in named_modules:
            continue
        original_module = named_modules[module_name]
        calibrator = EMAMinMaxCalibrator.from_state_dict(module_payload["state_dict"])
        scales = calibrator.scales_per_partition()
        if scales.shape[0] == 1 and num_partitions > 1:
            # Calibration was collected with a single simulated partition. Real TP
            # deployment would need per-partition statistics; for now we repeat the
            # single-partition scale across simulated partitions.
            scales = scales.repeat(num_partitions, 1)
        elif scales.shape[0] != num_partitions:
            raise ValueError(
                f"Calibration scales for {module_name} have shape {tuple(scales.shape)}, "
                f"which is incompatible with num_partitions={num_partitions}."
            )

        k = int(module_payload["k"])
        feature_dim = int(module_payload["feature_dim"])
        if mode == "int4":
            bf16_indices = torch.empty(0, dtype=torch.long)
        elif mode == "selected_bf16":
            bf16_indices = module_payload["topk_indices"].to(dtype=torch.long)
        else:
            bf16_indices = make_random_bf16_indices(feature_dim, k, seed=seed)

        output_dtype = getattr(getattr(original_module, "weight", None), "dtype", torch.float32)
        if isinstance(original_module, nn.Linear):
            replacements[module_name] = HybridQuantizedRowParallelLinear.from_linear(
                original_module,
                num_partitions=num_partitions,
                scales_per_partition=scales.to(dtype=output_dtype),
                bf16_feature_indices=bf16_indices,
                num_bits=num_bits,
                output_dtype=output_dtype,
            )
        elif Conv1D is not None and isinstance(original_module, Conv1D):
            replacements[module_name] = HybridQuantizedRowParallelConv1D.from_conv1d(
                original_module,
                num_partitions=num_partitions,
                scales_per_partition=scales.to(dtype=output_dtype),
                bf16_feature_indices=bf16_indices,
                num_bits=num_bits,
                output_dtype=output_dtype,
            )
        else:
            raise TypeError(f"Unsupported module type for replacement: {module_name} ({type(original_module)}).")
    return replacements
