from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math
from typing import Any, Literal, Protocol, Sequence

import torch
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

from .calibration import EMAMinMaxCalibrator
from .quantization import MATCHED_LOW_RANGE_BF16_MODE, RANGE_THRESHOLD_BF16_MODE, THRESHOLD_BF16_MODE
from .tp_linear import (
    HybridQuantizedRowParallelConv1D,
    HybridQuantizedRowParallelLinear,
    RowParallelConv1D,
    RowParallelLinear,
    ThreeTierRowParallelConv1D,
    ThreeTierRowParallelLinear,
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


class ModuleInputOutputCapture:
    """Capture module inputs via pre-hooks and outputs via forward hooks.

    By default captured tensors are copied to CPU for inexpensive storage.  Set
    ``store_on_cpu=False`` when a caller needs to reuse a captured tensor with
    the module that produced it (for example, simulated row-parallel partial
    computation on a CUDA model).
    """

    def __init__(
        self,
        model: nn.Module,
        module_names: Sequence[str],
        *,
        store_on_cpu: bool = True,
    ) -> None:
        self.module_names = list(module_names)
        self.store_on_cpu = store_on_cpu
        self.inputs: dict[str, list[Tensor]] = defaultdict(list)
        self.outputs: dict[str, list[Tensor]] = defaultdict(list)
        self._handles: list[RemovableHandle] = []

        wanted = set(self.module_names)
        found = set()
        for name, module in model.named_modules():
            if name in wanted:
                self._handles.append(module.register_forward_pre_hook(self._make_pre_hook(name)))
                self._handles.append(module.register_forward_hook(self._make_post_hook(name)))
                found.add(name)

        missing = sorted(wanted - found)
        if missing:
            self.remove()
            raise ValueError(f"Unknown module names for input/output capture: {missing}")

    def _make_pre_hook(self, module_name: str):
        def hook(_module: nn.Module, inputs: tuple[Any, ...]) -> None:
            if not inputs:
                return
            tensor = inputs[0]
            if not isinstance(tensor, torch.Tensor):
                return
            captured = tensor.detach()
            self.inputs[module_name].append(captured.cpu() if self.store_on_cpu else captured)

        return hook

    def _make_post_hook(self, module_name: str):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            if not isinstance(tensor, torch.Tensor):
                return
            captured = tensor.detach()
            self.outputs[module_name].append(captured.cpu() if self.store_on_cpu else captured)

        return hook

    def clear(self) -> None:
        self.inputs.clear()
        self.outputs.clear()

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def get_inputs(self, module_name: str) -> list[Tensor]:
        return list(self.inputs.get(module_name, []))

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


def derive_threshold_bf16_selection(
    calibration: dict[str, Any],
    *,
    model: nn.Module | None = None,
) -> dict[str, Any]:
    """Allocate the selected-BF16 budget globally by module-median range.

    Ranking is intentionally done once, on CPU, with an explicit secondary
    order of module name then feature index.  This makes equal boundary scores
    deterministic and avoids any sorting work in replacement forwards.
    """

    modules = calibration.get("modules")
    if not isinstance(modules, dict) or not modules:
        raise ValueError("threshold_bf16 requires a non-empty calibration 'modules' mapping.")

    named_modules = dict(model.named_modules()) if model is not None else {}
    entries: list[tuple[float, str, int]] = []
    module_feature_dims: dict[str, int] = {}
    target_count = 0
    for module_name in sorted(modules):
        payload = modules[module_name]
        if not isinstance(payload, dict):
            raise ValueError(f"Calibration entry for {module_name!r} must be a mapping.")
        if model is not None and module_name not in named_modules:
            raise ValueError(f"Calibrated target module {module_name!r} is not present in the model.")
        try:
            feature_dim = int(payload["feature_dim"])
            k = int(payload["k"])
            ranges = payload["aggregated_ranges"]
        except KeyError as exc:
            raise ValueError(f"Calibration entry for {module_name!r} is missing {exc.args[0]!r}.") from exc
        if feature_dim <= 0 or not 0 <= k <= feature_dim:
            raise ValueError(f"Invalid feature_dim/k for calibrated target module {module_name!r}.")
        if not isinstance(ranges, Tensor) or ranges.ndim != 1 or ranges.numel() != feature_dim:
            actual = tuple(ranges.shape) if isinstance(ranges, Tensor) else type(ranges).__name__
            raise ValueError(
                f"aggregated_ranges for {module_name!r} must have shape [{feature_dim}], got {actual}."
            )
        if model is not None:
            model_feature_dim = _module_feature_dim_for_threshold(named_modules[module_name])
            if model_feature_dim != feature_dim:
                raise ValueError(
                    f"Calibration feature dimension mismatch for {module_name!r}: "
                    f"artifact has {feature_dim}, but model requires {model_feature_dim}."
                )
        values = ranges.detach().to(device="cpu", dtype=torch.float32)
        median = values.median()
        if not torch.isfinite(median) or median.item() <= 0:
            raise ValueError(f"aggregated_ranges median for {module_name!r} must be finite and positive.")
        normalized = values / median
        if not torch.isfinite(normalized).all():
            raise ValueError(f"aggregated_ranges for {module_name!r} must be finite.")
        entries.extend((float(score), module_name, index) for index, score in enumerate(normalized.tolist()))
        module_feature_dims[module_name] = feature_dim
        target_count += k

    if target_count > len(entries):  # defensive; k validation above should imply this.
        raise ValueError("threshold_bf16 target BF16 count exceeds total calibrated features.")
    ranked = sorted(entries, key=lambda item: (-item[0], item[1], item[2]))
    selected = ranked[:target_count]
    next_excluded = ranked[target_count] if target_count < len(ranked) else None
    threshold = selected[-1][0] if selected else None
    boundary_tie = bool(next_excluded is not None and threshold == next_excluded[0])
    indices_by_module = {name: [] for name in module_feature_dims}
    for _score, module_name, index in selected:
        indices_by_module[module_name].append(index)
    per_module = {
        name: {
            "bf16_count": len(indices_by_module[name]),
            "bf16_fraction": len(indices_by_module[name]) / module_feature_dims[name],
            "feature_dim": module_feature_dims[name],
        }
        for name in sorted(module_feature_dims)
    }
    total_features = len(entries)
    return {
        "mode": THRESHOLD_BF16_MODE,
        "normalization_method": "module_median",
        "total_feature_count": total_features,
        "target_bf16_count": target_count,
        "actual_bf16_count": len(selected),
        "derived_threshold": threshold,
        "next_excluded_score": None if next_excluded is None else next_excluded[0],
        "boundary_tie": boundary_tie,
        "per_module": per_module,
        "global_bf16_fraction": len(selected) / total_features,
        "global_int4_fraction": (total_features - len(selected)) / total_features,
        "average_bits_per_value": 4.0 + 12.0 * len(selected) / total_features,
        "indices_by_module": {
            name: torch.tensor(indices, dtype=torch.long) for name, indices in indices_by_module.items()
        },
    }


def threshold_bf16_result_metadata(
    selection: dict[str, Any],
    *,
    calibration_path: str | None,
    calibration_sha256: str | None,
) -> dict[str, Any]:
    """Serialize the exact construction-time threshold allocation for results."""

    per_module = selection["per_module"]
    counts = {name: int(details["bf16_count"]) for name, details in per_module.items()}
    fractions = {name: float(details["bf16_fraction"]) for name, details in per_module.items()}
    return {
        "mode": THRESHOLD_BF16_MODE,
        "normalization_method": selection["normalization_method"],
        "total_feature_count": int(selection["total_feature_count"]),
        "target_bf16_count": int(selection["target_bf16_count"]),
        "actual_bf16_count": int(selection["actual_bf16_count"]),
        "global_bf16_fraction": float(selection["global_bf16_fraction"]),
        "global_int4_fraction": float(selection["global_int4_fraction"]),
        "average_bits_per_value": float(selection["average_bits_per_value"]),
        "derived_threshold": selection["derived_threshold"],
        "next_excluded_score": selection["next_excluded_score"],
        "boundary_tie": bool(selection["boundary_tie"]),
        "deterministic_tie_breaking": "normalized_score_desc,module_name_asc,feature_index_asc",
        "per_module_bf16_counts": counts,
        "per_module_bf16_fractions": fractions,
        "calibration_path": calibration_path,
        "calibration_sha256": calibration_sha256,
    }


def _threshold_entries(calibration: dict[str, Any], model: nn.Module | None) -> tuple[list[tuple[float, str, int]], dict[str, int]]:
    """Validate calibrated targets and return normalized feature scores once."""
    modules = calibration.get("modules")
    if not isinstance(modules, dict) or not modules:
        raise ValueError("Range-threshold modes require a non-empty calibration 'modules' mapping.")
    named = dict(model.named_modules()) if model is not None else {}
    entries: list[tuple[float, str, int]] = []
    dimensions: dict[str, int] = {}
    for name in sorted(modules):
        payload = modules[name]
        if model is not None and name not in named:
            raise ValueError(f"Calibrated target module {name!r} is not present in the model.")
        dim = int(payload["feature_dim"]); values = payload.get("aggregated_ranges")
        if not isinstance(values, Tensor) or values.ndim != 1 or values.numel() != dim:
            raise ValueError(f"aggregated_ranges for {name!r} must have shape [{dim}].")
        if model is not None and _module_feature_dim_for_threshold(named[name]) != dim:
            raise ValueError(f"Calibration feature dimension mismatch for {name!r}.")
        values = values.detach().to(device="cpu", dtype=torch.float32)
        if not torch.isfinite(values).all():
            raise ValueError(f"aggregated_ranges for {name!r} must be finite.")
        median = values.median()
        if not torch.isfinite(median) or median.item() <= 0:
            raise ValueError(f"aggregated_ranges median for {name!r} must be finite and positive.")
        normalized = values / median
        entries.extend((float(score), name, index) for index, score in enumerate(normalized.tolist()))
        dimensions[name] = dim
    return entries, dimensions


def derive_range_threshold_bf16_selection(
    calibration: dict[str, Any], *, threshold: float, mode: str, model: nn.Module | None = None,
) -> dict[str, Any]:
    """Construct semantic high-range or matched-low BF16 indices once."""
    if mode not in {RANGE_THRESHOLD_BF16_MODE, MATCHED_LOW_RANGE_BF16_MODE}:
        raise ValueError(f"Unsupported range-threshold mode: {mode}.")
    if threshold is None or not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("bf16_range_threshold must be finite and positive.")
    entries, dimensions = _threshold_entries(calibration, model)
    high = [entry for entry in entries if entry[0] >= threshold]
    if mode == RANGE_THRESHOLD_BF16_MODE:
        selected = high; direction = "high_range"; operator = ">="; tie = "module-local inclusive >= threshold"
    else:
        selected = sorted(entries, key=lambda item: (item[0], item[1], item[2]))[:len(high)]
        direction = "low_range"; operator = "matched globally smallest"; tie = "normalized_score_asc,module_name_asc,feature_index_asc"
    indices = {name: [] for name in dimensions}
    for _score, name, index in selected: indices[name].append(index)
    per_module = {name: {"bf16_count": len(indices[name]), "bf16_fraction": len(indices[name]) / dimensions[name], "feature_dim": dimensions[name]} for name in sorted(dimensions)}
    total = len(entries); count = len(selected)
    return {"mode": mode, "selection_direction": direction, "normalization_method": "module_median",
            "supplied_threshold": threshold, "comparison_operator": operator, "total_feature_count": total,
            "bf16_feature_count": count, "int4_feature_count": total - count,
            "global_bf16_fraction": count / total, "global_int4_fraction": (total-count) / total,
            "average_bits_per_value": 4.0 + 12.0 * count / total, "per_module": per_module,
            "minimum_module_count": min(row["bf16_count"] for row in per_module.values()),
            "median_module_count": float(torch.tensor([row["bf16_count"] for row in per_module.values()]).median()),
            "maximum_module_count": max(row["bf16_count"] for row in per_module.values()),
            "deterministic_tie_breaking": tie, "matched_high_range_count": len(high) if mode == MATCHED_LOW_RANGE_BF16_MODE else None,
            "indices_by_module": {name: torch.tensor(values, dtype=torch.long) for name, values in indices.items()}}


def range_threshold_bf16_result_metadata(selection: dict[str, Any], *, calibration_path: str | None, calibration_sha256: str | None) -> dict[str, Any]:
    """Serialize the exact explicit-threshold allocation used by replacements."""
    per_module = selection["per_module"]
    return {key: selection[key] for key in ("mode", "selection_direction", "normalization_method", "supplied_threshold", "comparison_operator", "total_feature_count", "bf16_feature_count", "int4_feature_count", "global_bf16_fraction", "global_int4_fraction", "average_bits_per_value", "minimum_module_count", "median_module_count", "maximum_module_count", "deterministic_tie_breaking", "matched_high_range_count")} | {
        "per_module_bf16_counts": {name: int(row["bf16_count"]) for name, row in per_module.items()},
        "per_module_bf16_fractions": {name: float(row["bf16_fraction"]) for name, row in per_module.items()},
        "calibration_path": calibration_path, "calibration_sha256": calibration_sha256}


def _module_feature_dim_for_threshold(module: nn.Module) -> int:
    if isinstance(module, nn.Linear):
        return int(module.out_features)
    if Conv1D is not None and isinstance(module, Conv1D):
        return int(module.weight.shape[1])
    raise TypeError(f"Unsupported calibrated target module type: {type(module)}")


def build_hybrid_replacements_from_calibration(
    model: nn.Module,
    calibration: dict,
    mode: str,
    num_partitions: int,
    num_bits: int = 4,
    seed: int = 0,
    int8_fraction: float = 0.015625,
    bf16_range_threshold: float | None = None,
) -> dict[str, nn.Module]:
    """Construct simulated row-parallel module replacements from calibration results."""

    if mode == "full":
        return {}
    if mode not in {"tp_uncompressed", "all_bf16", "int4", "selected_bf16", "random_bf16", THRESHOLD_BF16_MODE, RANGE_THRESHOLD_BF16_MODE, MATCHED_LOW_RANGE_BF16_MODE, "selected_bf16_int8", "selected_bf16_random_int8"}:
        raise ValueError(f"Unsupported mode: {mode}.")
    if not 0.0 <= int8_fraction <= 1.0:
        raise ValueError("int8_fraction must be in [0, 1].")

    threshold_selection = derive_threshold_bf16_selection(calibration, model=model) if mode == THRESHOLD_BF16_MODE else None
    range_selection = derive_range_threshold_bf16_selection(calibration, threshold=bf16_range_threshold, mode=mode, model=model) if mode in {RANGE_THRESHOLD_BF16_MODE, MATCHED_LOW_RANGE_BF16_MODE} else None
    replacements: dict[str, nn.Module] = {}
    named_modules = dict(model.named_modules())
    for module_name, module_payload in calibration["modules"].items():
        if module_name not in named_modules:
            if mode in {THRESHOLD_BF16_MODE, RANGE_THRESHOLD_BF16_MODE, MATCHED_LOW_RANGE_BF16_MODE}:
                raise ValueError(f"Calibrated target module {module_name!r} is not present in the model.")
            continue
        original_module = named_modules[module_name]
        calibrator = EMAMinMaxCalibrator.from_state_dict(module_payload["state_dict"])
        scales = calibrator.scales_per_partition()
        if scales.shape[0] == 1 and num_partitions > 1:
            # Calibration was collected with a single simulated partition. Real TP
            # deployment would need per-partition statistics; for now we repeat the
            # single-partition scale across simulated partitions.
            print(
                "Warning: repeating full-output calibration scales across partitions; "
                "this is less faithful than simulated row-parallel calibration."
            )
            scales = scales.repeat(num_partitions, 1)
        elif scales.shape[0] != num_partitions:
            raise ValueError(
                f"Calibration scales for {module_name} have shape {tuple(scales.shape)}, "
                f"which is incompatible with num_partitions={num_partitions}."
            )

        k = int(module_payload["k"])
        feature_dim = int(module_payload["feature_dim"])
        if k > feature_dim or k + math.floor(feature_dim * int8_fraction) > feature_dim:
            raise ValueError("BF16 and Int8 feature counts exceed feature dimension.")
        if mode == "tp_uncompressed":
            bf16_indices = None
        elif mode == "all_bf16":
            bf16_indices = torch.arange(feature_dim, dtype=torch.long)
        elif mode == "int4":
            bf16_indices = torch.empty(0, dtype=torch.long)
        elif mode == "selected_bf16":
            bf16_indices = module_payload["topk_indices"].to(dtype=torch.long)
        elif mode == THRESHOLD_BF16_MODE:
            assert threshold_selection is not None
            bf16_indices = threshold_selection["indices_by_module"][module_name]
        elif mode in {RANGE_THRESHOLD_BF16_MODE, MATCHED_LOW_RANGE_BF16_MODE}:
            assert range_selection is not None
            bf16_indices = range_selection["indices_by_module"][module_name]
        elif mode in {"selected_bf16_int8", "selected_bf16_random_int8"}:
            bf16_indices = module_payload["topk_indices"].to(dtype=torch.long)
        else:
            bf16_indices = make_random_bf16_indices(feature_dim, k, seed=seed)

        is_three_tier = mode in {"selected_bf16_int8", "selected_bf16_random_int8"}
        if is_three_tier:
            k_int8 = math.floor(feature_dim * int8_fraction)
            ranked = torch.argsort(module_payload["aggregated_ranges"], descending=True)
            complement = ranked[~torch.isin(ranked, bf16_indices)]
            if mode == "selected_bf16_int8":
                int8_indices = complement[:k_int8]
            else:
                generator = torch.Generator(device="cpu").manual_seed(seed)
                int8_indices = complement[torch.randperm(complement.numel(), generator=generator)[:k_int8]]
            int8_scales = calibrator.scales_per_partition(num_bits=8)
        output_dtype = getattr(getattr(original_module, "weight", None), "dtype", torch.float32)
        module_device = getattr(getattr(original_module, "weight", None), "device", torch.device("cpu"))
        if isinstance(original_module, nn.Linear):
            if mode == "tp_uncompressed":
                replacements[module_name] = RowParallelLinear.from_linear(
                    original_module,
                    num_partitions=num_partitions,
                )
            elif is_three_tier:
                replacements[module_name] = ThreeTierRowParallelLinear(
                    original_module.weight, original_module.bias, num_partitions,
                    scales.to(device=module_device, dtype=torch.float32), bf16_indices.to(device=module_device),
                    num_bits=num_bits, output_dtype=output_dtype,
                    int8_scales_per_partition=int8_scales.to(device=module_device, dtype=torch.float32),
                    int8_feature_indices=int8_indices.to(device=module_device),
                )
            else:
                replacements[module_name] = HybridQuantizedRowParallelLinear.from_linear(
                    original_module,
                    num_partitions=num_partitions,
                    # Calibration statistics/scales intentionally remain FP32.
                    # They were accumulated from model-dtype partials, while
                    # Int4 reconstruction and selected values use output_dtype.
                    scales_per_partition=scales.to(device=module_device, dtype=torch.float32),
                    bf16_feature_indices=bf16_indices.to(device=module_device),
                    num_bits=num_bits,
                    output_dtype=output_dtype,
                )
        elif Conv1D is not None and isinstance(original_module, Conv1D):
            if mode == "tp_uncompressed":
                replacements[module_name] = RowParallelConv1D.from_conv1d(
                    original_module,
                    num_partitions=num_partitions,
                )
            elif is_three_tier:
                replacements[module_name] = ThreeTierRowParallelConv1D(
                    original_module.weight, original_module.bias, num_partitions,
                    scales.to(device=module_device, dtype=torch.float32), bf16_indices.to(device=module_device),
                    num_bits=num_bits, output_dtype=output_dtype,
                    int8_scales_per_partition=int8_scales.to(device=module_device, dtype=torch.float32),
                    int8_feature_indices=int8_indices.to(device=module_device),
                )
            else:
                replacements[module_name] = HybridQuantizedRowParallelConv1D.from_conv1d(
                    original_module,
                    num_partitions=num_partitions,
                    scales_per_partition=scales.to(device=module_device, dtype=torch.float32),
                    bf16_feature_indices=bf16_indices.to(device=module_device),
                    num_bits=num_bits,
                    output_dtype=output_dtype,
                )
        else:
            raise TypeError(f"Unsupported module type for replacement: {module_name} ({type(original_module)}).")
    if threshold_selection is not None:
        # Retain the exact construction-time allocation for result writers.
        # It is ordinary Python provenance, not a buffer or a forward-path tier.
        for replacement in replacements.values():
            replacement.threshold_bf16_allocation = threshold_selection
    if range_selection is not None:
        for replacement in replacements.values():
            replacement.range_threshold_bf16_allocation = range_selection
    return replacements
