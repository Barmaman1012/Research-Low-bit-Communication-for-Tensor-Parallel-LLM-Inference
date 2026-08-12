from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lowbit_tp_comm.hooks import (
    build_hybrid_replacements_from_calibration,
    replace_modules_by_name,
)
from lowbit_tp_comm.dtypes import (
    DTYPE_CHOICES,
    ensure_dtype_supported,
    model_dtype_metadata,
    model_load_kwargs,
    resolve_dtype,
    validate_model_dtype,
    validate_module_devices_and_dtypes,
)

try:
    from transformers.pytorch_utils import Conv1D
except ImportError:  # pragma: no cover
    Conv1D = None

VALID_MODES = {"full", "tp_uncompressed", "all_bf16", "int4", "random_bf16", "selected_bf16", "threshold_bf16", "selected_bf16_int8", "selected_bf16_random_int8"}
DEFAULT_TASKS = ["arc_easy", "arc_challenge", "winogrande", "hellaswag", "boolq"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate simulated TP modes with lm-eval-harness.")
    parser.add_argument("--model_name", default="distilgpt2")
    parser.add_argument("--model_revision", default=None)
    parser.add_argument("--tokenizer_revision", default=None)
    parser.add_argument("--calibration_path", default=None)
    parser.add_argument(
        "--target_style",
        choices=["auto", "gpt2", "llama"],
        default="gpt2",
        help="Calibration provenance only; evaluation uses exact module names stored in the artifact.",
    )
    parser.add_argument("--mode", choices=sorted(VALID_MODES), default="full")
    parser.add_argument("--modes", default=None)
    parser.add_argument("--num_partitions", type=int, default=2)
    parser.add_argument("--num_bits", type=int, default=4)
    parser.add_argument("--int8_fraction", type=float, default=0.015625)
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=DTYPE_CHOICES, default="auto")
    parser.add_argument("--batch_size", default="1")
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def parse_csv_list(raw: str | None, default: list[str] | None = None) -> list[str]:
    if raw is None:
        return list(default or [])
    items = [item.strip() for item in raw.split(",") if item.strip()]
    return items if items else list(default or [])


def parse_modes(mode: str, modes: str | None) -> list[str]:
    if modes is None:
        return [mode]
    parsed = parse_csv_list(modes)
    invalid = [item for item in parsed if item not in VALID_MODES]
    if invalid:
        raise ValueError(f"Unsupported modes in --modes: {invalid}")
    return parsed


def safe_import_lm_eval():
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        print("lm-eval is not installed. Install it with: ./.venv/bin/pip install lm-eval", file=sys.stderr)
        raise
    return lm_eval, HFLM


def load_model_or_raise(
    model_name: str,
    device: torch.device,
    dtype_name: str = "auto",
    model_revision: str | None = None,
):
    requested_dtype = resolve_dtype(dtype_name)
    ensure_dtype_supported(requested_dtype, device)
    try:
        revision_kwargs = {"revision": model_revision} if model_revision is not None else {}
        model = AutoModelForCausalLM.from_pretrained(model_name, **revision_kwargs, **model_load_kwargs(dtype_name))
    except Exception:
        print(
            "Model loading failed. This model may be gated. "
            "Run `huggingface-cli login` or choose an open model.",
            file=sys.stderr,
        )
        raise
    model.eval()
    model.to(device)
    validate_model_dtype(model, requested_dtype)
    return model


def load_tokenizer_or_raise(model_name: str, tokenizer_revision: str | None = None):
    try:
        revision_kwargs = {"revision": tokenizer_revision} if tokenizer_revision is not None else {}
        tokenizer = AutoTokenizer.from_pretrained(model_name, **revision_kwargs)
    except Exception:
        print(
            "Tokenizer loading failed. This model may be gated. "
            "Run `huggingface-cli login` or choose an open model.",
            file=sys.stderr,
        )
        raise
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    return tokenizer


def _module_feature_dim(module: nn.Module) -> int:
    if isinstance(module, nn.Linear):
        return module.out_features
    if Conv1D is not None and isinstance(module, Conv1D):
        return int(module.weight.shape[1])
    raise TypeError(f"Unsupported calibrated module type: {type(module)}")


def validate_calibration_compatibility(
    model: nn.Module,
    calibration: dict[str, Any],
    *,
    model_name: str,
    num_partitions: int,
    dtype_name: str = "auto",
) -> None:
    """Reject calibration artifacts that do not match the evaluation model."""

    recorded_model_name = calibration.get("model_name")
    if recorded_model_name and recorded_model_name != model_name:
        raise ValueError(
            "Calibration model_name mismatch: artifact was created for "
            f"{recorded_model_name!r}, but evaluation requested {model_name!r}."
        )
    if dtype_name != "auto":
        dtype_metadata = calibration.get("dtype_metadata", {})
        recorded_dtype = dtype_metadata.get("requested_model_dtype") if isinstance(dtype_metadata, dict) else None
        if recorded_dtype is not None and recorded_dtype != dtype_name:
            raise ValueError(
                f"Calibration dtype mismatch: artifact requested {recorded_dtype!r}, evaluation requested {dtype_name!r}."
            )

    modules = calibration.get("modules")
    if not isinstance(modules, dict) or not modules:
        raise ValueError("Calibration artifact must contain a non-empty 'modules' mapping.")

    artifact_partitions = calibration.get("num_partitions")
    partition_specific = bool(calibration.get("simulated_row_parallel_calibration", False))
    if partition_specific and artifact_partitions != num_partitions:
        raise ValueError(
            "Calibration partition mismatch: artifact contains partition-specific statistics for "
            f"num_partitions={artifact_partitions}, but evaluation requested num_partitions={num_partitions}."
        )
    expected_partition_count = num_partitions if partition_specific else 1
    named_modules = dict(model.named_modules())
    for module_name, payload in modules.items():
        if module_name not in named_modules:
            raise ValueError(f"Calibrated module {module_name!r} is not present in the evaluation model.")
        if not isinstance(payload, dict) or "state_dict" not in payload:
            raise ValueError(f"Calibration entry for {module_name!r} is missing state_dict.")
        state = payload["state_dict"]
        if not isinstance(state, dict):
            raise ValueError(f"Calibration state_dict for {module_name!r} must be a mapping.")
        min_vals = state.get("min_vals")
        max_vals = state.get("max_vals")
        if not isinstance(min_vals, torch.Tensor) or not isinstance(max_vals, torch.Tensor):
            raise ValueError(f"Calibration state for {module_name!r} must contain tensor min_vals and max_vals.")
        if min_vals.ndim != 2 or max_vals.shape != min_vals.shape:
            raise ValueError(f"Calibration scale statistics for {module_name!r} must have matching [P, E] shapes.")
        if min_vals.shape[0] != expected_partition_count:
            raise ValueError(
                f"Calibration partition scales for {module_name!r} have P={min_vals.shape[0]}, "
                f"expected P={expected_partition_count}."
            )
        state_partitions = state.get("num_partitions")
        if state_partitions != expected_partition_count:
            raise ValueError(
                f"Calibration state for {module_name!r} records num_partitions={state_partitions}, "
                f"expected {expected_partition_count}."
            )
        model_feature_dim = _module_feature_dim(named_modules[module_name])
        artifact_feature_dim = payload.get("feature_dim")
        state_feature_dim = state.get("feature_dim")
        if artifact_feature_dim != model_feature_dim or state_feature_dim != model_feature_dim:
            raise ValueError(
                f"Calibration feature dimension mismatch for {module_name!r}: artifact has "
                f"feature_dim={artifact_feature_dim} (state={state_feature_dim}), "
                f"but model requires {model_feature_dim}."
            )
        if min_vals.shape[1] != model_feature_dim:
            raise ValueError(
                f"Calibration scale shape mismatch for {module_name!r}: got E={min_vals.shape[1]}, "
                f"but model requires E={model_feature_dim}."
            )


def build_model_for_mode(
    *,
    model_name: str,
    calibration_path: str | None,
    mode: str,
    num_partitions: int,
    num_bits: int,
    device: torch.device,
    seed: int,
    dtype_name: str = "auto",
    model_revision: str | None = None,
    int8_fraction: float = 0.015625,
):
    model = load_model_or_raise(model_name, device, dtype_name, model_revision)
    calibration = None
    if mode != "full":
        if calibration_path is None:
            raise ValueError("--calibration_path is required for non-full modes.")
        calibration = torch.load(calibration_path, map_location="cpu", weights_only=False)
        validate_calibration_compatibility(
            model,
            calibration,
            model_name=model_name,
            num_partitions=num_partitions,
            dtype_name=dtype_name,
        )
        replacements = build_hybrid_replacements_from_calibration(
            model,
            calibration=calibration,
            mode=mode,
            num_partitions=num_partitions,
            num_bits=num_bits,
            seed=seed,
            int8_fraction=int8_fraction,
        )
        replace_modules_by_name(model, replacements)
        # Replacements include calibration buffers loaded from CPU.  Move the
        # complete model again so all replacement parameters and buffers match
        # the requested execution device before lm-eval invokes it.
        model.to(device)
    validate_module_devices_and_dtypes(model, device, resolve_dtype(dtype_name))
    return model, calibration


def build_simple_evaluate_kwargs(
    simple_evaluate,
    *,
    model: Any,
    tasks: list[str],
    limit: int | None,
    batch_size: str,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    """Build lm-eval arguments using only parameters its installed API exposes."""

    supported = inspect.signature(simple_evaluate).parameters
    kwargs: dict[str, Any] = {
        "model": model,
        "tasks": tasks,
        "limit": limit,
        "batch_size": batch_size,
        "device": str(device),
    }
    requested_optional = {
        "num_fewshot": 0,
        "random_seed": seed,
        "numpy_random_seed": seed,
        "torch_random_seed": seed,
        "fewshot_random_seed": seed,
    }
    kwargs.update({name: value for name, value in requested_optional.items() if name in supported})
    return kwargs


def _git_commit_hash() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha256_file(path: str | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_run_metadata(
    *,
    args: argparse.Namespace,
    model: nn.Module,
    tokenizer: Any,
    lm_eval: Any,
    tasks: list[str],
    modes: list[str],
) -> dict[str, Any]:
    metadata = {
        "model_name": args.model_name,
        "model_revision": getattr(args, "model_revision", None),
        "resolved_model_revision": getattr(getattr(model, "config", None), "_commit_hash", None)
        or getattr(args, "model_revision", None),
        "tokenizer_name": getattr(tokenizer, "name_or_path", None),
        "tokenizer_revision": getattr(args, "tokenizer_revision", None),
        "resolved_tokenizer_revision": getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
        or getattr(args, "tokenizer_revision", None),
        "requested_device": args.device,
        "requested_model_dtype": getattr(args, "dtype", "auto"),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "lm_eval_version": getattr(lm_eval, "__version__", None),
        "python_version": platform.python_version(),
        "tasks": tasks,
        "num_fewshot": 0,
        "limit": args.limit,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "mode": args.mode,
        "modes": modes,
        "num_partitions": args.num_partitions,
        "num_bits": args.num_bits,
        "int8_fraction": getattr(args, "int8_fraction", 0.0),
        "calibration_path": args.calibration_path,
        "target_style": args.target_style,
        "git_commit": _git_commit_hash(),
    }
    metadata.update(model_dtype_metadata(model))
    if args.mode == "threshold_bf16":
        selection = next(
            (module.threshold_bf16_metadata for module in model.modules() if hasattr(module, "threshold_bf16_metadata")),
            None,
        )
        if selection is None:
            raise RuntimeError("threshold_bf16 replacements are missing construction-time allocation metadata.")
        metadata["threshold_bf16"] = {
            **selection,
            "calibration_path": args.calibration_path,
            "calibration_sha256": _sha256_file(args.calibration_path),
        }
    replacement_metadata: dict[str, Any] = {}
    for name, module in model.named_modules():
        if hasattr(module, "output_dtype") and hasattr(module, "scales_per_partition"):
            replacement_metadata[name] = {
                "reconstructed_output_dtype": str(module.output_dtype),
                "scale_dtype": str(module.scales_per_partition.dtype),
                "selected_feature_communication_dtype": str(module.output_dtype),
            }
            if hasattr(module, "int8_feature_indices"):
                feature_dim = int(module.out_features)
                bf16_count = int(module.bf16_feature_indices.numel())
                int8_count = int(module.int8_feature_indices.numel())
                replacement_metadata[name].update({
                    "selection_strategy": "calibrated_int8" if args.mode == "selected_bf16_int8" else "random_int8",
                    "bf16_feature_count": bf16_count,
                    "int8_feature_count": int8_count,
                    "int4_feature_count": feature_dim - bf16_count - int8_count,
                    "bf16_fraction": bf16_count / feature_dim,
                    "int8_fraction": int8_count / feature_dim,
                    "int4_fraction": (feature_dim - bf16_count - int8_count) / feature_dim,
                    "average_bits_per_value": 4 + 12 * bf16_count / feature_dim + 4 * int8_count / feature_dim,
                })
    metadata["replacement_dtypes"] = replacement_metadata
    return metadata


def extract_rows(mode: str, results: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_name, metrics in results.get("results", {}).items():
        for metric_name, value in metrics.items():
            metric_base = metric_name.split(",", maxsplit=1)[0]
            if metric_base.endswith("_stderr") or metric_base in {"sample_len", "num_samples", "num_fewshot"}:
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            rows.append(
                {
                    "mode": mode,
                    "task": task_name,
                    "metric": metric_name,
                    "value": numeric_value,
                }
            )
    return rows


def primary_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    task_metrics = {
        "arc_easy": "acc_norm,none",
        "arc_challenge": "acc_norm,none",
        "winogrande": "acc,none",
        "hellaswag": "acc,none",
        "boolq": "acc,none",
    }
    fallback_order = ["acc_norm,none", "acc,none", "exact_match,none"]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["mode"]), str(row["task"]))
        grouped.setdefault(key, []).append(row)

    selected: list[dict[str, Any]] = []
    for _key, task_rows in grouped.items():
        picked = None
        task_name = str(task_rows[0]["task"])
        preferred_order = [task_metrics[task_name], *fallback_order] if task_name in task_metrics else fallback_order
        for metric_name in dict.fromkeys(preferred_order):
            picked = next((row for row in task_rows if row["metric"] == metric_name), None)
            if picked is not None:
                break
        if picked is not None:
            selected.append(picked)
    return selected


def format_results_table(rows: list[dict[str, Any]]) -> str:
    lines = ["mode | task | metric | value"]
    for row in rows:
        lines.append(f"{row['mode']} | {row['task']} | {row['metric']} | {float(row['value']):.6f}")
    return "\n".join(lines)


def format_average_summary(rows: list[dict[str, Any]]) -> str:
    per_mode: dict[str, list[float]] = {}
    for row in primary_metric_rows(rows):
        per_mode.setdefault(str(row["mode"]), []).append(float(row["value"]))

    lines = ["", "mode | avg_primary_score"]
    for mode, values in per_mode.items():
        lines.append(f"{mode} | {statistics.mean(values):.6f}")
    return "\n".join(lines)


def make_json_serializable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(key): make_json_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if np is not None:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
    if isinstance(obj, (torch.dtype, torch.device, Path)):
        return str(obj)
    return str(obj)


def main() -> None:
    args = parse_args()
    lm_eval, HFLM = safe_import_lm_eval()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    tokenizer = load_tokenizer_or_raise(args.model_name, args.tokenizer_revision)
    tasks = parse_csv_list(args.tasks, default=DEFAULT_TASKS)
    modes = parse_modes(args.mode, args.modes)

    all_rows: list[dict[str, Any]] = []
    raw_results: dict[str, Any] = {}
    metadata_by_mode: dict[str, dict[str, Any]] = {}
    for mode in modes:
        model, _calibration = build_model_for_mode(
            model_name=args.model_name,
            calibration_path=args.calibration_path,
            mode=mode,
            num_partitions=args.num_partitions,
            num_bits=args.num_bits,
            device=device,
            seed=args.seed,
            dtype_name=args.dtype,
            model_revision=args.model_revision,
            int8_fraction=args.int8_fraction,
        )
        lm = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            device=str(next(model.parameters()).device),
        )
        results = lm_eval.simple_evaluate(
            **build_simple_evaluate_kwargs(
                lm_eval.simple_evaluate,
                model=lm,
                tasks=tasks,
                limit=args.limit,
                batch_size=args.batch_size,
                device=next(model.parameters()).device,
                seed=args.seed,
            )
        )
        raw_results[mode] = results
        metadata_by_mode[mode] = build_run_metadata(
            args=args,
            model=model,
            tokenizer=tokenizer,
            lm_eval=lm_eval,
            tasks=tasks,
            modes=modes,
        )
        all_rows.extend(extract_rows(mode, results))

    print(format_results_table(all_rows))
    print(format_average_summary(all_rows))

    if args.output_path is not None:
        payload = make_json_serializable(
            {"metadata": metadata_by_mode, "rows": all_rows, "raw_results": raw_results}
        )
        with open(args.output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nSaved results to {args.output_path}")


if __name__ == "__main__":
    main()
