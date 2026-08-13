from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from huggingface_hub.errors import HfUriError
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lowbit_tp_comm.calibration import EMAMinMaxCalibrator
from lowbit_tp_comm.calibration_data import VALID_SAMPLING_STRATEGIES, prepare_calibration_data
from lowbit_tp_comm.dtypes import DTYPE_CHOICES, ensure_dtype_supported, model_load_kwargs, resolve_dtype, validate_model_dtype, validate_module_devices_and_dtypes
from lowbit_tp_comm.hooks import (
    ModuleInputOutputCapture,
    derive_threshold_bf16_selection,
    list_candidate_sync_modules,
    threshold_bf16_result_metadata,
    derive_range_threshold_bf16_selection,
    range_threshold_bf16_result_metadata,
)
from lowbit_tp_comm.quantization import (
    dequantize_symmetric,
    get_qmin_qmax,
    quantization_error_stats,
    quantize_symmetric,
    multi_tier_quant_dequant,
)
from lowbit_tp_comm.tp_linear import compute_row_parallel_partials_for_module, make_random_bf16_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose quantization behavior on TP partial outputs.")
    parser.add_argument("--model_name", default="distilgpt2")
    parser.add_argument("--calibration_path", default="calibration-distilgpt2-tp2.pt")
    parser.add_argument("--target_style", choices=["auto", "gpt2", "llama"], default="gpt2")
    parser.add_argument("--num_partitions", type=int, default=2)
    parser.add_argument("--num_sequences", type=int, default=16)
    parser.add_argument("--sequence_length", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", choices=["int4", "random_bf16", "selected_bf16", "threshold_bf16", "range_threshold_bf16", "matched_low_range_bf16", "selected_bf16_int8", "selected_bf16_random_int8"], default="selected_bf16")
    parser.add_argument("--top_modules", type=int, default=12)
    parser.add_argument("--num_bits", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--int8_fraction", type=float, default=0.015625)
    parser.add_argument("--bf16_range_threshold", type=float, default=None)
    parser.add_argument("--dtype", choices=DTYPE_CHOICES, default="auto")
    parser.add_argument("--sampling_strategy", choices=VALID_SAMPLING_STRATEGIES, default="random_token_chunks")
    parser.add_argument("--dataset_revision", default=None)
    parser.add_argument("--model_revision", default=None)
    parser.add_argument("--tokenizer_revision", default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--exclude_calibration_chunks", action="store_true")
    return parser.parse_args()


def load_text_dataset(dataset_name: str, dataset_config: str, split: str, revision: str | None = None):
    try:
        return load_dataset(dataset_name, dataset_config, split=split, revision=revision)
    except HfUriError:
        if "/" in dataset_name:
            raise
        fallback_name = f"Salesforce/{dataset_name}"
        print(f"Retrying dataset load with namespaced path: {fallback_name}")
        return load_dataset(fallback_name, dataset_config, split=split, revision=revision)


def select_nonempty_texts(dataset, num_sequences: int) -> list[str]:
    texts: list[str] = []
    for row in dataset:
        text = row.get("text", "")
        if isinstance(text, str) and text.strip():
            texts.append(text)
        if len(texts) >= num_sequences:
            break
    if len(texts) < num_sequences:
        raise ValueError(f"Requested {num_sequences} non-empty sequences, found only {len(texts)}.")
    return texts


def choose_selected_indices(module_payload: dict[str, Any], mode: str, seed: int, threshold_indices: torch.Tensor | None = None) -> torch.Tensor:
    feature_dim = int(module_payload["feature_dim"])
    k = int(module_payload["k"])
    if mode in {"selected_bf16", "selected_bf16_int8", "selected_bf16_random_int8"}:
        return module_payload["topk_indices"].to(dtype=torch.long)
    if mode == "random_bf16":
        return make_random_bf16_indices(feature_dim, k, seed=seed)
    if mode in {"threshold_bf16", "range_threshold_bf16", "matched_low_range_bf16"}:
        if threshold_indices is None:
            raise ValueError(f"{mode} selection was not derived.")
        return threshold_indices.to(dtype=torch.long)
    return torch.empty(0, dtype=torch.long)


def choose_int8_indices(module_payload: dict[str, Any], mode: str, seed: int, int8_fraction: float) -> torch.Tensor:
    if mode not in {"selected_bf16_int8", "selected_bf16_random_int8"}:
        return torch.empty(0, dtype=torch.long)
    feature_dim, k = int(module_payload["feature_dim"]), int(module_payload["k"])
    k_int8 = int(feature_dim * int8_fraction)
    bf16 = module_payload["topk_indices"].to(dtype=torch.long)
    ranked = torch.argsort(module_payload["aggregated_ranges"], descending=True)
    complement = ranked[~torch.isin(ranked, bf16)]
    if mode == "selected_bf16_int8":
        return complement[:k_int8]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return complement[torch.randperm(complement.numel(), generator=generator)[:k_int8]]


def flatten_records(records: list[torch.Tensor]) -> torch.Tensor:
    flattened = [record.reshape(-1) for record in records]
    return torch.cat(flattened, dim=0) if flattened else torch.empty(0, dtype=torch.float32)


def _tier_mask(feature_dim: int, indices: torch.Tensor, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(feature_dim, dtype=torch.bool, device=device)
    if indices.numel():
        mask[indices.to(device=device, dtype=torch.long)] = True
    return mask


def _payload_saturation(q: torch.Tensor, mask: torch.Tensor, qmin: int, qmax: int, prefix: str) -> dict[str, float]:
    """Empty payloads report zero saturation rather than NaN."""

    payload = q[..., mask]
    if payload.numel() == 0:
        return {f"{prefix}_payload_saturation_low_rate": 0.0, f"{prefix}_payload_saturation_high_rate": 0.0}
    return {
        f"{prefix}_payload_saturation_low_rate": float((payload == qmin).float().mean().item()),
        f"{prefix}_payload_saturation_high_rate": float((payload == qmax).float().mean().item()),
    }


def three_tier_diagnostic_stats(
    original: torch.Tensor, reconstructed: torch.Tensor, q4: torch.Tensor, q8: torch.Tensor,
    bf16_indices: torch.Tensor, int8_indices: torch.Tensor,
) -> dict[str, float]:
    """Payload-specific saturation and reconstruction errors for three tiers."""

    feature_dim = original.shape[-1]
    bf16_mask = _tier_mask(feature_dim, bf16_indices, original.device)
    int8_mask = _tier_mask(feature_dim, int8_indices, original.device)
    int4_mask = ~(bf16_mask | int8_mask)
    stats = quantization_error_stats(original, reconstructed)
    error = (reconstructed.float() - original.float()).abs()
    for name, mask in (("bf16", bf16_mask), ("int8", int8_mask), ("int4", int4_mask)):
        values = error[..., mask]
        stats[f"{name}_mean_abs_error"] = float(values.mean().item()) if values.numel() else 0.0
        stats[f"{name}_max_abs_error"] = float(values.max().item()) if values.numel() else 0.0
        stats[f"{name}_fraction"] = float(mask.float().mean().item())
    stats.update(_payload_saturation(q4, int4_mask, -8, 7, "int4"))
    stats.update(_payload_saturation(q8, int8_mask, -128, 127, "int8"))
    stats["average_bits_per_value"] = 16 * stats["bf16_fraction"] + 8 * stats["int8_fraction"] + 4 * stats["int4_fraction"]
    return stats


def main() -> None:
    args = parse_args()
    if args.mode in {"range_threshold_bf16", "matched_low_range_bf16"} and (args.bf16_range_threshold is None or not math.isfinite(args.bf16_range_threshold) or args.bf16_range_threshold <= 0):
        raise ValueError("--bf16_range_threshold must be finite and positive for range-threshold modes.")
    if args.mode not in {"range_threshold_bf16", "matched_low_range_bf16"} and args.bf16_range_threshold is not None:
        raise ValueError("--bf16_range_threshold is valid only with range_threshold_bf16 or matched_low_range_bf16.")
    device = torch.device(args.device)
    requested_dtype = resolve_dtype(args.dtype)
    ensure_dtype_supported(requested_dtype, device)
    qmin, qmax = get_qmin_qmax(args.num_bits)

    calibration = torch.load(args.calibration_path, map_location="cpu", weights_only=False)
    module_payloads = list(calibration["modules"].items())[: args.top_modules]

    try:
        tokenizer_kwargs = {"revision": args.tokenizer_revision} if args.tokenizer_revision is not None else {}
        model_kwargs = {"revision": args.model_revision} if args.model_revision is not None else {}
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, **tokenizer_kwargs)
        model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs, **model_load_kwargs(args.dtype))
    except Exception:
        print(
            "Model loading failed. This model may be gated. "
            "Run `huggingface-cli login` or choose an open model.",
            file=sys.stderr,
        )
        raise
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model.eval()
    model.to(device)
    validate_model_dtype(model, requested_dtype)
    validate_module_devices_and_dtypes(model, device, requested_dtype)

    threshold_selection = (
        derive_threshold_bf16_selection(calibration, model=model) if args.mode == "threshold_bf16" else None
    )
    range_selection = (derive_range_threshold_bf16_selection(calibration, threshold=args.bf16_range_threshold, mode=args.mode, model=model)
                       if args.mode in {"range_threshold_bf16", "matched_low_range_bf16"} else None)

    candidate_names = [name for name, _module in list_candidate_sync_modules(model, target_style=args.target_style)]
    module_names = [name for name, _payload in module_payloads if name in candidate_names]
    capture = ModuleInputOutputCapture(model, module_names, store_on_cpu=device.type != "cuda")
    module_lookup = dict(model.named_modules())

    dataset = load_text_dataset("wikitext", "wikitext-2-raw-v1", split="test", revision=args.dataset_revision)
    calibration_ids = calibration.get("selected_chunk_ids", []) if args.exclude_calibration_chunks else []
    prepared_data = prepare_calibration_data(
        dataset, tokenizer, num_sequences=args.num_sequences, sequence_length=args.sequence_length,
        sampling_strategy=args.sampling_strategy, seed=args.seed, excluded_chunk_ids=set(calibration_ids),
    )

    diagnostics: dict[str, dict[int, dict[str, list[torch.Tensor]]]] = {}
    for module_name in module_names:
        diagnostics[module_name] = {
            partition_idx: {
                "original": [],
                "reconstructed": [],
                "q": [],
                "q8": [],
                "scale": [],
            }
            for partition_idx in range(args.num_partitions)
        }
        diagnostics[module_name]["aggregate"] = {"original": [], "reconstructed": []}

    try:
        with torch.no_grad():
            for prepared_inputs in prepared_data.inputs:
                capture.clear()
                encoded = {key: value.to(device) for key, value in prepared_inputs.items()}
                model(**encoded)

                for module_name, module_payload in module_payloads:
                    if module_name not in module_lookup:
                        continue
                    inputs = capture.get_inputs(module_name)
                    if not inputs:
                        continue
                    module = module_lookup[module_name]
                    calibrator = EMAMinMaxCalibrator.from_state_dict(module_payload["state_dict"])
                    scales = calibrator.scales_per_partition()
                    selection = threshold_selection or range_selection
                    threshold_indices = None if selection is None else selection["indices_by_module"][module_name]
                    selected_indices = choose_selected_indices(module_payload, args.mode, args.seed, threshold_indices)
                    int8_indices = choose_int8_indices(module_payload, args.mode, args.seed, args.int8_fraction)
                    int8_scales = calibrator.scales_per_partition(num_bits=8)
                    for input_tensor in inputs:
                        partials = compute_row_parallel_partials_for_module(module, input_tensor, args.num_partitions)
                        reconstructed_partials: list[torch.Tensor] = []
                        for partition_idx, partial in enumerate(partials):
                            scale = scales[partition_idx].to(device=partial.device, dtype=torch.float32)
                            q = quantize_symmetric(partial, scale, num_bits=args.num_bits)
                            q8 = quantize_symmetric(partial, int8_scales[partition_idx].to(device=partial.device), num_bits=8)
                            if int8_indices.numel():
                                reconstructed = multi_tier_quant_dequant(
                                    partial, scale, int8_scales[partition_idx].to(device=partial.device),
                                    selected_indices, int8_indices, partial.dtype,
                                )
                            else:
                                reconstructed = dequantize_symmetric(q, scale, dtype=partial.dtype)
                            if selected_indices.numel() > 0:
                                indices = selected_indices.to(device=partial.device, dtype=torch.long)
                                reconstructed[..., indices] = partial[..., indices]
                            reconstructed_partials.append(reconstructed)
                            diagnostics[module_name][partition_idx]["original"].append(partial.to(torch.float32).cpu())
                            diagnostics[module_name][partition_idx]["reconstructed"].append(reconstructed.to(torch.float32).cpu())
                            diagnostics[module_name][partition_idx]["q"].append(q.cpu())
                            diagnostics[module_name][partition_idx]["q8"].append(q8.cpu())
                            diagnostics[module_name][partition_idx]["scale"].append(scale.cpu())
                        original_sum = torch.stack(partials, dim=0).sum(dim=0)
                        reconstructed_sum = torch.stack(reconstructed_partials, dim=0).sum(dim=0)
                        bias = getattr(module, "bias", None)
                        if bias is not None:
                            original_sum = original_sum + bias.to(original_sum.dtype)
                            reconstructed_sum = reconstructed_sum + bias.to(reconstructed_sum.dtype)
                        diagnostics[module_name]["aggregate"]["original"].append(original_sum.float().cpu())
                        diagnostics[module_name]["aggregate"]["reconstructed"].append(reconstructed_sum.float().cpu())
    finally:
        capture.remove()

    print(
        f"model_name={args.model_name}, mode={args.mode}, num_partitions={args.num_partitions}, "
        f"num_sequences={args.num_sequences}, sequence_length={args.sequence_length}, num_bits={args.num_bits}"
    )
    calibration_sha256 = hashlib.sha256(Path(args.calibration_path).read_bytes()).hexdigest()
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
    output: dict[str, Any] = {"provenance": {
        "mode": args.mode, "calibration_path": args.calibration_path, "int8_fraction": args.int8_fraction,
        "model_name": args.model_name, "model_revision": args.model_revision,
        "tokenizer_revision": args.tokenizer_revision, "dataset_revision": args.dataset_revision,
        "sampling_strategy": args.sampling_strategy, "sampling_seed": args.seed,
        "selected_chunk_ids": prepared_data.selected_chunk_ids,
        "total_available_chunk_count": prepared_data.total_available_chunks,
        "excluded_calibration_chunk_ids": calibration_ids,
        "padding_used": prepared_data.padding_used,
        "separator_token_policy": prepared_data.separator_token_policy,
        "requested_dtype": args.dtype, "actual_parameter_dtype": str(next(model.parameters()).dtype),
        "calibration_sha256": calibration_sha256, "git_commit": git_commit,
    }, "modules": {}}
    if threshold_selection is not None:
        output["provenance"]["threshold_bf16"] = threshold_bf16_result_metadata(
            threshold_selection,
            calibration_path=args.calibration_path,
            calibration_sha256=calibration_sha256,
        )
    if range_selection is not None:
        output["provenance"][args.mode] = range_threshold_bf16_result_metadata(range_selection, calibration_path=args.calibration_path, calibration_sha256=calibration_sha256)
    for module_name, module_payload in module_payloads:
        if module_name not in diagnostics:
            continue
        selection = threshold_selection or range_selection
        threshold_indices = None if selection is None else selection["indices_by_module"][module_name]
        selected_indices = choose_selected_indices(module_payload, args.mode, args.seed, threshold_indices)
        int8_indices = choose_int8_indices(module_payload, args.mode, args.seed, args.int8_fraction)
        is_three_tier = args.mode in {"selected_bf16_int8", "selected_bf16_random_int8"}
        print(f"\nmodule={module_name}")
        print(f"selected_fraction={selected_indices.numel() / int(module_payload['feature_dim']):.6f}")
        output["modules"][module_name] = {}
        for partition_idx, partition_data in diagnostics[module_name].items():
            if partition_idx == "aggregate":
                continue
            if not partition_data["original"]:
                continue
            original = torch.cat(partition_data["original"], dim=0)
            reconstructed = torch.cat(partition_data["reconstructed"], dim=0)
            q = torch.cat(partition_data["q"], dim=0)
            q8 = torch.cat(partition_data["q8"], dim=0)
            scale = partition_data["scale"][0]
            if is_three_tier:
                stats = three_tier_diagnostic_stats(original, reconstructed, q, q8, selected_indices, int8_indices)
            else:
                stats = quantization_error_stats(original, reconstructed, q=q, qmin=qmin, qmax=qmax, selected_indices=selected_indices)
                int4_mask = ~_tier_mask(original.shape[-1], selected_indices, original.device)
                stats.update(_payload_saturation(q, int4_mask, qmin, qmax, "int4"))
            output["modules"][module_name][str(partition_idx)] = stats
            print(f"  partition={partition_idx}")
            print(f"    partial_mean={float(original.mean().item()):.6f}")
            print(f"    partial_std={float(original.std(unbiased=False).item()):.6f}")
            print(f"    partial_min={float(original.min().item()):.6f}")
            print(f"    partial_max={float(original.max().item()):.6f}")
            print(f"    scale_min={float(scale.min().item()):.6f}")
            print(f"    scale_median={float(scale.median().item()):.6f}")
            print(f"    scale_max={float(scale.max().item()):.6f}")
            print(f"    quantized_min={stats.get('quantized_min', 0.0):.6f}")
            print(f"    quantized_max={stats.get('quantized_max', 0.0):.6f}")
            print(f"    saturation_low_rate={stats.get('saturation_low_rate', 0.0):.6f}")
            print(f"    saturation_high_rate={stats.get('saturation_high_rate', 0.0):.6f}")
            print(f"    int4_payload_saturation_low_rate={stats['int4_payload_saturation_low_rate']:.6f}")
            print(f"    int4_payload_saturation_high_rate={stats['int4_payload_saturation_high_rate']:.6f}")
            print(f"    mean_abs_error={stats['mean_abs_error']:.6f}")
            print(f"    max_abs_error={stats['max_abs_error']:.6f}")
            print(f"    rmse={stats['rmse']:.6f}")
            print(f"    mean_signed_error={stats['mean_signed_error']:.6f}")
            print(f"    relative_rmse={stats['relative_rmse']:.6f}")
            if is_three_tier:
                for key in ("bf16_mean_abs_error", "int8_mean_abs_error", "int4_mean_abs_error", "int8_payload_saturation_low_rate", "int8_payload_saturation_high_rate", "average_bits_per_value"):
                    print(f"    {key}={stats[key]:.6f}")
            if selected_indices.numel() > 0:
                print(f"    selected_mean_abs_error={stats['selected_mean_abs_error']:.6f}")
                print(f"    non_selected_mean_abs_error={stats['non_selected_mean_abs_error']:.6f}")
        aggregate = diagnostics[module_name]["aggregate"]
        if aggregate["original"]:
            aggregate_original = torch.cat(aggregate["original"], dim=0)
            aggregate_reconstructed = torch.cat(aggregate["reconstructed"], dim=0)
            aggregate_stats = quantization_error_stats(aggregate_original, aggregate_reconstructed)
            partition_stats = [value for key, value in output["modules"][module_name].items() if key != "aggregate"]
            for key in (
                "saturation_low_rate", "saturation_high_rate", "int4_payload_saturation_low_rate",
                "int4_payload_saturation_high_rate", "int8_payload_saturation_low_rate",
                "int8_payload_saturation_high_rate", "selected_mean_abs_error", "non_selected_mean_abs_error",
            ):
                values = [float(stats[key]) for stats in partition_stats if key in stats]
                if values:
                    aggregate_stats[key] = statistics.mean(values)
            if is_three_tier:
                bf16_mask = _tier_mask(aggregate_original.shape[-1], selected_indices, aggregate_original.device)
                int8_mask = _tier_mask(aggregate_original.shape[-1], int8_indices, aggregate_original.device)
                int4_mask = ~(bf16_mask | int8_mask)
                absolute = (aggregate_reconstructed - aggregate_original).abs()
                for name, mask in (("bf16", bf16_mask), ("int8", int8_mask), ("int4", int4_mask)):
                    values = absolute[..., mask]
                    aggregate_stats[f"{name}_mean_abs_error"] = float(values.mean().item()) if values.numel() else 0.0
                    aggregate_stats[f"{name}_max_abs_error"] = float(values.max().item()) if values.numel() else 0.0
                    aggregate_stats[f"{name}_fraction"] = float(mask.float().mean().item())
                aggregate_stats["average_bits_per_value"] = (
                    16 * aggregate_stats["bf16_fraction"] + 8 * aggregate_stats["int8_fraction"] + 4 * aggregate_stats["int4_fraction"]
                )
            output["modules"][module_name]["aggregate"] = aggregate_stats
            print(f"  aggregate_rmse={aggregate_stats['rmse']:.6f}")
    if args.output_path:
        with open(args.output_path, "w", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2)


if __name__ == "__main__":
    main()
