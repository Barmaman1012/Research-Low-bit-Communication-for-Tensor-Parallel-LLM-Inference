from __future__ import annotations

import argparse
import statistics
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
from lowbit_tp_comm.hooks import ModuleInputOutputCapture, list_candidate_sync_modules
from lowbit_tp_comm.quantization import (
    dequantize_symmetric,
    get_qmin_qmax,
    quantization_error_stats,
    quantize_symmetric,
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
    parser.add_argument("--mode", choices=["int4", "random_bf16", "selected_bf16"], default="selected_bf16")
    parser.add_argument("--top_modules", type=int, default=12)
    parser.add_argument("--num_bits", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_text_dataset(dataset_name: str, dataset_config: str, split: str):
    try:
        return load_dataset(dataset_name, dataset_config, split=split)
    except HfUriError:
        if "/" in dataset_name:
            raise
        fallback_name = f"Salesforce/{dataset_name}"
        print(f"Retrying dataset load with namespaced path: {fallback_name}")
        return load_dataset(fallback_name, dataset_config, split=split)


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


def choose_selected_indices(module_payload: dict[str, Any], mode: str, seed: int) -> torch.Tensor:
    feature_dim = int(module_payload["feature_dim"])
    k = int(module_payload["k"])
    if mode == "selected_bf16":
        return module_payload["topk_indices"].to(dtype=torch.long)
    if mode == "random_bf16":
        return make_random_bf16_indices(feature_dim, k, seed=seed)
    return torch.empty(0, dtype=torch.long)


def flatten_records(records: list[torch.Tensor]) -> torch.Tensor:
    flattened = [record.reshape(-1) for record in records]
    return torch.cat(flattened, dim=0) if flattened else torch.empty(0, dtype=torch.float32)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    qmin, qmax = get_qmin_qmax(args.num_bits)

    calibration = torch.load(args.calibration_path, map_location="cpu", weights_only=False)
    module_payloads = list(calibration["modules"].items())[: args.top_modules]

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForCausalLM.from_pretrained(args.model_name)
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

    candidate_names = [name for name, _module in list_candidate_sync_modules(model, target_style=args.target_style)]
    module_names = [name for name, _payload in module_payloads if name in candidate_names]
    capture = ModuleInputOutputCapture(model, module_names)
    module_lookup = dict(model.named_modules())

    dataset = load_text_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = select_nonempty_texts(dataset, args.num_sequences)

    diagnostics: dict[str, dict[int, dict[str, list[torch.Tensor]]]] = {}
    for module_name in module_names:
        diagnostics[module_name] = {
            partition_idx: {
                "original": [],
                "reconstructed": [],
                "q": [],
                "scale": [],
            }
            for partition_idx in range(args.num_partitions)
        }

    try:
        with torch.no_grad():
            for text in texts:
                capture.clear()
                encoded = tokenizer(
                    text,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=args.sequence_length,
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
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
                    selected_indices = choose_selected_indices(module_payload, args.mode, args.seed)
                    for input_tensor in inputs:
                        partials = compute_row_parallel_partials_for_module(module, input_tensor, args.num_partitions)
                        for partition_idx, partial in enumerate(partials):
                            scale = scales[partition_idx]
                            q = quantize_symmetric(partial, scale, num_bits=args.num_bits)
                            reconstructed = dequantize_symmetric(q, scale, dtype=torch.float32)
                            if selected_indices.numel() > 0:
                                reconstructed[..., selected_indices] = partial[..., selected_indices]
                            diagnostics[module_name][partition_idx]["original"].append(partial.to(torch.float32).cpu())
                            diagnostics[module_name][partition_idx]["reconstructed"].append(reconstructed.cpu())
                            diagnostics[module_name][partition_idx]["q"].append(q.cpu())
                            diagnostics[module_name][partition_idx]["scale"].append(scale.cpu())
    finally:
        capture.remove()

    print(
        f"model_name={args.model_name}, mode={args.mode}, num_partitions={args.num_partitions}, "
        f"num_sequences={args.num_sequences}, sequence_length={args.sequence_length}, num_bits={args.num_bits}"
    )
    for module_name, module_payload in module_payloads:
        if module_name not in diagnostics:
            continue
        selected_indices = choose_selected_indices(module_payload, args.mode, args.seed)
        print(f"\nmodule={module_name}")
        print(f"selected_fraction={selected_indices.numel() / int(module_payload['feature_dim']):.6f}")
        for partition_idx, partition_data in diagnostics[module_name].items():
            if not partition_data["original"]:
                continue
            original = torch.cat(partition_data["original"], dim=0)
            reconstructed = torch.cat(partition_data["reconstructed"], dim=0)
            q = torch.cat(partition_data["q"], dim=0)
            scale = partition_data["scale"][0]
            stats = quantization_error_stats(
                original,
                reconstructed,
                q=q,
                qmin=qmin,
                qmax=qmax,
                selected_indices=selected_indices,
            )
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
            print(f"    mean_abs_error={stats['mean_abs_error']:.6f}")
            print(f"    max_abs_error={stats['max_abs_error']:.6f}")
            print(f"    rmse={stats['rmse']:.6f}")
            print(f"    mean_signed_error={stats['mean_signed_error']:.6f}")
            print(f"    relative_rmse={stats['relative_rmse']:.6f}")
            if selected_indices.numel() > 0:
                print(f"    selected_mean_abs_error={stats['selected_mean_abs_error']:.6f}")
                print(f"    non_selected_mean_abs_error={stats['non_selected_mean_abs_error']:.6f}")


if __name__ == "__main__":
    main()
