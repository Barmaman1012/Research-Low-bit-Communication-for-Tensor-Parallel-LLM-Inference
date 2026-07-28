from __future__ import annotations

import argparse
import math
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
from lowbit_tp_comm.dtypes import (
    DTYPE_CHOICES,
    ensure_dtype_supported,
    model_dtype_metadata,
    model_load_kwargs,
    resolve_dtype,
    validate_model_dtype,
)
from lowbit_tp_comm.hooks import ActivationCapture, ModuleInputOutputCapture, list_candidate_sync_modules
from lowbit_tp_comm.tp_linear import compute_row_parallel_partials_for_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate real model activations for TP compression.")
    parser.add_argument("--model_name", default="sshleifer/tiny-gpt2")
    parser.add_argument("--dataset_name", default="wikitext")
    parser.add_argument("--dataset_config", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="train")
    parser.add_argument("--num_sequences", type=int, default=32)
    parser.add_argument("--sequence_length", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.01)
    parser.add_argument("--k_fraction", type=float, default=0.015625)
    parser.add_argument("--output_path", default="calibration.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=DTYPE_CHOICES, default="auto")
    parser.add_argument("--patterns", nargs="*", default=None)
    parser.add_argument("--target_style", choices=["auto", "gpt2", "llama"], default="auto")
    parser.add_argument("--simulate_row_parallel_calibration", action="store_true")
    parser.add_argument("--num_partitions", type=int, default=2)
    return parser.parse_args()


def select_nonempty_texts(dataset, num_sequences: int) -> list[str]:
    selected: list[str] = []
    for row in dataset:
        text = row.get("text", "")
        if isinstance(text, str) and text.strip():
            selected.append(text)
        if len(selected) >= num_sequences:
            break
    if len(selected) < num_sequences:
        raise ValueError(f"Requested {num_sequences} non-empty sequences, found only {len(selected)}.")
    return selected


def build_inputs(
    tokenizer,
    texts: list[str],
    sequence_length: int,
    device: torch.device,
) -> dict[str, Any]:
    """Tokenize text and place every model input on the execution device."""

    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=sequence_length,
    )
    # Tokenizers normally return BatchEncoding, but tests and downstream
    # callers may provide a plain mapping.  Move tensor values explicitly so
    # both forms work while retaining any non-tensor metadata unchanged.
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in encoded.items()
    }


def load_text_dataset(dataset_name: str, dataset_config: str, split: str):
    try:
        return load_dataset(dataset_name, dataset_config, split=split)
    except HfUriError:
        if "/" in dataset_name:
            raise
        fallback_name = f"Salesforce/{dataset_name}"
        print(f"Retrying dataset load with namespaced path: {fallback_name}")
        return load_dataset(fallback_name, dataset_config, split=split)


def build_dtype_metadata(
    *,
    requested_dtype_name: str,
    model: torch.nn.Module,
    calibrators: dict[str, EMAMinMaxCalibrator],
    observed_partial_dtypes: set[str],
) -> dict[str, Any]:
    """Describe execution dtype separately from intentionally FP32 statistics."""

    requested_dtype = resolve_dtype(requested_dtype_name)
    model_dtype = next(model.parameters()).dtype
    return {
        "requested_model_dtype": requested_dtype_name,
        **model_dtype_metadata(model),
        "partial_output_dtype": (
            next(iter(observed_partial_dtypes)) if len(observed_partial_dtypes) == 1 else sorted(observed_partial_dtypes)
        ),
        "calibration_statistics_dtype": str(next(iter(calibrators.values())).min_vals.dtype)
        if calibrators
        else str(torch.float32),
        "intended_selected_feature_communication_dtype": str(requested_dtype or model_dtype),
        "intended_reconstructed_output_dtype": str(requested_dtype or model_dtype),
    }


def main() -> None:
    args = parse_args()

    device = torch.device(args.device)
    requested_dtype = resolve_dtype(args.dtype)
    ensure_dtype_supported(requested_dtype, device)
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_load_kwargs(args.dtype))
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
    model_device = next(model.parameters()).device

    dataset = load_text_dataset(args.dataset_name, args.dataset_config, split=args.split)
    texts = select_nonempty_texts(dataset, args.num_sequences)

    candidate_modules = list_candidate_sync_modules(
        model,
        patterns=args.patterns,
        target_style=args.target_style,
    )
    if not candidate_modules:
        raise ValueError("No candidate synchronization modules were found.")

    candidate_names = [name for name, _module in candidate_modules]
    print("Discovered candidate modules:")
    for name in candidate_names:
        print(f"- {name}")

    if args.simulate_row_parallel_calibration:
        # These inputs are reused with the target module's weights below, so
        # they must remain on the model's execution device.
        capture = ModuleInputOutputCapture(model, candidate_names, store_on_cpu=False)
    else:
        capture = ActivationCapture(model, candidate_names)
    calibrators: dict[str, EMAMinMaxCalibrator] = {}
    observed_partial_dtypes: set[str] = set()

    try:
        with torch.no_grad():
            for index, text in enumerate(texts, start=1):
                capture.clear()
                encoded = build_inputs(tokenizer, [text], args.sequence_length, model_device)
                model(**encoded)

                for module_name in candidate_names:
                    if args.simulate_row_parallel_calibration:
                        module = dict(candidate_modules)[module_name]
                        input_tensors = capture.get_inputs(module_name)
                        for input_tensor in input_tensors:
                            partial_outputs = compute_row_parallel_partials_for_module(
                                module,
                                input_tensor,
                                num_partitions=args.num_partitions,
                            )
                            observed_partial_dtypes.update(str(partial.dtype) for partial in partial_outputs)
                            feature_dim = partial_outputs[0].shape[-1]
                            if module_name not in calibrators:
                                calibrators[module_name] = EMAMinMaxCalibrator(
                                    num_partitions=args.num_partitions,
                                    feature_dim=feature_dim,
                                    gamma=args.gamma,
                                    device="cpu",
                                )
                            calibrators[module_name].update(partial_outputs)
                    else:
                        outputs = capture.get_outputs(module_name)
                        for output_tensor in outputs:
                            observed_partial_dtypes.add(str(output_tensor.dtype))
                            feature_dim = output_tensor.shape[-1]
                            if module_name not in calibrators:
                                calibrators[module_name] = EMAMinMaxCalibrator(
                                    num_partitions=1,
                                    feature_dim=feature_dim,
                                    gamma=args.gamma,
                                    device="cpu",
                                )
                            calibrators[module_name].update([output_tensor])

                if index % 8 == 0 or index == len(texts):
                    print(f"Processed {index}/{len(texts)} sequences")
    finally:
        capture.remove()

    module_payload: dict[str, dict[str, torch.Tensor | int | dict]] = {}
    print()
    print("Calibration summary:")
    for module_name, calibrator in calibrators.items():
        aggregated_ranges = calibrator.aggregated_ranges()
        feature_dim = calibrator.feature_dim
        k = min(feature_dim, max(1, math.floor(feature_dim * args.k_fraction))) if feature_dim > 0 else 0
        topk_indices = calibrator.topk_features(k)
        _scales = calibrator.scales_per_partition()

        top_values, _ = torch.topk(aggregated_ranges, k=min(10, feature_dim))
        print(f"Module: {module_name}")
        print(f"  feature_dim={feature_dim}")
        print(f"  k={k}")
        print(f"  top_aggregated_ranges={top_values.tolist()}")
        print(f"  selected_indices={topk_indices.tolist()}")

        module_payload[module_name] = {
            "state_dict": calibrator.state_dict(),
            "aggregated_ranges": aggregated_ranges.clone(),
            "topk_indices": topk_indices.clone(),
            "k": k,
            "feature_dim": feature_dim,
        }

    payload = {
        "model_name": args.model_name,
        "gamma": args.gamma,
        "k_fraction": args.k_fraction,
        "num_sequences": args.num_sequences,
        "sequence_length": args.sequence_length,
        "simulated_row_parallel_calibration": args.simulate_row_parallel_calibration,
        "num_partitions": args.num_partitions if args.simulate_row_parallel_calibration else 1,
        # Min/max EMA and derived scales are FP32 for stable calibration, but
        # their inputs above are partial outputs from the requested model dtype.
        "dtype_metadata": build_dtype_metadata(
            requested_dtype_name=args.dtype,
            model=model,
            calibrators=calibrators,
            observed_partial_dtypes=observed_partial_dtypes,
        ),
        "modules": module_payload,
    }
    torch.save(payload, args.output_path)
    print()
    print(f"Saved calibration to {args.output_path}")


if __name__ == "__main__":
    main()
