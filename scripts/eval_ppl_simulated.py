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

from lowbit_tp_comm.hooks import (
    build_hybrid_replacements_from_calibration,
    list_candidate_sync_modules,
    replace_modules_by_name,
)

VALID_MODES = {"full", "int4", "random_bf16", "selected_bf16"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate perplexity with simulated TP communication compression.")
    parser.add_argument("--model_name", default="sshleifer/tiny-gpt2")
    parser.add_argument("--calibration_path", default="calibration.pt")
    parser.add_argument("--dataset_name", default="wikitext")
    parser.add_argument("--dataset_config", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--num_sequences", type=int, default=32)
    parser.add_argument("--sequence_length", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num_partitions", type=int, default=2)
    parser.add_argument("--mode", choices=sorted(VALID_MODES), default="full")
    parser.add_argument("--modes", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose_bits", action="store_true")
    parser.add_argument("--target_style", choices=["auto", "gpt2", "llama"], default="auto")
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


def parse_modes(mode: str, modes: str | None) -> list[str]:
    if modes is None:
        return [mode]

    parsed = [item.strip() for item in modes.split(",") if item.strip()]
    if not parsed:
        raise ValueError("--modes was provided but no valid modes were parsed.")
    invalid = [item for item in parsed if item not in VALID_MODES]
    if invalid:
        raise ValueError(f"Unsupported modes in --modes: {invalid}")
    return parsed


def compute_module_avg_bits(feature_dim: int, k: int, num_bits: int = 4) -> float:
    if feature_dim <= 0:
        raise ValueError(f"feature_dim must be positive, got {feature_dim}.")
    selected_fraction = min(max(k, 0), feature_dim) / feature_dim
    return selected_fraction * 16.0 + (1.0 - selected_fraction) * float(num_bits)


def compute_bits_summary(
    mode: str,
    calibration: dict[str, Any] | None,
    num_bits: int = 4,
) -> tuple[float, list[dict[str, float | int | str]]]:
    if mode == "full":
        return 16.0, []
    if mode == "int4":
        return float(num_bits), []
    if calibration is None:
        raise ValueError("Calibration payload is required for hybrid modes.")

    module_rows: list[dict[str, float | int | str]] = []
    for module_name, module_payload in calibration["modules"].items():
        feature_dim = int(module_payload["feature_dim"])
        k = int(module_payload["k"])
        selected_fraction = k / feature_dim if feature_dim > 0 else 0.0
        avg_bits = compute_module_avg_bits(feature_dim, k, num_bits=num_bits)
        module_rows.append(
            {
                "module_name": module_name,
                "feature_dim": feature_dim,
                "k": k,
                "selected_fraction": selected_fraction,
                "avg_bits": avg_bits,
            }
        )

    avg_bits = sum(float(row["avg_bits"]) for row in module_rows) / len(module_rows) if module_rows else float(num_bits)
    return avg_bits, module_rows


def build_model_for_mode(
    model_name: str,
    mode: str,
    calibration_path: str,
    num_partitions: int,
    seed: int,
    device: torch.device,
    target_style: str,
) -> tuple[torch.nn.Module, dict[str, Any] | None]:
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    except Exception:
        print(
            "Model loading failed. This model may be gated. "
            "Run `huggingface-cli login` or choose an open model.",
            file=sys.stderr,
        )
        raise
    model.eval()

    calibration = None
    if mode != "full":
        calibration = torch.load(calibration_path, map_location="cpu", weights_only=False)
        candidate_names = {
            name for name, _module in list_candidate_sync_modules(model, target_style=target_style)
        }
        filtered_modules = {
            module_name: payload
            for module_name, payload in calibration["modules"].items()
            if module_name in candidate_names
        }
        calibration = {**calibration, "modules": filtered_modules}
        replacements = build_hybrid_replacements_from_calibration(
            model,
            calibration=calibration,
            mode=mode,
            num_partitions=num_partitions,
            seed=seed,
        )
        replace_modules_by_name(model, replacements)

    model.to(device)
    return model, calibration


def evaluate_mode(
    *,
    model_name: str,
    mode: str,
    calibration_path: str,
    texts: list[str],
    tokenizer,
    device: torch.device,
    num_partitions: int,
    seed: int,
    sequence_length: int,
    verbose_bits: bool,
    target_style: str,
) -> dict[str, float | str]:
    model, calibration = build_model_for_mode(
        model_name=model_name,
        mode=mode,
        calibration_path=calibration_path,
        num_partitions=num_partitions,
        seed=seed,
        device=device,
        target_style=target_style,
    )
    avg_bits, module_rows = compute_bits_summary(mode, calibration)

    if verbose_bits and module_rows:
        print(f"Bit summary for mode={mode}:")
        for row in module_rows:
            print(
                f"- {row['module_name']}: E={row['feature_dim']}, k={row['k']}, "
                f"selected_fraction={float(row['selected_fraction']):.6f}, avg_bits={float(row['avg_bits']):.6f}"
            )

    losses: list[float] = []
    with torch.no_grad():
        for index, text in enumerate(texts, start=1):
            encoded = tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=sequence_length,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded, labels=encoded["input_ids"])
            losses.append(float(outputs.loss.item()))
            if index % 8 == 0 or index == len(texts):
                print(f"[{mode}] Processed {index}/{len(texts)} sequences")

    avg_loss = sum(losses) / len(losses)
    perplexity = math.exp(avg_loss)
    return {
        "mode": mode,
        "avg_loss": avg_loss,
        "perplexity": perplexity,
        "avg_bits_per_value": avg_bits,
    }


def print_single_result(result: dict[str, float | str], model_name: str, num_sequences: int, sequence_length: int) -> None:
    print()
    print(f"mode={result['mode']}")
    print(f"model_name={model_name}")
    print(f"num_sequences={num_sequences}")
    print(f"sequence_length={sequence_length}")
    print(f"avg_loss={float(result['avg_loss']):.6f}")
    print(f"perplexity={float(result['perplexity']):.6f}")
    print(f"avg_bits_per_value={float(result['avg_bits_per_value']):.6f}")


def print_comparison_table(results: list[dict[str, float | str]]) -> None:
    full_ppl = None
    for result in results:
        if result["mode"] == "full":
            full_ppl = float(result["perplexity"])
            break

    print()
    print("mode | avg_loss | perplexity | avg_bits_per_value | relative_ppl_to_full")
    for result in results:
        relative = "N/A" if full_ppl is None else f"{float(result['perplexity']) / full_ppl:.6f}"
        print(
            f"{result['mode']} | {float(result['avg_loss']):.6f} | "
            f"{float(result['perplexity']):.6f} | {float(result['avg_bits_per_value']):.6f} | {relative}"
        )


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    modes = parse_modes(args.mode, args.modes)

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    except Exception:
        print(
            "Tokenizer loading failed. This model may be gated. "
            "Run `huggingface-cli login` or choose an open model.",
            file=sys.stderr,
        )
        raise
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    dataset = load_text_dataset(args.dataset_name, args.dataset_config, split=args.split)
    texts = select_nonempty_texts(dataset, args.num_sequences)

    results = [
        evaluate_mode(
            model_name=args.model_name,
            mode=mode,
            calibration_path=args.calibration_path,
            texts=texts,
            tokenizer=tokenizer,
            device=device,
            num_partitions=args.num_partitions,
            seed=args.seed,
            sequence_length=args.sequence_length,
            verbose_bits=args.verbose_bits,
            target_style=args.target_style,
        )
        for mode in modes
    ]

    if len(results) == 1:
        print_single_result(results[0], args.model_name, args.num_sequences, args.sequence_length)
    else:
        print_comparison_table(results)


if __name__ == "__main__":
    main()
