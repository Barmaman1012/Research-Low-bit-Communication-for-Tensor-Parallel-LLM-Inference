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

from lowbit_tp_comm.hooks import build_hybrid_replacements_from_calibration, replace_modules_by_name

MODES = ["full", "int4", "random_bf16", "selected_bf16"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare per-sequence losses across quantization modes.")
    parser.add_argument("--model_name", default="distilgpt2")
    parser.add_argument("--calibration_path", default="calibration-distilgpt2-tp2.pt")
    parser.add_argument("--target_style", choices=["auto", "gpt2", "llama"], default="gpt2")
    parser.add_argument("--num_partitions", type=int, default=2)
    parser.add_argument("--num_sequences", type=int, default=64)
    parser.add_argument("--sequence_length", type=int, default=128)
    parser.add_argument("--device", default="cpu")
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


def load_model(model_name: str, device: torch.device):
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
    model.to(device)
    return model


def build_model_for_mode(
    mode: str,
    model_name: str,
    calibration: dict[str, Any],
    device: torch.device,
    num_partitions: int,
    target_style: str,
    num_bits: int,
    seed: int,
):
    model = load_model(model_name, device)
    if mode != "full":
        replacements = build_hybrid_replacements_from_calibration(
            model,
            calibration=calibration,
            mode=mode,
            num_partitions=num_partitions,
            num_bits=num_bits,
            seed=seed,
        )
        replace_modules_by_name(model, replacements)
    return model


def summarize_deltas(name: str, deltas: list[float]) -> None:
    better = sum(delta < 0 for delta in deltas)
    worse = sum(delta > 0 for delta in deltas)
    indexed = list(enumerate(deltas))
    most_improved = sorted(indexed, key=lambda item: item[1])[:5]
    most_degraded = sorted(indexed, key=lambda item: item[1], reverse=True)[:5]

    print(f"\n{name}:")
    print(f"  mean_delta={statistics.mean(deltas):.6f}")
    print(f"  median_delta={statistics.median(deltas):.6f}")
    print(f"  better_than_full={better}")
    print(f"  worse_than_full={worse}")
    print(f"  worst_improved={most_improved}")
    print(f"  worst_degraded={most_degraded}")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

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

    calibration = torch.load(args.calibration_path, map_location="cpu", weights_only=False)
    dataset = load_text_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = select_nonempty_texts(dataset, args.num_sequences)

    mode_losses: dict[str, list[float]] = {}
    for mode in MODES:
        model = build_model_for_mode(
            mode,
            model_name=args.model_name,
            calibration=calibration,
            device=device,
            num_partitions=args.num_partitions,
            target_style=args.target_style,
            num_bits=args.num_bits,
            seed=args.seed,
        )
        losses: list[float] = []
        with torch.no_grad():
            for text in texts:
                encoded = tokenizer(
                    text,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=args.sequence_length,
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                outputs = model(**encoded, labels=encoded["input_ids"])
                losses.append(float(outputs.loss.item()))
        mode_losses[mode] = losses

    print("sequence_index | full_loss | int4_loss | random_bf16_loss | selected_bf16_loss | int4_minus_full | selected_minus_full")
    for index in range(len(texts)):
        full_loss = mode_losses["full"][index]
        int4_loss = mode_losses["int4"][index]
        random_loss = mode_losses["random_bf16"][index]
        selected_loss = mode_losses["selected_bf16"][index]
        print(
            f"{index} | {full_loss:.6f} | {int4_loss:.6f} | {random_loss:.6f} | {selected_loss:.6f} | "
            f"{int4_loss - full_loss:.6f} | {selected_loss - full_loss:.6f}"
        )

    summarize_deltas("int4_minus_full", [a - b for a, b in zip(mode_losses["int4"], mode_losses["full"], strict=True)])
    summarize_deltas(
        "random_bf16_minus_full",
        [a - b for a, b in zip(mode_losses["random_bf16"], mode_losses["full"], strict=True)],
    )
    summarize_deltas(
        "selected_bf16_minus_full",
        [a - b for a, b in zip(mode_losses["selected_bf16"], mode_losses["full"], strict=True)],
    )


if __name__ == "__main__":
    main()
