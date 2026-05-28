from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lowbit_tp_comm.hooks import build_hybrid_replacements_from_calibration, replace_modules_by_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare original logits to replacement modes.")
    parser.add_argument("--model_name", default="distilgpt2")
    parser.add_argument("--calibration_path", default="calibration-distilgpt2.pt")
    parser.add_argument("--target_style", choices=["auto", "gpt2", "llama"], default="gpt2")
    parser.add_argument("--num_partitions", type=int, default=2)
    parser.add_argument("--sequence_length", type=int, default=64)
    parser.add_argument("--text", default="The quick brown fox jumps over the lazy dog.")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def load_model_or_raise(model_name: str, device: torch.device):
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


def compare_logits(reference: torch.Tensor, candidate: torch.Tensor) -> tuple[float, float, float]:
    diff = (reference - candidate).abs()
    max_abs_diff = float(diff.max().item())
    mean_abs_diff = float(diff.mean().item())
    relative_max_diff = max_abs_diff / max(float(reference.abs().max().item()), 1e-12)
    return max_abs_diff, mean_abs_diff, relative_max_diff


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

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
    encoded = tokenizer(
        args.text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=args.sequence_length,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    calibration = torch.load(args.calibration_path, map_location="cpu", weights_only=False)

    reference_model = load_model_or_raise(args.model_name, device)
    with torch.no_grad():
        reference_logits = reference_model(**encoded).logits.detach().cpu()

    for mode in ["tp_uncompressed", "all_bf16"]:
        candidate_model = load_model_or_raise(args.model_name, device)
        replacements = build_hybrid_replacements_from_calibration(
            candidate_model,
            calibration=calibration,
            mode=mode,
            num_partitions=args.num_partitions,
        )
        replace_modules_by_name(candidate_model, replacements)
        with torch.no_grad():
            candidate_logits = candidate_model(**encoded).logits.detach().cpu()
        max_abs_diff, mean_abs_diff, relative_max_diff = compare_logits(reference_logits, candidate_logits)
        print(f"mode={mode}")
        print(f"max_abs_diff={max_abs_diff:.10f}")
        print(f"mean_abs_diff={mean_abs_diff:.10f}")
        print(f"relative_max_diff={relative_max_diff:.10f}")
        print()


if __name__ == "__main__":
    main()
