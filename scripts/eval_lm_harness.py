from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lowbit_tp_comm.hooks import build_hybrid_replacements_from_calibration, replace_modules_by_name

VALID_MODES = {"full", "tp_uncompressed", "all_bf16", "int4", "random_bf16", "selected_bf16"}
DEFAULT_TASKS = ["arc_easy", "arc_challenge", "winogrande", "hellaswag", "boolq"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate simulated TP modes with lm-eval-harness.")
    parser.add_argument("--model_name", default="distilgpt2")
    parser.add_argument("--calibration_path", default=None)
    parser.add_argument("--target_style", choices=["auto", "gpt2", "llama"], default="gpt2")
    parser.add_argument("--mode", choices=sorted(VALID_MODES), default="full")
    parser.add_argument("--modes", default=None)
    parser.add_argument("--num_partitions", type=int, default=2)
    parser.add_argument("--num_bits", type=int, default=4)
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cpu")
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


def load_tokenizer_or_raise(model_name: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception:
        print(
            "Tokenizer loading failed. This model may be gated. "
            "Run `huggingface-cli login` or choose an open model.",
            file=sys.stderr,
        )
        raise
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    return tokenizer


def build_model_for_mode(
    *,
    model_name: str,
    calibration_path: str | None,
    target_style: str,
    mode: str,
    num_partitions: int,
    num_bits: int,
    device: torch.device,
    seed: int,
):
    model = load_model_or_raise(model_name, device)
    calibration = None
    if mode != "full":
        if calibration_path is None:
            raise ValueError("--calibration_path is required for non-full modes.")
        calibration = torch.load(calibration_path, map_location="cpu", weights_only=False)
        replacements = build_hybrid_replacements_from_calibration(
            model,
            calibration=calibration,
            mode=mode,
            num_partitions=num_partitions,
            num_bits=num_bits,
            seed=seed,
        )
        replace_modules_by_name(model, replacements)
        # Replacements include calibration buffers loaded from CPU.  Move the
        # complete model again so all replacement parameters and buffers match
        # the requested execution device before lm-eval invokes it.
        model.to(device)
    return model, calibration


def extract_rows(mode: str, results: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_name, metrics in results.get("results", {}).items():
        for metric_name, value in metrics.items():
            if metric_name.endswith(",stderr"):
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
    preferred_order = ["acc_norm,none", "acc,none", "exact_match,none"]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["mode"]), str(row["task"]))
        grouped.setdefault(key, []).append(row)

    selected: list[dict[str, Any]] = []
    for _key, task_rows in grouped.items():
        picked = None
        for metric_name in preferred_order:
            picked = next((row for row in task_rows if row["metric"] == metric_name), None)
            if picked is not None:
                break
        if picked is None and task_rows:
            picked = task_rows[0]
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
    tokenizer = load_tokenizer_or_raise(args.model_name)
    tasks = parse_csv_list(args.tasks, default=DEFAULT_TASKS)
    modes = parse_modes(args.mode, args.modes)

    all_rows: list[dict[str, Any]] = []
    raw_results: dict[str, Any] = {}
    for mode in modes:
        model, _calibration = build_model_for_mode(
            model_name=args.model_name,
            calibration_path=args.calibration_path,
            target_style=args.target_style,
            mode=mode,
            num_partitions=args.num_partitions,
            num_bits=args.num_bits,
            device=device,
            seed=args.seed,
        )
        lm = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            device=str(device),
        )
        results = lm_eval.simple_evaluate(
            model=lm,
            tasks=tasks,
            limit=args.limit,
            batch_size=args.batch_size,
            device=str(device),
        )
        raw_results[mode] = results
        all_rows.extend(extract_rows(mode, results))

    print(format_results_table(all_rows))
    print(format_average_summary(all_rows))

    if args.output_path is not None:
        payload = make_json_serializable({"rows": all_rows, "raw_results": raw_results})
        with open(args.output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nSaved results to {args.output_path}")


if __name__ == "__main__":
    main()
