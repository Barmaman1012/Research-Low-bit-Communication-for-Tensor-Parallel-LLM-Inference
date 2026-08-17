"""CPU-only validation and summary utilities for the locked range-sweep campaign."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from three_model_campaign import configuration_names, load_manifest, model_lookup, output_path  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_calibration(payload: dict[str, Any], spec: dict[str, Any], calibration: dict[str, Any]) -> dict[str, Any]:
    """Validate the exact fields consumed by simulation replacements."""
    required = {
        "model_name": spec["model_id"], "model_revision": spec["model_revision"],
        "tokenizer_revision": spec["tokenizer_revision"], "dataset_name": calibration["dataset_name"],
        "dataset_config": calibration["dataset_config"], "dataset_revision": calibration["dataset_revision"],
        "split": calibration["split"], "sampling_strategy": calibration["sampling_strategy"],
        "sampling_seed": calibration["sampling_seed"], "num_sequences": calibration["num_sequences"],
        "sequence_length": calibration["sequence_length"], "gamma": calibration["gamma"],
        "k_fraction": calibration["k_fraction"], "num_partitions": spec["num_partitions"],
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise ValueError(f"Calibration {key} mismatch: expected {expected!r}, got {payload.get(key)!r}.")
    if not payload.get("simulated_row_parallel_calibration"):
        raise ValueError("Calibration was not produced with simulated row-parallel calibration.")
    dtype = payload.get("dtype_metadata", {}).get("requested_model_dtype")
    if dtype != spec["dtype"]:
        raise ValueError(f"Calibration dtype mismatch: expected {spec['dtype']!r}, got {dtype!r}.")
    modules = payload.get("modules")
    if not isinstance(modules, dict) or not modules:
        raise ValueError("Calibration modules mapping is missing or empty.")
    dimensions: dict[str, int] = {}
    for name, module in modules.items():
        feature_dim, k = int(module["feature_dim"]), int(module["k"])
        ranges, indices = module["aggregated_ranges"], module["topk_indices"]
        if not isinstance(ranges, torch.Tensor) or ranges.ndim != 1 or ranges.numel() != feature_dim:
            raise ValueError(f"{name}: aggregated_ranges does not match feature_dim.")
        if not torch.isfinite(ranges).all() or float(ranges.median()) <= 0:
            raise ValueError(f"{name}: ranges must be finite with a positive median.")
        if not isinstance(indices, torch.Tensor) or indices.ndim != 1 or indices.numel() != k:
            raise ValueError(f"{name}: topk_indices does not match k.")
        indices = indices.to(dtype=torch.long, device="cpu")
        if indices.unique().numel() != k or (k and (indices.min() < 0 or indices.max() >= feature_dim)):
            raise ValueError(f"{name}: topk_indices must be unique and in range.")
        dimensions[name] = feature_dim
    return {"module_count": len(modules), "module_feature_dimensions": dimensions,
            "total_feature_count": sum(dimensions.values())}


def run_calibration_validation(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    spec = model_lookup(args.model_key, manifest, args.lock)
    path = Path(args.calibration_path)
    if not path.exists():
        raise FileNotFoundError(f"Calibration artifact does not exist: {path}")
    output = Path(args.output_path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    report = validate_calibration(payload, spec, manifest["calibration"])
    digest = sha256(path)
    sums_path = path.parent / "SHA256SUMS"
    if not sums_path.exists():
        raise FileNotFoundError(f"Calibration checksum file is required: {sums_path}")
    recorded = sums_path.read_text(encoding="utf-8").strip().split(maxsplit=1)
    if len(recorded) != 2 or recorded[0] != digest:
        raise ValueError(f"Calibration SHA256SUMS does not match {path}.")
    report.update({"model_key": args.model_key, "calibration_path": str(path), "calibration_sha256": digest,
                   "sha256sums_path": str(sums_path),
                   "model_revision": spec["model_revision"], "tokenizer_revision": spec["tokenizer_revision"]})
    if args.analysis_output_dir:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "analyze_calibration_ranges.py"),
                        "--calibration_path", str(path), "--output_dir", args.analysis_output_dir,
                        "--thresholds", args.thresholds], check=True, cwd=ROOT)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    calibration = sub.add_parser("calibration")
    calibration.add_argument("--manifest", default=str(ROOT / "experiments" / "three_model_range_sweep.yaml"))
    calibration.add_argument("--lock", required=True); calibration.add_argument("--model-key", required=True)
    calibration.add_argument("--calibration-path", required=True); calibration.add_argument("--output-path", required=True)
    calibration.add_argument("--analysis-output-dir", required=True)
    calibration.add_argument("--thresholds", required=True)
    args = parser.parse_args()
    if args.command == "calibration": run_calibration_validation(args)


if __name__ == "__main__":
    main()
