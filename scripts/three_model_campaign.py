"""Validated command construction for the three-model range-sweep campaign.

This module is intentionally the single place that reads the YAML manifest and
immutable revision lock.  Slurm wrappers call ``run-stage``; tests use the pure
command-building functions.  It never submits Slurm jobs itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "experiments" / "three_model_range_sweep.yaml"
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
STAGES = ("b", "c", "d", "f", "g")


def dependency_graph(through_stage: str) -> list[tuple[str, str | None]]:
    """Logical Slurm graph; arrays are represented by their parent job IDs."""
    order = ["b", "c", "d", "f", "g"]
    if through_stage not in order:
        raise ValueError(f"Unsupported terminal stage {through_stage!r}.")
    end = order.index(through_stage) + 1
    return [(stage, None if index == 0 else order[index - 1]) for index, stage in enumerate(order[:end])]


def load_manifest(path: str | Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("Campaign manifest must be a mapping.")
    return manifest


def _valid_revision(value: Any) -> bool:
    return isinstance(value, str) and bool(REVISION_RE.fullmatch(value))


def validate_lock(manifest: dict[str, Any], lock_path: str | Path | None = None) -> dict[str, Any]:
    """Read and strictly validate the immutable model/tokenizer revision lock."""
    path = Path(lock_path or ROOT / manifest["experiment"]["revision_lock"])
    if not path.exists():
        raise ValueError(f"Immutable revision lock is required: {path}")
    try:
        lock = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid revision lock JSON: {path}") from exc
    if not isinstance(lock, dict):
        raise ValueError("Revision lock must map model keys to immutable revisions.")
    for key, spec in manifest["models"].items():
        entry = lock.get(key)
        if not isinstance(entry, dict):
            raise ValueError(f"Revision lock is missing model entry {key!r}.")
        if entry.get("model_id") != spec["model_id"]:
            raise ValueError(f"Revision lock model_id disagrees with manifest for {key!r}.")
        for field in ("model_revision", "tokenizer_revision"):
            value = entry.get(field)
            if value in {"main", "RESOLVE_STAGE_A"} or not _valid_revision(value):
                raise ValueError(f"{key} has invalid immutable {field}: {value!r}.")
    return lock


def model_lookup(
    model_key: str, manifest: dict[str, Any], lock_path: str | Path | None = None
) -> dict[str, Any]:
    """Return the complete locked campaign specification for one model key."""
    if model_key not in manifest["models"]:
        raise ValueError(f"Unknown campaign model key {model_key!r}.")
    lock = validate_lock(manifest, lock_path)
    spec = manifest["models"][model_key]
    calibration = manifest["calibration"]
    return {
        "model_key": model_key,
        "model_id": spec["model_id"],
        "model_revision": lock[model_key]["model_revision"],
        "tokenizer_revision": lock[model_key]["tokenizer_revision"],
        "target_style": spec["target_style"],
        "num_partitions": int(spec["num_partitions"]),
        "dtype": spec["dtype"],
        "resources": dict(spec["resources"]),
        "calibration_path": calibration_path(manifest, model_key),
    }


def configuration_names(manifest: dict[str, Any]) -> list[str]:
    return [
        *manifest["modes"],
        *[f"range_threshold_bf16-t{threshold:.1f}" for threshold in manifest["range_thresholds"]],
    ]


def configuration_for_index(manifest: dict[str, Any], array_index: int) -> tuple[str, float | None]:
    configurations = configuration_names(manifest)
    if array_index < 0 or array_index >= len(configurations):
        raise ValueError(f"SLURM_ARRAY_TASK_ID must be in [0, {len(configurations) - 1}], got {array_index}.")
    config = configurations[array_index]
    if config.startswith("range_threshold_bf16-t"):
        return "range_threshold_bf16", float(config.rsplit("t", 1)[1])
    return config, None


def calibration_path(manifest: dict[str, Any], model_key: str) -> Path:
    return ROOT / manifest["outputs"]["root"] / manifest["outputs"]["calibration_pattern"].format(model=model_key)


def output_path(manifest: dict[str, Any], model_key: str, stage: str, configuration: str) -> Path:
    return ROOT / manifest["outputs"]["root"] / manifest["outputs"]["result_pattern"].format(
        model=model_key, stage=stage, configuration=configuration
    )


def stage_output_path(manifest: dict[str, Any], model_key: str, stage: str, job_id: str) -> Path:
    return ROOT / manifest["outputs"]["root"] / model_key / stage / f"{stage}-{job_id}.json"


def command_for_stage(
    stage: str,
    model_key: str,
    manifest: dict[str, Any],
    lock_path: str | Path,
    *,
    array_index: int | None = None,
    job_id: str = "manual",
) -> list[str]:
    """Build the exact research-script command for a stage, without executing it."""
    if stage not in STAGES:
        raise ValueError(f"Unsupported stage {stage!r}.")
    spec = model_lookup(model_key, manifest, lock_path)
    python = str(ROOT / ".venv-gpu310" / "bin" / "python")
    base = [python]
    if stage == "b":
        return base + [
            str(ROOT / "scripts" / "model_loading_smoke.py"), "--model_name", spec["model_id"],
            "--model_revision", spec["model_revision"], "--tokenizer_revision", spec["tokenizer_revision"],
            "--target_style", spec["target_style"], "--output_path", str(stage_output_path(manifest, model_key, "load_smoke", job_id)),
        ]
    if stage == "c":
        c = manifest["calibration"]
        return base + [
            str(ROOT / "scripts" / "calibrate_model.py"), "--model_name", spec["model_id"],
            "--model_revision", spec["model_revision"], "--tokenizer_revision", spec["tokenizer_revision"],
            "--dataset_name", c["dataset_name"], "--dataset_config", c["dataset_config"],
            "--dataset_revision", c["dataset_revision"], "--split", c["split"],
            "--sampling_strategy", c["sampling_strategy"], "--seed", str(c["sampling_seed"]),
            "--num_sequences", str(c["num_sequences"]), "--sequence_length", str(c["sequence_length"]),
            "--gamma", str(c["gamma"]), "--k_fraction", str(c["k_fraction"]),
            "--simulate_row_parallel_calibration", "--num_partitions", str(spec["num_partitions"]),
            "--dtype", spec["dtype"], "--device", "cuda", "--target_style", spec["target_style"],
            "--output_path", str(spec["calibration_path"]),
        ]
    if stage == "d":
        validation = stage_output_path(manifest, model_key, "validation", job_id)
        analysis = ROOT / manifest["outputs"]["root"] / model_key / "range_analysis"
        return base + [str(ROOT / "scripts" / "validate_three_model_campaign.py"), "calibration",
                       "--manifest", str(DEFAULT_MANIFEST), "--lock", str(lock_path), "--model-key", model_key,
                       "--calibration-path", str(spec["calibration_path"]), "--output-path", str(validation),
                       "--analysis-output-dir", str(analysis),
                       "--thresholds", ",".join(str(value) for value in manifest["range_thresholds"])]
    if array_index is None:
        raise ValueError(f"Stage {stage} requires an array index.")
    mode, threshold = configuration_for_index(manifest, array_index)
    config = configuration_names(manifest)[array_index]
    eval_stage = "smoke" if stage == "f" else "full"
    output = output_path(manifest, model_key, eval_stage, config)
    e = manifest["evaluation"]
    tasks = e["smoke_tasks"] if stage == "f" else e["tasks"]
    command = base + [
        str(ROOT / "scripts" / "eval_lm_harness.py"), "--model_name", spec["model_id"],
        "--model_revision", spec["model_revision"], "--tokenizer_revision", spec["tokenizer_revision"],
        "--calibration_path", str(spec["calibration_path"]), "--target_style", spec["target_style"],
        "--mode", mode, "--num_partitions", str(spec["num_partitions"]), "--dtype", spec["dtype"],
        "--device", "cuda", "--tasks", ",".join(tasks), "--batch_size", str(e["batch_size"]),
        "--seed", str(e["seed"]), "--output_path", str(output),
    ]
    if stage == "f":
        command += ["--limit", str(e["smoke_limit"])]
    if threshold is not None:
        command += ["--bf16_range_threshold", str(threshold)]
    return command


def checksum_indices(indices: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, indices)).encode()).hexdigest()


def _refuse_overwrite(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing campaign output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def run_stage(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    command = command_for_stage(args.stage, args.model_key, manifest, args.lock, array_index=args.array_index, job_id=args.job_id)
    if args.stage == "c":
        _refuse_overwrite(calibration_path(manifest, args.model_key))
    elif args.stage in {"b", "d"}:
        _refuse_overwrite(stage_output_path(manifest, args.model_key, "load_smoke" if args.stage == "b" else "validation", args.job_id))
    else:
        _refuse_overwrite(output_path(manifest, args.model_key, "smoke" if args.stage == "f" else "full", configuration_names(manifest)[args.array_index]))
    subprocess.run(command, check=True, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    lookup = sub.add_parser("lookup"); lookup.add_argument("--model-key", required=True); lookup.add_argument("--manifest", default=str(DEFAULT_MANIFEST)); lookup.add_argument("--lock", required=True)
    listing = sub.add_parser("list-configurations"); listing.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    build = sub.add_parser("print-command")
    build.add_argument("stage", choices=STAGES); build.add_argument("--model-key", required=True); build.add_argument("--manifest", default=str(DEFAULT_MANIFEST)); build.add_argument("--lock", required=True); build.add_argument("--array-index", type=int); build.add_argument("--job-id", default="manual")
    run = sub.add_parser("run-stage")
    run.add_argument("stage", choices=STAGES); run.add_argument("--model-key", required=True); run.add_argument("--manifest", default=str(DEFAULT_MANIFEST)); run.add_argument("--lock", required=True); run.add_argument("--array-index", type=int); run.add_argument("--job-id", required=True)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    if args.command == "lookup":
        print(json.dumps(model_lookup(args.model_key, manifest, args.lock), indent=2, default=str)); return
    if args.command == "list-configurations":
        print("\n".join(configuration_names(manifest))); return
    if args.command == "print-command":
        print(json.dumps(command_for_stage(args.stage, args.model_key, manifest, args.lock, array_index=args.array_index, job_id=args.job_id))); return
    run_stage(args)


if __name__ == "__main__":
    main()
