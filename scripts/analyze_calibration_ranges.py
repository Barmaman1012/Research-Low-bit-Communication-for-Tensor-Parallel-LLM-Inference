"""CPU-only descriptive analysis of calibrated TP feature ranges."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

THRESHOLDS = [1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.40, 1.50, 1.75, 2.0, 2.5, 3.0, 4.0, 8.0, 16.0]
EQUAL_BUDGET_THRESHOLD = 1.4867687225341797
MODULE_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)\.(self_attn\.o_proj|mlp\.down_proj)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze calibration ranges on CPU; does not load a model.")
    parser.add_argument("--calibration_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--thresholds",
        default=None,
        help="Comma-separated positive finite median-normalized thresholds; preserves declared order.",
    )
    return parser.parse_args()


def parse_thresholds(raw: str | None) -> list[float]:
    """Parse an explicit grid without changing the legacy default when omitted."""
    if raw is None:
        return list(THRESHOLDS)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("--thresholds must be a non-empty comma-separated list.")
    values: list[float] = []
    for item in raw.split(","):
        token = item.strip()
        if not token:
            raise ValueError("--thresholds contains an empty threshold.")
        try:
            value = float(token)
        except ValueError as exc:
            raise ValueError(f"Invalid threshold {token!r} in --thresholds.") from exc
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"Thresholds must be finite and strictly positive; got {token!r}.")
        if value in values:
            raise ValueError(f"Duplicate threshold {value:g} in --thresholds.")
        values.append(value)
    return values


def parse_module_name(module_name: str) -> tuple[int, str]:
    match = MODULE_PATTERN.search(module_name)
    if match is None:
        raise ValueError(
            f"Unsupported calibrated module name {module_name!r}; expected a model.layers.N self_attn.o_proj or mlp.down_proj target."
        )
    return int(match.group(1)), match.group(2)


def validate_ranges(module_name: str, ranges: Any, feature_dim: int) -> torch.Tensor:
    if not isinstance(ranges, torch.Tensor) or ranges.ndim != 1 or ranges.numel() != feature_dim:
        shape = tuple(ranges.shape) if isinstance(ranges, torch.Tensor) else type(ranges).__name__
        raise ValueError(f"aggregated_ranges for {module_name!r} must have shape [{feature_dim}], got {shape}.")
    values = ranges.detach().to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(values).all():
        raise ValueError(f"aggregated_ranges for {module_name!r} contains non-finite values.")
    if values.numel() == 0:
        raise ValueError(f"aggregated_ranges for {module_name!r} must not be empty.")
    if values.median().item() <= 0:
        raise ValueError(f"aggregated_ranges median for {module_name!r} must be positive.")
    return values


def threshold_column_name(threshold: float) -> str:
    return f"count_ge_{threshold:g}x_median".replace(".", "_")


def extract_module_records(
    calibration: dict[str, Any], thresholds: list[float] | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    thresholds = list(THRESHOLDS if thresholds is None else thresholds)
    modules = calibration.get("modules")
    if not isinstance(modules, dict) or not modules:
        raise ValueError("Calibration artifact must contain a non-empty 'modules' mapping.")
    records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for module_name in sorted(modules):
        payload = modules[module_name]
        feature_dim = int(payload["feature_dim"])
        ranges = validate_ranges(module_name, payload["aggregated_ranges"], feature_dim)
        layer, module_type = parse_module_name(module_name)
        selected = payload["topk_indices"].detach().to(dtype=torch.long, device="cpu")
        if selected.ndim != 1 or selected.numel() != int(payload["k"]) or selected.numel() != selected.unique().numel():
            raise ValueError(f"topk_indices for {module_name!r} is incompatible with k.")
        if selected.numel() and (selected.min() < 0 or selected.max() >= feature_dim):
            raise ValueError(f"topk_indices for {module_name!r} contains out-of-range indices.")
        selected_mask = torch.zeros(feature_dim, dtype=torch.bool)
        selected_mask[selected] = True
        minimum, median, mean, maximum = (float(ranges.min()), float(ranges.median()), float(ranges.mean()), float(ranges.max()))
        median_norm, mean_norm = ranges / median, ranges / mean
        # Stable feature-index secondary ordering makes equal values reproducible.
        ranks = torch.empty(feature_dim, dtype=torch.long)
        ranks[torch.argsort(ranges, descending=True, stable=True)] = torch.arange(1, feature_dim + 1)
        sorted_values = torch.sort(ranges, descending=True).values
        topk_boundary = float(sorted_values[int(payload["k"]) - 1]) if int(payload["k"]) else None
        concentration = float(sorted_values[: int(payload["k"])].sum() / ranges.sum()) if ranges.sum() > 0 else 0.0
        summary: dict[str, Any] = {
            "module_name": module_name, "transformer_layer": layer, "module_type": module_type,
            "feature_dim": feature_dim, "k": int(payload["k"]), "minimum": minimum, "median": median,
            "mean": mean, "maximum": maximum, "q90": float(torch.quantile(ranges, .90)),
            "q95": float(torch.quantile(ranges, .95)), "q98": float(torch.quantile(ranges, .98)),
            "q99": float(torch.quantile(ranges, .99)), "q99_5": float(torch.quantile(ranges, .995)),
            "maximum_over_median": maximum / median, "existing_topk_boundary": topk_boundary,
            "existing_topk_concentration": concentration,
        }
        for threshold in thresholds:
            count = int((median_norm >= threshold).sum())
            summary[threshold_column_name(float(threshold))] = count
            summary[threshold_column_name(float(threshold)).replace("count_", "fraction_")] = count / feature_dim
        summaries.append(summary)
        for index in range(feature_dim):
            records.append({
                "module_name": module_name, "transformer_layer": layer, "module_type": module_type,
                "feature_index": index, "raw_aggregated_range": float(ranges[index]), "module_min": minimum,
                "module_median": median, "module_mean": mean, "normalized_by_median": float(median_norm[index]),
                "normalized_by_mean": float(mean_norm[index]), "within_module_rank": int(ranks[index]),
                "within_module_percentile": float((feature_dim - ranks[index] + 1) / feature_dim),
                "existing_selected_bf16": bool(selected_mask[index]),
            })
    return records, summaries


def threshold_summary(
    records: list[dict[str, Any]], module_summaries: list[dict[str, Any]], thresholds: list[float] = THRESHOLDS
) -> list[dict[str, Any]]:
    total = sum(int(row["feature_dim"]) for row in module_summaries)
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        counts = [sum(record["normalized_by_median"] >= threshold for record in records if record["module_name"] == row["module_name"])
                  for row in module_summaries]
        selected = sum(counts)
        fraction = selected / total
        rows.append({"threshold": threshold, "selected_count": selected, "bf16_fraction": fraction,
                     "int4_fraction": 1 - fraction, "average_bits_per_value": 4 + 12 * fraction,
                     "minimum_module_count": min(counts), "median_module_count": float(torch.tensor(counts).median()),
                     "maximum_module_count": max(counts)})
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], *, gzip_output: bool = False) -> None:
    if not rows:
        raise ValueError("Refusing to write an empty analysis table.")
    opener = gzip.open if gzip_output else open
    with opener(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _plots(records: list[dict[str, Any]], summaries: list[dict[str, Any]], thresholds: list[dict[str, Any]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised only in CLI environments without plotting support.
        raise RuntimeError("Plot generation requires matplotlib. Install it in the analysis environment.") from exc
    raw = torch.tensor([r["raw_aggregated_range"] for r in records])
    norm = torch.tensor([r["normalized_by_median"] for r in records])
    types = {kind: torch.tensor([r["normalized_by_median"] for r in records if r["module_type"] == kind]) for kind in ("self_attn.o_proj", "mlp.down_proj")}
    def save(name: str) -> None: plt.tight_layout(); plt.savefig(output_dir / name, dpi=180); plt.close()
    plt.hist(torch.log10(raw).numpy(), bins=100); plt.xlabel("log10(raw range)"); plt.ylabel("features"); save("hist_log10_raw_range.png")
    plt.hist(norm.numpy(), bins=150, range=(0, float(torch.quantile(norm, .995)))); plt.xlabel("range / module median"); plt.ylabel("features"); save("hist_median_normalized_range.png")
    grid = torch.logspace(-2, math.log10(max(16., float(norm.max()))), 300); survival = torch.tensor([(norm >= t).float().mean() for t in grid])
    plt.semilogx(grid.numpy(), survival.numpy()); plt.xlabel("range / module median"); plt.ylabel("fraction above threshold"); save("survival_median_normalized_range.png")
    values = torch.sort(norm).values; plt.plot(values.numpy(), torch.arange(1, len(values)+1).numpy()/len(values)); plt.xlabel("range / module median"); plt.ylabel("empirical CDF"); save("cdf_median_normalized_range.png")
    curves = torch.stack([torch.sort(torch.tensor([r["normalized_by_median"] for r in records if r["module_name"] == s["module_name"]])).values for s in summaries])
    for curve in curves: plt.plot(curve.numpy(), color="steelblue", alpha=.12)
    plt.yscale("log"); plt.xlabel("sorted feature rank"); plt.ylabel("range / module median"); save("all_module_sorted_normalized_curves.png")
    x = torch.arange(curves.shape[1]); plt.plot(x, curves.median(dim=0).values); plt.fill_between(x.numpy(), torch.quantile(curves,.1,dim=0).numpy(), torch.quantile(curves,.9,dim=0).numpy(), alpha=.25); plt.yscale("log"); plt.xlabel("sorted feature rank"); plt.ylabel("range / module median"); save("median_sorted_normalized_curve_band.png")
    for kind, values_by_type in types.items(): plt.hist(values_by_type.numpy(), bins=120, range=(0,float(torch.quantile(values_by_type,.995))), alpha=.65, label=kind)
    plt.legend(); plt.xlabel("range / module median"); save("module_type_normalized_summaries.png")
    heat = torch.tensor([[s["q90"]/s["median"], s["q95"]/s["median"], s["q99"]/s["median"], s["maximum_over_median"]] for s in summaries])
    plt.imshow(heat, aspect="auto"); plt.colorbar(label="median-normalized range"); plt.yticks(range(len(summaries)), [f"L{s['transformer_layer']} {s['module_type']}" for s in summaries], fontsize=5); plt.xticks(range(4), ["q90","q95","q99","max"]); save("layer_module_normalized_quantile_heatmap.png")
    labels = [str(row["threshold"]) for row in thresholds]; xs = range(len(labels))
    for summary in summaries:
        counts = [sum(record["normalized_by_median"] >= row["threshold"] for record in records if record["module_name"] == summary["module_name"])
                  for row in thresholds]
        plt.plot(xs, counts, alpha=.25)
    plt.xticks(xs, labels, rotation=45); plt.ylabel("BF16 features per module"); plt.xlabel("threshold"); save("per_module_bf16_counts_by_threshold.png")
    plt.plot([r["threshold"] for r in thresholds], [r["bf16_fraction"] for r in thresholds], marker="o"); plt.axvline(EQUAL_BUDGET_THRESHOLD, color="red", linestyle="--", label="Equal-budget boundary, not a precision requirement."); plt.xscale("log"); plt.legend(fontsize=7); plt.xlabel("threshold"); plt.ylabel("BF16 fraction"); save("threshold_vs_bf16_fraction.png")
    plt.plot([r["threshold"] for r in thresholds], [r["average_bits_per_value"] for r in thresholds], marker="o"); plt.axvline(EQUAL_BUDGET_THRESHOLD, color="red", linestyle="--", label="Equal-budget boundary, not a precision requirement."); plt.xscale("log"); plt.legend(fontsize=7); plt.xlabel("threshold"); plt.ylabel("theoretical average bits/value"); save("threshold_vs_average_bits.png")


def main() -> None:
    args = parse_args(); calibration_path = Path(args.calibration_path); output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = parse_thresholds(args.thresholds)
    calibration = torch.load(calibration_path, map_location="cpu", weights_only=False)
    records, summaries = extract_module_records(calibration, thresholds=thresholds)
    threshold_rows = threshold_summary(records, summaries, thresholds=thresholds)
    _write_csv(output_dir / "calibration_feature_ranges.csv.gz", records, gzip_output=True)
    _write_csv(output_dir / "calibration_module_ranges.csv", summaries)
    _write_csv(output_dir / "threshold_summary.csv", threshold_rows)
    sha256 = hashlib.sha256(calibration_path.read_bytes()).hexdigest()
    provenance = {"calibration_path": str(calibration_path), "calibration_sha256": sha256,
                  "model_revision": calibration.get("resolved_model_revision", calibration.get("model_revision")),
                  "tokenizer_revision": calibration.get("resolved_tokenizer_revision", calibration.get("tokenizer_revision")),
                  "calibration_parameters": {key: calibration.get(key) for key in ("gamma", "k_fraction", "num_sequences", "sequence_length", "num_partitions", "sampling_strategy", "seed")},
                  "analyzed_thresholds": thresholds,
                  "source_git_commit": calibration.get("git_commit"), "analysis_git_commit": _git_commit(),
                  "python_version": platform.python_version(), "torch_version": torch.__version__}
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    _plots(records, summaries, threshold_rows, output_dir)
    print(f"Analyzed {len(summaries)} modules and {len(records)} module-feature entries into {output_dir}")


if __name__ == "__main__":
    main()
