from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a saved calibration artifact.")
    parser.add_argument("--calibration_path", default="calibration.pt")
    parser.add_argument("--topn", type=int, default=10)
    return parser.parse_args()


def format_calibration_summary(calibration: dict[str, Any], topn: int) -> str:
    lines = [
        f"model_name={calibration['model_name']}",
        f"gamma={calibration['gamma']}",
        f"k_fraction={calibration['k_fraction']}",
        f"num_sequences={calibration['num_sequences']}",
        f"sequence_length={calibration['sequence_length']}",
        f"num_modules={len(calibration['modules'])}",
    ]

    for module_name, module_payload in calibration["modules"].items():
        aggregated_ranges = module_payload["aggregated_ranges"].detach().cpu().float()
        sorted_values = torch.sort(aggregated_ranges, descending=True).values
        top_values = sorted_values[: min(topn, sorted_values.numel())].tolist()
        median_range = statistics.median(aggregated_ranges.tolist()) if aggregated_ranges.numel() > 0 else 0.0
        largest_range = float(sorted_values[0].item()) if aggregated_ranges.numel() > 0 else 0.0
        ratio = float("inf") if median_range == 0.0 and largest_range > 0.0 else (
            1.0 if median_range == 0.0 else largest_range / median_range
        )
        feature_dim = int(module_payload["feature_dim"])
        k = int(module_payload["k"])
        selected_fraction = (k / feature_dim) if feature_dim > 0 else 0.0

        lines.extend(
            [
                f"module={module_name}",
                f"  feature_dim={feature_dim}",
                f"  k={k}",
                f"  selected_fraction={selected_fraction:.6f}",
                f"  top_selected_indices={module_payload['topk_indices'].tolist()}",
                f"  top{min(topn, len(top_values))}_aggregated_ranges={top_values}",
                f"  largest_to_median_range_ratio={ratio}",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    calibration = torch.load(args.calibration_path, map_location="cpu", weights_only=False)
    print(format_calibration_summary(calibration, topn=args.topn))


if __name__ == "__main__":
    main()
