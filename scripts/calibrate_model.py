from __future__ import annotations

import argparse
import sys
from pathlib import Path

from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lowbit_tp_comm.calibration import CalibrationRunner
from lowbit_tp_comm.config import CalibrationConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Placeholder calibration entry point.")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CalibrationConfig(
        enabled=True,
        dataset_name=args.dataset_name,
        num_samples=args.num_samples,
        sequence_length=args.sequence_length,
    )
    runner = CalibrationRunner(config)
    result = runner.calibrate_module(module=nn.Identity())
    print("calibration scaffold")
    print(f"dataset_name={config.dataset_name}")
    print(f"num_samples={config.num_samples}")
    print(f"status={result.metadata['status']}")


if __name__ == "__main__":
    main()
