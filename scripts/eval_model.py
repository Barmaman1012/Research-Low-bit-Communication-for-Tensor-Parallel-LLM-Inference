from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lowbit_tp_comm.config import EvaluationConfig
from lowbit_tp_comm.evaluation import LLMEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Placeholder evaluation entry point.")
    parser.add_argument("--model-name-or-path", default="gpt2")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = EvaluationConfig(
        model_name_or_path=args.model_name_or_path,
        batch_size=args.batch_size,
        device=args.device,
    )
    evaluator = LLMEvaluator(config)
    result = evaluator.evaluate()
    print("evaluation scaffold")
    print(f"model_name_or_path={config.model_name_or_path}")
    print(f"batch_size={config.batch_size}")
    print(f"status={result.metadata['status']}")


if __name__ == "__main__":
    main()
