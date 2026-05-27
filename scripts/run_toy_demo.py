from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lowbit_tp_comm.config import TensorParallelConfig
from lowbit_tp_comm.tp_linear import SimulatedTPLinear


def main() -> None:
    tp_config = TensorParallelConfig(tp_degree=2, shard_dimension="column")
    layer = SimulatedTPLinear(
        in_features=8,
        out_features=16,
        tp_config=tp_config,
        layer_name="toy_mlp_up_proj",
    )
    inputs = torch.randn(2, 8)
    outputs = layer(inputs)

    print("lowbit_tp_comm toy demo")
    print(f"input_shape={tuple(inputs.shape)}")
    print(f"output_shape={tuple(outputs.shape)}")
    print(f"tp_degree={layer.tp_config.tp_degree}")
    print("status=skeleton_only")


if __name__ == "__main__":
    main()
