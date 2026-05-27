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


def test_simulated_tp_linear_produces_expected_shape() -> None:
    layer = SimulatedTPLinear(
        in_features=8,
        out_features=12,
        tp_config=TensorParallelConfig(tp_degree=2),
    )
    inputs = torch.randn(3, 8)

    outputs = layer(inputs)

    assert outputs.shape == (3, 12)
    assert layer.shard_spec.tp_degree == 2
