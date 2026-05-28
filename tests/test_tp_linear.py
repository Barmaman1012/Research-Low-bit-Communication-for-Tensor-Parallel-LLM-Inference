from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lowbit_tp_comm.quantization import get_symmetric_scale
from lowbit_tp_comm.tp_linear import TensorParallelLinearSimulator, compute_partition_minmax


def _make_simulator() -> tuple[TensorParallelLinearSimulator, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    weight = torch.randn(8, 6, generator=generator, dtype=torch.float32)
    bias = torch.randn(6, generator=generator, dtype=torch.float32)
    simulator = TensorParallelLinearSimulator(weight=weight, bias=bias, num_partitions=2)
    x = torch.randn(3, 5, 8, generator=generator, dtype=torch.float32)
    return simulator, x


def test_full_and_uncompressed_tp_outputs_match() -> None:
    simulator, x = _make_simulator()

    full = simulator.forward_full(x)
    tp = simulator.forward_tp_uncompressed(x)

    assert torch.allclose(full, tp)


def test_hybrid_quantized_output_has_same_shape() -> None:
    simulator, x = _make_simulator()
    partials = simulator.compute_partials(x)
    stats = compute_partition_minmax(partials)
    scales_per_partition = [
        get_symmetric_scale(partition_stats.min_vals, partition_stats.max_vals)
        for partition_stats in stats
    ]

    hybrid = simulator.forward_tp_hybrid_quantized(
        x,
        scales_per_partition=scales_per_partition,
        bf16_feature_indices=[],
    )

    assert hybrid.shape == simulator.forward_full(x).shape


def test_all_selected_bf16_features_match_full_tp() -> None:
    simulator, x = _make_simulator()
    partials = simulator.compute_partials(x)
    stats = compute_partition_minmax(partials)
    scales_per_partition = [
        get_symmetric_scale(partition_stats.min_vals, partition_stats.max_vals)
        for partition_stats in stats
    ]
    all_features = torch.arange(simulator.out_features, dtype=torch.long)

    hybrid = simulator.forward_tp_hybrid_quantized(
        x,
        scales_per_partition=scales_per_partition,
        bf16_feature_indices=all_features,
        output_dtype=torch.float32,
    )
    tp = simulator.forward_tp_uncompressed(x)

    assert torch.allclose(hybrid, tp)


def test_no_selected_features_differs_slightly_but_stays_finite() -> None:
    simulator, x = _make_simulator()
    partials = simulator.compute_partials(x)
    stats = compute_partition_minmax(partials)
    scales_per_partition = [
        get_symmetric_scale(partition_stats.min_vals, partition_stats.max_vals)
        for partition_stats in stats
    ]

    hybrid = simulator.forward_tp_hybrid_quantized(
        x,
        scales_per_partition=scales_per_partition,
        bf16_feature_indices=[],
        output_dtype=torch.float32,
    )
    tp = simulator.forward_tp_uncompressed(x)

    assert hybrid.shape == tp.shape
    assert torch.isfinite(hybrid).all()
    assert not torch.equal(hybrid, tp)
