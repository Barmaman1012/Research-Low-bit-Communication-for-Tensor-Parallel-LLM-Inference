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


def format_tensor(values: torch.Tensor, precision: int = 4) -> str:
    rounded = [round(float(value), precision) for value in values.tolist()]
    return str(rounded)


def max_abs_error(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    return float((reference - candidate).abs().max().item())


def main() -> None:
    torch.manual_seed(0)

    d = 8
    e = 6
    num_partitions = 2
    seq_len = 4
    num_calibration_samples = 8
    top_k_bf16 = 1

    x = torch.randn(seq_len, d)
    weight = torch.randn(d, e)
    bias = torch.randn(e)

    # Make one output feature an artificial outlier so its synchronized partials
    # have much larger range than the other features.
    outlier_feature_index = 2
    weight[:, outlier_feature_index] *= 8.0

    simulator = TensorParallelLinearSimulator(
        weight=weight,
        bias=bias,
        num_partitions=num_partitions,
    )

    print("Toy demo: low-bit communication for tensor-parallel linear layers")
    print("This is a single-process simulation of TP synchronization compression.")
    print()
    print(f"Setup: D={d}, E={e}, S={seq_len}, num_partitions={num_partitions}")
    print(f"Artificial outlier feature index: {outlier_feature_index}")
    print()

    full_output = simulator.forward_full(x)
    tp_uncompressed_output = simulator.forward_tp_uncompressed(x)

    print("Step 1-2: full linear vs uncompressed tensor-parallel simulation")
    print("Each partition computes a partial output, then the partials are summed.")
    print()

    partition_min_vals: list[torch.Tensor] = []
    partition_max_vals: list[torch.Tensor] = []

    print("Step 3-4: calibration over random input samples")
    print("We collect per-partition, per-feature min/max statistics from partial outputs.")
    for sample_idx in range(num_calibration_samples):
        calibration_x = torch.randn(seq_len, d)
        calibration_partials = simulator.compute_partials(calibration_x)
        partition_stats = compute_partition_minmax(calibration_partials)

        if sample_idx == 0:
            partition_min_vals = [stats.min_vals.clone() for stats in partition_stats]
            partition_max_vals = [stats.max_vals.clone() for stats in partition_stats]
        else:
            partition_min_vals = [
                torch.minimum(running_min, stats.min_vals)
                for running_min, stats in zip(partition_min_vals, partition_stats, strict=True)
            ]
            partition_max_vals = [
                torch.maximum(running_max, stats.max_vals)
                for running_max, stats in zip(partition_max_vals, partition_stats, strict=True)
            ]

    aggregated_ranges = torch.zeros(e, dtype=weight.dtype)
    scales_per_partition: list[torch.Tensor] = []

    print()
    print("Step 5: aggregate symmetric ranges across partitions")
    print("R_j_bar = sum_i 2 * max(abs(min_ij), abs(max_ij))")
    for partition_idx, (min_vals, max_vals) in enumerate(
        zip(partition_min_vals, partition_max_vals, strict=True)
    ):
        symmetric_abs = torch.maximum(min_vals.abs(), max_vals.abs())
        aggregated_ranges += 2.0 * symmetric_abs
        scales_per_partition.append(get_symmetric_scale(min_vals, max_vals))
        print(
            f"Partition {partition_idx}: symmetric_abs per feature = "
            f"{format_tensor(symmetric_abs)}"
        )

    selected_feature_indices = torch.topk(aggregated_ranges, k=top_k_bf16).indices

    print()
    print("Step 6: select the top-k highest-range features to keep in BF16")
    print(f"Aggregated ranges: {format_tensor(aggregated_ranges)}")
    print(f"Selected BF16 feature indices (k={top_k_bf16}): {selected_feature_indices.tolist()}")

    hybrid_output = simulator.forward_tp_hybrid_quantized(
        x,
        scales_per_partition=scales_per_partition,
        bf16_feature_indices=selected_feature_indices,
        output_dtype=torch.float32,
    )
    pure_int4_output = simulator.forward_tp_hybrid_quantized(
        x,
        scales_per_partition=scales_per_partition,
        bf16_feature_indices=[],
        output_dtype=torch.float32,
    )

    uncompressed_error = max_abs_error(full_output, tp_uncompressed_output)
    pure_int4_error = max_abs_error(full_output, pure_int4_output)
    hybrid_error = max_abs_error(full_output, hybrid_output)

    print()
    print("Step 7-8: compare pure Int4 communication against hybrid Int4 + selected BF16")
    print("This demo simulates Int4 numerically. It does not bit-pack values yet.")
    print()
    print(f"Max absolute error, uncompressed TP vs full: {uncompressed_error:.8f}")
    print(f"Max absolute error, pure Int4 vs full:     {pure_int4_error:.8f}")
    print(f"Max absolute error, hybrid vs full:        {hybrid_error:.8f}")

    print()
    print("Expected interpretation:")
    print("- Uncompressed TP should be nearly identical to the full linear output.")
    print("- Pure Int4 introduces reconstruction error in synchronized partials.")
    print("- Keeping the outlier feature in BF16 should usually reduce that error.")


if __name__ == "__main__":
    main()
