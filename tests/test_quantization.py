from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lowbit_tp_comm.quantization import (
    dequantize_symmetric,
    get_symmetric_scale,
    hybrid_quant_dequant,
    quantization_error_stats,
    quantize_symmetric,
)


def test_quantize_symmetric_outputs_signed_int4_range() -> None:
    x = torch.tensor(
        [[-10.0, -1.0, 0.0, 1.0, 10.0], [4.1, -4.1, 0.5, -0.5, 2.0]],
        dtype=torch.float32,
    )
    min_vals = x.amin(dim=0)
    max_vals = x.amax(dim=0)

    scale = get_symmetric_scale(min_vals, max_vals)
    q = quantize_symmetric(x, scale)

    assert q.dtype == torch.int8
    assert int(q.min().item()) >= -8
    assert int(q.max().item()) <= 7


def test_dequantize_symmetric_preserves_shape() -> None:
    x = torch.randn(2, 3, 8, dtype=torch.float32)
    min_vals = x.amin(dim=(0, 1))
    max_vals = x.amax(dim=(0, 1))

    scale = get_symmetric_scale(min_vals, max_vals)
    q = quantize_symmetric(x, scale)
    restored = dequantize_symmetric(q, scale, dtype=torch.float32)

    assert restored.shape == x.shape


def test_hybrid_quant_dequant_preserves_selected_features() -> None:
    x = torch.tensor(
        [
            [[-1.3, 0.25, 2.9, -0.75], [0.8, -1.0, 1.7, 3.2]],
            [[1.2, -0.4, -2.1, 0.9], [-0.6, 0.5, 0.1, -3.0]],
        ],
        dtype=torch.float32,
    )
    min_vals = x.amin(dim=(0, 1))
    max_vals = x.amax(dim=(0, 1))
    selected = torch.tensor([1, 3], dtype=torch.long)

    scale = get_symmetric_scale(min_vals, max_vals)
    reconstructed = hybrid_quant_dequant(
        x,
        scale,
        bf16_feature_indices=selected,
        output_dtype=torch.bfloat16,
    )

    assert reconstructed.dtype == torch.bfloat16
    assert torch.equal(reconstructed[..., selected], x[..., selected].to(torch.bfloat16))


def test_hybrid_quant_dequant_non_selected_features_are_approximate() -> None:
    x = torch.tensor(
        [[-1.3, 0.25, 2.9, -0.75], [0.8, -1.0, 1.7, 3.2], [1.2, -0.4, -2.1, 0.9]],
        dtype=torch.float32,
    )
    min_vals = x.amin(dim=0)
    max_vals = x.amax(dim=0)
    selected = [1]
    non_selected = torch.tensor([0, 2, 3], dtype=torch.long)

    scale = get_symmetric_scale(min_vals, max_vals)
    reconstructed = hybrid_quant_dequant(
        x,
        scale,
        bf16_feature_indices=selected,
        output_dtype=torch.float32,
    )

    assert torch.allclose(reconstructed[..., non_selected], x[..., non_selected], atol=scale.max().item())
    assert not torch.equal(reconstructed[..., non_selected], x[..., non_selected])


def test_quantization_error_stats_zero_error_when_tensors_match() -> None:
    x = torch.randn(2, 3, 4)
    stats = quantization_error_stats(x, x)

    assert stats["mean_abs_error"] == 0.0
    assert stats["max_abs_error"] == 0.0
    assert stats["rmse"] == 0.0


def test_quantization_error_stats_saturation_rates_are_computed() -> None:
    original = torch.zeros(2, 2)
    reconstructed = torch.zeros(2, 2)
    q = torch.tensor([[-8, -8], [7, 0]], dtype=torch.int8)

    stats = quantization_error_stats(original, reconstructed, q=q, qmin=-8, qmax=7)

    assert stats["saturation_low_rate"] == 0.5
    assert stats["saturation_high_rate"] == 0.25


def test_quantization_error_stats_selected_split_is_reported() -> None:
    original = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
    reconstructed = torch.tensor([[1.0, 3.0, 3.0, 6.0]], dtype=torch.float32)

    stats = quantization_error_stats(
        original,
        reconstructed,
        selected_indices=torch.tensor([0, 2], dtype=torch.long),
    )

    assert stats["selected_fraction"] == 0.5
    assert stats["selected_mean_abs_error"] == 0.0
    assert stats["non_selected_mean_abs_error"] > 0.0
