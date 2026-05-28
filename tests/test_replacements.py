from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lowbit_tp_comm.hooks import build_hybrid_replacements_from_calibration, replace_modules_by_name
from lowbit_tp_comm.tp_linear import (
    HybridQuantizedRowParallelConv1D,
    HybridQuantizedRowParallelLinear,
)

try:
    from transformers.pytorch_utils import Conv1D
except ImportError:  # pragma: no cover
    Conv1D = None


class NestedTinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Module()
        self.block.inner = nn.Module()
        self.block.inner.down_proj = nn.Linear(4, 6)


def test_replace_modules_by_nested_name() -> None:
    model = NestedTinyModel()
    replacement = nn.Identity()

    replace_modules_by_name(model, {"block.inner.down_proj": replacement})

    assert model.block.inner.down_proj is replacement


def test_hybrid_row_parallel_linear_output_shape() -> None:
    linear = nn.Linear(8, 6)
    scales = torch.ones(2, 6, dtype=linear.weight.dtype)
    module = HybridQuantizedRowParallelLinear.from_linear(
        linear,
        num_partitions=2,
        scales_per_partition=scales,
        bf16_feature_indices=None,
    )

    x = torch.randn(3, 5, 8)
    y = module(x)

    assert y.shape == (3, 5, 6)


def test_all_selected_bf16_matches_original_linear() -> None:
    linear = nn.Linear(8, 6)
    scales = torch.ones(2, 6, dtype=linear.weight.dtype)
    selected = torch.arange(6, dtype=torch.long)
    module = HybridQuantizedRowParallelLinear.from_linear(
        linear,
        num_partitions=2,
        scales_per_partition=scales,
        bf16_feature_indices=selected,
        output_dtype=linear.weight.dtype,
    )
    x = torch.randn(4, 8)

    expected = linear(x)
    actual = module(x)

    assert torch.allclose(actual, expected)


@pytest.mark.skipif(Conv1D is None, reason="transformers Conv1D is not importable")
def test_conv1d_replacement_matches_shape() -> None:
    conv = Conv1D(6, 8)
    scales = torch.ones(2, 6, dtype=conv.weight.dtype)
    selected = torch.arange(6, dtype=torch.long)
    module = HybridQuantizedRowParallelConv1D.from_conv1d(
        conv,
        num_partitions=2,
        scales_per_partition=scales,
        bf16_feature_indices=selected,
        output_dtype=conv.weight.dtype,
    )
    x = torch.randn(2, 4, 8)

    expected = conv(x)
    actual = module(x)

    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected)


def test_build_hybrid_replacements_from_fake_calibration() -> None:
    model = NestedTinyModel()
    calibration = {
        "modules": {
            "block.inner.down_proj": {
                "state_dict": {
                    "min_vals": torch.tensor([[-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]], dtype=torch.float32),
                    "max_vals": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=torch.float32),
                    "initialized": True,
                    "gamma": 0.01,
                    "num_partitions": 1,
                    "feature_dim": 6,
                },
                "aggregated_ranges": torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0, 12.0], dtype=torch.float32),
                "topk_indices": torch.tensor([5], dtype=torch.long),
                "k": 1,
                "feature_dim": 6,
            }
        }
    }

    replacements = build_hybrid_replacements_from_calibration(
        model,
        calibration=calibration,
        mode="selected_bf16",
        num_partitions=2,
        seed=0,
    )

    assert "block.inner.down_proj" in replacements
    replacement = replacements["block.inner.down_proj"]
    assert isinstance(replacement, HybridQuantizedRowParallelLinear)
    assert replacement.scales_per_partition.shape == (2, 6)
    assert torch.equal(replacement.bf16_feature_indices, torch.tensor([5], dtype=torch.long))
