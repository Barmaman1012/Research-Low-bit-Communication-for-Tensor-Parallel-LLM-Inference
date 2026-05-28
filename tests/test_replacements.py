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
    RowParallelConv1D,
    RowParallelLinear,
    compute_row_parallel_partials_for_module,
)

try:
    from transformers.pytorch_utils import Conv1D
except ImportError:  # pragma: no cover
    Conv1D = None

try:
    from transformers import GPT2Config, GPT2LMHeadModel
except ImportError:  # pragma: no cover
    GPT2Config = None
    GPT2LMHeadModel = None


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


def test_tp_uncompressed_linear_matches_original_linear() -> None:
    linear = nn.Linear(8, 6)
    module = RowParallelLinear.from_linear(linear, num_partitions=2)
    x = torch.randn(4, 8)

    expected = linear(x)
    actual = module(x)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


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

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


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


@pytest.mark.skipif(Conv1D is None, reason="transformers Conv1D is not importable")
def test_tp_uncompressed_conv1d_matches_original_conv1d() -> None:
    conv = Conv1D(6, 8)
    module = RowParallelConv1D.from_conv1d(conv, num_partitions=2)
    x = torch.randn(2, 4, 8)

    expected = conv(x)
    actual = module(x)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


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


def test_build_hybrid_replacements_uses_matching_partition_scales_without_repeat() -> None:
    model = NestedTinyModel()
    calibration = {
        "modules": {
            "block.inner.down_proj": {
                "state_dict": {
                    "min_vals": torch.tensor(
                        [[-1.0, -2.0, -3.0, -4.0, -5.0, -6.0], [-0.5, -1.5, -2.5, -3.5, -4.5, -5.5]],
                        dtype=torch.float32,
                    ),
                    "max_vals": torch.tensor(
                        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]],
                        dtype=torch.float32,
                    ),
                    "initialized": True,
                    "gamma": 0.01,
                    "num_partitions": 2,
                    "feature_dim": 6,
                },
                "aggregated_ranges": torch.tensor([3.0, 7.0, 11.0, 15.0, 19.0, 23.0], dtype=torch.float32),
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
    replacement = replacements["block.inner.down_proj"]
    expected_scales = torch.tensor(
        [
            [1.0 / 7.0, 2.0 / 7.0, 3.0 / 7.0, 4.0 / 7.0, 5.0 / 7.0, 6.0 / 7.0],
            [0.5 / 7.0, 1.5 / 7.0, 2.5 / 7.0, 3.5 / 7.0, 4.5 / 7.0, 5.5 / 7.0],
        ],
        dtype=replacement.scales_per_partition.dtype,
    )

    assert torch.allclose(replacement.scales_per_partition, expected_scales)


@pytest.mark.skipif(GPT2LMHeadModel is None or GPT2Config is None, reason="transformers GPT2 model is not importable")
def test_tiny_gpt2_tp_uncompressed_and_all_bf16_logits_match_original() -> None:
    config = GPT2Config(
        vocab_size=64,
        n_positions=32,
        n_ctx=32,
        n_embd=16,
        n_layer=2,
        n_head=2,
    )
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
    calibration = {
        "modules": {
            f"transformer.h.{layer_idx}.attn.c_proj": {
                "state_dict": {
                    "min_vals": torch.full((1, 16), -1.0),
                    "max_vals": torch.full((1, 16), 1.0),
                    "initialized": True,
                    "gamma": 0.01,
                    "num_partitions": 1,
                    "feature_dim": 16,
                },
                "aggregated_ranges": torch.ones(16),
                "topk_indices": torch.arange(2, dtype=torch.long),
                "k": 2,
                "feature_dim": 16,
            }
            for layer_idx in range(2)
        }
    }
    calibration["modules"].update(
        {
            f"transformer.h.{layer_idx}.mlp.c_proj": {
                "state_dict": {
                    "min_vals": torch.full((1, 16), -1.0),
                    "max_vals": torch.full((1, 16), 1.0),
                    "initialized": True,
                    "gamma": 0.01,
                    "num_partitions": 1,
                    "feature_dim": 16,
                },
                "aggregated_ranges": torch.ones(16),
                "topk_indices": torch.arange(2, dtype=torch.long),
                "k": 2,
                "feature_dim": 16,
            }
            for layer_idx in range(2)
        }
    )

    reference_model = GPT2LMHeadModel(config).eval()
    state_dict = reference_model.state_dict()
    tp_model = GPT2LMHeadModel(config).eval()
    tp_model.load_state_dict(state_dict)
    all_bf16_model = GPT2LMHeadModel(config).eval()
    all_bf16_model.load_state_dict(state_dict)

    tp_replacements = build_hybrid_replacements_from_calibration(
        tp_model,
        calibration=calibration,
        mode="tp_uncompressed",
        num_partitions=2,
    )
    replace_modules_by_name(tp_model, tp_replacements)

    all_bf16_replacements = build_hybrid_replacements_from_calibration(
        all_bf16_model,
        calibration=calibration,
        mode="all_bf16",
        num_partitions=2,
    )
    replace_modules_by_name(all_bf16_model, all_bf16_replacements)

    with torch.no_grad():
        reference_logits = reference_model(input_ids=input_ids).logits
        tp_logits = tp_model(input_ids=input_ids).logits
        all_bf16_logits = all_bf16_model(input_ids=input_ids).logits

    assert torch.allclose(tp_logits, reference_logits, atol=1e-5, rtol=1e-5)
    assert torch.allclose(all_bf16_logits, reference_logits, atol=1e-5, rtol=1e-5)


def test_compute_row_parallel_partials_for_linear_sum_to_original_without_bias() -> None:
    linear = nn.Linear(8, 6)
    x = torch.randn(3, 5, 8)

    partials = compute_row_parallel_partials_for_module(linear, x, num_partitions=2)
    reconstructed = torch.stack(partials, dim=0).sum(dim=0)
    expected = torch.matmul(x, linear.weight.transpose(0, 1))

    assert len(partials) == 2
    assert torch.allclose(reconstructed, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(Conv1D is None, reason="transformers Conv1D is not importable")
def test_compute_row_parallel_partials_for_conv1d_sum_to_original_without_bias() -> None:
    conv = Conv1D(6, 8)
    x = torch.randn(2, 4, 8)

    partials = compute_row_parallel_partials_for_module(conv, x, num_partitions=2)
    reconstructed = torch.stack(partials, dim=0).sum(dim=0)
    expected = torch.matmul(x, conv.weight)

    assert len(partials) == 2
    assert torch.allclose(reconstructed, expected, atol=1e-6, rtol=1e-6)
