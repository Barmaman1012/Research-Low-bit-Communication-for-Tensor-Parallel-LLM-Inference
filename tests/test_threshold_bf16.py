from __future__ import annotations

import torch
import pytest
from torch import nn

from lowbit_tp_comm.hooks import build_hybrid_replacements_from_calibration, derive_threshold_bf16_selection
from lowbit_tp_comm.quantization import hybrid_quant_dequant
from lowbit_tp_comm.tp_linear import HybridQuantizedRowParallelLinear


class TinyTargets(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a_proj = nn.Linear(4, 6)
        self.b_proj = nn.Linear(4, 6)


def _payload(ranges: list[float], k: int) -> dict:
    width = len(ranges)
    return {
        "state_dict": {
            "min_vals": -torch.ones(2, width), "max_vals": torch.ones(2, width),
            "initialized": True, "gamma": 0.01, "num_partitions": 2, "feature_dim": width,
        },
        "aggregated_ranges": torch.tensor(ranges, dtype=torch.float32),
        "topk_indices": torch.arange(k), "k": k, "feature_dim": width,
    }


def _calibration() -> dict:
    return {"modules": {
        "a_proj": _payload([1, 1, 1, 2, 3, 4], 2),
        "b_proj": _payload([1, 1, 1, 1, 1, 8], 2),
    }}


def test_threshold_global_target_variable_allocations_masks_and_budget() -> None:
    selection = derive_threshold_bf16_selection(_calibration())
    assert selection["target_bf16_count"] == selection["actual_bf16_count"] == 4
    assert selection["per_module"]["a_proj"]["bf16_count"] == 3
    assert selection["per_module"]["b_proj"]["bf16_count"] == 1
    assert selection["average_bits_per_value"] == 8.0
    for name, details in selection["per_module"].items():
        selected = selection["indices_by_module"][name]
        mask = torch.zeros(details["feature_dim"], dtype=torch.bool)
        mask[selected] = True
        unselected = torch.arange(details["feature_dim"])[~mask]
        assert selected.unique().numel() == selected.numel()
        assert int(mask.sum()) == details["bf16_count"]
        assert not torch.isin(selected, unselected).any()
        assert torch.equal(torch.sort(torch.cat((selected, unselected))).values, torch.arange(details["feature_dim"]))


def test_threshold_fixture_matches_current_artifact_boundary_values() -> None:
    threshold = 1.4867687225341797
    excluded = 1.4865761995315552
    calibration = {"modules": {"proj": _payload([0.5, 0.75, 1.0, excluded, threshold], 1)}}
    selection = derive_threshold_bf16_selection(calibration)
    assert selection["target_bf16_count"] == 1
    assert selection["derived_threshold"] == threshold
    assert selection["next_excluded_score"] == excluded
    assert selection["boundary_tie"] is False


def test_threshold_ties_are_deterministic_by_module_then_feature_index() -> None:
    calibration = {"modules": {
        "z_proj": _payload([1, 2, 2], 1),
        "a_proj": _payload([1, 2, 2], 1),
    }}
    first = derive_threshold_bf16_selection(calibration)
    second = derive_threshold_bf16_selection(calibration)
    assert torch.equal(first["indices_by_module"]["a_proj"], torch.tensor([1, 2]))
    assert first["indices_by_module"]["z_proj"].numel() == 0
    assert first["boundary_tie"] is True
    assert torch.equal(first["indices_by_module"]["a_proj"], second["indices_by_module"]["a_proj"])


def test_threshold_reuses_two_tier_restoration_without_int8() -> None:
    model = TinyTargets()
    replacements = build_hybrid_replacements_from_calibration(model, _calibration(), "threshold_bf16", 2)
    replacement = replacements["a_proj"]
    assert isinstance(replacement, HybridQuantizedRowParallelLinear)
    assert not hasattr(replacement, "int8_feature_indices")
    assert not any("int8" in name for name, _buffer in replacement.named_buffers())
    partial = torch.randn(2, 6)
    actual = hybrid_quant_dequant(partial, torch.ones(6), replacement.bf16_feature_indices, output_dtype=torch.bfloat16)
    expected = partial.to(torch.bfloat16)
    assert torch.equal(actual[..., replacement.bf16_feature_indices], expected[..., replacement.bf16_feature_indices])


def test_threshold_rejects_missing_module_and_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="not present"):
        build_hybrid_replacements_from_calibration(TinyTargets(), {"modules": {"missing": _payload([1, 2], 1)}}, "threshold_bf16", 2)
    calibration = _calibration()
    calibration["modules"]["a_proj"]["feature_dim"] = 5
    with pytest.raises(ValueError, match="shape|dimension"):
        build_hybrid_replacements_from_calibration(TinyTargets(), calibration, "threshold_bf16", 2)


def test_selected_bf16_behavior_is_unchanged() -> None:
    model = TinyTargets()
    replacement = build_hybrid_replacements_from_calibration(model, _calibration(), "selected_bf16", 2)["a_proj"]
    assert torch.equal(replacement.bf16_feature_indices, torch.tensor([0, 1]))
