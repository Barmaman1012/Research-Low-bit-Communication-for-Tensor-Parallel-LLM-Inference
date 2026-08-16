from __future__ import annotations

import torch
import pytest
from torch import nn

from lowbit_tp_comm.hooks import build_hybrid_replacements_from_calibration, derive_range_threshold_bf16_selection, derive_threshold_bf16_selection, range_threshold_bf16_result_metadata
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


def test_explicit_high_threshold_is_inclusive_and_variable() -> None:
    selection = derive_range_threshold_bf16_selection(_calibration(), threshold=2.0, mode="range_threshold_bf16")
    assert selection["bf16_feature_count"] == 4
    assert selection["indices_by_module"]["a_proj"].tolist() == [3, 4, 5]
    assert selection["indices_by_module"]["b_proj"].tolist() == [5]
    assert selection["per_module"]["a_proj"]["bf16_count"] != selection["per_module"]["b_proj"]["bf16_count"]


def test_threshold_one_selects_at_or_above_module_median() -> None:
    selection = derive_range_threshold_bf16_selection(_calibration(), threshold=1.0, mode="range_threshold_bf16")
    assert selection["bf16_feature_count"] == 12
    assert selection["average_bits_per_value"] == 16.0


@pytest.mark.parametrize("threshold, expected", [(1.2, 4), (1.3, 4), (2.0, 4), (4.0, 2)])
def test_explicit_threshold_counts_and_bits(threshold: float, expected: int) -> None:
    selection = derive_range_threshold_bf16_selection(_calibration(), threshold=threshold, mode="range_threshold_bf16")
    assert selection["bf16_feature_count"] == expected
    assert selection["int4_feature_count"] + expected == selection["total_feature_count"]
    assert selection["average_bits_per_value"] == 16 * selection["global_bf16_fraction"] + 4 * selection["global_int4_fraction"]


def test_matched_low_range_has_equal_budget_and_global_lowest_ties() -> None:
    high = derive_range_threshold_bf16_selection(_calibration(), threshold=2.0, mode="range_threshold_bf16")
    low = derive_range_threshold_bf16_selection(_calibration(), threshold=2.0, mode="matched_low_range_bf16")
    assert low["bf16_feature_count"] == low["matched_high_range_count"] == high["bf16_feature_count"]
    assert low["average_bits_per_value"] == high["average_bits_per_value"]
    assert low["indices_by_module"]["a_proj"].tolist() == [0, 1, 2]
    assert low["indices_by_module"]["b_proj"].tolist() == [0]


def test_explicit_modes_reuse_two_tier_and_validate_threshold() -> None:
    model = TinyTargets()
    replacement = build_hybrid_replacements_from_calibration(model, _calibration(), "range_threshold_bf16", 2, bf16_range_threshold=2.0)["a_proj"]
    assert not hasattr(replacement, "int8_feature_indices")
    with pytest.raises(ValueError, match="finite and positive"):
        derive_range_threshold_bf16_selection(_calibration(), threshold=float("nan"), mode="range_threshold_bf16")
    with pytest.raises(ValueError, match="finite and positive"):
        derive_range_threshold_bf16_selection(_calibration(), threshold=0.0, mode="matched_low_range_bf16")


def test_explicit_metadata_invariants() -> None:
    selection = derive_range_threshold_bf16_selection(_calibration(), threshold=2.0, mode="matched_low_range_bf16")
    metadata = range_threshold_bf16_result_metadata(selection, calibration_path="fixture.pt", calibration_sha256="abc")
    assert sum(metadata["per_module_bf16_counts"].values()) == metadata["bf16_feature_count"]
    assert metadata["bf16_feature_count"] == metadata["matched_high_range_count"]
    assert metadata["global_bf16_fraction"] == metadata["bf16_feature_count"] / metadata["total_feature_count"]


def test_random_bf16_uses_shared_reproducible_generator_per_model() -> None:
    first = build_hybrid_replacements_from_calibration(TinyTargets(), _calibration(), "random_bf16", 2, seed=17)
    second = build_hybrid_replacements_from_calibration(TinyTargets(), _calibration(), "random_bf16", 2, seed=17)
    changed = build_hybrid_replacements_from_calibration(TinyTargets(), _calibration(), "random_bf16", 2, seed=18)
    mapping = {name: replacement.bf16_feature_indices for name, replacement in first.items()}
    assert not torch.equal(mapping["a_proj"], mapping["b_proj"])
    for name, indices in mapping.items():
        assert torch.equal(indices, second[name].bf16_feature_indices)
        assert indices.numel() == 2 and indices.unique().numel() == 2
        assert int(indices.min()) >= 0 and int(indices.max()) < 6
    assert any(not torch.equal(first[name].bf16_feature_indices, changed[name].bf16_feature_indices) for name in first)
    assert first["a_proj"].random_bf16_metadata["selection_strategy"] == "shared_stateful_generator_per_model"


def test_deprecated_threshold_alias_matches_canonical_global_equal_budget() -> None:
    with pytest.warns(DeprecationWarning):
        alias = build_hybrid_replacements_from_calibration(TinyTargets(), _calibration(), "threshold_bf16", 2)
    canonical = build_hybrid_replacements_from_calibration(TinyTargets(), _calibration(), "global_equal_budget_bf16", 2)
    for name in alias:
        assert torch.equal(alias[name].bf16_feature_indices, canonical[name].bf16_feature_indices)
