from __future__ import annotations

import importlib.util
import csv
import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location("analyze_calibration_ranges", ROOT / "scripts" / "analyze_calibration_ranges.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _payload(values: list[float], selected: list[int] = [2]) -> dict:
    return {"aggregated_ranges": torch.tensor(values), "topk_indices": torch.tensor(selected),
            "k": len(selected), "feature_dim": len(values)}


def test_extracts_normalizes_ranks_and_marks_existing_selection() -> None:
    module = _module()
    calibration = {"modules": {
        "model.layers.2.self_attn.o_proj": _payload([1, 2, 4, 8]),
        "model.layers.2.mlp.down_proj": _payload([2, 2, 2, 6]),
    }}
    records, summaries = module.extract_module_records(calibration)
    assert len(records) == 8
    attention = [row for row in records if row["module_type"] == "self_attn.o_proj"]
    assert attention[2]["normalized_by_median"] == 2.0
    assert attention[3]["within_module_rank"] == 1
    assert attention[2]["existing_selected_bf16"] is True
    summary = next(row for row in summaries if row["module_type"] == "self_attn.o_proj")
    assert summary["minimum"] == 1.0
    assert summary["median"] == 2.0
    assert summary["count_ge_2x_median"] == 2


def test_threshold_counts_and_average_bit_calculation() -> None:
    module = _module()
    calibration = {"modules": {
        "model.layers.0.self_attn.o_proj": _payload([1, 2, 4, 8]),
        "model.layers.0.mlp.down_proj": _payload([1, 1, 1, 4]),
    }}
    records, summaries = module.extract_module_records(calibration)
    row = module.threshold_summary(records, summaries, thresholds=[2.0])[0]
    assert row["selected_count"] == 3
    assert row["bf16_fraction"] == 3 / 8
    assert row["int4_fraction"] == 5 / 8
    assert row["average_bits_per_value"] == 16 * (3 / 8) + 4 * (5 / 8)
    assert row["minimum_module_count"] == 1
    assert row["maximum_module_count"] == 2


def test_explicit_thresholds_drive_csv_inclusively_and_do_not_modify_calibration(tmp_path, monkeypatch) -> None:
    module = _module()
    calibration_path = tmp_path / "calibration.pt"
    calibration = {"modules": {
        "model.layers.0.self_attn.o_proj": _payload([1, 2, 4, 8]),
        "model.layers.0.mlp.down_proj": _payload([1, 1, 1, 4]),
    }}
    torch.save(calibration, calibration_path)
    before = hashlib.sha256(calibration_path.read_bytes()).hexdigest()
    output_dir = tmp_path / "analysis"
    monkeypatch.setattr(module, "_plots", lambda *_args: None)
    monkeypatch.setattr(sys, "argv", ["analyze_calibration_ranges.py", "--calibration_path", str(calibration_path),
                                       "--output_dir", str(output_dir), "--thresholds", "1.0,1.6,2.1,2.3"])
    module.main()
    with (output_dir / "threshold_summary.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [float(row["threshold"]) for row in rows] == [1.0, 1.6, 2.1, 2.3]
    # [1, 2, 4, 8] has median 2, so 2 is included by >= 1.0.
    assert int(rows[0]["selected_count"]) == 7
    provenance = json.loads((output_dir / "provenance.json").read_text())
    assert provenance["analyzed_thresholds"] == [1.0, 1.6, 2.1, 2.3]
    assert hashlib.sha256(calibration_path.read_bytes()).hexdigest() == before


@pytest.mark.parametrize("raw", ["", "1.0,,2.0", "zero", "0", "-1", "nan", "inf", "1.0,1.0"])
def test_rejects_malformed_nonpositive_nonfinite_or_duplicate_thresholds(raw: str) -> None:
    module = _module()
    with pytest.raises(ValueError):
        module.parse_thresholds(raw)


def test_legacy_default_threshold_grid_is_unchanged() -> None:
    module = _module()
    assert module.parse_thresholds(None) == module.THRESHOLDS
    calibration = {"modules": {"model.layers.0.mlp.down_proj": _payload([1, 2, 4, 8])}}
    _records, summaries = module.extract_module_records(calibration)
    assert "count_ge_1_05x_median" in summaries[0]


def test_module_name_parsing() -> None:
    module = _module()
    assert module.parse_module_name("model.layers.39.mlp.down_proj") == (39, "mlp.down_proj")
    assert module.parse_module_name("model.layers.3.self_attn.o_proj") == (3, "self_attn.o_proj")
    with pytest.raises(ValueError, match="Unsupported"):
        module.parse_module_name("transformer.h.0.attn.c_proj")


@pytest.mark.parametrize("values, error", [([0, 0], "positive"), ([1, float("nan")], "non-finite"), ([1, float("inf")], "non-finite")])
def test_rejects_zero_or_nonfinite_ranges(values: list[float], error: str) -> None:
    module = _module()
    calibration = {"modules": {"model.layers.0.mlp.down_proj": _payload(values)}}
    with pytest.raises(ValueError, match=error):
        module.extract_module_records(calibration)
