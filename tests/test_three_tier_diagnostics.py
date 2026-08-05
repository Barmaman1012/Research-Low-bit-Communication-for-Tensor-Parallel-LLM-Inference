from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def _diagnostics():
    spec = importlib.util.spec_from_file_location("diagnose_quantization", ROOT / "scripts" / "diagnose_quantization.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_three_tier_diagnostic_masks_and_empty_int8_are_payload_specific() -> None:
    module = _diagnostics()
    original = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    reconstructed = torch.tensor([[1.0, 2.1, 3.5, 4.5]])
    q4 = torch.tensor([[-8, 7, -8, 7]], dtype=torch.int8)
    q8 = torch.tensor([[-128, 127, -128, 127]], dtype=torch.int8)
    stats = module.three_tier_diagnostic_stats(original, reconstructed, q4, q8, torch.tensor([0]), torch.tensor([1]))

    assert stats["bf16_mean_abs_error"] == 0.0
    assert stats["int8_mean_abs_error"] > 0
    assert stats["int4_mean_abs_error"] > stats["int8_mean_abs_error"]
    assert stats["int4_payload_saturation_low_rate"] == 0.5
    assert stats["int4_payload_saturation_high_rate"] == 0.5
    assert stats["int8_payload_saturation_low_rate"] == 0.0
    assert stats["int8_payload_saturation_high_rate"] == 1.0
    empty = module.three_tier_diagnostic_stats(original, reconstructed, q4, q8, torch.tensor([0]), torch.empty(0, dtype=torch.long))
    assert empty["int8_fraction"] == 0.0
    assert empty["int8_payload_saturation_low_rate"] == 0.0
    assert empty["int8_payload_saturation_high_rate"] == 0.0
