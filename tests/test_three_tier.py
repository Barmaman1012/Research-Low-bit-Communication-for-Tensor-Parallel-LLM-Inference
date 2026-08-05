from __future__ import annotations

import torch
import pytest
import importlib.util
from pathlib import Path

from lowbit_tp_comm.quantization import get_qmin_qmax, get_symmetric_scale, multi_tier_quant_dequant


ROOT = Path(__file__).resolve().parents[1]


def _ppl_module():
    spec = importlib.util.spec_from_file_location("eval_ppl_simulated", ROOT / "scripts" / "eval_ppl_simulated.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_int8_signed_range_and_scale() -> None:
    assert get_qmin_qmax(8) == (-128, 127)
    scale = get_symmetric_scale(torch.tensor([-254.0]), torch.tensor([127.0]), num_bits=8)
    assert scale.item() == 2.0


def test_multi_tier_groups_are_disjoint_and_restore_bf16() -> None:
    x = torch.tensor([[-1.1, -0.6, 0.6, 1.1]])
    restored = multi_tier_quant_dequant(x, torch.ones(4), torch.ones(4) / 127, [0], [1], torch.bfloat16)
    assert restored.dtype == torch.bfloat16
    assert restored[..., 0].item() == x[..., 0].to(torch.bfloat16).item()
    with pytest.raises(ValueError, match="disjoint"):
        multi_tier_quant_dequant(x, torch.ones(4), torch.ones(4), [0], [0])


def test_empty_int8_tier_matches_existing_bf16_int4_semantics() -> None:
    from lowbit_tp_comm.quantization import hybrid_quant_dequant

    x = torch.randn(2, 4)
    scale = torch.ones(4)
    selected = torch.tensor([1])
    assert torch.equal(
        multi_tier_quant_dequant(x, scale, torch.ones(4) / 127, selected, [], torch.bfloat16),
        hybrid_quant_dequant(x, scale, selected, output_dtype=torch.bfloat16),
    )


def test_three_tier_bits_and_zero_int8_equivalence() -> None:
    module = _ppl_module()
    calibration = {"modules": {"proj": {"feature_dim": 5120, "k": 80}}}
    bits, rows = module.compute_bits_summary("selected_bf16_int8", calibration, int8_fraction=1 / 64)
    zero_bits, _ = module.compute_bits_summary("selected_bf16_int8", calibration, int8_fraction=0)
    original_bits, _ = module.compute_bits_summary("selected_bf16", calibration)
    random_bits, _ = module.compute_bits_summary("selected_bf16_random_int8", calibration, int8_fraction=1 / 64)
    assert bits == random_bits == 4.25
    assert rows[0]["k_int8"] == 80
    assert zero_bits == original_bits == 4.1875


def test_original_selected_mode_does_not_create_an_int8_path() -> None:
    from torch import nn
    from lowbit_tp_comm.hooks import build_hybrid_replacements_from_calibration

    model = nn.Module()
    model.proj = nn.Linear(8, 64)
    calibration = {"modules": {"proj": {"state_dict": {
        "min_vals": torch.full((2, 64), -1.0), "max_vals": torch.ones(2, 64), "initialized": True,
        "gamma": 0.01, "num_partitions": 2, "feature_dim": 64,
    }, "aggregated_ranges": torch.arange(64.0), "topk_indices": torch.tensor([63]), "k": 1, "feature_dim": 64}}}
    replacement = build_hybrid_replacements_from_calibration(
        model, calibration, "selected_bf16", 2, int8_fraction=1 / 64
    )["proj"]
    assert not hasattr(replacement, "int8_feature_indices")
