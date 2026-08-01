from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lowbit_tp_comm.dtypes import (
    canonicalize_device,
    model_load_kwargs,
    resolve_dtype,
    validate_model_dtype,
    validate_module_devices_and_dtypes,
)
from lowbit_tp_comm.calibration import EMAMinMaxCalibrator
from lowbit_tp_comm.quantization import hybrid_quant_dequant
from lowbit_tp_comm.tp_linear import HybridQuantizedRowParallelLinear, compute_row_parallel_partials_for_module


def _load_script_module(script_name: str):
    path = ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name[:-3], path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("name", "expected"),
    [("auto", None), ("float32", torch.float32), ("float16", torch.float16), ("bfloat16", torch.bfloat16)],
)
def test_dtype_string_parsing_and_transformers_forwarding(name: str, expected: torch.dtype | None) -> None:
    assert resolve_dtype(name) is expected
    assert model_load_kwargs(name)["dtype"] == ("auto" if expected is None else expected)


def test_dtype_string_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="Unsupported dtype"):
        resolve_dtype("fp8")


def test_bfloat16_partials_selected_features_and_reconstruction_stay_bfloat16() -> None:
    linear = nn.Linear(8, 6).to(torch.bfloat16)
    x = torch.randn(2, 3, 8, dtype=torch.bfloat16)
    partials = compute_row_parallel_partials_for_module(linear, x, num_partitions=2)
    assert all(partial.dtype == torch.bfloat16 for partial in partials)

    scales = torch.ones(2, 6, dtype=torch.float32)
    selected = torch.tensor([1, 4], dtype=torch.long)
    replacement = HybridQuantizedRowParallelLinear.from_linear(
        linear, num_partitions=2, scales_per_partition=scales, bf16_feature_indices=selected, output_dtype=torch.bfloat16
    )
    reconstructed = hybrid_quant_dequant(partials[0], scales[0], selected, output_dtype=torch.bfloat16)
    assert reconstructed.dtype == torch.bfloat16
    assert torch.equal(reconstructed[..., selected], partials[0][..., selected])
    assert replacement(x).dtype == torch.bfloat16
    assert replacement.scales_per_partition.dtype == torch.float32


def test_explicit_dtype_validation_rejects_silent_fp32_model() -> None:
    with pytest.raises(ValueError, match="Requested model dtype"):
        validate_model_dtype(nn.Linear(2, 2), torch.bfloat16)


def test_module_device_validation_accepts_cpu_model() -> None:
    validate_module_devices_and_dtypes(nn.Linear(2, 2), torch.device("cpu"), torch.float32)


def test_module_device_validation_rejects_actual_device_mismatch() -> None:
    model = nn.Linear(2, 2, device="meta")

    with pytest.raises(ValueError, match=r"is on meta, expected cpu"):
        validate_module_devices_and_dtypes(model, torch.device("cpu"), None)


def test_explicit_cuda_index_is_not_rewritten() -> None:
    assert canonicalize_device(torch.device("cuda:1")) == torch.device("cuda:1")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_module_device_validation_accepts_implicit_current_cuda_device() -> None:
    with torch.cuda.device(0):
        model = nn.Linear(2, 2).to(torch.device("cuda:0"))

        validate_module_devices_and_dtypes(model, torch.device("cuda"), torch.float32)
        assert next(model.parameters()).device == torch.device("cuda:0")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_module_device_validation_accepts_explicit_cuda_zero() -> None:
    model = nn.Linear(2, 2).to(torch.device("cuda:0"))

    validate_module_devices_and_dtypes(model, torch.device("cuda:0"), torch.float32)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="At least two CUDA devices are required",
)
def test_module_device_validation_preserves_explicit_nondefault_cuda_device() -> None:
    model = nn.Linear(2, 2).to(torch.device("cuda:1"))

    validate_module_devices_and_dtypes(model, torch.device("cuda:1"), torch.float32)


def test_calibration_evaluation_dtype_mismatch_is_rejected() -> None:
    eval_module = _load_script_module("eval_ppl_simulated.py")
    with pytest.raises(ValueError, match="Calibration dtype mismatch"):
        eval_module.validate_calibration_dtype(
            {"dtype_metadata": {"requested_model_dtype": "float32"}}, "bfloat16"
        )


def test_calibration_metadata_records_bf16_execution_and_fp32_statistics() -> None:
    calibration_script = _load_script_module("calibrate_model.py")
    calibrator = EMAMinMaxCalibrator(num_partitions=1, feature_dim=2)
    metadata = calibration_script.build_dtype_metadata(
        requested_dtype_name="bfloat16",
        model=nn.Linear(2, 2).to(torch.bfloat16),
        calibrators={"proj": calibrator},
        observed_partial_dtypes={"torch.bfloat16"},
    )
    assert metadata["requested_model_dtype"] == "bfloat16"
    assert metadata["model_parameter_dtype"] == "torch.bfloat16"
    assert metadata["partial_output_dtype"] == "torch.bfloat16"
    assert metadata["calibration_statistics_dtype"] == "torch.float32"
    assert metadata["intended_selected_feature_communication_dtype"] == "torch.bfloat16"


def test_analytical_bits_only_reports_4_1875_for_16_bit_selected_path() -> None:
    eval_module = _load_script_module("eval_ppl_simulated.py")
    assert eval_module.compute_module_avg_bits(64, 1, selected_bits=16) == 4.1875
    assert eval_module.compute_module_avg_bits(64, 1, selected_bits=32) == 4.4375


def test_lm_loader_forwards_requested_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    harness = _load_script_module("eval_lm_harness.py")
    captured: dict[str, object] = {}

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(_name: str, **kwargs):
            captured.update(kwargs)
            return nn.Linear(2, 2).to(torch.bfloat16)

    monkeypatch.setattr(harness, "AutoModelForCausalLM", FakeAutoModel)
    model = harness.load_model_or_raise("fake", torch.device("cpu"), "bfloat16")
    assert captured == {"dtype": torch.bfloat16}
    assert next(model.parameters()).dtype == torch.bfloat16
