from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_script_module(script_name: str):
    script_path = ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bits_calculation_examples() -> None:
    eval_module = _load_script_module("eval_ppl_simulated.py")

    assert eval_module.compute_module_avg_bits(64, 1) == 4.1875
    assert eval_module.compute_module_avg_bits(2, 1) == 10.0

    int4_bits, _ = eval_module.compute_bits_summary("int4", calibration=None)
    full_bits, _ = eval_module.compute_bits_summary("full", calibration=None)
    assert int4_bits == 4.0
    assert full_bits == 16.0


def test_inspect_calibration_formats_fake_payload() -> None:
    inspect_module = _load_script_module("inspect_calibration.py")
    fake_calibration = {
        "model_name": "fake-model",
        "gamma": 0.01,
        "k_fraction": 0.015625,
        "num_sequences": 8,
        "sequence_length": 64,
        "modules": {
            "layer.down_proj": {
                "aggregated_ranges": __import__("torch").tensor([1.0, 4.0, 2.0, 3.0]),
                "topk_indices": __import__("torch").tensor([1]),
                "k": 1,
                "feature_dim": 4,
            }
        },
    }

    output = inspect_module.format_calibration_summary(fake_calibration, topn=3)

    assert "model_name=fake-model" in output
    assert "module=layer.down_proj" in output
    assert "selected_fraction=0.250000" in output
    assert "top_selected_indices=[1]" in output


def test_eval_lm_harness_table_formatting() -> None:
    harness_module = _load_script_module("eval_lm_harness.py")
    rows = [
        {"mode": "full", "task": "arc_easy", "metric": "acc_norm,none", "value": 0.5},
        {"mode": "int4", "task": "arc_easy", "metric": "acc_norm,none", "value": 0.4},
    ]

    table = harness_module.format_results_table(rows)
    summary = harness_module.format_average_summary(rows)

    assert "mode | task | metric | value" in table
    assert "full | arc_easy | acc_norm,none | 0.500000" in table
    assert "mode | avg_primary_score" in summary
    assert "int4 | 0.400000" in summary


def test_eval_lm_harness_uses_zero_shot_and_supported_deterministic_seeds() -> None:
    harness_module = _load_script_module("eval_lm_harness.py")

    def simple_evaluate(
        *,
        model,
        tasks,
        limit,
        batch_size,
        device,
        num_fewshot,
        random_seed,
        numpy_random_seed,
        torch_random_seed,
        fewshot_random_seed,
    ):
        return None

    kwargs = harness_module.build_simple_evaluate_kwargs(
        simple_evaluate,
        model="model",
        tasks=["arc_easy"],
        limit=10,
        batch_size="1",
        device=torch.device("cpu"),
        seed=7,
    )

    assert kwargs["num_fewshot"] == 0
    assert kwargs["random_seed"] == 7
    assert kwargs["numpy_random_seed"] == 7
    assert kwargs["torch_random_seed"] == 7
    assert kwargs["fewshot_random_seed"] == 7


def test_eval_lm_harness_omits_unsupported_reproducibility_arguments() -> None:
    harness_module = _load_script_module("eval_lm_harness.py")

    def simple_evaluate(*, model, tasks, limit, batch_size, device):
        return None

    kwargs = harness_module.build_simple_evaluate_kwargs(
        simple_evaluate,
        model="model",
        tasks=["arc_easy"],
        limit=10,
        batch_size="1",
        device=torch.device("cpu"),
        seed=7,
    )

    assert "num_fewshot" not in kwargs
    assert not {"random_seed", "numpy_random_seed", "torch_random_seed", "fewshot_random_seed"} & set(kwargs)


def test_eval_lm_harness_filters_reporting_metrics_and_selects_primary_scores() -> None:
    harness_module = _load_script_module("eval_lm_harness.py")
    results = {
        "results": {
            "arc_easy": {
                "acc,none": 0.4,
                "acc_stderr,none": 0.02,
                "acc_norm,none": 0.5,
                "acc_norm_stderr,none": 0.03,
                "sample_len,none": 17.0,
            },
            "boolq": {"acc,none": 0.6, "sample_len,none": 4.0},
            "other": {"exact_match,none": 0.7, "f1,none": 0.8},
        }
    }

    rows = harness_module.extract_rows("full", results)
    metrics = {row["metric"] for row in rows}
    primary = harness_module.primary_metric_rows(rows)

    assert metrics == {"acc,none", "acc_norm,none", "exact_match,none", "f1,none"}
    assert {(row["task"], row["metric"]) for row in primary} == {
        ("arc_easy", "acc_norm,none"),
        ("boolq", "acc,none"),
        ("other", "exact_match,none"),
    }


def test_eval_lm_harness_metadata_is_json_serializable() -> None:
    harness_module = _load_script_module("eval_lm_harness.py")
    args = SimpleNamespace(
        model_name="fake-model",
        device="cpu",
        limit=10,
        batch_size="1",
        seed=3,
        mode="full",
        num_partitions=2,
        num_bits=4,
        calibration_path="calibration.pt",
        target_style="gpt2",
    )
    tokenizer = SimpleNamespace(name_or_path="fake-tokenizer")
    lm_eval = SimpleNamespace(__version__="test-version")

    metadata = harness_module.build_run_metadata(
        args=args,
        model=nn.Linear(4, 4),
        tokenizer=tokenizer,
        lm_eval=lm_eval,
        tasks=["arc_easy"],
        modes=["full"],
    )

    assert metadata["num_fewshot"] == 0
    assert metadata["actual_model_device"] == "cpu"
    assert metadata["model_parameter_dtype"] == "torch.float32"
    json.dumps(harness_module.make_json_serializable(metadata))


def test_eval_lm_harness_rejects_partition_specific_calibration_mismatch() -> None:
    harness_module = _load_script_module("eval_lm_harness.py")
    model = nn.Module()
    model.proj = nn.Linear(4, 4)
    calibration = {
        "model_name": "fake-model",
        "simulated_row_parallel_calibration": True,
        "num_partitions": 2,
        "modules": {
            "proj": {
                "feature_dim": 4,
                "state_dict": {
                    "min_vals": torch.zeros(2, 4),
                    "max_vals": torch.ones(2, 4),
                    "num_partitions": 2,
                    "feature_dim": 4,
                },
            }
        },
    }

    with pytest.raises(ValueError, match="Calibration partition mismatch"):
        harness_module.validate_calibration_compatibility(
            model, calibration, model_name="fake-model", num_partitions=4
        )


def test_eval_lm_harness_json_serialization_helper() -> None:
    harness_module = _load_script_module("eval_lm_harness.py")
    torch = __import__("torch")
    obj = {
        "dtype": torch.float32,
        "tensor": torch.tensor([[1.0, 2.0]]),
        "nested": {"items": [torch.tensor(3.0), "x"]},
    }
    try:
        import numpy as np

        obj["numpy_scalar"] = np.float32(1.25)
        obj["numpy_array"] = np.array([1, 2, 3])
    except ImportError:
        pass

    serializable = harness_module.make_json_serializable(obj)
    json.dumps(serializable)
