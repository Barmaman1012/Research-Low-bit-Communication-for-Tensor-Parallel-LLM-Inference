from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lowbit_tp_comm.hooks import build_hybrid_replacements_from_calibration, replace_modules_by_name

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


def test_eval_lm_harness_parser_accepts_revision_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    harness_module = _load_script_module("eval_lm_harness.py")
    monkeypatch.setattr(sys, "argv", ["eval_lm_harness.py", "--model_revision", "model-sha", "--tokenizer_revision", "tokenizer-sha"])

    args = harness_module.parse_args()

    assert args.model_revision == "model-sha"
    assert args.tokenizer_revision == "tokenizer-sha"


def test_eval_lm_harness_loaders_forward_optional_revisions(monkeypatch: pytest.MonkeyPatch) -> None:
    harness_module = _load_script_module("eval_lm_harness.py")
    captured: dict[str, dict] = {}

    class FakeModelLoader:
        @staticmethod
        def from_pretrained(_name, **kwargs):
            captured["model"] = kwargs
            return nn.Linear(2, 2)

    class FakeTokenizer:
        eos_token = "<eos>"
        pad_token = None

    class FakeTokenizerLoader:
        @staticmethod
        def from_pretrained(_name, **kwargs):
            captured["tokenizer"] = kwargs
            return FakeTokenizer()

    monkeypatch.setattr(harness_module, "AutoModelForCausalLM", FakeModelLoader)
    monkeypatch.setattr(harness_module, "AutoTokenizer", FakeTokenizerLoader)
    harness_module.load_model_or_raise("fake", torch.device("cpu"), model_revision="model-sha")
    harness_module.load_tokenizer_or_raise("fake", tokenizer_revision="tokenizer-sha")

    assert captured["model"]["revision"] == "model-sha"
    assert captured["tokenizer"]["revision"] == "tokenizer-sha"


def test_eval_lm_harness_omits_revision_kwargs_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    harness_module = _load_script_module("eval_lm_harness.py")
    captured: dict[str, dict] = {}

    class FakeModelLoader:
        @staticmethod
        def from_pretrained(_name, **kwargs):
            captured["model"] = kwargs
            return nn.Linear(2, 2)

    monkeypatch.setattr(harness_module, "AutoModelForCausalLM", FakeModelLoader)
    harness_module.load_model_or_raise("fake", torch.device("cpu"))

    assert "revision" not in captured["model"]


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
            "hellaswag": {"acc,none": 0.7, "acc_norm,none": 0.8},
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
        ("hellaswag", "acc,none"),
        ("other", "exact_match,none"),
    }


def test_eval_lm_harness_metadata_is_json_serializable() -> None:
    harness_module = _load_script_module("eval_lm_harness.py")
    args = SimpleNamespace(
        model_name="fake-model",
        model_revision="requested-model-sha",
        tokenizer_revision="requested-tokenizer-sha",
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
    tokenizer = SimpleNamespace(name_or_path="fake-tokenizer", init_kwargs={"_commit_hash": "resolved-tokenizer-sha"})
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
    assert metadata["model_revision"] == "requested-model-sha"
    assert metadata["resolved_model_revision"] == "requested-model-sha"
    assert metadata["tokenizer_revision"] == "requested-tokenizer-sha"
    assert metadata["resolved_tokenizer_revision"] == "resolved-tokenizer-sha"
    json.dumps(harness_module.make_json_serializable(metadata))


def test_threshold_bf16_harness_serializes_exact_allocation_metadata(tmp_path: Path) -> None:
    harness_module = _load_script_module("eval_lm_harness.py")
    calibration_path = tmp_path / "calibration.pt"
    calibration_path.write_bytes(b"threshold-metadata-fixture")
    model = nn.Module()
    model.a_proj = nn.Linear(4, 4)
    model.b_proj = nn.Linear(4, 4)

    def payload(ranges: list[float], k: int) -> dict:
        width = len(ranges)
        return {
            "state_dict": {
                "min_vals": -torch.ones(2, width), "max_vals": torch.ones(2, width),
                "initialized": True, "gamma": 0.01, "num_partitions": 2, "feature_dim": width,
            },
            "aggregated_ranges": torch.tensor(ranges), "topk_indices": torch.arange(k),
            "k": k, "feature_dim": width,
        }

    calibration = {"modules": {"a_proj": payload([1, 1, 2, 4], 1), "b_proj": payload([1, 1, 1, 8], 1)}}
    replace_modules_by_name(model, build_hybrid_replacements_from_calibration(model, calibration, "threshold_bf16", 2))
    args = SimpleNamespace(
        model_name="fake-model", model_revision=None, tokenizer_revision=None, device="cpu", limit=1,
        batch_size="1", seed=0, mode="full", num_partitions=2, num_bits=4,
        calibration_path=str(calibration_path), target_style="llama", int8_fraction=0.0,
    )
    metadata = harness_module.build_run_metadata(
        args=args, model=model, tokenizer=SimpleNamespace(name_or_path="fake", init_kwargs={}),
        lm_eval=SimpleNamespace(__version__="test"), tasks=["arc_easy"], modes=["full", "threshold_bf16"],
        mode="threshold_bf16",
    )
    serialized = harness_module.make_json_serializable({"metadata": {"threshold_bf16": metadata}})
    json.loads(json.dumps(serialized))
    allocation = serialized["metadata"]["threshold_bf16"]["threshold_bf16"]
    required = {
        "mode", "normalization_method", "total_feature_count", "target_bf16_count", "actual_bf16_count",
        "global_bf16_fraction", "global_int4_fraction", "average_bits_per_value", "derived_threshold",
        "next_excluded_score", "boundary_tie", "deterministic_tie_breaking", "per_module_bf16_counts",
        "per_module_bf16_fractions", "calibration_path", "calibration_sha256",
    }
    assert required <= allocation.keys()
    assert allocation["mode"] == "threshold_bf16"
    assert allocation["normalization_method"] == "module_median"
    assert sum(allocation["per_module_bf16_counts"].values()) == allocation["actual_bf16_count"]
    assert allocation["actual_bf16_count"] == allocation["target_bf16_count"]
    assert allocation["global_bf16_fraction"] == allocation["actual_bf16_count"] / allocation["total_feature_count"]
    assert allocation["global_int4_fraction"] == 1 - allocation["global_bf16_fraction"]
    assert allocation["average_bits_per_value"] == (
        16 * allocation["global_bf16_fraction"] + 4 * allocation["global_int4_fraction"]
    )


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
