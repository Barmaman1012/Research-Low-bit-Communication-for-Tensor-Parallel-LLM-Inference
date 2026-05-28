from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

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
