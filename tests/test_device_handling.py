from __future__ import annotations

import pytest
import torch
from torch import nn

from lowbit_tp_comm.hooks import ModuleInputOutputCapture
from lowbit_tp_comm.tp_linear import compute_row_parallel_partials_for_module
from scripts.calibrate_model import build_inputs


class DummyTokenizer:
    pad_token = None
    eos_token = "<eos>"

    def __call__(self, texts, **_kwargs):
        return {
            "input_ids": torch.arange(len(texts) * 4, dtype=torch.long).reshape(len(texts), 4),
            "attention_mask": torch.ones(len(texts), 4, dtype=torch.long),
        }


class MetadataTokenizer(DummyTokenizer):
    def __call__(self, texts, **kwargs):
        encoded = super().__call__(texts, **kwargs)
        encoded["metadata"] = "preserve-me"
        return encoded


def test_calibration_build_inputs_moves_every_tensor_to_cpu() -> None:
    encoded = build_inputs(DummyTokenizer(), ["test"], sequence_length=4, device=torch.device("cpu"))

    assert encoded
    assert all(tensor.device.type == "cpu" for tensor in encoded.values())


def test_calibration_build_inputs_preserves_non_tensor_values() -> None:
    encoded = build_inputs(MetadataTokenizer(), ["test"], sequence_length=4, device=torch.device("cpu"))

    assert encoded["metadata"] == "preserve-me"
    assert encoded["input_ids"].device.type == "cpu"


def test_captured_cpu_inputs_match_cpu_module_weights_for_partials() -> None:
    module = nn.Linear(8, 6, bias=False)
    capture = ModuleInputOutputCapture(module, [""], store_on_cpu=False)
    inputs = torch.randn(2, 8)

    module(inputs)
    captured = capture.get_inputs("")[0]
    partials = compute_row_parallel_partials_for_module(module, captured, num_partitions=2)

    assert captured.device == module.weight.device == torch.device("cpu")
    assert all(partial.device == module.weight.device for partial in partials)
    capture.remove()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_calibration_inputs_and_captured_partials_stay_on_cuda() -> None:
    # Tensor.device is concrete (cuda:0), while torch.device("cuda") is not
    # equal to torch.device("cuda:0") despite selecting the same default GPU.
    device = torch.device("cuda:0")
    encoded = build_inputs(DummyTokenizer(), ["test"], sequence_length=4, device=device)
    module = nn.Linear(8, 6, bias=False).to(device)
    capture = ModuleInputOutputCapture(module, [""], store_on_cpu=False)
    inputs = torch.randn(2, 8, device=device)

    module(inputs)
    captured = capture.get_inputs("")[0]
    partials = compute_row_parallel_partials_for_module(module, captured, num_partitions=2)

    assert all(tensor.device == device for tensor in encoded.values())
    assert captured.device == module.weight.device == device
    assert all(partial.device == device for partial in partials)
    capture.remove()
