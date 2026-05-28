from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lowbit_tp_comm.hooks import ActivationCapture, list_candidate_sync_modules


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.down_proj = nn.Linear(4, 4)
        self.other = nn.Linear(4, 4)
        self.block = nn.Module()
        self.block.out_proj = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down_proj(x)
        x = self.other(x)
        return self.block.out_proj(x)


class TargetStyleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Module()])
        self.layers[0].self_attn = nn.Module()
        self.layers[0].self_attn.o_proj = nn.Linear(4, 4)
        self.layers[0].mlp = nn.Module()
        self.layers[0].mlp.down_proj = nn.Linear(4, 4)
        self.layers[0].attn = nn.Module()
        self.layers[0].attn.c_proj = nn.Linear(4, 4)
        self.layers[0].mlp.c_proj = nn.Linear(4, 4)


def test_list_candidate_sync_modules_finds_expected_names() -> None:
    model = TinyModel()

    matches = list_candidate_sync_modules(model)
    names = [name for name, _module in matches]

    assert "down_proj" in names
    assert "block.out_proj" in names
    assert "other" not in names


def test_list_candidate_sync_modules_llama_style_finds_o_proj_and_down_proj() -> None:
    model = TargetStyleModel()

    matches = list_candidate_sync_modules(model, target_style="llama")
    names = [name for name, _module in matches]

    assert "layers.0.self_attn.o_proj" in names
    assert "layers.0.mlp.down_proj" in names
    assert "layers.0.attn.c_proj" not in names
    assert "layers.0.mlp.c_proj" not in names


def test_list_candidate_sync_modules_gpt2_style_finds_attn_and_mlp_c_proj() -> None:
    model = TargetStyleModel()

    matches = list_candidate_sync_modules(model, target_style="gpt2")
    names = [name for name, _module in matches]

    assert "layers.0.attn.c_proj" in names
    assert "layers.0.mlp.c_proj" in names
    assert "layers.0.self_attn.o_proj" not in names
    assert "layers.0.mlp.down_proj" not in names


def test_activation_capture_stores_output_tensors() -> None:
    model = TinyModel()
    capture = ActivationCapture(model, ["down_proj", "block.out_proj"])
    x = torch.randn(2, 4)

    model(x)

    down_outputs = capture.get_outputs("down_proj")
    out_outputs = capture.get_outputs("block.out_proj")
    assert len(down_outputs) == 1
    assert len(out_outputs) == 1
    assert down_outputs[0].shape == (2, 4)
    assert out_outputs[0].shape == (2, 4)
    assert down_outputs[0].device.type == "cpu"
    capture.remove()


def test_activation_capture_clear_empties_outputs() -> None:
    model = TinyModel()
    capture = ActivationCapture(model, ["down_proj"])

    model(torch.randn(2, 4))
    assert len(capture.get_outputs("down_proj")) == 1

    capture.clear()

    assert capture.get_outputs("down_proj") == []
    capture.remove()


def test_activation_capture_remove_disables_future_capture() -> None:
    model = TinyModel()
    capture = ActivationCapture(model, ["down_proj"])

    capture.remove()
    model(torch.randn(2, 4))

    assert capture.get_outputs("down_proj") == []
