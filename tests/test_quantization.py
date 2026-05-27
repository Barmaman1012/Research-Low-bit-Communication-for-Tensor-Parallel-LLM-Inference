from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lowbit_tp_comm.quantization import IdentityQuantizer


def test_identity_quantizer_round_trip_preserves_tensor() -> None:
    quantizer = IdentityQuantizer()
    tensor = torch.randn(4, 8)

    qtensor = quantizer.quantize(tensor)
    restored = quantizer.dequantize(qtensor)

    assert restored.shape == tensor.shape
    assert torch.equal(restored, tensor)
    assert qtensor.metadata["scheme"] == "identity"
