from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


TorchDTypeName = Literal["float32", "float16", "bfloat16"]
ShardDimension = Literal["row", "column"]


@dataclass(slots=True)
class QuantizationConfig:
    """Configuration for future communication quantization experiments."""

    enabled: bool = False
    num_bits: int = 4
    group_size: int = 128
    symmetric: bool = True
    keep_bf16_ratio: float = 0.0


@dataclass(slots=True)
class CalibrationConfig:
    """Controls activation-statistics collection for later calibration passes."""

    enabled: bool = False
    num_samples: int = 128
    sequence_length: int = 512
    dataset_name: str | None = None
    dataset_split: str = "validation"


@dataclass(slots=True)
class TensorParallelConfig:
    """Simulation settings for a virtual tensor-parallel execution environment."""

    tp_degree: int = 2
    shard_dimension: ShardDimension = "column"
    simulate_collectives: bool = True
    record_communication: bool = True


@dataclass(slots=True)
class EvaluationConfig:
    """High-level evaluation parameters for model and dataset wiring."""

    model_name_or_path: str = "gpt2"
    tokenizer_name_or_path: str | None = None
    batch_size: int = 1
    max_new_tokens: int = 32
    dtype: TorchDTypeName = "bfloat16"
    device: str = "cpu"


@dataclass(slots=True)
class ProjectConfig:
    """Top-level project configuration bundle."""

    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    tensor_parallel: TensorParallelConfig = field(default_factory=TensorParallelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
