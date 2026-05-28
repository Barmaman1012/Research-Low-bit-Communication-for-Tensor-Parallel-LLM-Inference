"""Top-level package exports for the low-bit tensor-parallel communication scaffold."""

from .calibration import (
    CalibrationResult,
    CalibrationRunner,
    CalibrationState,
    EMAMinMaxCalibrator,
    TensorStats,
)
from .config import (
    CalibrationConfig,
    EvaluationConfig,
    ProjectConfig,
    QuantizationConfig,
    TensorParallelConfig,
)
from .evaluation import EvaluationResult, LLMEvaluator
from .hooks import (
    ActivationCapture,
    HookContext,
    NoOpCommunicationHook,
    build_hybrid_replacements_from_calibration,
    list_candidate_sync_modules,
    replace_modules_by_name,
)
from .quantization import (
    IdentityQuantizer,
    QuantizedTensor,
    dequantize_symmetric,
    get_symmetric_scale,
    hybrid_quant_dequant,
    quantize_symmetric,
)
from .tp_linear import (
    HybridQuantizedRowParallelConv1D,
    HybridQuantizedRowParallelLinear,
    PartitionMinMax,
    SimulatedTPLinear,
    TensorParallelLinearSimulator,
    compute_partition_minmax,
    make_random_bf16_indices,
)

__all__ = [
    "CalibrationConfig",
    "CalibrationResult",
    "CalibrationRunner",
    "CalibrationState",
    "EMAMinMaxCalibrator",
    "ActivationCapture",
    "EvaluationConfig",
    "EvaluationResult",
    "HybridQuantizedRowParallelConv1D",
    "HybridQuantizedRowParallelLinear",
    "HookContext",
    "IdentityQuantizer",
    "LLMEvaluator",
    "NoOpCommunicationHook",
    "ProjectConfig",
    "QuantizationConfig",
    "QuantizedTensor",
    "PartitionMinMax",
    "SimulatedTPLinear",
    "TensorParallelLinearSimulator",
    "TensorParallelConfig",
    "TensorStats",
    "build_hybrid_replacements_from_calibration",
    "compute_partition_minmax",
    "dequantize_symmetric",
    "get_symmetric_scale",
    "hybrid_quant_dequant",
    "list_candidate_sync_modules",
    "make_random_bf16_indices",
    "quantize_symmetric",
    "replace_modules_by_name",
]
