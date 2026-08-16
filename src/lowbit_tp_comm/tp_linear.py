from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from .quantization import hybrid_quant_dequant, multi_tier_quant_dequant


# ``threshold_bf16`` uses HybridQuantizedRowParallel{Linear,Conv1D} directly.
# No extra layer class or Int8 buffers are needed because the reconstruction is
# exactly the existing selected-BF16 + Int4 reconstruction.
# The explicit range-threshold modes use these same classes too.

try:
    from transformers.pytorch_utils import Conv1D
except ImportError:  # pragma: no cover - transformers is part of requirements, but keep import safe.
    Conv1D = None


@dataclass(slots=True)
class PartitionMinMax:
    """Per-partition feature statistics for future calibration steps."""

    min_vals: Tensor
    max_vals: Tensor


def compute_partition_minmax(partials: list[Tensor]) -> list[PartitionMinMax]:
    """Return per-feature min/max statistics for each partition partial output."""

    stats: list[PartitionMinMax] = []
    for partial in partials:
        if partial.ndim < 2:
            raise ValueError(f"Expected partial outputs with shape [..., E], got ndim={partial.ndim}.")

        reduce_dims = tuple(range(partial.ndim - 1))
        stats.append(
            PartitionMinMax(
                min_vals=partial.amin(dim=reduce_dims),
                max_vals=partial.amax(dim=reduce_dims),
            )
        )
    return stats


def make_random_bf16_indices(
    feature_dim: int,
    k: int,
    seed: int = 0,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Choose k random output-feature indices without replacement."""

    if feature_dim <= 0:
        raise ValueError(f"feature_dim must be positive, got {feature_dim}.")
    if k <= 0:
        return torch.empty(0, dtype=torch.long, device=device)

    clamped_k = min(k, feature_dim)
    resolved_generator = generator or torch.Generator(device="cpu").manual_seed(seed)
    return torch.randperm(feature_dim, generator=resolved_generator, device="cpu")[:clamped_k].to(device=device)


def compute_row_parallel_partials_for_module(
    module: nn.Module,
    x: Tensor,
    num_partitions: int,
) -> list[Tensor]:
    """Compute row-parallel partial outputs for supported projection modules.

    Bias is intentionally excluded. The returned partials sum to the pre-bias
    projection output.
    """

    if num_partitions <= 0:
        raise ValueError(f"num_partitions must be positive, got {num_partitions}.")

    if isinstance(module, nn.Linear):
        in_features = module.in_features
        out_features = module.out_features
        if x.shape[-1] != in_features:
            raise ValueError(f"Expected input trailing dimension {in_features}, got {x.shape[-1]}.")
        if in_features % num_partitions != 0:
            raise ValueError(
                f"Input dimension {in_features} must be divisible by num_partitions={num_partitions}."
            )
        partition_size = in_features // num_partitions
        x_parts = list(x.split(partition_size, dim=-1))
        weight_parts = list(module.weight.split(partition_size, dim=1))
        partials = [
            torch.matmul(x_i, w_i.transpose(0, 1))
            for x_i, w_i in zip(x_parts, weight_parts, strict=True)
        ]
    elif Conv1D is not None and isinstance(module, Conv1D):
        in_features, out_features = module.weight.shape
        if x.shape[-1] != in_features:
            raise ValueError(f"Expected input trailing dimension {in_features}, got {x.shape[-1]}.")
        if in_features % num_partitions != 0:
            raise ValueError(
                f"Input dimension {in_features} must be divisible by num_partitions={num_partitions}."
            )
        partition_size = in_features // num_partitions
        x_parts = list(x.split(partition_size, dim=-1))
        weight_parts = list(module.weight.split(partition_size, dim=0))
        partials = [torch.matmul(x_i, w_i) for x_i, w_i in zip(x_parts, weight_parts, strict=True)]
    else:
        raise TypeError(f"Unsupported module type for row-parallel partials: {type(module)}")

    expected_shape = (*x.shape[:-1], out_features)
    for partial in partials:
        if tuple(partial.shape) != expected_shape:
            raise ValueError(
                f"Expected partial output shape {expected_shape}, got {tuple(partial.shape)}."
            )
    return partials


class TensorParallelLinearSimulator:
    """Single-process simulation of input-sharded tensor parallel linear layers.

    The full linear is y = x @ W + b with x shape [..., D] and W shape [D, E].
    We split D across partitions, compute partial_i = x_i @ W_i, optionally
    simulate communication quantization on each partial, then sum the partials.
    """

    def __init__(
        self,
        weight: Tensor,
        bias: Tensor | None = None,
        num_partitions: int = 2,
    ) -> None:
        if weight.ndim != 2:
            raise ValueError(f"weight must have shape [D, E], got {tuple(weight.shape)}.")
        if num_partitions <= 0:
            raise ValueError(f"num_partitions must be positive, got {num_partitions}.")

        in_features, out_features = weight.shape
        if in_features % num_partitions != 0:
            raise ValueError(
                f"Input dimension D={in_features} must be divisible by num_partitions={num_partitions}."
            )
        if bias is not None and bias.shape != (out_features,):
            raise ValueError(f"bias must have shape [{out_features}], got {tuple(bias.shape)}.")

        self.weight = weight
        self.bias = bias
        self.num_partitions = num_partitions
        self.in_features = in_features
        self.out_features = out_features
        self.partition_size = in_features // num_partitions
        self.weight_partitions = list(weight.split(self.partition_size, dim=0))

    def _split_input(self, x: Tensor) -> list[Tensor]:
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected input trailing dimension {self.in_features}, got {x.shape[-1]}."
            )
        return list(x.split(self.partition_size, dim=-1))

    def _compute_partials(self, x: Tensor) -> list[Tensor]:
        x_partitions = self._split_input(x)
        return [x_i @ w_i for x_i, w_i in zip(x_partitions, self.weight_partitions, strict=True)]

    def compute_partials(self, x: Tensor) -> list[Tensor]:
        """Compute per-partition partial outputs before synchronization."""

        return self._compute_partials(x)

    def forward_full(self, x: Tensor) -> Tensor:
        y = x @ self.weight
        if self.bias is not None:
            y = y + self.bias
        return y

    def forward_tp_uncompressed(self, x: Tensor) -> Tensor:
        partials = self.compute_partials(x)
        y = torch.stack(partials, dim=0).sum(dim=0)
        if self.bias is not None:
            y = y + self.bias
        return y

    def forward_tp_hybrid_quantized(
        self,
        x: Tensor,
        scales_per_partition: list[Tensor],
        bf16_feature_indices: Tensor | list[int] | None,
        output_dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        partials = self.compute_partials(x)
        if len(scales_per_partition) != len(partials):
            raise ValueError(
                f"Expected {len(partials)} scale tensors, got {len(scales_per_partition)}."
            )

        reconstructed_partials = [
            hybrid_quant_dequant(
                partial,
                scale,
                bf16_feature_indices=bf16_feature_indices,
                output_dtype=output_dtype,
            )
            for partial, scale in zip(partials, scales_per_partition, strict=True)
        ]
        y = torch.stack(reconstructed_partials, dim=0).sum(dim=0)
        if self.bias is not None:
            y = y + self.bias.to(output_dtype)
        return y


class _HybridQuantizedRowParallelBase(nn.Module):
    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        bias: Tensor | None,
        num_partitions: int,
        scales_per_partition: Tensor,
        bf16_feature_indices: Optional[Tensor],
        num_bits: int,
        output_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if num_partitions <= 0:
            raise ValueError(f"num_partitions must be positive, got {num_partitions}.")
        if in_features % num_partitions != 0:
            raise ValueError(
                f"Input dimension {in_features} must be divisible by num_partitions={num_partitions}."
            )
        if scales_per_partition.shape != (num_partitions, out_features):
            raise ValueError(
                "scales_per_partition must have shape "
                f"({num_partitions}, {out_features}), got {tuple(scales_per_partition.shape)}."
            )

        if bf16_feature_indices is None:
            bf16_feature_indices = torch.empty(0, dtype=torch.long)
        else:
            bf16_feature_indices = bf16_feature_indices.detach().to(dtype=torch.long)

        self.in_features = in_features
        self.out_features = out_features
        self.num_partitions = num_partitions
        self.partition_size = in_features // num_partitions
        self.num_bits = num_bits
        self.output_dtype = output_dtype
        self.register_buffer("scales_per_partition", scales_per_partition.detach().clone())
        self.register_buffer("bf16_feature_indices", bf16_feature_indices)
        if bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)

    def _split_input(self, x: Tensor) -> list[Tensor]:
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected input trailing dimension {self.in_features}, got {x.shape[-1]}."
            )
        return list(x.split(self.partition_size, dim=-1))

    def _validate_output_dtype(self, output: Tensor) -> Tensor:
        if output.dtype != self.output_dtype:
            raise RuntimeError(
                f"Hybrid TP replacement produced {output.dtype}, expected model-compatible {self.output_dtype}."
            )
        return output


class _RowParallelBase(nn.Module):
    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        bias: Tensor | None,
        num_partitions: int,
    ) -> None:
        super().__init__()
        if num_partitions <= 0:
            raise ValueError(f"num_partitions must be positive, got {num_partitions}.")
        if in_features % num_partitions != 0:
            raise ValueError(
                f"Input dimension {in_features} must be divisible by num_partitions={num_partitions}."
            )
        self.in_features = in_features
        self.out_features = out_features
        self.num_partitions = num_partitions
        self.partition_size = in_features // num_partitions
        if bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)

    def _split_input(self, x: Tensor) -> list[Tensor]:
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected input trailing dimension {self.in_features}, got {x.shape[-1]}."
            )
        return list(x.split(self.partition_size, dim=-1))


class RowParallelLinear(_RowParallelBase):
    """Exact single-process row-parallel replacement for nn.Linear."""

    def __init__(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        num_partitions: int,
    ) -> None:
        if weight.ndim != 2:
            raise ValueError(f"weight must have shape [out_features, in_features], got {tuple(weight.shape)}.")
        out_features, in_features = weight.shape
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            num_partitions=num_partitions,
        )
        self.weight = nn.Parameter(weight.detach().clone(), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        x_partitions = self._split_input(x)
        weight_partitions = self.weight.split(self.partition_size, dim=1)
        partials = [
            torch.matmul(x_i, weight_i.transpose(0, 1))
            for x_i, weight_i in zip(x_partitions, weight_partitions, strict=True)
        ]
        y = torch.stack(partials, dim=0).sum(dim=0)
        if self.bias is not None:
            y = y + self.bias
        return y

    @classmethod
    def from_linear(cls, linear: nn.Linear, num_partitions: int) -> "RowParallelLinear":
        return cls(weight=linear.weight, bias=linear.bias, num_partitions=num_partitions)


class RowParallelConv1D(_RowParallelBase):
    """Exact single-process row-parallel replacement for GPT-2 style Conv1D."""

    def __init__(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        num_partitions: int,
    ) -> None:
        if weight.ndim != 2:
            raise ValueError(f"weight must have shape [in_features, out_features], got {tuple(weight.shape)}.")
        in_features, out_features = weight.shape
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            num_partitions=num_partitions,
        )
        self.weight = nn.Parameter(weight.detach().clone(), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        x_partitions = self._split_input(x)
        weight_partitions = self.weight.split(self.partition_size, dim=0)
        partials = [torch.matmul(x_i, weight_i) for x_i, weight_i in zip(x_partitions, weight_partitions, strict=True)]
        y = torch.stack(partials, dim=0).sum(dim=0)
        if self.bias is not None:
            y = y + self.bias
        return y

    @classmethod
    def from_conv1d(cls, conv1d: Conv1D, num_partitions: int) -> "RowParallelConv1D":
        return cls(weight=conv1d.weight, bias=conv1d.bias, num_partitions=num_partitions)


class HybridQuantizedRowParallelLinear(_HybridQuantizedRowParallelBase):
    """Simulated row-parallel replacement for nn.Linear."""

    def __init__(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        num_partitions: int,
        scales_per_partition: torch.Tensor,
        bf16_feature_indices: Optional[torch.Tensor] = None,
        num_bits: int = 4,
        output_dtype: torch.dtype = torch.float32,
    ) -> None:
        if weight.ndim != 2:
            raise ValueError(f"weight must have shape [out_features, in_features], got {tuple(weight.shape)}.")
        out_features, in_features = weight.shape
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            num_partitions=num_partitions,
            scales_per_partition=scales_per_partition,
            bf16_feature_indices=bf16_feature_indices,
            num_bits=num_bits,
            output_dtype=output_dtype,
        )
        self.weight = nn.Parameter(weight.detach().clone(), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        x_partitions = self._split_input(x)
        weight_partitions = self.weight.split(self.partition_size, dim=1)
        partials = [
            torch.matmul(x_i, weight_i.transpose(0, 1))
            for x_i, weight_i in zip(x_partitions, weight_partitions, strict=True)
        ]
        reconstructed = [
            hybrid_quant_dequant(
                partial,
                self.scales_per_partition[i],
                bf16_feature_indices=self.bf16_feature_indices,
                num_bits=self.num_bits,
                output_dtype=self.output_dtype,
            )
            for i, partial in enumerate(partials)
        ]
        y = torch.stack(reconstructed, dim=0).sum(dim=0)
        if self.bias is not None:
            y = y + self.bias.to(self.output_dtype)
        return self._validate_output_dtype(y)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        num_partitions: int,
        scales_per_partition: torch.Tensor,
        bf16_feature_indices: Optional[torch.Tensor],
        num_bits: int = 4,
        output_dtype: torch.dtype = torch.float32,
    ) -> "HybridQuantizedRowParallelLinear":
        return cls(
            weight=linear.weight,
            bias=linear.bias,
            num_partitions=num_partitions,
            scales_per_partition=scales_per_partition,
            bf16_feature_indices=bf16_feature_indices,
            num_bits=num_bits,
            output_dtype=output_dtype,
        )


class HybridQuantizedRowParallelConv1D(_HybridQuantizedRowParallelBase):
    """Simulated row-parallel replacement for GPT-2 style Conv1D."""

    def __init__(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        num_partitions: int,
        scales_per_partition: torch.Tensor,
        bf16_feature_indices: Optional[torch.Tensor] = None,
        num_bits: int = 4,
        output_dtype: torch.dtype = torch.float32,
    ) -> None:
        if weight.ndim != 2:
            raise ValueError(f"weight must have shape [in_features, out_features], got {tuple(weight.shape)}.")
        in_features, out_features = weight.shape
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            num_partitions=num_partitions,
            scales_per_partition=scales_per_partition,
            bf16_feature_indices=bf16_feature_indices,
            num_bits=num_bits,
            output_dtype=output_dtype,
        )
        self.weight = nn.Parameter(weight.detach().clone(), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        x_partitions = self._split_input(x)
        weight_partitions = self.weight.split(self.partition_size, dim=0)
        partials = [torch.matmul(x_i, weight_i) for x_i, weight_i in zip(x_partitions, weight_partitions, strict=True)]
        reconstructed = [
            hybrid_quant_dequant(
                partial,
                self.scales_per_partition[i],
                bf16_feature_indices=self.bf16_feature_indices,
                num_bits=self.num_bits,
                output_dtype=self.output_dtype,
            )
            for i, partial in enumerate(partials)
        ]
        y = torch.stack(reconstructed, dim=0).sum(dim=0)
        if self.bias is not None:
            y = y + self.bias.to(self.output_dtype)
        return self._validate_output_dtype(y)

    @classmethod
    def from_conv1d(
        cls,
        conv1d: Conv1D,
        num_partitions: int,
        scales_per_partition: torch.Tensor,
        bf16_feature_indices: Optional[torch.Tensor],
        num_bits: int = 4,
        output_dtype: torch.dtype = torch.float32,
    ) -> "HybridQuantizedRowParallelConv1D":
        return cls(
            weight=conv1d.weight,
            bias=conv1d.bias,
            num_partitions=num_partitions,
            scales_per_partition=scales_per_partition,
            bf16_feature_indices=bf16_feature_indices,
            num_bits=num_bits,
            output_dtype=output_dtype,
        )


class ThreeTierRowParallelLinear(HybridQuantizedRowParallelLinear):
    """Experimental BF16 + Int8 + Int4 row-parallel ``nn.Linear`` simulation."""

    def __init__(self, *args, int8_scales_per_partition: Tensor, int8_feature_indices: Tensor, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if int8_scales_per_partition.shape != self.scales_per_partition.shape:
            raise ValueError("Int8 scales must match Int4 per-partition scale shape.")
        self.register_buffer("int8_scales_per_partition", int8_scales_per_partition.detach().clone())
        self.register_buffer("int8_feature_indices", int8_feature_indices.detach().to(dtype=torch.long))

    def forward(self, x: Tensor) -> Tensor:
        parts = self._split_input(x)
        weights = self.weight.split(self.partition_size, dim=1)
        reconstructed = [
            multi_tier_quant_dequant(
                torch.matmul(x_i, w_i.transpose(0, 1)), self.scales_per_partition[i],
                self.int8_scales_per_partition[i], self.bf16_feature_indices, self.int8_feature_indices,
                self.output_dtype,
            )
            for i, (x_i, w_i) in enumerate(zip(parts, weights, strict=True))
        ]
        y = torch.stack(reconstructed, dim=0).sum(dim=0)
        if self.bias is not None:
            y = y + self.bias.to(self.output_dtype)
        return self._validate_output_dtype(y)


class ThreeTierRowParallelConv1D(HybridQuantizedRowParallelConv1D):
    """Experimental BF16 + Int8 + Int4 GPT-2 Conv1D simulation."""

    def __init__(self, *args, int8_scales_per_partition: Tensor, int8_feature_indices: Tensor, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if int8_scales_per_partition.shape != self.scales_per_partition.shape:
            raise ValueError("Int8 scales must match Int4 per-partition scale shape.")
        self.register_buffer("int8_scales_per_partition", int8_scales_per_partition.detach().clone())
        self.register_buffer("int8_feature_indices", int8_feature_indices.detach().to(dtype=torch.long))

    def forward(self, x: Tensor) -> Tensor:
        parts = self._split_input(x)
        weights = self.weight.split(self.partition_size, dim=0)
        reconstructed = [
            multi_tier_quant_dequant(
                torch.matmul(x_i, w_i), self.scales_per_partition[i], self.int8_scales_per_partition[i],
                self.bf16_feature_indices, self.int8_feature_indices, self.output_dtype,
            )
            for i, (x_i, w_i) in enumerate(zip(parts, weights, strict=True))
        ]
        y = torch.stack(reconstructed, dim=0).sum(dim=0)
        if self.bias is not None:
            y = y + self.bias.to(self.output_dtype)
        return self._validate_output_dtype(y)


class SimulatedTPLinear(nn.Module):
    """Compatibility wrapper around TensorParallelLinearSimulator."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        num_partitions: int = 2,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        weight = torch.empty(in_features, out_features, dtype=dtype or torch.float32)
        nn.init.kaiming_uniform_(weight.t(), a=5**0.5)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=weight.dtype)) if bias else None
        self.num_partitions = num_partitions

    def simulator(self) -> TensorParallelLinearSimulator:
        bias = self.bias if self.bias is not None else None
        return TensorParallelLinearSimulator(self.weight, bias, self.num_partitions)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.simulator().forward_tp_uncompressed(inputs)
