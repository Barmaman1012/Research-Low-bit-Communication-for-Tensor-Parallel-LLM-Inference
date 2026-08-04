"""Deterministic calibration-sequence preparation shared by scripts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal

import torch


SamplingStrategy = Literal["legacy_first_records", "random_token_chunks"]
VALID_SAMPLING_STRATEGIES = ("legacy_first_records", "random_token_chunks")


@dataclass(slots=True)
class PreparedCalibrationData:
    """Model-ready sequences and compact provenance for a calibration pass."""

    inputs: list[dict[str, torch.Tensor]]
    selected_chunk_ids: list[int]
    total_available_chunks: int
    padding_used: bool
    separator_token_policy: str


def _nonempty_texts(dataset: Iterable[dict[str, Any]]) -> list[str]:
    return [text for row in dataset if isinstance((text := row.get("text", "")), str) and text.strip()]


def _legacy_inputs(tokenizer: Any, texts: list[str], sequence_length: int) -> list[dict[str, torch.Tensor]]:
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    return [
        {
            key: value
            for key, value in tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=sequence_length,
            ).items()
            if isinstance(value, torch.Tensor)
        }
        for text in texts
    ]


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    values = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if isinstance(values, torch.Tensor):
        values = values.tolist()
    return list(values)


def prepare_calibration_data(
    dataset: Iterable[dict[str, Any]],
    tokenizer: Any,
    *,
    num_sequences: int,
    sequence_length: int,
    sampling_strategy: SamplingStrategy,
    seed: int,
    excluded_chunk_ids: set[int] | None = None,
) -> PreparedCalibrationData:
    """Prepare legacy padded records or seeded exact token chunks.

    Random chunks are selected without replacement in the seeded ``randperm``
    order; that order is intentionally retained because EMA is order-sensitive.
    """

    if sampling_strategy not in VALID_SAMPLING_STRATEGIES:
        raise ValueError(f"Unsupported sampling strategy: {sampling_strategy!r}.")
    if num_sequences <= 0 or sequence_length <= 0:
        raise ValueError("num_sequences and sequence_length must be positive.")

    texts = _nonempty_texts(dataset)
    if sampling_strategy == "legacy_first_records":
        selected = texts[:num_sequences]
        if len(selected) < num_sequences:
            raise ValueError(f"Requested {num_sequences} non-empty sequences, found only {len(selected)}.")
        return PreparedCalibrationData(
            inputs=_legacy_inputs(tokenizer, selected, sequence_length),
            selected_chunk_ids=list(range(num_sequences)),
            total_available_chunks=len(texts),
            padding_used=True,
            separator_token_policy="none (legacy individual records)",
        )

    separator_id = tokenizer.eos_token_id
    if separator_id is None:
        separator_id = tokenizer.sep_token_id
    if separator_id is None:
        raise ValueError("random_token_chunks requires tokenizer.eos_token_id or tokenizer.sep_token_id.")
    stream: list[int] = []
    for index, text in enumerate(texts):
        if index:
            stream.append(int(separator_id))
        stream.extend(_token_ids(tokenizer, text))
    total_chunks = len(stream) // sequence_length
    excluded = excluded_chunk_ids or set()
    candidates = [chunk_id for chunk_id in range(total_chunks) if chunk_id not in excluded]
    if len(candidates) < num_sequences:
        raise ValueError(
            f"Requested {num_sequences} chunks, but only {len(candidates)} are available after exclusions."
        )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(len(candidates), generator=generator)[:num_sequences].tolist()
    selected_ids = [candidates[index] for index in order]
    inputs = []
    for chunk_id in selected_ids:
        start = chunk_id * sequence_length
        ids = torch.tensor(stream[start : start + sequence_length], dtype=torch.long).unsqueeze(0)
        inputs.append({"input_ids": ids, "attention_mask": torch.ones_like(ids)})
    return PreparedCalibrationData(
        inputs=inputs,
        selected_chunk_ids=selected_ids,
        total_available_chunks=total_chunks,
        padding_used=False,
        separator_token_policy="eos_between_nonempty_records",
    )
