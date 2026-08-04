from __future__ import annotations

import torch

from lowbit_tp_comm.calibration import EMAMinMaxCalibrator
from lowbit_tp_comm.calibration_data import prepare_calibration_data


class Tokenizer:
    eos_token = "<eos>"
    eos_token_id = 99
    sep_token_id = None
    pad_token = None

    def __call__(self, text, **kwargs):
        if kwargs.get("add_special_tokens") is False:
            return {"input_ids": [ord(char) for char in text]}
        values = [ord(char) for char in text]
        length = kwargs["max_length"]
        values = (values[:length] + [self.eos_token_id] * length)[:length]
        return {"input_ids": torch.tensor([values]), "attention_mask": torch.tensor([[1] * min(len(text), length) + [0] * max(0, length - len(text))])}


DATASET = [{"text": "ab"}, {"text": ""}, {"text": "cd"}, {"text": "ef"}, {"text": "gh"}]


def test_random_token_chunks_are_seeded_unique_exact_and_drop_tail() -> None:
    first = prepare_calibration_data(DATASET, Tokenizer(), num_sequences=2, sequence_length=3, sampling_strategy="random_token_chunks", seed=4)
    second = prepare_calibration_data(DATASET, Tokenizer(), num_sequences=2, sequence_length=3, sampling_strategy="random_token_chunks", seed=4)
    changed = prepare_calibration_data(DATASET, Tokenizer(), num_sequences=2, sequence_length=3, sampling_strategy="random_token_chunks", seed=5)

    assert first.selected_chunk_ids == second.selected_chunk_ids
    assert first.selected_chunk_ids != changed.selected_chunk_ids
    assert len(set(first.selected_chunk_ids)) == 2
    assert first.total_available_chunks == 3  # 11 stream tokens, final two are dropped.
    assert first.padding_used is False
    assert all(sample["input_ids"].shape == (1, 3) for sample in first.inputs)
    assert all(torch.equal(sample["attention_mask"], torch.ones_like(sample["input_ids"])) for sample in first.inputs)


def test_legacy_first_records_keeps_padded_record_semantics() -> None:
    prepared = prepare_calibration_data(DATASET, Tokenizer(), num_sequences=2, sequence_length=4, sampling_strategy="legacy_first_records", seed=999)

    assert prepared.selected_chunk_ids == [0, 1]
    assert prepared.padding_used is True
    assert prepared.inputs[0]["attention_mask"].tolist() == [[1, 1, 0, 0]]


def test_random_chunks_present_only_real_positions_to_ema() -> None:
    prepared = prepare_calibration_data(DATASET, Tokenizer(), num_sequences=1, sequence_length=3, sampling_strategy="random_token_chunks", seed=0)
    calibrator = EMAMinMaxCalibrator(num_partitions=1, feature_dim=1)
    values = prepared.inputs[0]["input_ids"].float().unsqueeze(-1)
    calibrator.update([values])

    assert calibrator.min_vals.item() == values.min().item()
    assert calibrator.max_vals.item() == values.max().item()
