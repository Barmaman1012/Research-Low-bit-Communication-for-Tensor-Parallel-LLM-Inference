from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lowbit_tp_comm.calibration import EMAMinMaxCalibrator


def test_first_update_initializes_min_and_max_directly() -> None:
    calibrator = EMAMinMaxCalibrator(num_partitions=2, feature_dim=3, gamma=0.1)
    partials = [
        torch.tensor([[1.0, -2.0, 3.0], [0.5, 4.0, -1.0]], dtype=torch.float32),
        torch.tensor([[-3.0, 1.0, 2.0], [2.0, -4.0, 5.0]], dtype=torch.float32),
    ]

    calibrator.update(partials)

    expected_min = torch.tensor([[0.5, -2.0, -1.0], [-3.0, -4.0, 2.0]], dtype=torch.float32)
    expected_max = torch.tensor([[1.0, 4.0, 3.0], [2.0, 1.0, 5.0]], dtype=torch.float32)
    assert calibrator.initialized is True
    assert torch.equal(calibrator.min_vals, expected_min)
    assert torch.equal(calibrator.max_vals, expected_max)


def test_second_update_applies_ema_correctly() -> None:
    calibrator = EMAMinMaxCalibrator(num_partitions=2, feature_dim=2, gamma=0.25)
    first = [
        torch.tensor([[1.0, -2.0], [3.0, 4.0]], dtype=torch.float32),
        torch.tensor([[-1.0, 5.0], [2.0, -3.0]], dtype=torch.float32),
    ]
    second = [
        torch.tensor([[-3.0, 6.0], [0.0, 1.0]], dtype=torch.float32),
        torch.tensor([[4.0, -2.0], [1.0, 7.0]], dtype=torch.float32),
    ]

    calibrator.update(first)
    calibrator.update(second)

    first_min = torch.tensor([[1.0, -2.0], [-1.0, -3.0]], dtype=torch.float32)
    first_max = torch.tensor([[3.0, 4.0], [2.0, 5.0]], dtype=torch.float32)
    second_min = torch.tensor([[-3.0, 1.0], [1.0, -2.0]], dtype=torch.float32)
    second_max = torch.tensor([[0.0, 6.0], [4.0, 7.0]], dtype=torch.float32)
    expected_min = 0.75 * first_min + 0.25 * second_min
    expected_max = 0.75 * first_max + 0.25 * second_max

    assert torch.allclose(calibrator.min_vals, expected_min)
    assert torch.allclose(calibrator.max_vals, expected_max)


def test_ranges_shape_is_num_partitions_by_feature_dim() -> None:
    calibrator = EMAMinMaxCalibrator(num_partitions=3, feature_dim=4)
    partials = [torch.randn(2, 5, 4) for _ in range(3)]

    calibrator.update(partials)

    assert calibrator.ranges().shape == (3, 4)


def test_aggregated_ranges_shape_is_feature_dim() -> None:
    calibrator = EMAMinMaxCalibrator(num_partitions=2, feature_dim=5)
    partials = [torch.randn(4, 5), torch.randn(4, 5)]

    calibrator.update(partials)

    assert calibrator.aggregated_ranges().shape == (5,)


def test_topk_features_identifies_known_outlier() -> None:
    calibrator = EMAMinMaxCalibrator(num_partitions=2, feature_dim=4)
    partials = [
        torch.tensor([[1.0, -2.0, 10.0, 0.5], [-1.0, 1.0, -9.0, -0.5]], dtype=torch.float32),
        torch.tensor([[0.5, 0.0, 8.0, -1.0], [-0.25, 0.2, -7.0, 1.0]], dtype=torch.float32),
    ]

    calibrator.update(partials)

    top_feature = calibrator.topk_features(1)
    assert torch.equal(top_feature, torch.tensor([2], dtype=torch.long))


def test_topk_features_with_zero_returns_empty_tensor() -> None:
    calibrator = EMAMinMaxCalibrator(num_partitions=1, feature_dim=3)

    result = calibrator.topk_features(0)

    assert result.dtype == torch.long
    assert result.numel() == 0


def test_scales_per_partition_returns_positive_finite_scales() -> None:
    calibrator = EMAMinMaxCalibrator(num_partitions=2, feature_dim=3)
    partials = [
        torch.tensor([[1.0, -2.0, 3.0], [0.5, 4.0, -1.0]], dtype=torch.float32),
        torch.tensor([[-3.0, 1.0, 2.0], [2.0, -4.0, 5.0]], dtype=torch.float32),
    ]

    calibrator.update(partials)
    scales = calibrator.scales_per_partition()

    assert scales.shape == (2, 3)
    assert torch.isfinite(scales).all()
    assert torch.all(scales > 0)


def test_invalid_input_shapes_raise_value_error() -> None:
    calibrator = EMAMinMaxCalibrator(num_partitions=2, feature_dim=3)

    with pytest.raises(ValueError):
        calibrator.update([torch.randn(4, 3)])

    with pytest.raises(ValueError):
        calibrator.update([torch.randn(4, 3), torch.randn(4, 4)])

    with pytest.raises(ValueError):
        calibrator.update([torch.randn(3), torch.randn(3, 3)])


def test_state_dict_round_trip_restores_calibrator() -> None:
    calibrator = EMAMinMaxCalibrator(num_partitions=2, feature_dim=3, gamma=0.2)
    partials = [
        torch.tensor([[1.0, -2.0, 3.0], [0.5, 4.0, -1.0]], dtype=torch.float32),
        torch.tensor([[-3.0, 1.0, 2.0], [2.0, -4.0, 5.0]], dtype=torch.float32),
    ]
    calibrator.update(partials)

    restored = EMAMinMaxCalibrator.from_state_dict(calibrator.state_dict())

    assert restored.gamma == calibrator.gamma
    assert restored.num_partitions == calibrator.num_partitions
    assert restored.feature_dim == calibrator.feature_dim
    assert restored.initialized == calibrator.initialized
    assert torch.equal(restored.min_vals, calibrator.min_vals)
    assert torch.equal(restored.max_vals, calibrator.max_vals)
