"""Đối chiếu điểm mask với kết quả tính tay và từ chối dữ liệu không hợp lệ."""

import numpy as np
import pytest

from src.utils.metrics import compare_masks, foreground_ratio


def test_mask_scores_match_hand_computed_confusion_matrix():
    prediction = np.array([[1, 1, 0], [0, 0, 1]])
    reference = np.array([[255, 0, 255], [0, 0, 255]])
    scores = compare_masks(prediction, reference)
    assert (scores.tp, scores.fp, scores.fn, scores.tn) == (2, 1, 1, 2)
    assert scores.iou == pytest.approx(1 / 2)
    assert scores.dice == scores.pixel_accuracy == pytest.approx(2 / 3)
    assert foreground_ratio(prediction) == pytest.approx(1 / 2)


def test_all_background_masks_are_valid_and_have_perfect_agreement():
    mask = np.zeros((2, 3), np.uint8)
    scores = compare_masks(mask, mask)
    assert scores.iou == scores.dice == scores.pixel_accuracy == 1.0
    assert foreground_ratio(mask) == 0.0


@pytest.mark.parametrize("mask", [
    np.empty((0, 0)), np.array([[np.nan]]), np.array([[np.inf]]),
    np.array([[-1]]), np.array([[1j]]), np.array([["1"]]),
    np.zeros((2, 2, 3)), np.array([0, 1]),
])
def test_metrics_reject_invalid_masks(mask):
    with pytest.raises(ValueError):
        compare_masks(mask, mask)
    with pytest.raises(ValueError):
        foreground_ratio(mask)


def test_different_mask_dimensions_are_not_broadcast():
    with pytest.raises(ValueError, match="không khớp"):
        compare_masks(np.zeros((2, 3)), np.zeros((3, 2)))
