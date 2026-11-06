"""Hậu xử lý phải dùng cùng quy ước nhị phân cho bool, 0/1 và 0/255."""

import numpy as np
import pytest

from src.processing.postprocessing import cutout_on_color, cutout_rgba, invert_mask, overlay_mask


@pytest.mark.parametrize("dtype,level", [(bool, True), (np.uint8, 1), (np.uint8, 255)])
def test_invert_mask_is_a_binary_complement_for_all_mask_representations(dtype, level):
    mask = np.array([[0, level]], dtype=dtype)
    np.testing.assert_array_equal(invert_mask(mask), [[255, 0]])
    np.testing.assert_array_equal(invert_mask(invert_mask(mask)), [[0, 255]])


@pytest.mark.parametrize("mask", [np.array([[np.nan]]), np.array([[-1]]),
                                  np.array([[1j]]), np.array([["1"]]), np.empty((0, 1))])
def test_postprocessing_rejects_invalid_masks_consistently(mask):
    image = np.full(mask.shape + (3,), 50, np.uint8)
    for operation in (invert_mask, lambda m: cutout_rgba(image, m),
                      lambda m: cutout_on_color(image, m), lambda m: overlay_mask(image, m)):
        with pytest.raises(ValueError):
            operation(mask)


def test_cutouts_and_overlay_agree_on_boolean_foreground():
    image = np.full((1, 2, 3), [40, 80, 120], np.uint8)
    mask = np.array([[False, True]])
    rgba = cutout_rgba(image, mask)
    np.testing.assert_array_equal(rgba[..., 3], [[0, 255]])
    np.testing.assert_array_equal(cutout_on_color(image, mask), [[[255, 255, 255], [40, 80, 120]]])
    np.testing.assert_array_equal(overlay_mask(image, mask, alpha=1.0), [[[40, 80, 120], [235, 104, 52]]])
