"""Kiểm tra biên tiền xử lý để không làm biến đổi âm thầm dữ liệu sai."""

import numpy as np
import pytest

from src.processing.preprocessing import limit_size, resize_to, to_grayscale


@pytest.mark.parametrize("image", [
    np.array([[-1, 256]]), np.array([[1.5]]), np.array([[np.nan]]),
    np.array([[np.inf]]), np.array([[1j]]), np.array([["1"]]),
    np.empty((0, 2), np.uint8), np.zeros((2,), np.uint8), np.zeros((2, 2, 4), np.uint8),
])
def test_preprocessing_rejects_invalid_images_without_casting(image):
    for operation in (to_grayscale, lambda x: limit_size(x, 800), lambda x: resize_to(x, (1, 1))):
        with pytest.raises(ValueError):
            operation(image)


@pytest.mark.parametrize("max_side", [True, 1.5, 0, -1, np.nan, "800"])
def test_limit_size_rejects_invalid_limits_even_without_resize(max_side):
    with pytest.raises(ValueError):
        limit_size(np.zeros((2, 2), np.uint8), max_side)


@pytest.mark.parametrize("size", [(0, 1), (-1, 1), (1.5, 2), (True, 2), (1,), (1, 2, 3), None])
def test_resize_rejects_invalid_target_sizes(size):
    with pytest.raises(ValueError):
        resize_to(np.zeros((2, 2), np.uint8), size)


def test_invalid_methods_are_rejected_even_when_no_conversion_is_needed():
    image = np.zeros((2, 2), np.uint8)
    with pytest.raises(ValueError):
        to_grayscale(image, "unknown")
    with pytest.raises(ValueError):
        limit_size(image, 800, "unknown")
    with pytest.raises(ValueError):
        resize_to(image, (2, 2), "unknown")


def test_valid_integer_arrays_are_preserved_and_nearest_resize_keeps_mask_values():
    gray = np.array([[0, 128], [255, 40]], dtype=np.int16)
    np.testing.assert_array_equal(to_grayscale(gray), gray)
    assert to_grayscale(gray).dtype == np.uint8
    mask = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    resized = resize_to(mask, (7, 5), "nearest")
    assert resized.shape == (5, 7)
    assert set(np.unique(resized)) == {0, 255}
