"""Hồi quy từ review: mask xuất/nhập và gom pixel xám ở nhiều kiểu dữ liệu."""

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from src.algorithms.kmeans import unique_pixels
from src.processing.pipeline import otsu_export_files, prepare_input, run_otsu
from src.utils.image_io import InputError, load_mask_bytes, load_path, matrix_to_csv_bytes, to_png_bytes

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("suffix", ["png", "csv"])
@pytest.mark.parametrize("pixels", [
    np.array([[0, 1, 1], [1, 0, 1]], np.uint8),
    np.zeros((2, 3), np.uint8),
    np.ones((2, 3), np.uint8),
])
def test_logical_mask_files_preserve_foreground(suffix, pixels):
    data = to_png_bytes(pixels) if suffix == "png" else matrix_to_csv_bytes(pixels)
    np.testing.assert_array_equal(load_mask_bytes(data, f"mask.{suffix}"), pixels * 255)


@pytest.mark.parametrize("pixels", [[[0, 1, 1]], [[1, 1, 1]], [[0, 0, 0]]])
def test_16bit_logical_mask_is_not_quantized_to_background(pixels):
    buffer = io.BytesIO()
    Image.fromarray(np.array(pixels, np.uint16)).save(buffer, format="PNG")
    np.testing.assert_array_equal(load_mask_bytes(buffer.getvalue()), np.array(pixels) * 255)


@pytest.mark.parametrize("pixels", [[[0, 256]], [[256, 256]], [[1, 300]]])
def test_dark_16bit_gray_mask_is_not_misidentified_as_logical_after_quantization(pixels):
    buffer = io.BytesIO()
    Image.fromarray(np.array(pixels, np.uint16)).save(buffer, format="PNG")
    np.testing.assert_array_equal(load_mask_bytes(buffer.getvalue()), np.zeros((1, 2), np.uint8))


def test_gray_mask_image_still_uses_absolute_brightness_outside_logical_encoding():
    pixels = np.array([[0, 1, 127, 128, 200, 255]], np.uint8)
    np.testing.assert_array_equal(load_mask_bytes(to_png_bytes(pixels)), [[0, 0, 0, 255, 255, 255]])


@pytest.mark.parametrize("pixels", [[[0, 1, 2]], [[0, 127, 255]], [[0, 1, 255]]])
def test_ground_truth_csv_rejects_cluster_labels_or_grayscale_values(pixels):
    with pytest.raises(InputError, match="mask nhị phân"):
        load_mask_bytes(matrix_to_csv_bytes(np.array(pixels)), "labels.csv")


def test_ground_truth_csv_preserves_0255_mask():
    pixels = np.array([[0, 255, 255], [255, 0, 255]], np.uint8)
    np.testing.assert_array_equal(load_mask_bytes(matrix_to_csv_bytes(pixels), "mask.csv"), pixels)


def test_otsu_exported_csv_can_be_used_as_the_same_mask():
    prepared = prepare_input(load_path(ROOT / "data/sample_images/circle_black_on_white.png"), 800)
    out = run_otsu(prepared, polarity="dark")
    files = otsu_export_files(prepared, out)
    mask = load_mask_bytes(files["otsu_binary_matrix.csv"][0], "otsu_binary_matrix.csv")
    assert mask.any()
    np.testing.assert_array_equal(mask, out.mask)


@pytest.mark.parametrize("suffix", ["png", "csv"])
def test_comparison_accepts_logical_ground_truth_and_reports_real_scores(suffix):
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    app.session_state["_test_page"] = "compare"
    app.session_state["sample_id"] = "circle_black_on_white"
    app.run()
    next(b for b in app.button if b.label.startswith("Chạy K-means")).click().run()
    gt = load_mask_bytes((ROOT / "data/ground_truth/circle_black_on_white_gt.png").read_bytes())
    pixels = gt // 255
    content = to_png_bytes(pixels) if suffix == "png" else matrix_to_csv_bytes(pixels)
    mime = "image/png" if suffix == "png" else "text/csv"
    app.file_uploader[0].set_value((f"logical_gt.{suffix}", content, mime)).run()
    assert not app.exception and not app.error
    assert any(f"ground truth tải lên (logical_gt.{suffix})" in c.value for c in app.caption)
    rows = next(df.value for df in app.dataframe if "IoU" in df.value.columns)
    np.testing.assert_array_equal(rows["IoU"], [1.0, 1.0])
    np.testing.assert_array_equal(app.session_state["cmp_uploaded_gt"]["mask"], gt)


def test_comparison_rejects_multiclass_csv_without_using_sample_ground_truth():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    app.session_state["_test_page"] = "compare"
    app.session_state["sample_id"] = "circle_black_on_white"
    app.run()
    next(b for b in app.button if b.label.startswith("Chạy K-means")).click().run()
    assert any("IoU" in df.value.columns for df in app.dataframe)
    app.file_uploader[0].set_value(("cluster_labels.csv", b"0,1,2\n2,1,0", "text/csv")).run()
    assert not app.exception
    assert any("mask nhị phân" in error.value for error in app.error)
    assert not any("IoU" in df.value.columns for df in app.dataframe)


@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.float64, np.bool_])
def test_grayscale_grouping_preserves_noncontiguous_input_and_inverse(dtype):
    pixels = np.random.default_rng(16).integers(0, 256, (37, 53), dtype=np.uint8).astype(dtype)
    features = pixels.T.reshape(-1, 1)[::-2]  # negative stride, repeated and missing gray levels
    original = features.copy()
    values, inverse, counts = unique_pixels(features)
    expected = np.unique(features, axis=0, return_inverse=True, return_counts=True)
    np.testing.assert_array_equal(values, expected[0])
    np.testing.assert_array_equal(inverse, expected[1].ravel())
    np.testing.assert_array_equal(counts, expected[2])
    np.testing.assert_array_equal(values[inverse], features)
    np.testing.assert_array_equal(features, original)
