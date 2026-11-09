"""Kiểm tra hợp đồng dữ liệu giữa thuật toán, pipeline và tệp người dùng tải về."""

import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from src.processing.pipeline import (
    KMeansParams,
    MaskPostprocess,
    kmeans_outputs,
    prepare_input,
    run_kmeans,
)
from src.processing.postprocessing import mask_from_labels
from src.utils.image_io import LoadedImage, parse_csv_matrix

ROOT = Path(__file__).resolve().parents[1]


def prepared_gray():
    pixels = np.array([[0, 0, 200, 200], [0, 0, 200, 200]], np.uint8)
    return prepare_input(LoadedImage(pixels, "two.png", "PNG", 0, "L"), 800)


@pytest.mark.parametrize("selection", [[0.5], [True], ["1"], [-1], [2], None])
def test_pipeline_rejects_invalid_foreground_instead_of_changing_selection(selection):
    prepared = prepared_gray()
    result = run_kmeans(prepared, KMeansParams(k=2))
    with pytest.raises(ValueError, match="cụm"):
        kmeans_outputs(prepared, result, selection)


@pytest.mark.parametrize("labels", [np.array([0, 1]), np.array([[0.5, 1]]),
                                    np.array([[-1, 1]]), np.empty((0, 2), int)])
def test_label_masks_require_nonempty_integer_matrices(labels):
    with pytest.raises(ValueError, match="Nhãn"):
        mask_from_labels(labels, [1])


@pytest.mark.parametrize("size", [True, 1.5, 0, -1, 2])
@pytest.mark.parametrize("operation", ["none", "open"])
def test_postprocess_rejects_invalid_window_at_construction(size, operation):
    with pytest.raises(ValueError, match="lẻ"):
        MaskPostprocess(operation, size)


@pytest.mark.parametrize("median_size", [True, 1.0, np.float64(1)])
def test_no_filter_setting_still_requires_integer_window(median_size):
    prepared = prepared_gray()
    loaded = LoadedImage(prepared.image, "two.png", "PNG", 0, "L")
    with pytest.raises(ValueError):
        prepare_input(loaded, 800, median_size=median_size)


def test_unknown_feature_is_rejected_also_for_grayscale():
    with pytest.raises(ValueError, match="đặc trưng"):
        run_kmeans(prepared_gray(), KMeansParams(feature="invalid"))


@pytest.mark.parametrize("text", ["١,٢,٢٥٥\n０,１２,２５５", "007,+12,3.0\n0,255,001"])
def test_csv_conversion_preserves_valid_noncanonical_integer_notation(text):
    expected = [[1, 2, 255], [0, 12, 255]] if text.startswith("١") else [[7, 12, 3], [0, 255, 1]]
    actual = parse_csv_matrix(text)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.uint8


def compare_app():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    app.session_state["_test_page"] = "compare"
    app.run()
    next(b for b in app.button if b.label.startswith("Chạy K-means")).click().run()
    assert not app.exception
    return app


def test_compare_exports_both_current_masks_and_refreshes_selection():
    app = compare_app()
    app.multiselect[0].set_value([]).run()
    assert not app.exception
    files = app.session_state["cmp_result"]["_export_cache"]["files"]
    with ZipFile(io.BytesIO(files["comparison_results.zip"][0])) as archive:
        km = np.asarray(Image.open(io.BytesIO(archive.read("kmeans_mask.png"))))
        otsu = np.asarray(Image.open(io.BytesIO(archive.read("otsu_mask.png"))))
        assert not km.any()
        np.testing.assert_array_equal(otsu, app.session_state["cmp_result"]["otsu"].mask)
        assert json.loads(archive.read("kmeans_stats.json"))["cum_doi_tuong"] == []
        np.testing.assert_array_equal(
            parse_csv_matrix(archive.read("otsu_binary_matrix.csv").decode()) * 255, otsu
        )
    app.radio(key="cmp_otsu_polarity").set_value("Lớp tối (≤ t)").run()
    files = app.session_state["cmp_result"]["_export_cache"]["files"]
    with ZipFile(io.BytesIO(files["comparison_results.zip"][0])) as archive:
        assert json.loads(archive.read("otsu_stats.json"))["lua_chon_cuc_tinh"] == "dark"
    app.selectbox(key="sample_id").set_value("uniform_gray").run()
    assert not app.exception and not app.get("download_button")


def test_otsu_controls_explain_which_settings_apply():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    app.session_state["_test_page"] = "otsu"
    app.run()
    assert app.slider(key="otsu_manual_t").disabled
    app.radio(key="otsu_mode").set_value("Thủ công (đối chiếu)").run()
    assert not app.slider(key="otsu_manual_t").disabled
    app.radio(key="otsu_classes").set_value("3 lớp — Multi-Otsu (2 ngưỡng)").run()
    assert app.radio(key="otsu_mode").disabled
    assert app.radio(key="otsu_polarity").disabled
    assert app.slider(key="otsu_manual_t").disabled
    assert not app.exception
