"""Kiểm thử tích hợp: dữ liệu mẫu -> pipeline -> tệp xuất -> đọc lại; và giao diện Streamlit (AppTest)."""

import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.processing.pipeline import (
    KMeansParams,
    default_foreground,
    fit_ground_truth,
    kmeans_export_files,
    kmeans_outputs,
    otsu_export_files,
    prepare_input,
    run_kmeans,
    run_otsu,
)
from src.utils.image_io import load_path, parse_csv_matrix
from src.utils.metrics import compare_masks

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MANIFEST = json.loads((DATA / "samples.json").read_text(encoding="utf-8"))


def load_sample(sid, max_side=800):
    entry = next(s for s in MANIFEST if s["id"] == sid)
    prepared = prepare_input(load_path(DATA / entry["file"]), max_side)
    gt = None
    if entry["ground_truth"]:
        gt = fit_ground_truth(np.asarray(Image.open(DATA / entry["ground_truth"]).convert("L")), prepared)
    return entry, prepared, gt


def test_all_samples_exist_and_load():
    assert len(MANIFEST) >= 10
    for entry in MANIFEST:
        loaded = load_path(DATA / entry["file"])
        assert loaded.array.dtype == np.uint8 and loaded.array.size > 0
        if entry["ground_truth"]:
            gt = np.asarray(Image.open(DATA / entry["ground_truth"]))
            assert gt.shape == loaded.array.shape[:2]
            assert set(np.unique(gt).tolist()) <= {0, 255}


def test_circle_perfect_segmentation_both_methods():
    _, prepared, gt = load_sample("circle_black_on_white")
    otsu = run_otsu(prepared, polarity="dark")  # đối tượng là hình tròn đen -> lớp tối
    assert run_otsu(prepared, polarity="auto").invert  # gợi ý viền ảnh: nền trắng -> đối tượng tối
    assert compare_masks(otsu.mask, gt).iou == 1.0
    km = run_kmeans(prepared, KMeansParams(k=2, feature="gray"))
    out = kmeans_outputs(prepared, km, default_foreground(km))
    assert out.foreground_clusters == [0]  # gợi ý: cụm tối (không chạm viền) là đối tượng
    assert compare_masks(out.mask, gt).iou == 1.0


def test_textbook_csv_threshold():
    _, prepared, _ = load_sample("textbook_otsu_5x6")
    assert run_otsu(prepared).threshold_used == 1


def test_csv_sample_segmented_by_both():
    _, prepared, gt = load_sample("csv_rectangle_32x32")
    assert compare_masks(run_otsu(prepared).mask, gt).iou == 1.0
    km = run_kmeans(prepared, KMeansParams(k=2))
    assert compare_masks(kmeans_outputs(prepared, km, [1]).mask, gt).iou == 1.0


def test_uniform_image_is_handled_without_crash():
    _, prepared, _ = load_sample("uniform_gray")
    otsu = run_otsu(prepared)
    assert not otsu.result.valid and otsu.warnings
    km = run_kmeans(prepared, KMeansParams(k=4))
    assert km.core.k_effective == 1
    out = kmeans_outputs(prepared, km, default_foreground(km))
    assert out.foreground_clusters == [] and np.all(out.mask == 0)


def test_rgb_kmeans_on_multi_shapes():
    _, prepared, gt = load_sample("multi_shapes_color")
    km = run_kmeans(prepared, KMeansParams(k=5, feature="rgb"))
    out = kmeans_outputs(prepared, km, default_foreground(km))
    assert compare_masks(out.mask, gt).iou > 0.98


def test_resize_limit_applied_and_reported():
    _, prepared, gt = load_sample("rgb_regions", max_side=120)
    assert prepared.resize.resized
    assert max(prepared.image.shape[:2]) == 120
    km = run_kmeans(prepared, KMeansParams(k=3))
    assert km.labels.shape == prepared.image.shape[:2]


def test_ground_truth_resized_to_processing_size():
    _, prepared, gt = load_sample("circle_black_on_white", max_side=100)
    assert gt.shape == prepared.gray.shape == (100, 100)


def test_export_files_reopen_with_same_content():
    _, prepared, _ = load_sample("multi_shapes_color")
    params = KMeansParams(k=4)
    km = run_kmeans(prepared, params)
    out = kmeans_outputs(prepared, km, [0, 1])
    files = kmeans_export_files(prepared, params, km, out)
    seg = np.asarray(Image.open(io.BytesIO(files["kmeans_segmented.png"][0])))
    np.testing.assert_array_equal(seg, out.segmented)
    mask = np.asarray(Image.open(io.BytesIO(files["kmeans_mask.png"][0])))
    np.testing.assert_array_equal(mask, out.mask)
    cut = Image.open(io.BytesIO(files["kmeans_cutout.png"][0]))
    assert cut.mode == "RGBA" and cut.size == (prepared.image.shape[1], prepared.image.shape[0])
    labels = parse_csv_matrix(files["kmeans_labels.csv"][0].decode())
    np.testing.assert_array_equal(labels, km.labels)
    stats = json.loads(files["kmeans_stats.json"][0])
    assert stats["k_thuc_te"] == 4 and sum(stats["so_pixel_moi_cum"]) == km.labels.size

    otsu = run_otsu(prepared, polarity="dark")
    ofiles = otsu_export_files(prepared, otsu)
    np.testing.assert_array_equal(np.asarray(Image.open(io.BytesIO(ofiles["otsu_mask.png"][0]))), otsu.mask)
    matrix = parse_csv_matrix(ofiles["otsu_binary_matrix.csv"][0].decode())
    np.testing.assert_array_equal(matrix * 255, otsu.mask)
    assert json.loads(ofiles["otsu_stats.json"][0])["nguong_otsu"] == otsu.result.threshold


def test_changing_parameters_changes_result():
    _, prepared, _ = load_sample("rgb_regions")
    a = run_kmeans(prepared, KMeansParams(k=2))
    b = run_kmeans(prepared, KMeansParams(k=5))
    assert a.core.k_effective == 2 and b.core.k_effective == 5
    assert b.core.sse < a.core.sse
    manual = run_otsu(prepared, manual_threshold=50)
    auto = run_otsu(prepared)
    assert manual.threshold_used == 50 and manual.mode == "manual"
    assert not np.array_equal(manual.mask, auto.mask)


def test_no_opencv_or_library_algorithms_imported():
    import re

    banned_import = re.compile(r"^\s*(import|from)\s+(cv2|sklearn|skimage|scipy)\b", re.MULTILINE)
    banned_call = re.compile(r"\b(threshold_otsu|KMeans)\s*\(")
    files = list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").rglob("*.py")) + [ROOT / "app.py"]
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert not banned_import.search(text), f"{path} import thư viện bị cấm"
        assert not banned_call.search(text), f"{path} gọi hàm thuật toán có sẵn"
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    assert "opencv" not in req and "scikit" not in req


# ----------------------- Kiểm thử giao diện (Streamlit AppTest) -----------------------

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


def _app(page: str):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.session_state["_test_page"] = page
    return at


@pytest.mark.parametrize("page", ["overview", "kmeans", "otsu", "compare", "library", "theory", "guide"])
def test_every_page_renders_without_exception(page):
    at = _app(page).run()
    assert not at.exception, [e.value for e in at.exception]


def test_kmeans_page_run_button_and_downloads():
    at = _app("kmeans").run()
    run = [b for b in at.button if b.label.startswith("Chạy K-means")]
    assert run, "Không thấy nút chạy K-means"
    run[0].click().run()
    assert not at.exception
    assert any("Số vòng lặp" in m.label for m in at.metric)
    downloads = at.get("download_button")
    assert len(downloads) == 7
    assert {d.proto.label for d in downloads} >= {"⬇ Ảnh phân đoạn (PNG)", "⬇ Mask nhị phân (PNG)", "⬇ Ma trận nhãn cụm (CSV)"}
    # Thay đổi K rồi chạy lại: số cụm hiển thị phải đổi theo
    at.slider(key="km_k").set_value(4)  # widget trong form: đặt giá trị rồi submit cùng một lần chạy
    [b for b in at.button if b.label.startswith("Chạy K-means")][0].click()
    at.run()
    assert not at.exception
    k_metric = next(m for m in at.metric if m.label.startswith("Số cụm"))
    assert k_metric.value.startswith("4")


def test_otsu_page_shows_threshold():
    at = _app("otsu").run()
    [b for b in at.button if b.label.startswith("Chạy Otsu")][0].click().run()
    assert not at.exception
    assert any(m.label.startswith("Ngưỡng Otsu") for m in at.metric)
    assert len(at.get("download_button")) == 7


def test_compare_page_without_ground_truth_does_not_invent_scores():
    at = _app("compare")
    at.session_state["sample_id"] = "rgb_regions"  # ảnh mẫu không có ground truth
    at.run()
    [b for b in at.button if b.label.startswith("Chạy K-means")][0].click()
    at.run()
    assert not at.exception
    assert any("không tính" in i.value for i in at.info)
    table_cols = [list(df.value.columns) for df in at.dataframe]
    assert not any("IoU" in cols for cols in table_cols)


def test_compare_page_with_ground_truth_reports_iou():
    at = _app("compare")
    at.session_state["sample_id"] = "circle_black_on_white"
    at.run()
    at.slider(key="cmp_k").set_value(2)
    [b for b in at.button if b.label.startswith("Chạy K-means")][0].click()
    at.run()
    assert not at.exception
    assert any("IoU" in list(df.value.columns) for df in at.dataframe)


def test_invalid_csv_upload_shows_error_message():
    at = _app("otsu")
    at.session_state["input_source"] = "Tải tệp lên"
    at.session_state["_test_upload"] = ("invalid_text.csv", (DATA / "sample_csv" / "invalid_text.csv").read_bytes())
    at.run()
    assert not at.exception
    assert any("không phải số" in e.value for e in at.error)
