"""Kiểm thử tích hợp: dữ liệu mẫu -> pipeline -> tệp xuất -> đọc lại; và giao diện Streamlit (AppTest)."""

import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.processing.pipeline import (
    TIMING_SCOPE,
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
from src.utils.image_io import load_image_bytes, load_path, parse_csv_matrix, to_png_bytes
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


@pytest.mark.parametrize("polarity", ["bright", "dark", "auto"])
def test_uniform_auto_otsu_always_has_empty_foreground(polarity):
    _, prepared, _ = load_sample("uniform_gray")
    out = run_otsu(prepared, polarity=polarity)
    assert not out.result.valid and out.mode == "otsu"
    assert not out.invert and out.foreground_ratio == 0.0
    assert np.all(out.binary == 0) and np.all(out.mask == 0)
    assert np.all(out.cutout_rgba[..., 3] == 0)
    assert any("mask tự động" in w for w in out.warnings)
    files = otsu_export_files(prepared, out)
    stats = json.loads(files["otsu_stats.json"][0])
    assert stats["canh_bao"] == out.warnings
    assert stats["ty_le_foreground"] == 0.0


def test_uniform_manual_threshold_still_allows_explicit_selection():
    _, prepared, _ = load_sample("uniform_gray")
    before = prepared.image.copy()
    out = run_otsu(prepared, polarity="dark", manual_threshold=200)
    assert out.mode == "manual" and out.invert
    assert np.all(out.mask == 255) and out.foreground_ratio == 1.0
    np.testing.assert_array_equal(prepared.image, before)


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
    assert stats["pham_vi_thoi_gian"] == TIMING_SCOPE

    otsu = run_otsu(prepared, polarity="dark")
    ofiles = otsu_export_files(prepared, otsu)
    np.testing.assert_array_equal(np.asarray(Image.open(io.BytesIO(ofiles["otsu_mask.png"][0]))), otsu.mask)
    matrix = parse_csv_matrix(ofiles["otsu_binary_matrix.csv"][0].decode())
    np.testing.assert_array_equal(matrix * 255, otsu.mask)
    assert json.loads(ofiles["otsu_stats.json"][0])["nguong_otsu"] == otsu.result.threshold
    assert json.loads(ofiles["otsu_stats.json"][0])["pham_vi_thoi_gian"] == TIMING_SCOPE


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


@pytest.mark.parametrize("data", [b"9" * 400, b"1.00000000000000001", b'"1,2'])
def test_extreme_csv_upload_is_reported_without_exception(data):
    at = _app("otsu")
    at.session_state["input_source"] = "Tải tệp lên"
    at.session_state["_test_upload"] = ("invalid.csv", data)
    at.run()
    assert not at.exception and at.error


@pytest.mark.parametrize("threshold", [1.5, "128", True])
def test_pipeline_does_not_truncate_invalid_manual_threshold(threshold):
    _, prepared, _ = load_sample("uniform_gray")
    with pytest.raises(ValueError):
        run_otsu(prepared, manual_threshold=threshold)


@pytest.mark.parametrize("sid,min_kmeans,min_otsu", [
    ("circle_black_on_white", 1.0, 1.0), ("bright_rect_on_dark", 1.0, 1.0),
    ("multi_shapes_color", 1.0, 0.79), ("gray_levels_objects", 1.0, 0.66),
    ("noisy_circle", 0.83, 0.83), ("low_contrast", 0.95, 0.95),
    ("tiny_10x10", 1.0, 1.0), ("csv_rectangle_32x32", 1.0, 1.0),
])
def test_reference_samples_keep_expected_segmentation(sid, min_kmeans, min_otsu):
    entry, prepared, gt = load_sample(sid)
    before = prepared.image.copy()
    params = KMeansParams(**entry["suggested"])
    result = run_kmeans(prepared, params)
    out = kmeans_outputs(prepared, result, default_foreground(result))
    otsu = run_otsu(prepared, polarity="auto")
    assert result.core.converged
    assert compare_masks(out.mask, gt).iou >= min_kmeans
    assert compare_masks(otsu.mask, gt).iou >= min_otsu
    np.testing.assert_array_equal(prepared.image, before)


def test_large_encoded_rgb_image_is_resized_and_exported_at_reported_size():
    image = np.full((3000, 4000, 3), 240, np.uint8)
    image[800:2200, 1200:2800] = (30, 60, 90)
    gt = np.zeros((3000, 4000), np.uint8)
    gt[800:2200, 1200:2800] = 255
    prepared = prepare_input(load_image_bytes(to_png_bytes(image), "large.png"), 800)
    gt = fit_ground_truth(gt, prepared)
    assert prepared.resize.original_size == (4000, 3000)
    assert prepared.resize.processed_size == (800, 600)
    params = KMeansParams(k=2)
    result = run_kmeans(prepared, params)
    out = kmeans_outputs(prepared, result, default_foreground(result))
    otsu = run_otsu(prepared, polarity="auto")
    assert compare_masks(out.mask, gt).iou == compare_masks(otsu.mask, gt).iou == 1.0
    for files in (kmeans_export_files(prepared, params, result, out), otsu_export_files(prepared, otsu)):
        for name, (data, mime) in files.items():
            if mime == "image/png":
                assert Image.open(io.BytesIO(data)).size == (800, 600), name


@pytest.mark.parametrize("sid,max_iter,warning", [("uniform_gray", 100, "không chia được hai lớp"),
                                                ("rgb_regions", 1, "Chưa hội tụ")])
def test_compare_surfaces_algorithm_warnings(sid, max_iter, warning):
    at = _app("compare")
    at.session_state["sample_id"] = sid
    at.run()
    at.number_input(key="cmp_max_iter").set_value(max_iter)
    next(b for b in at.button if b.label.startswith("Chạy K-means")).click().run()
    assert not at.exception
    assert any(warning in w.value for w in at.warning)


@pytest.mark.parametrize("page,function", [("kmeans", "run_kmeans"), ("otsu", "run_otsu"),
                                          ("compare", "run_kmeans")])
def test_processing_memory_failure_is_reported_without_stale_downloads(monkeypatch, page, function):
    import importlib

    at = _app(page).run()
    label = "Chạy Otsu" if page == "otsu" else "Chạy K-means"
    next(b for b in at.button if b.label.startswith(label)).click().run()

    def fail(*args, **kwargs):
        raise MemoryError("Không đủ bộ nhớ kiểm thử")

    monkeypatch.setattr(importlib.import_module(f"src.ui.page_{page}"), function, fail)
    next(b for b in at.button if b.label.startswith(label)).click().run()
    assert not at.exception and at.error
    assert len(at.get("download_button")) == 0


def test_active_dependencies_are_compatible_and_do_not_require_opencv():
    from importlib import metadata
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    roots = [Requirement(line.split("#", 1)[0].strip()) for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
             if line.split("#", 1)[0].strip()]
    for requirement in roots:
        assert metadata.version(requirement.name) in requirement.specifier
    queue, seen = [canonicalize_name(requirement.name) for requirement in roots], set()
    while queue:
        name = queue.pop()
        if name in seen:
            continue
        assert "opencv" not in name and name != "cv2", name
        seen.add(name)
        for raw in metadata.requires(name) or []:
            requirement = Requirement(raw)
            if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
                queue.append(canonicalize_name(requirement.name))


def test_rerender_reuses_current_decoded_image(monkeypatch):
    from src.ui import state

    calls, original = [], state._load

    def tracked(data, name):
        calls.append(name)
        return original(data, name)

    monkeypatch.setattr(state, "_load", tracked)
    at = _app("kmeans").run()
    loaded = at.session_state["_input_ctx"].loaded
    at.run()
    at.selectbox(key="gray_method").set_value("average").run()
    assert not at.exception and len(calls) == 1
    assert at.session_state["_input_ctx"].loaded is loaded
    at.selectbox(key="sample_id").set_value("bright_rect_on_dark").run()
    assert not at.exception and len(calls) == 2


def test_input_change_releases_results_and_returning_to_old_image_does_not_restore_downloads():
    at = _app("kmeans").run()
    next(b for b in at.button if b.label.startswith("Chạy K-means")).click().run()
    assert len(at.get("download_button")) == 7
    assert "_export_cache" in at.session_state["km_result"]
    at.selectbox(key="sample_id").set_value("bright_rect_on_dark").run()
    assert not at.exception and len(at.get("download_button")) == 0
    assert set(at.session_state["km_result"]) == {"signature", "_stale"}
    at.selectbox(key="sample_id").set_value("multi_shapes_color").run()
    assert not at.exception and len(at.get("download_button")) == 0
    assert any("kết quả cũ" in w.value for w in at.warning)
    next(b for b in at.button if b.label.startswith("Chạy K-means")).click().run()
    assert not at.exception and len(at.get("download_button")) == 7


def test_upload_filename_change_invalidates_export_metadata():
    data = (DATA / "sample_images" / "bright_rect_on_dark.png").read_bytes()
    at = _app("otsu")
    at.session_state["input_source"] = "Tải tệp lên"
    at.session_state["_test_upload"] = ("first.png", data)
    at.run()
    next(b for b in at.button if b.label.startswith("Chạy Otsu")).click().run()
    at.session_state["_test_upload"] = ("second.png", data)
    at.run()
    assert not at.exception and len(at.get("download_button")) == 0
    next(b for b in at.button if b.label.startswith("Chạy Otsu")).click().run()
    files = at.session_state["otsu_result"]["_export_cache"]["files"]
    assert json.loads(files["otsu_stats.json"][0])["anh"] == "second.png"


def test_kmeans_exports_are_reused_until_foreground_changes(monkeypatch):
    from src.ui import page_kmeans

    calls, original = [], page_kmeans.kmeans_export_files

    def tracked(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(page_kmeans, "kmeans_export_files", tracked)
    at = _app("kmeans").run()
    next(b for b in at.button if b.label.startswith("Chạy K-means")).click().run()
    at.run()
    assert not at.exception and len(calls) == 1
    at.multiselect[0].set_value([]).run()
    assert not at.exception and len(calls) == 2
    files = at.session_state["km_result"]["_export_cache"]["files"]
    assert json.loads(files["kmeans_stats.json"][0])["cum_doi_tuong"] == []
    assert not np.asarray(Image.open(io.BytesIO(files["kmeans_mask.png"][0]))).any()
    at.run()
    assert len(calls) == 2


def test_otsu_exports_are_reused_until_a_new_run(monkeypatch):
    from src.ui import page_otsu

    calls, original = [], page_otsu.otsu_export_files

    def tracked(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(page_otsu, "otsu_export_files", tracked)
    at = _app("otsu").run()
    next(b for b in at.button if b.label.startswith("Chạy Otsu")).click().run()
    at.run()
    assert not at.exception and len(calls) == 1
    at.radio(key="otsu_mode").set_value("Thủ công (đối chiếu)")
    at.slider(key="otsu_manual_t").set_value(50)
    next(b for b in at.button if b.label.startswith("Chạy Otsu")).click().run()
    assert not at.exception and len(calls) == 2
    files = at.session_state["otsu_result"]["_export_cache"]["files"]
    assert json.loads(files["otsu_stats.json"][0])["nguong_su_dung"] == 50


@pytest.mark.parametrize("page,function", [("kmeans", "kmeans_export_files"), ("otsu", "otsu_export_files")])
def test_export_failure_is_visible_and_does_not_offer_stale_files(monkeypatch, page, function):
    import importlib

    module = importlib.import_module(f"src.ui.page_{page}")
    original = getattr(module, function)

    def fail(*args):
        raise OSError("Lỗi mã hoá kiểm thử")

    at = _app(page).run()
    monkeypatch.setattr(module, function, fail)
    label = "Chạy Otsu" if page == "otsu" else "Chạy K-means"
    next(b for b in at.button if b.label.startswith(label)).click().run()
    assert not at.exception and any("Không tạo được tệp" in e.value for e in at.error)
    assert len(at.get("download_button")) == 0
    monkeypatch.setattr(module, function, original)
    at.run()
    assert not at.exception and len(at.get("download_button")) == 7


def test_decode_memory_failure_clears_input_and_existing_downloads(monkeypatch):
    from src.ui import sidebar

    at = _app("otsu").run()
    next(b for b in at.button if b.label.startswith("Chạy Otsu")).click().run()
    assert len(at.get("download_button")) == 7

    def fail(*args, **kwargs):
        raise MemoryError("Không đủ RAM giải mã")

    monkeypatch.setattr(sidebar, "build_context", fail)
    at.run()
    assert not at.exception
    assert any("thu nhỏ ảnh trước khi tải lên" in e.value for e in at.error)
    assert at.session_state["_input_ctx"] is None
    assert len(at.get("download_button")) == 0


def test_ground_truth_upload_is_bound_to_the_current_input():
    at = _app("compare")
    at.session_state["sample_id"] = "circle_black_on_white"
    at.run()
    gt_bytes = (DATA / "ground_truth" / "circle_black_on_white_gt.png").read_bytes()
    at.file_uploader[0].set_value(("reference.png", gt_bytes, "image/png")).run()
    assert at.file_uploader[0].value.name == "reference.png"
    uploader_id = at.file_uploader[0].proto.id
    at.run()
    assert at.file_uploader[0].value.name == "reference.png"
    assert at.file_uploader[0].proto.id == uploader_id
    # Hai ảnh cùng kích thước: kiểm tra shape đơn thuần không phát hiện GT cũ.
    at.selectbox(key="sample_id").set_value("noisy_circle").run()
    assert not at.exception
    assert at.file_uploader[0].proto.id != uploader_id
    assert at.file_uploader[0].value is None


def test_ground_truth_memory_error_does_not_crash_or_report_scores(monkeypatch):
    from src.ui import page_compare

    at = _app("compare").run()
    next(b for b in at.button if b.label.startswith("Chạy K-means")).click().run()

    def fail(*args, **kwargs):
        raise MemoryError("Không đủ RAM để đọc mask")

    monkeypatch.setattr(page_compare, "load_mask_bytes", fail)
    at.file_uploader[0].set_value(("reference.png", to_png_bytes(np.ones((2, 2), np.uint8)), "image/png")).run()
    assert not at.exception
    assert any("Không dùng được ground-truth" in error.value for error in at.error)
    assert not any("IoU" in list(df.value.columns) for df in at.dataframe)


def test_compare_reuses_otsu_until_polarity_changes(monkeypatch):
    from src.ui import page_compare

    calls, original = [], page_compare.run_otsu

    def tracked(*args, **kwargs):
        calls.append(kwargs["polarity"])
        return original(*args, **kwargs)

    monkeypatch.setattr(page_compare, "run_otsu", tracked)
    at = _app("compare").run()
    next(b for b in at.button if b.label.startswith("Chạy K-means")).click().run()
    assert not at.exception and calls == ["auto"]
    at.multiselect[0].set_value([]).run()
    assert not at.exception and calls == ["auto"]
    at.radio(key="cmp_otsu_polarity").set_value("Lớp sáng (> t)").run()
    assert not at.exception and calls == ["auto", "bright"]
    at.run()
    assert calls == ["auto", "bright"]


def test_missing_sample_file_is_reported_and_upload_still_works(monkeypatch, tmp_path):
    from src.ui import sidebar

    monkeypatch.setattr(sidebar, "DATA_DIR", tmp_path)
    at = _app("otsu").run()
    assert not at.exception
    assert any("ảnh mẫu" in e.value for e in at.error)
    assert at.session_state["_input_ctx"] is None
    at.session_state["_test_upload"] = ("valid.png", to_png_bytes(np.zeros((3, 4), np.uint8)))
    at.radio(key="input_source").set_value("Tải tệp lên").run()
    assert not at.exception and at.session_state["_input_ctx"] is not None


@pytest.mark.parametrize("gt", [np.array(1), np.empty((0, 2)), np.zeros((2, 2, 3))])
def test_ground_truth_requires_a_nonempty_matrix(gt):
    _, prepared, _ = load_sample("circle_black_on_white")
    with pytest.raises(ValueError, match="2 chiều"):
        fit_ground_truth(gt, prepared)


def test_plot_is_closed_even_when_streamlit_rendering_fails(monkeypatch):
    import matplotlib.pyplot as plt
    from src.ui import components

    fig = plt.figure()

    def fail(*args, **kwargs):
        raise OSError("Không render được biểu đồ")

    monkeypatch.setattr(components.st, "pyplot", fail)
    with pytest.raises(OSError):
        components.show_figure(fig)
    assert not plt.fignum_exists(fig.number)


@pytest.mark.parametrize("page", ["kmeans", "compare"])
def test_postprocessing_memory_error_is_visible_without_downloads(monkeypatch, page):
    import importlib

    at = _app(page).run()
    next(b for b in at.button if b.label.startswith("Chạy K-means")).click().run()

    def fail(*args, **kwargs):
        raise MemoryError("Không đủ RAM để tạo ảnh tách nền")

    monkeypatch.setattr(importlib.import_module(f"src.ui.page_{page}"), "kmeans_outputs", fail)
    at.multiselect[0].set_value([]).run()
    assert not at.exception
    assert any("tạo ảnh kết quả" in error.value for error in at.error)
    assert len(at.get("download_button")) == 0
    result_key = "km_result" if page == "kmeans" else "cmp_result"
    assert "_export_cache" not in at.session_state[result_key]


@pytest.mark.parametrize("page", ["otsu", "library", "theory"])
@pytest.mark.parametrize("catalog", ["{invalid", "null", '[{"id":"broken"}]'])
def test_corrupt_sample_catalog_is_reported_and_does_not_block_upload(monkeypatch, tmp_path, page, catalog):
    from src.ui import state

    broken = tmp_path / "samples.json"
    broken.write_text(catalog, encoding="utf-8")
    monkeypatch.setattr(state, "MANIFEST_PATH", broken)
    state.load_manifest.clear()
    try:
        at = _app(page).run()
        assert not at.exception
        assert any("danh mục ảnh mẫu" in error.value.lower() for error in at.error)
        at.radio(key="input_source").set_value("Tải tệp lên").run()
        at.file_uploader(key="uploader").set_value(("valid.png", to_png_bytes(np.zeros((3, 4), np.uint8)), "image/png")).run()
        assert not at.exception
        assert at.session_state["_input_ctx"].loaded.source_name == "valid.png"
    finally:
        state.load_manifest.clear()


@pytest.mark.parametrize("page", ["library", "theory"])
def test_missing_demo_image_does_not_crash_other_content(monkeypatch, tmp_path, page):
    import importlib

    monkeypatch.setattr(importlib.import_module(f"src.ui.page_{page}"), "DATA_DIR", tmp_path)
    at = _app(page).run()
    assert not at.exception
    assert any("Không đọc được ảnh mẫu" in error.value for error in at.error)


def test_theory_variance_table_has_numeric_arrow_compatible_values(monkeypatch, caplog):
    import pandas as pd
    import pyarrow as pa
    from src.ui import page_theory

    tables, original = [], page_theory.st.dataframe

    def capture(data, *args, **kwargs):
        if isinstance(data, pd.DataFrame) and "f(g) = σ²_B" in data:
            tables.append(data.copy())
        return original(data, *args, **kwargs)

    monkeypatch.setattr(page_theory.st, "dataframe", capture)
    at = _app("theory").run()
    assert not at.exception and len(tables) == 1
    table = tables[0]
    pa.Table.from_pandas(table, preserve_index=False)
    assert pd.api.types.is_numeric_dtype(table["f(g) = σ²_B"])
    invalid = table.loc[table["g"] == 5].iloc[0]
    assert not invalid["Ngưỡng hợp lệ"]
    assert pd.isna(invalid["f(g) = σ²_B"])
    assert "Serialization of dataframe to Arrow table was unsuccessful" not in caplog.text


@pytest.mark.parametrize("dtype,level", [(bool, True), (np.uint8, 1), (np.uint8, 255)])
def test_ground_truth_resize_preserves_binary_mask_representations(dtype, level):
    prepared = prepare_input(load_image_bytes(to_png_bytes(np.zeros((4, 4), np.uint8)), "image.png"), 2)
    gt = np.zeros((4, 4), dtype=dtype)
    gt[:2, :2] = level
    fitted = fit_ground_truth(gt, prepared)
    np.testing.assert_array_equal(fitted, [[255, 0], [0, 0]])
    assert fitted.dtype == np.uint8


@pytest.mark.parametrize("method", ["average", "luminance"])
def test_kmeans_json_records_grayscale_method_needed_to_reproduce_the_run(method):
    image = np.array([[[200, 0, 0], [0, 200, 0]]], np.uint8)
    prepared = prepare_input(load_image_bytes(to_png_bytes(image), "color.png"), 800, method)
    params = KMeansParams(k=2, feature="gray")
    result = run_kmeans(prepared, params)
    out = kmeans_outputs(prepared, result, default_foreground(result))
    stats = json.loads(kmeans_export_files(prepared, params, result, out)["kmeans_stats.json"][0])
    assert stats["chuyen_xam"] == method


def test_compare_16bit_transparent_input_and_all_white_16bit_ground_truth():
    raw = np.full((32, 48), 1000, np.uint16)
    raw[8:24, 12:24] = 2000
    raw[8:24, 24:36] = 3000
    image_bytes, gt_bytes = io.BytesIO(), io.BytesIO()
    Image.fromarray(raw).save(image_bytes, "PNG", transparency=1000)
    Image.fromarray(np.full(raw.shape, 65535, np.uint16)).save(gt_bytes, "PNG")
    at = _app("compare").run()
    at.radio(key="input_source").set_value("Tải tệp lên").run()
    at.file_uploader(key="uploader").set_value(("depth.png", image_bytes.getvalue(), "image/png")).run()
    next(b for b in at.button if b.label.startswith("Chạy K-means")).click().run()
    uploader = next(u for u in at.file_uploader if u.key.startswith("cmp_gt_upload_"))
    uploader.set_value(("white16.png", gt_bytes.getvalue(), "image/png")).run()
    assert not at.exception
    scores = next(df.value for df in at.dataframe if "IoU" in df.value.columns)
    assert scores["IoU"].tolist() == [0.125, 0.125]


def test_uploaded_ground_truth_is_decoded_once_until_the_file_changes(monkeypatch):
    from src.ui import page_compare

    calls, original = [], page_compare.load_mask_bytes

    def tracked(data, name):
        calls.append(name)
        return original(data, name)

    monkeypatch.setattr(page_compare, "load_mask_bytes", tracked)
    at = _app("compare")
    at.session_state["sample_id"] = "circle_black_on_white"
    at.run()
    next(b for b in at.button if b.label.startswith("Chạy K-means")).click().run()
    gt_bytes = (DATA / "ground_truth" / "circle_black_on_white_gt.png").read_bytes()
    at.file_uploader[0].set_value(("first.png", gt_bytes, "image/png")).run()
    at.multiselect[0].set_value([]).run()
    assert not at.exception and calls == ["first.png"]
    at.file_uploader[0].set_value(("second.png", gt_bytes, "image/png")).run()
    assert not at.exception and calls == ["first.png", "second.png"]
    at.file_uploader[0].clear().run()
    assert "cmp_uploaded_gt" not in at.session_state


# ---------------- Hồi quy: giữ lựa chọn người dùng khi chuyển trang ----------------

def _navigate(at, page: str):
    at.session_state["_test_page"] = page
    return at.run()


def _km_export(at) -> dict:
    files = at.session_state["km_result"]["_export_cache"]["files"]
    return json.loads(files["kmeans_stats.json"][0])


@pytest.mark.parametrize("selection", [[0], []])
def test_kmeans_foreground_selection_survives_navigation(selection):
    at = _app("kmeans").run()
    next(b for b in at.button if b.label.startswith("Chạy K-means")).click().run()
    sse = at.session_state["km_result"]["result"].core.sse
    at.multiselect[0].set_value(selection).run()
    assert _km_export(at)["cum_doi_tuong"] == selection
    _navigate(at, "guide")
    _navigate(at, "kmeans")
    assert not at.exception
    assert at.multiselect[0].value == selection
    assert _km_export(at)["cum_doi_tuong"] == selection
    assert at.session_state["km_result"]["result"].core.sse == sse  # không chạy lại thuật toán


def test_kmeans_form_parameters_survive_navigation():
    at = _app("kmeans").run()
    at.slider(key="km_k").set_value(5)
    next(b for b in at.button if b.label.startswith("Chạy K-means")).click().run()
    _navigate(at, "otsu")
    _navigate(at, "kmeans")
    assert not at.exception
    assert at.slider(key="km_k").value == 5
    assert at.session_state["km_result"]["params"].k == 5


def test_compare_polarity_and_foreground_survive_navigation():
    at = _app("compare").run()
    next(b for b in at.button if b.label.startswith("Chạy K-means")).click().run()
    at.radio(key="cmp_otsu_polarity").set_value("Lớp sáng (> t)").run()
    at.multiselect[0].set_value([1]).run()
    _navigate(at, "guide")
    _navigate(at, "compare")
    assert not at.exception
    assert at.radio(key="cmp_otsu_polarity").value == "Lớp sáng (> t)"
    assert at.session_state["cmp_result"]["otsu"].polarity == "bright"
    assert at.multiselect[0].value == [1]


def test_otsu_form_survives_navigation():
    at = _app("otsu").run()
    at.radio(key="otsu_polarity").set_value("Lớp tối (≤ t)")
    next(b for b in at.button if b.label.startswith("Chạy Otsu")).click().run()
    _navigate(at, "theory")
    _navigate(at, "otsu")
    assert not at.exception
    assert at.radio(key="otsu_polarity").value == "Lớp tối (≤ t)"
    assert at.session_state["otsu_result"]["out"].polarity == "dark"


def test_new_kmeans_run_starts_from_suggested_foreground():
    at = _app("kmeans").run()
    run_button = lambda: next(b for b in at.button if b.label.startswith("Chạy K-means"))
    run_button().click().run()
    suggested = list(at.multiselect[0].value)
    at.multiselect[0].set_value([]).run()
    run_button().click().run()
    assert at.multiselect[0].value == suggested


def test_otsu_histogram_table_lists_all_256_levels():
    at = _app("otsu").run()
    next(b for b in at.button if b.label.startswith("Chạy Otsu")).click().run()
    tables = [df.value for df in at.dataframe if "Mức xám g" in df.value.columns]
    assert tables and list(tables[0]["Mức xám g"]) == list(range(256))


# ---------------- Hồi quy: ground truth tải lên qua điều hướng (BUG-005) và mask trong suốt (RISK-004) ----------------

def _gt_source(at) -> str | None:
    return next((c.value for c in at.caption if c.value.startswith("Tính trên ")), None)


def _iou_rows(at) -> dict | None:
    for df in at.dataframe:
        if "IoU" in df.value.columns:
            return dict(zip(df.value["Phương pháp"], df.value["IoU"]))
    return None


def _opposite_gt() -> bytes:
    gt = np.asarray(Image.open(DATA / "ground_truth" / "circle_black_on_white_gt.png"))
    return to_png_bytes(np.where(gt > 127, 0, 255).astype(np.uint8))


def _compare_with_uploaded_gt(run_first: bool = True):
    at = _app("compare")
    at.session_state["sample_id"] = "circle_black_on_white"
    at.run()
    if run_first:
        next(b for b in at.button if b.label.startswith("Chạy K-means")).click().run()
    at.file_uploader[0].set_value(("opposite_gt.png", _opposite_gt(), "image/png")).run()
    return at


def test_uploaded_ground_truth_survives_navigation():
    at = _compare_with_uploaded_gt()
    assert _gt_source(at) == "Tính trên ground truth tải lên (opposite_gt.png)."
    before = _iou_rows(at)
    assert before is not None and before["K-means"] < 0.5  # GT đảo ngược: khác hẳn GT mẫu (IoU = 1)
    _navigate(at, "guide")
    _navigate(at, "compare")
    assert not at.exception
    assert _gt_source(at) == "Tính trên ground truth tải lên (opposite_gt.png)."
    assert _iou_rows(at) == before
    assert any("opposite_gt.png" in c.value for c in at.caption if c.value.startswith("Đang dùng ground truth"))
    # Thao tác khác sau khi quay lại (widget upload đang rỗng) không được làm mất GT.
    at.multiselect[0].set_value([0]).run()
    assert _gt_source(at) == "Tính trên ground truth tải lên (opposite_gt.png)."


def test_uploaded_ground_truth_before_first_run_survives_navigation():
    at = _compare_with_uploaded_gt(run_first=False)
    _navigate(at, "theory")
    _navigate(at, "compare")
    next(b for b in at.button if b.label.startswith("Chạy K-means")).click().run()
    assert not at.exception
    assert _gt_source(at) == "Tính trên ground truth tải lên (opposite_gt.png)."


def test_explicit_removal_of_uploaded_ground_truth_returns_to_sample_ground_truth():
    at = _compare_with_uploaded_gt()
    _navigate(at, "guide")
    _navigate(at, "compare")
    next(b for b in at.button if b.label == "Bỏ ground truth tải lên").click().run()
    assert not at.exception
    assert "cmp_uploaded_gt" not in at.session_state
    assert _gt_source(at) == "Tính trên ground truth của ảnh mẫu."
    assert _iou_rows(at)["K-means"] == 1.0


def test_clearing_uploader_removes_uploaded_ground_truth():
    at = _compare_with_uploaded_gt()
    at.file_uploader[0].clear().run()
    assert not at.exception
    assert _gt_source(at) == "Tính trên ground truth của ảnh mẫu."


def test_uploaded_ground_truth_is_dropped_when_input_changes():
    at = _compare_with_uploaded_gt()
    at.selectbox(key="sample_id").set_value("noisy_circle").run()
    assert "cmp_uploaded_gt" not in at.session_state
    at.selectbox(key="sample_id").set_value("circle_black_on_white").run()
    assert not at.exception
    assert "cmp_uploaded_gt" not in at.session_state
    next(b for b in at.button if b.label.startswith("Chạy K-means")).click().run()  # kết quả cũ đã bị ẩn khi đổi ảnh
    assert _gt_source(at) == "Tính trên ground truth của ảnh mẫu."


def test_transparent_ground_truth_upload_is_rejected_without_fallback_scores():
    rgba = np.zeros((4, 4, 4), np.uint8)
    rgba[1:3, 1:3] = (255, 255, 255, 255)  # 4 pixel đối tượng đục, 12 pixel nền trong suốt
    at = _app("compare")
    at.session_state["sample_id"] = "circle_black_on_white"
    at.run()
    next(b for b in at.button if b.label.startswith("Chạy K-means")).click().run()
    at.file_uploader[0].set_value(("alpha_gt.png", to_png_bytes(rgba), "image/png")).run()
    assert not at.exception
    assert any("vùng trong suốt" in e.value for e in at.error)
    assert _iou_rows(at) is None  # không âm thầm dùng GT mẫu thay cho tệp lỗi
    assert at.session_state["cmp_uploaded_gt"]["mask"] is None


def _compare_with_transparent_gt():
    rgba = np.zeros((4, 4, 4), np.uint8)
    rgba[1:3, 1:3] = (255, 255, 255, 255)
    at = _app("compare")
    at.session_state["sample_id"] = "circle_black_on_white"
    at.run()
    next(b for b in at.button if b.label.startswith("Chạy K-means")).click().run()
    at.file_uploader[0].set_value(("alpha_gt.png", to_png_bytes(rgba), "image/png")).run()
    return at


def test_failed_ground_truth_upload_survives_navigation():
    at = _compare_with_transparent_gt()
    _navigate(at, "guide")
    _navigate(at, "compare")
    assert not at.exception
    assert any("vùng trong suốt" in e.value for e in at.error)
    assert _iou_rows(at) is None  # không quay về GT mẫu với IoU = 1.0
    assert any("alpha_gt.png" in c.value for c in at.caption if c.value.startswith("Tệp ground truth đã tải lên bị lỗi"))
    at.multiselect[0].set_value([0]).run()
    assert _iou_rows(at) is None


def test_failed_ground_truth_is_cleared_by_explicit_removal():
    at = _compare_with_transparent_gt()
    _navigate(at, "guide")
    _navigate(at, "compare")
    next(b for b in at.button if b.label == "Bỏ ground truth tải lên").click().run()
    assert not at.exception and not at.error
    assert "cmp_uploaded_gt" not in at.session_state
    assert _iou_rows(at)["K-means"] == 1.0


def test_failed_ground_truth_is_cleared_by_uploader_removal_or_replacement():
    at = _compare_with_transparent_gt()
    at.file_uploader[0].clear().run()
    assert "cmp_uploaded_gt" not in at.session_state and _iou_rows(at)["K-means"] == 1.0
    at = _compare_with_transparent_gt()
    at.file_uploader[0].set_value(("opposite_gt.png", _opposite_gt(), "image/png")).run()
    assert not at.error
    assert _gt_source(at) == "Tính trên ground truth tải lên (opposite_gt.png)."


def test_failed_ground_truth_is_cleared_when_input_changes():
    at = _compare_with_transparent_gt()
    at.selectbox(key="sample_id").set_value("noisy_circle").run()
    assert "cmp_uploaded_gt" not in at.session_state
