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
