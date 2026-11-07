"""Kiểm thử các cải tiến sau phản biện: Lab, K > 8, Multi-Otsu, heuristic nền, lọc trung vị,
hình thái học và đọc CSV vector hoá."""

import json
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from src.algorithms.kmeans import image_to_features, segment_image_kmeans
from src.algorithms.otsu import apply_multi_threshold, multi_otsu_from_histogram, otsu_threshold
from src.processing.filters import apply_morphology, closing, dilate, erode, median_filter, opening
from src.processing.pipeline import (
    KMeansParams,
    MaskPostprocess,
    default_multi_otsu_foreground,
    kmeans_export_files,
    kmeans_outputs,
    multi_otsu_export_files,
    multi_otsu_outputs,
    prepare_input,
    run_kmeans,
    run_multi_otsu,
    run_otsu,
)
from src.processing.postprocessing import (
    CLUSTER_PALETTE_HEX,
    cluster_palette_hex,
    colorize_labels,
    suggest_foreground_clusters,
)
from src.utils.color_space import lab_to_rgb, rgb_to_lab
from src.utils.image_io import InputError, load_mask_bytes, load_path, parse_csv_matrix
from src.utils.metrics import compare_masks

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


# ------------------------------- CIE L*a*b* -------------------------------

@pytest.mark.parametrize("rgb,lab", [
    ((255, 255, 255), (100.0, 0.0, 0.0)),
    ((0, 0, 0), (0.0, 0.0, 0.0)),
    ((255, 0, 0), (53.2408, 80.0925, 67.2032)),
    ((0, 255, 0), (87.7347, -86.1827, 83.1793)),
    ((0, 0, 255), (32.2970, 79.1875, -107.8602)),
])
def test_rgb_to_lab_matches_reference_values(rgb, lab):
    np.testing.assert_allclose(rgb_to_lab(np.array(rgb, float)), lab, atol=2e-3)


def test_lab_round_trip_recovers_rgb():
    rgb = np.random.default_rng(0).integers(0, 256, (5000, 3)).astype(float)
    np.testing.assert_allclose(lab_to_rgb(rgb_to_lab(rgb)), rgb, atol=1e-6)


@pytest.mark.parametrize("bad", [np.zeros((2, 2)), np.full((1, 3), 256.0), np.full((1, 3), np.nan)])
def test_rgb_to_lab_rejects_invalid_input(bad):
    with pytest.raises(ValueError):
        rgb_to_lab(bad)


def test_kmeans_lab_feature_on_image():
    img = np.zeros((20, 30, 3), np.uint8)
    img[:, :10] = (230, 30, 40)
    img[:, 10:20] = (30, 160, 60)
    img[:, 20:] = (20, 30, 200)
    res = segment_image_kmeans(img, 3, feature="lab", seed=0)
    assert res.feature == "lab" and res.core.converged
    # Mỗi vùng màu đồng nhất là một cụm; tâm ở thang Lab, sắp tăng dần theo L*.
    assert len({int(res.labels[0, c]) for c in (0, 15, 25)}) == 3
    assert np.all(np.diff(res.centers[:, 0]) > 0)
    np.testing.assert_allclose(res.centers, rgb_to_lab(res.centers_rgb), atol=1e-6)
    # Ảnh phân đoạn được tô bằng màu sRGB của tâm: vùng đồng nhất giữ nguyên màu.
    np.testing.assert_array_equal(res.segmented, img)


def test_lab_feature_requires_color_image():
    with pytest.raises(ValueError):
        image_to_features(np.zeros((4, 4), np.uint8), "lab")
    with pytest.raises(ValueError):
        segment_image_kmeans(np.zeros((4, 4), np.uint8), 2, feature="lab")
    assert image_to_features(np.zeros((2, 2, 3), np.uint8), "lab").shape == (4, 3)


def test_pipeline_lab_cluster_table_and_exports():
    prepared = prepare_input(load_path(DATA / "sample_images" / "multi_shapes_color.png"), 200)
    result = run_kmeans(prepared, KMeansParams(k=5, feature="lab"))
    out = kmeans_outputs(prepared, result, [1, 2])
    files = kmeans_export_files(prepared, KMeansParams(k=5, feature="lab"), result, out)
    stats = json.loads(files["kmeans_stats.json"][0])
    assert stats["tham_so"]["feature_thuc_te"] == "lab"
    assert stats["khong_gian_tam_cum"].startswith("CIE")
    assert len(stats["mau_hien_thi_tam_cum_rgb"]) == result.core.k_effective


# ------------------------------- K > 8, bảng màu -------------------------------

def test_palette_extends_beyond_eight_with_distinct_colors():
    colors = cluster_palette_hex(24)
    assert colors[:8] == CLUSTER_PALETTE_HEX and len(set(colors)) == 24
    labels = np.arange(16).reshape(4, 4)
    rgb = colorize_labels(labels)
    assert rgb.shape == (4, 4, 3) and len({tuple(p) for p in rgb.reshape(-1, 3)}) == 16
    with pytest.raises(ValueError):
        colorize_labels(np.array([[-1]]))


def test_image_kmeans_with_twelve_clusters():
    img = np.repeat(np.arange(0, 240, 20, dtype=np.uint8), 5)[None, :].repeat(4, axis=0)
    res = segment_image_kmeans(img, 12, feature="gray", seed=0)
    assert res.core.k_effective == 12 and np.unique(res.labels).size == 12


# ------------------------------- Heuristic nền -------------------------------

def test_closeup_object_touching_all_borders_stays_foreground():
    # Đối tượng (1) chiếm phần lớn khung hình và chạm cả 4 viền; nền (0) chỉ ở 4 góc.
    yy, xx = np.mgrid[:100, :100]
    labels = (np.hypot(yy - 49.5, xx - 49.5) < 62).astype(np.int64)
    border = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    assert border.mean() > 0.5  # quy tắc cũ "chiếm viền nhiều nhất là nền" sẽ chọn sai
    assert suggest_foreground_clusters(labels, 2) == [1]


def test_multi_colored_background_is_all_background():
    # Trời (0) ở trên, cỏ (1) ở dưới, đối tượng (2) ở giữa chạm hai mép bên.
    labels = np.zeros((90, 90), np.int64)
    labels[45:] = 1
    labels[25:65, :] = 2
    assert suggest_foreground_clusters(labels, 3) == [2]


def test_suggestion_falls_back_when_border_equals_center():
    labels = np.zeros((8, 8), np.int64)
    assert suggest_foreground_clusters(labels, 2) == [1]


# ------------------------------- Multi-Otsu -------------------------------

def _brute_force_score(h, a, b):
    lv = np.arange(256)
    parts = [slice(0, a + 1), slice(a + 1, b + 1), slice(b + 1, 256)]
    cs = [h[p].sum() for p in parts]
    if min(cs) == 0:
        return None
    return sum((lv[p] * h[p]).sum() ** 2 / c for p, c in zip(parts, cs))


@pytest.mark.parametrize("seed", range(4))
def test_multi_otsu_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    h = rng.integers(0, 40, 256) * (rng.random(256) < 0.15)
    h[[5, 120, 240]] += 1  # ít nhất 3 mức xám
    res = multi_otsu_from_histogram(h)
    best = max(v for a in range(256) for b in range(a + 1, 256) if (v := _brute_force_score(h, a, b)) is not None)
    assert res.valid
    assert _brute_force_score(h, *res.thresholds) == pytest.approx(best, rel=1e-12)
    assert sum(res.class_counts) == h.sum()
    assert 0 <= res.separability <= 1


def test_multi_otsu_separates_three_gray_levels():
    gray = np.repeat(np.array([10, 120, 230], np.uint8), [50, 30, 20])[None, :]
    res = multi_otsu_from_histogram(np.bincount(gray.ravel(), minlength=256))
    t1, t2 = res.thresholds
    assert 10 <= t1 < 120 <= t2 < 230
    np.testing.assert_array_equal(np.unique(apply_multi_threshold(gray, res.thresholds)), [0, 1, 2])
    assert res.separability == pytest.approx(1.0)


def test_multi_otsu_with_two_levels_is_invalid_but_safe():
    h = np.zeros(256, np.int64)
    h[[20, 200]] = 5
    res = multi_otsu_from_histogram(h)
    assert not res.valid and res.thresholds[0] == res.thresholds[1] and res.warnings


def test_multi_otsu_fixes_gray_levels_sample():
    prepared = prepare_input(load_path(DATA / "sample_images" / "gray_levels_objects.png"), 800)
    gt = load_mask_bytes((DATA / "ground_truth" / "gray_levels_objects_gt.png").read_bytes())
    binary_iou = compare_masks(run_otsu(prepared, polarity="auto").mask, gt).iou
    result, labels = run_multi_otsu(prepared)
    out = multi_otsu_outputs(prepared, result, labels, default_multi_otsu_foreground(labels))
    assert binary_iou < 0.7 and compare_masks(out.mask, gt).iou == pytest.approx(1.0)
    files = multi_otsu_export_files(prepared, out)
    assert json.loads(files["multiotsu_stats.json"][0])["nguong"] == list(result.thresholds)


def test_apply_multi_threshold_validates_thresholds():
    with pytest.raises(ValueError):
        apply_multi_threshold(np.zeros((2, 2), np.uint8), (200, 100))
    with pytest.raises(ValueError):
        apply_multi_threshold(np.zeros((2, 2), np.uint8), (10,))


# ------------------------------- Lọc trung vị -------------------------------

def _median_brute(img, k):
    r = k // 2
    pad = np.pad(img, ((r, r), (r, r)) + ((0, 0),) * (img.ndim - 2), mode="edge")
    out = np.empty_like(img)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            out[y, x] = np.median(pad[y : y + k, x : x + k].reshape(k * k, -1), axis=0).squeeze()
    return out


@pytest.mark.parametrize("shape,k", [((13, 17), 3), ((9, 11, 3), 5), ((6, 7), 7)])
def test_median_filter_matches_brute_force(shape, k):
    img = np.random.default_rng(1).integers(0, 256, shape).astype(np.uint8)
    np.testing.assert_array_equal(median_filter(img, k), _median_brute(img, k))


def test_median_filter_removes_salt_and_pepper():
    img = np.full((20, 20), 100, np.uint8)
    img[5, 5], img[10, 12] = 255, 0
    np.testing.assert_array_equal(median_filter(img, 3), np.full((20, 20), 100, np.uint8))


def test_median_filter_handles_blocks_taller_than_row_chunk():
    img = np.random.default_rng(2).integers(0, 256, (300, 8)).astype(np.uint8)
    np.testing.assert_array_equal(median_filter(img, 3), _median_brute(img, 3))


@pytest.mark.parametrize("size", [0, 2, 4, -3, 2.0, True])
def test_filters_reject_even_or_invalid_sizes(size):
    with pytest.raises(ValueError):
        median_filter(np.zeros((4, 4), np.uint8), size)
    with pytest.raises(ValueError):
        dilate(np.zeros((4, 4), bool), size)


# ------------------------------- Hình thái học -------------------------------

def _morph_brute(mask, k, op):
    r = k // 2
    pad = np.pad(mask, r, mode="edge")
    f = np.max if op == "dilate" else np.min
    return np.array([[f(pad[y : y + k, x : x + k]) for x in range(mask.shape[1])] for y in range(mask.shape[0])])


@pytest.mark.parametrize("k", [3, 5])
def test_erode_and_dilate_match_brute_force(k):
    mask = np.random.default_rng(3).random((15, 19)) > 0.6
    np.testing.assert_array_equal(dilate(mask, k), _morph_brute(mask, k, "dilate"))
    np.testing.assert_array_equal(erode(mask, k), _morph_brute(mask, k, "erode"))


def test_opening_removes_specks_and_closing_fills_holes():
    mask = np.zeros((30, 30), bool)
    mask[8:22, 8:22] = True
    noisy = mask.copy()
    noisy[2, 2] = True  # đốm nhiễu ngoài đối tượng
    noisy[15, 15] = False  # lỗ thủng trong đối tượng
    np.testing.assert_array_equal(opening(noisy, 3)[2, 2], False)
    np.testing.assert_array_equal(closing(noisy, 3), closing(mask, 3) | closing(noisy, 3))
    assert closing(noisy, 3)[15, 15]
    cleaned = apply_morphology(noisy, "open_close", 3)
    np.testing.assert_array_equal(cleaned, np.where(mask, 255, 0))


def test_object_touching_border_is_not_eroded_from_edge():
    mask = np.zeros((10, 10), bool)
    mask[:, :4] = True  # đối tượng chạm mép trái
    np.testing.assert_array_equal(opening(mask, 3), mask)


def test_morphology_improves_noisy_circle_sample():
    prepared = prepare_input(load_path(DATA / "sample_images" / "noisy_circle.png"), 800)
    gt = load_mask_bytes((DATA / "ground_truth" / "noisy_circle_gt.png").read_bytes())
    plain = compare_masks(run_otsu(prepared, polarity="auto").mask, gt).iou
    post = run_otsu(prepared, polarity="auto", postprocess=MaskPostprocess("open_close", 5))
    assert compare_masks(post.mask, gt).iou > plain
    assert post.raw_mask is not None and not np.array_equal(post.raw_mask, post.mask)


def test_median_preprocessing_improves_noisy_circle_sample():
    gt = load_mask_bytes((DATA / "ground_truth" / "noisy_circle_gt.png").read_bytes())
    loaded = load_path(DATA / "sample_images" / "noisy_circle.png")
    plain = compare_masks(run_otsu(prepare_input(loaded, 800), polarity="auto").mask, gt).iou
    filtered = prepare_input(loaded, 800, median_size=5)
    assert filtered.median_size == 5
    assert compare_masks(run_otsu(filtered, polarity="auto").mask, gt).iou > plain


def test_mask_postprocess_rejects_unknown_operation():
    with pytest.raises(ValueError):
        MaskPostprocess("blur")


# ------------------------------- CSV vector hoá -------------------------------

def test_csv_fast_and_slow_paths_agree():
    text = "0,7,255\n007,+12,3.0\n1, 2 ,3\n"
    np.testing.assert_array_equal(parse_csv_matrix(text), [[0, 7, 255], [7, 12, 3], [1, 2, 3]])


@pytest.mark.parametrize("text,message", [
    ("1,2\n3,256\n", "dòng 2, cột 2"),
    ("1,2\n3,-1\n", "dòng 2, cột 2"),
    ("1,2\n3,x\n", "dòng 2, cột 2"),
    ("1,2\n3,1.5\n", "dòng 2, cột 2"),
])
def test_csv_errors_still_point_to_the_right_cell(text, message):
    with pytest.raises(InputError, match=message):
        parse_csv_matrix(text)


def test_large_csv_preserves_all_values():
    m = np.random.default_rng(0).integers(0, 256, (2000, 2000))
    text = "\n".join(",".join(map(str, row)) for row in m)
    np.testing.assert_array_equal(parse_csv_matrix(text), m)


# ------------------------------- Giao diện -------------------------------

def _app(page: str):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.session_state["_test_page"] = page
    return at


def test_otsu_page_multi_otsu_mode():
    at = _app("otsu")
    at.session_state["sample_id"] = "gray_levels_objects"
    at.run()
    at.radio(key="otsu_classes").set_value("3 lớp — Multi-Otsu (2 ngưỡng)")
    at.selectbox(key="otsu_morph").set_value("close")
    next(b for b in at.button if b.label.startswith("Chạy Otsu")).click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(m.label == "Ngưỡng t2" for m in at.metric)
    assert at.multiselect[0].value == [1, 2]
    assert len(at.get("download_button")) == 5
    assert any("IoU = 1.0000" in s.value for s in at.success)


def test_kmeans_page_lab_and_large_k_with_morphology():
    at = _app("kmeans").run()
    at.radio(key="km_feature").set_value("CIE L*a*b* (3 kênh, theo cảm nhận)")
    at.slider(key="km_k").set_value(12)
    next(b for b in at.button if b.label.startswith("Chạy K-means")).click().run()
    assert not at.exception, [e.value for e in at.exception]
    at.selectbox(key="km_morph").set_value("open").run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["km_result"]["result"].feature == "lab"


def test_sidebar_median_filter_changes_signature():
    at = _app("otsu").run()
    first = at.session_state["_input_ctx"].signature
    at.selectbox(key="median_size").set_value(3).run()
    assert not at.exception
    ctx = at.session_state["_input_ctx"]
    assert ctx.signature != first and ctx.prepared.median_size == 3


# ------------------------- Hồi quy sau review 07/11/2026 -------------------------

def test_median_keeps_unfiltered_source_for_display_and_cutout():
    loaded = load_path(DATA / "sample_images/noisy_circle.png")
    plain = prepare_input(loaded, 800)
    filtered = prepare_input(loaded, 800, median_size=5)
    assert not plain.is_filtered and filtered.is_filtered
    np.testing.assert_array_equal(filtered.original, plain.image)  # ảnh gốc = ảnh chưa lọc
    assert not np.array_equal(filtered.image, filtered.original)  # thuật toán nhận ảnh đã lọc
    np.testing.assert_array_equal(filtered.image, median_filter(plain.image, 5))
    from src.processing.preprocessing import to_grayscale
    np.testing.assert_array_equal(filtered.gray, to_grayscale(filtered.image, filtered.gray_method))
    res = run_kmeans(filtered, KMeansParams(k=2))
    out = kmeans_outputs(filtered, res, [1])
    fg = out.mask > 0
    src = filtered.original if filtered.original.ndim == 3 else np.stack([filtered.original] * 3, -1)
    np.testing.assert_array_equal(out.cutout_rgba[fg][:, :3], src[fg])  # màu đối tượng = pixel nguồn
    stats = run_otsu(filtered, "auto")
    assert stats.mask.shape == filtered.image.shape[:2]


@pytest.mark.parametrize("text,where", [('"1,2",3\n"4,5",6\n', "dòng 1, cột 1"),
                                        ('1;2\n"3,4";5\n', "dòng 2, cột 1")])
def test_csv_quoted_comma_reports_cell(text, where):
    with pytest.raises(InputError, match=where):
        parse_csv_matrix(text)


def test_csv_quoted_plain_numbers_still_fast_path_ok():
    np.testing.assert_array_equal(parse_csv_matrix('"1",2\n3,"4"\n'), [[1, 2], [3, 4]])


def test_multi_otsu_variance_matrix_is_stable_for_skewed_histogram():
    from fractions import Fraction

    h = np.zeros(256, dtype=np.int64)
    h[253], h[254], h[255] = 2 ** 52 - 2, 1, 1
    r = multi_otsu_from_histogram(h)
    assert r.thresholds == (253, 254)
    n = int(h.sum())
    mu = Fraction(253 * h[253] + 254 + 255, n)
    exact = sum(Fraction(int(c), n) * (Fraction(v) - mu) ** 2
                for c, v in ((h[253], 253), (1, 254), (1, 255)))
    got = r.between_class_variance[253, 254]
    assert got == pytest.approx(float(exact), rel=1e-6)
    assert got == pytest.approx(r.max_variance, rel=1e-6)


@pytest.mark.parametrize("seed", range(5))
def test_multi_otsu_variance_matrix_matches_fraction(seed):
    from fractions import Fraction

    h = np.random.default_rng(seed).integers(0, 50, 256)
    r = multi_otsu_from_histogram(h)
    t1, t2 = r.thresholds
    n = int(h.sum())
    lv = np.arange(256)
    mu = Fraction(int((lv * h).sum()), n)
    exact = Fraction(0)
    for lo, hi in ((0, t1 + 1), (t1 + 1, t2 + 1), (t2 + 1, 256)):
        c, m = int(h[lo:hi].sum()), int((lv[lo:hi] * h[lo:hi]).sum())
        exact += Fraction(c, n) * (Fraction(m, c) - mu) ** 2
    assert r.between_class_variance[t1, t2] == pytest.approx(float(exact), rel=1e-12)
    assert np.nanmax(r.between_class_variance) == pytest.approx(r.max_variance, rel=1e-12)


def test_kmeans_reset_defaults_also_resets_morphology():
    at = _app("kmeans").run()
    next(b for b in at.button if b.label.startswith("Chạy K-means")).click().run()
    at.selectbox(key="km_morph").set_value("open_close")
    at.selectbox(key="km_morph_size").set_value(7).run()
    next(b for b in at.button if b.label == "Khôi phục mặc định").click().run()
    next(b for b in at.button if b.label.startswith("Chạy K-means")).click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.selectbox(key="km_morph").value == "none"
    assert at.selectbox(key="km_morph_size").value == 3
    assert at.session_state["_persist__km_morph"] == "none"


def test_source_panels_show_unfiltered_original_and_filtered_input():
    from src.ui.components import source_panels

    loaded = load_path(DATA / "sample_images/noisy_circle.png")
    plain = source_panels(prepare_input(loaded, 800), with_gray=True)
    assert [c for _, c in plain] == ["Ảnh gốc", "Ảnh xám"]
    prep = prepare_input(loaded, 800, median_size=3)
    panels = source_panels(prep, with_gray=True)
    assert [c for _, c in panels] == ["Ảnh gốc", "Sau lọc trung vị 3×3 (đầu vào thuật toán)", "Ảnh xám (sau lọc)"]
    assert panels[0][0] is prep.original and panels[1][0] is prep.image


def test_otsu_page_renders_with_median_filter():
    at = _app("otsu").run()
    at.selectbox(key="median_size").set_value(3).run()
    next(b for b in at.button if b.label.startswith("Chạy Otsu")).click().run()
    assert not at.exception, [e.value for e in at.exception]
