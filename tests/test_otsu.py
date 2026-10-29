"""Kiểm thử thuật toán Otsu tự cài đặt."""

import numpy as np
import pytest

from src.algorithms.otsu import apply_threshold, compute_histogram, otsu_from_histogram, otsu_threshold


def brute_force_otsu(gray):
    """Cài đặt trực tiếp theo định nghĩa w0*w1*(mu0-mu1)^2, duyệt từng ngưỡng — dùng để đối chiếu."""
    g = gray.ravel().astype(float)
    best, ts = -1.0, []
    for t in range(256):
        c0, c1 = g[g <= t], g[g > t]
        if c0.size == 0 or c1.size == 0:
            continue
        w0, w1 = c0.size / g.size, c1.size / g.size
        v = w0 * w1 * (c0.mean() - c1.mean()) ** 2
        if v > best + 1e-9:
            best, ts = v, [t]
        elif abs(v - best) <= 1e-9:
            ts.append(t)
    return best, ts


def test_textbook_example_section_3_1_6():
    # Ví dụ giáo trình mục 3.1.6 (trang 32): ngưỡng cần tách θ = 1, f(θ) ≈ 1.66
    img = np.array([[0, 1, 2, 3, 4, 5],
                    [0, 0, 1, 2, 3, 4],
                    [0, 0, 0, 1, 2, 3],
                    [0, 0, 0, 0, 1, 2],
                    [0, 0, 0, 0, 0, 1]], dtype=np.uint8)
    res = otsu_threshold(img)
    assert res.threshold == 1
    # Giá trị chính xác: 20/10 * (5/20 - 35/30)^2 = 1.6806 (giáo trình làm tròn thành 1.66)
    assert res.max_variance == pytest.approx(2 * (0.25 - 35 / 30) ** 2)
    # Đối chiếu f(0) = 1.36 (giáo trình ghi 1.35 do làm tròn)
    assert res.between_class_variance[0] == pytest.approx((35 / 30) ** 2)


def test_hand_computed_small_matrix():
    # Mức 10 (x3), 20 (x1), 200 (x4). t = 20 tách {10,10,10,20} | {200 x4}
    img = np.array([[10, 10, 10, 20], [200, 200, 200, 200]], dtype=np.uint8)
    res = otsu_threshold(img)
    # Ngưỡng tối ưu thuộc đoạn [20, 199] (cùng một phân hoạch), chọn floor((20+199)/2) = 109
    assert res.tied_thresholds == list(range(20, 200))
    assert res.threshold == 109
    w0 = w1 = 0.5
    mu0, mu1 = 12.5, 200.0
    assert res.max_variance == pytest.approx(w0 * w1 * (mu0 - mu1) ** 2)
    assert res.mean0 == pytest.approx(mu0) and res.mean1 == pytest.approx(mu1)


def test_two_level_image():
    img = np.full((40, 40), 50, dtype=np.uint8)
    img[10:30, 10:30] = 200
    res = otsu_threshold(img)
    assert res.valid
    assert 50 <= res.threshold < 200
    binary = apply_threshold(img, res.threshold)
    np.testing.assert_array_equal(binary == 255, img == 200)
    assert res.separability == pytest.approx(1.0)


def test_all_black_image():
    res = otsu_threshold(np.zeros((16, 16), dtype=np.uint8))
    assert not res.valid
    assert res.threshold == 0
    assert np.all(apply_threshold(np.zeros((16, 16), np.uint8), res.threshold) == 0)
    assert res.warnings


def test_all_white_image():
    img = np.full((16, 16), 255, dtype=np.uint8)
    res = otsu_threshold(img)
    assert not res.valid
    assert res.threshold == 255
    assert np.all(apply_threshold(img, res.threshold) == 0)  # cả ảnh là một lớp
    assert np.all(res.between_class_variance == 0)


def test_empty_class_thresholds_have_zero_variance_and_no_nan():
    img = np.array([[100, 100, 150, 150]], dtype=np.uint8)
    res = otsu_threshold(img)
    assert np.all(np.isfinite(res.between_class_variance))
    assert not res.valid_thresholds[:100].any()  # t < 100: lớp C0 rỗng
    assert not res.valid_thresholds[150:].any()  # t >= 150: lớp C1 rỗng
    assert res.valid_thresholds[100:150].all()
    assert 100 <= res.threshold < 150


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_random_images_match_brute_force(seed):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, (30, 40), dtype=np.uint8)
    res = otsu_threshold(img)
    best, ts = brute_force_otsu(img)
    assert res.max_variance == pytest.approx(best, rel=1e-9)
    assert res.threshold in range(min(ts), max(ts) + 1)


def test_bimodal_random_image():
    rng = np.random.default_rng(10)
    img = np.concatenate([rng.normal(60, 10, 5000), rng.normal(180, 10, 5000)])
    img = np.clip(np.round(img), 0, 255).astype(np.uint8).reshape(100, 100)
    res = otsu_threshold(img)
    assert 105 <= res.threshold <= 135


def test_binary_mask_values_are_0_or_255():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (50, 50), dtype=np.uint8)
    res = otsu_threshold(img)
    b = apply_threshold(img, res.threshold)
    assert b.dtype == np.uint8
    assert set(np.unique(b).tolist()) <= {0, 255}
    assert np.count_nonzero(b) == np.count_nonzero(img > res.threshold)


def test_histogram_counts():
    img = np.array([[0, 0, 255], [7, 7, 7]], dtype=np.uint8)
    h = compute_histogram(img)
    assert h.shape == (256,) and h.sum() == 6
    assert h[0] == 2 and h[7] == 3 and h[255] == 1


def test_invalid_inputs():
    with pytest.raises(ValueError):
        compute_histogram(np.zeros((4, 4, 3), np.uint8))
    with pytest.raises(ValueError):
        compute_histogram(np.array([[1.5, 2.0]]))
    with pytest.raises(ValueError):
        otsu_from_histogram(np.zeros(256))
    with pytest.raises(ValueError):
        otsu_from_histogram(np.ones(10))
    with pytest.raises(ValueError):
        apply_threshold(np.zeros((2, 2), np.uint8), 300)


def test_polarity_suggestion_from_border():
    from src.processing.pipeline import suggest_dark_foreground

    bright_border = np.full((10, 10), 255, np.uint8)
    bright_border[3:7, 3:7] = 0
    assert suggest_dark_foreground(bright_border)  # nền (viền) sáng -> đối tượng là lớp tối
    assert not suggest_dark_foreground(255 - bright_border)
