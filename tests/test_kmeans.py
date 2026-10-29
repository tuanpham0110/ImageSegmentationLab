"""Kiểm thử thuật toán K-means tự cài đặt."""

import numpy as np
import pytest

from src.algorithms.kmeans import assign_labels, kmeans, segment_image_kmeans, unique_rows


def two_blobs(n=200, seed=1):
    rng = np.random.default_rng(seed)
    a = rng.normal([20.0, 20.0], 2.0, (n, 2))
    b = rng.normal([200.0, 180.0], 2.0, (n, 2))
    return np.vstack([a, b])


def test_two_separated_groups_are_recovered():
    x = two_blobs()
    res = kmeans(x, 2, seed=0)
    assert res.k_effective == 2
    # Mỗi nhóm 200 điểm phải thuộc trọn một cụm
    assert len(set(res.labels[:200])) == 1
    assert len(set(res.labels[200:])) == 1
    assert res.labels[0] != res.labels[-1]
    np.testing.assert_allclose(res.centers[0], [20, 20], atol=1.0)
    np.testing.assert_allclose(res.centers[1], [200, 180], atol=1.0)
    assert res.converged


def test_hand_computed_1d_example():
    # Dữ liệu 1 chiều: {1, 2, 3} và {10, 11, 12} -> tâm 2 và 11, SSE = 2 * (1 + 0 + 1) = 4
    x = np.array([1, 2, 3, 10, 11, 12], dtype=float)
    res = kmeans(x, 2, seed=3)
    np.testing.assert_allclose(res.centers.ravel(), [2.0, 11.0])
    assert res.labels.tolist() == [0, 0, 0, 1, 1, 1]
    assert res.sse == pytest.approx(4.0)


def test_k_larger_than_distinct_values_is_reduced_with_warning():
    x = np.array([0, 0, 0, 255, 255], dtype=float)
    res = kmeans(x, 5, seed=0)
    assert res.k_requested == 5
    assert res.k_effective == 2
    assert any("giảm" in w for w in res.warnings)
    assert sorted(res.centers.ravel().tolist()) == [0.0, 255.0]


def test_k_larger_than_number_of_samples():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    res = kmeans(x, 8, seed=0)
    assert res.k_effective == 2
    assert res.sse == pytest.approx(0.0)


def test_uniform_image_gives_single_cluster():
    img = np.full((20, 30), 128, dtype=np.uint8)
    res = segment_image_kmeans(img, 3, feature="gray", seed=0)
    assert res.core.k_effective == 1
    assert np.all(res.labels == 0)
    assert np.all(res.segmented == 128)
    assert res.core.sse == 0.0


def test_empty_cluster_is_reinitialised():
    # Khởi tạo random có thể đặt 2 tâm ở cùng nhóm; ép tình huống cụm rỗng bằng dữ liệu lệch:
    # 3 tâm nhưng dữ liệu có 3 giá trị, trong đó một nhóm rất lớn.
    x = np.concatenate([np.zeros(1000), np.full(1000, 1.0), np.full(1, 255.0)])
    for seed in range(20):
        res = kmeans(x, 3, seed=seed, init="random")
        assert res.counts.min() > 0, "Không được còn cụm rỗng trong kết quả cuối"
        assert res.counts.sum() == x.size


def test_max_iter_is_respected():
    rng = np.random.default_rng(0)
    x = rng.random((5000, 3)) * 255
    res = kmeans(x, 8, max_iter=2, tol=0.0, seed=0, init="random")
    assert res.n_iter == 2
    assert not res.converged
    assert len(res.sse_history) == 2


def test_convergence_threshold_controls_stopping():
    rng = np.random.default_rng(0)
    x = rng.random((5000, 2)) * 255
    strict = kmeans(x, 5, max_iter=300, tol=0.0, seed=0)
    loose = kmeans(x, 5, max_iter=300, tol=50.0, seed=0)
    assert loose.converged and loose.n_iter <= strict.n_iter
    assert loose.shift_history[-1] <= 50.0


def test_seed_reproducibility():
    rng = np.random.default_rng(5)
    x = rng.random((3000, 3)) * 255
    a = kmeans(x, 4, seed=42)
    b = kmeans(x, 4, seed=42)
    np.testing.assert_array_equal(a.labels, b.labels)
    np.testing.assert_array_equal(a.centers, b.centers)


def test_sse_is_non_increasing():
    rng = np.random.default_rng(7)
    x = rng.random((4000, 3)) * 255
    res = kmeans(x, 6, seed=1, init="random", tol=0.0)
    diffs = np.diff(res.sse_history)
    assert np.all(diffs <= 1e-6 * res.sse_history[0])


def test_output_shapes_and_pixel_counts_rgb():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (37, 53, 3), dtype=np.uint8)
    res = segment_image_kmeans(img, 4, feature="rgb", seed=0)
    assert res.labels.shape == (37, 53)
    assert res.segmented.shape == (37, 53, 3)
    assert res.segmented.dtype == np.uint8
    assert res.core.counts.sum() == 37 * 53
    assert res.core.counts.tolist() == np.bincount(res.labels.ravel(), minlength=4).tolist()


def test_labels_consistent_with_centers_and_sse():
    rng = np.random.default_rng(3)
    x = rng.random((2000, 3)) * 255
    res = kmeans(x, 5, seed=0)
    labels, d2 = assign_labels(x, res.centers)
    np.testing.assert_array_equal(labels, res.labels)
    assert res.sse == pytest.approx(d2.sum())


def test_clusters_sorted_by_brightness():
    img = np.zeros((10, 30), dtype=np.uint8)
    img[:, 10:20] = 128
    img[:, 20:] = 250
    res = segment_image_kmeans(img, 3, feature="gray", seed=9)
    assert np.all(np.diff(res.centers.ravel()) > 0)
    assert res.labels[0, 0] == 0 and res.labels[0, 15] == 1 and res.labels[0, 25] == 2


def test_chunked_distance_equals_full():
    rng = np.random.default_rng(0)
    x = rng.random((1000, 3)) * 255
    c = rng.random((4, 3)) * 255
    l1, d1 = assign_labels(x, c, chunk_size=7)
    l2, d2 = assign_labels(x, c, chunk_size=10_000)
    np.testing.assert_array_equal(l1, l2)
    np.testing.assert_allclose(d1, d2)
    brute = ((x[:, None, :] - c[None, :, :]) ** 2).sum(axis=2)
    np.testing.assert_array_equal(l1, brute.argmin(axis=1))


def test_unique_rows_packed_matches_numpy():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 4, (500, 3)).astype(float) * 60
    np.testing.assert_array_equal(unique_rows(x), np.unique(x, axis=0))


def test_weighted_equals_expanded_data():
    # Giá trị phân biệt + trọng số phải cho cùng nghiệm với dữ liệu khai triển (cùng tâm khởi tạo -> cùng kết quả)
    values = np.array([[0.0], [10.0], [20.0], [200.0], [210.0]])
    weights = np.array([5, 1, 3, 2, 4])
    expanded = np.repeat(values, weights, axis=0)
    a = kmeans(values, 2, weights=weights, init="random", seed=1)
    b = kmeans(expanded, 2, init="random", seed=1)
    np.testing.assert_allclose(a.centers, b.centers)
    assert a.sse == pytest.approx(b.sse)
    assert a.counts.tolist() == b.counts.tolist() == [9, 6]
    np.testing.assert_allclose(a.centers.ravel(), [(0 * 5 + 10 + 20 * 3) / 9, (200 * 2 + 210 * 4) / 6])


def test_image_result_is_exact_kmeans_on_all_pixels():
    """Kết quả trên ảnh (chạy trên giá trị phân biệt) phải là một điểm dừng hợp lệ của K-means trên mọi pixel."""
    rng = np.random.default_rng(11)
    img = (rng.integers(0, 6, (40, 50, 3)) * 50).astype(np.uint8)
    res = segment_image_kmeans(img, 4, feature="rgb", seed=2, tol=0.0)
    x = img.reshape(-1, 3).astype(float)
    labels, d2 = assign_labels(x, res.centers)
    np.testing.assert_array_equal(labels, res.labels.ravel())  # nhãn = tâm gần nhất của từng pixel
    assert res.core.sse == pytest.approx(d2.sum())  # SSE tính trên toàn bộ pixel
    for c in range(4):  # tâm = trung bình các pixel của cụm
        np.testing.assert_allclose(res.centers[c], x[labels == c].mean(axis=0), atol=1e-9)
    assert res.core.counts.sum() == img.shape[0] * img.shape[1]


def test_invalid_weights_rejected():
    with pytest.raises(ValueError):
        kmeans(np.arange(4.0), 2, weights=np.array([1, 1, 0, 1]))
    with pytest.raises(ValueError):
        kmeans(np.arange(4.0), 2, weights=np.ones(3))


@pytest.mark.parametrize("bad_k", [0, 9, 2.5, -1])
def test_invalid_k_rejected(bad_k):
    with pytest.raises(ValueError):
        kmeans(np.arange(10.0), bad_k)


def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        kmeans(np.arange(10.0), 2, max_iter=0)
    with pytest.raises(ValueError):
        kmeans(np.arange(10.0), 2, tol=-1)
    with pytest.raises(ValueError):
        kmeans(np.array([1.0, np.nan]), 2)
    with pytest.raises(ValueError):
        segment_image_kmeans(np.zeros((4, 4), np.uint8), 2, feature="rgb")
