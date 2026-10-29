"""Thuật toán phân cụm K-means tự cài đặt bằng NumPy.

Căn cứ lý thuyết: Giáo trình Xử lý ảnh (PTIT), mục 6.2.4.2 "Thuật toán K trung bình":
mỗi vòng lặp gồm (1) gán mỗi phần tử vào tâm gần nhất theo khoảng cách Euclid,
(2) cập nhật tâm cụm bằng trung bình các phần tử của cụm (công thức 6.10),
dừng khi tâm cụm không còn thay đổi.

Module này KHÔNG phụ thuộc giao diện và KHÔNG dùng sklearn/OpenCV.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

# Số điểm xử lý mỗi lần khi tính ma trận khoảng cách, để bộ nhớ tạm luôn nhỏ:
# với chunk 262 144 điểm và K = 8, ma trận khoảng cách chỉ ~16 MB (float64).
DEFAULT_CHUNK_SIZE = 262_144

MIN_K = 1
MAX_K = 8


@dataclass
class KMeansResult:
    """Kết quả phân cụm trên dữ liệu dạng (N, D)."""

    labels: np.ndarray  # (N,) int32, nhãn cụm 0..k_effective-1
    centers: np.ndarray  # (k_effective, D) float64
    counts: np.ndarray  # (k_effective,) int64, số điểm mỗi cụm
    sse: float  # tổng bình phương khoảng cách tới tâm (inertia)
    n_iter: int  # số vòng lặp thực tế đã chạy
    converged: bool  # True nếu dừng do độ dịch tâm <= tol
    k_requested: int
    k_effective: int
    sse_history: list[float] = field(default_factory=list)
    shift_history: list[float] = field(default_factory=list)
    centers_history: list[np.ndarray] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    init: str = "kmeans++"
    seed: int | None = None
    warnings: list[str] = field(default_factory=list)


def _validate_params(k: int, max_iter: int, tol: float, init: str) -> None:
    if not isinstance(k, (int, np.integer)) or not MIN_K <= int(k) <= MAX_K:
        raise ValueError(f"K phải là số nguyên trong khoảng [{MIN_K}, {MAX_K}], nhận được {k!r}.")
    if not isinstance(max_iter, (int, np.integer)) or int(max_iter) < 1:
        raise ValueError(f"Số vòng lặp tối đa phải là số nguyên >= 1, nhận được {max_iter!r}.")
    if not np.isfinite(tol) or tol < 0:
        raise ValueError(f"Ngưỡng hội tụ phải là số thực >= 0, nhận được {tol!r}.")
    if init not in ("kmeans++", "random"):
        raise ValueError(f"Phương pháp khởi tạo không hợp lệ: {init!r} (chỉ hỗ trợ 'kmeans++' hoặc 'random').")


def unique_rows(data: np.ndarray) -> np.ndarray:
    """Trả về các vector đặc trưng phân biệt của dữ liệu (N, D).

    Với dữ liệu nguyên 0..255 (trường hợp ảnh), gộp D kênh thành một số nguyên
    để np.unique chạy trên mảng 1 chiều — nhanh hơn nhiều so với unique theo hàng.
    """
    if data.shape[0] == 0:
        return data.copy()
    is_byte_valued = (
        data.shape[1] <= 3
        and np.all(data >= 0)
        and np.all(data <= 255)
        and np.all(data == np.round(data))
    )
    if is_byte_valued:
        ints = data.astype(np.int64)
        packed = np.zeros(ints.shape[0], dtype=np.int64)
        for c in range(ints.shape[1]):
            packed = packed * 256 + ints[:, c]
        uniq = np.unique(packed)
        out = np.empty((uniq.size, data.shape[1]), dtype=np.float64)
        for c in range(data.shape[1] - 1, -1, -1):
            out[:, c] = uniq % 256
            uniq = uniq // 256
        return out
    return np.unique(data, axis=0)


def assign_labels(
    data: np.ndarray, centers: np.ndarray, chunk_size: int = DEFAULT_CHUNK_SIZE
) -> tuple[np.ndarray, np.ndarray]:
    """Gán mỗi điểm vào tâm gần nhất (khoảng cách Euclid).

    Dùng khai triển ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2 và xử lý theo từng khối
    `chunk_size` điểm để không tạo ma trận khoảng cách N x K quá lớn.

    Returns:
        labels: (N,) int32 — chỉ số tâm gần nhất (hoà thì lấy chỉ số nhỏ nhất).
        min_sq_dist: (N,) float64 — bình phương khoảng cách tới tâm đó.
    """
    n = data.shape[0]
    labels = np.empty(n, dtype=np.int32)
    min_sq = np.empty(n, dtype=np.float64)
    center_sq = np.einsum("kd,kd->k", centers, centers)
    for start in range(0, n, chunk_size):
        block = data[start : start + chunk_size]
        block_sq = np.einsum("nd,nd->n", block, block)
        d2 = block_sq[:, None] - 2.0 * (block @ centers.T) + center_sq[None, :]
        np.maximum(d2, 0.0, out=d2)  # loại sai số làm tròn âm
        idx = np.argmin(d2, axis=1)
        labels[start : start + chunk_size] = idx
        min_sq[start : start + chunk_size] = d2[np.arange(block.shape[0]), idx]
    return labels, min_sq


def _init_random(distinct: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Chọn ngẫu nhiên k vector PHÂN BIỆT làm tâm ban đầu (tránh hai tâm trùng nhau)."""
    idx = rng.choice(distinct.shape[0], size=k, replace=False)
    return distinct[idx].astype(np.float64)


def _sample_index(prob_mass: np.ndarray, rng: np.random.Generator) -> int:
    """Lấy một chỉ số với xác suất tỉ lệ prob_mass (dùng hàm phân phối tích luỹ)."""
    cdf = np.cumsum(prob_mass)
    idx = int(np.searchsorted(cdf, rng.random() * cdf[-1], side="right"))
    return min(idx, prob_mass.size - 1)


def _init_kmeans_pp(data: np.ndarray, weights: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Khởi tạo K-means++ (Arthur & Vassilvitskii, 2007) — kiến thức bổ sung ngoài giáo trình.

    Tâm đầu tiên chọn ngẫu nhiên; mỗi tâm tiếp theo được chọn với xác suất tỉ lệ với
    w(x) * D(x)^2 — D(x) là khoảng cách từ x tới tâm gần nhất đã chọn, w(x) là trọng số
    (số pixel có cùng giá trị x). Điểm trùng tâm đã chọn có xác suất 0 nên các tâm luôn
    phân biệt (khi số giá trị phân biệt >= k).
    """
    centers = np.empty((k, data.shape[1]), dtype=np.float64)
    centers[0] = data[_sample_index(weights, rng)]
    closest = np.einsum("nd,nd->n", data - centers[0], data - centers[0])
    for j in range(1, k):
        mass = weights * closest
        if mass.sum() <= 0:  # không còn điểm nào khác các tâm đã chọn
            raise RuntimeError("K-means++: không đủ điểm phân biệt để khởi tạo.")
        centers[j] = data[_sample_index(mass, rng)]
        diff = data - centers[j]
        np.minimum(closest, np.einsum("nd,nd->n", diff, diff), out=closest)
    return centers


def _relabel_by_brightness(centers: np.ndarray) -> np.ndarray:
    """Thứ tự cụm tăng dần theo độ sáng của tâm — để kết quả ổn định và dễ đọc.

    Với 3 kênh dùng độ chói BT.601 (0.299R + 0.587G + 0.114B); với 1 kênh dùng giá trị.
    """
    if centers.shape[1] == 3:
        key = centers @ np.array([0.299, 0.587, 0.114])
    else:
        key = centers.mean(axis=1)
    return np.argsort(key, kind="stable")


def kmeans(
    data: np.ndarray,
    k: int,
    max_iter: int = 100,
    tol: float = 1e-4,
    seed: int | None = 0,
    init: str = "kmeans++",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    weights: np.ndarray | None = None,
) -> KMeansResult:
    """Phân cụm K-means (thuật toán Lloyd) cho dữ liệu (N, D).

    Có trọng số: điểm i được tính như w_i điểm trùng nhau, tức hàm mục tiêu là
    J = sum_i w_i * ||x_i - Z_c(i)||^2 và tâm Z_k = sum w_i x_i / sum w_i. Với ảnh, truyền
    các giá trị pixel PHÂN BIỆT và w = số pixel của mỗi giá trị cho kết quả tương đương
    K-means trên toàn bộ pixel nhưng nhanh hơn nhiều (ảnh xám chỉ còn tối đa 256 điểm).

    Args:
        data: mảng (N, D) hoặc (N,) các vector đặc trưng.
        weights: (N,) trọng số dương; None = mọi điểm trọng số 1.
        k: số cụm yêu cầu (1..8). Nếu dữ liệu có ít giá trị phân biệt hơn k,
           k được giảm xuống bằng số giá trị phân biệt (có cảnh báo).
        max_iter: số vòng lặp tối đa.
        tol: ngưỡng hội tụ — dừng khi độ dịch chuyển lớn nhất của tâm (chuẩn Euclid) <= tol.
        seed: hạt giống ngẫu nhiên để tái lập kết quả.
        init: 'kmeans++' hoặc 'random'.
        chunk_size: số điểm mỗi khối khi tính khoảng cách.

    Returns:
        KMeansResult — nhãn được sắp xếp sao cho cụm 0 có tâm tối nhất.
    """
    _validate_params(k, max_iter, tol, init)
    k = int(k)
    max_iter = int(max_iter)
    x = np.asarray(data, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("Dữ liệu phải là mảng 2 chiều (N, D) với N >= 1.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Dữ liệu chứa giá trị NaN hoặc vô cực.")
    if weights is None:
        w = np.ones(x.shape[0], dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (x.shape[0],) or not np.all(np.isfinite(w)) or np.any(w <= 0):
            raise ValueError("Trọng số phải là mảng (N,) các số dương hữu hạn.")

    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    warnings: list[str] = []

    distinct = unique_rows(x)
    k_eff = min(k, distinct.shape[0])
    if k_eff < k:
        warnings.append(
            f"Dữ liệu chỉ có {distinct.shape[0]} giá trị phân biệt nên K được giảm từ {k} xuống {k_eff}."
        )

    centers = _init_kmeans_pp(x, w, k_eff, rng) if init == "kmeans++" else _init_random(distinct, k_eff, rng)

    sse_history: list[float] = []
    shift_history: list[float] = []
    centers_history: list[np.ndarray] = [centers.copy()]
    converged = False
    n_iter = 0
    n_empty_events = 0

    for it in range(1, max_iter + 1):
        n_iter = it
        # Bước 1: gán nhãn theo tâm gần nhất
        labels, min_sq = assign_labels(x, centers, chunk_size)
        sse_history.append(float(np.dot(w, min_sq)))

        # Bước 2: cập nhật tâm = trung bình (có trọng số) các điểm trong cụm, dùng np.bincount theo từng chiều
        counts = np.bincount(labels, weights=w, minlength=k_eff)
        new_centers = np.empty_like(centers)
        for d in range(x.shape[1]):
            sums = np.bincount(labels, weights=w * x[:, d], minlength=k_eff)
            with np.errstate(invalid="ignore", divide="ignore"):
                new_centers[:, d] = sums / counts

        # Xử lý cụm rỗng: đặt tâm vào điểm xa tâm hiện tại của nó nhất (điểm "khó" nhất).
        empty = np.flatnonzero(counts == 0)
        if empty.size:
            n_empty_events += int(empty.size)
            order = np.argsort(min_sq)[::-1]  # điểm xa nhất trước
            used: list[np.ndarray] = [new_centers[j] for j in range(k_eff) if counts[j] > 0]
            pos = 0
            for j in empty:
                while pos < order.size and any(np.array_equal(x[order[pos]], u) for u in used):
                    pos += 1
                if pos >= order.size:  # không thể xảy ra khi k_eff <= số giá trị phân biệt
                    new_centers[j] = centers[j]
                    continue
                new_centers[j] = x[order[pos]]
                used.append(new_centers[j])
                pos += 1

        shift = float(np.max(np.sqrt(np.sum((new_centers - centers) ** 2, axis=1))))
        shift_history.append(shift)
        centers = new_centers
        centers_history.append(centers.copy())
        if shift <= tol and empty.size == 0:
            converged = True
            break

    if n_empty_events:
        warnings.append(f"Đã xử lý {n_empty_events} lần cụm rỗng (khởi tạo lại tâm tại điểm xa nhất).")
    if not converged:
        warnings.append(f"Chưa hội tụ sau {max_iter} vòng lặp (độ dịch tâm cuối = {shift_history[-1]:.4g}).")

    # Gán nhãn lần cuối theo tâm cuối cùng để nhãn, số điểm và SSE nhất quán với tâm báo cáo.
    labels, min_sq = assign_labels(x, centers, chunk_size)

    # Sắp xếp lại thứ tự cụm theo độ sáng tâm
    order = _relabel_by_brightness(centers)
    remap = np.empty(k_eff, dtype=np.int32)
    remap[order] = np.arange(k_eff, dtype=np.int32)
    labels = remap[labels]
    centers = centers[order]
    centers_history = [c[order] for c in centers_history]
    counts_f = np.bincount(labels, weights=w, minlength=k_eff)
    counts = np.rint(counts_f).astype(np.int64) if np.all(w == np.rint(w)) else counts_f

    return KMeansResult(
        labels=labels,
        centers=centers,
        counts=counts,
        sse=float(np.dot(w, min_sq)),
        n_iter=n_iter,
        converged=converged,
        k_requested=k,
        k_effective=k_eff,
        sse_history=sse_history,
        shift_history=shift_history,
        centers_history=centers_history,
        elapsed_seconds=time.perf_counter() - t0,
        init=init,
        seed=seed,
        warnings=warnings,
    )


@dataclass
class ImageKMeansResult:
    """Kết quả K-means trên ảnh."""

    labels: np.ndarray  # (H, W) int32
    segmented: np.ndarray  # (H, W) hoặc (H, W, 3) uint8 — mỗi pixel thay bằng màu tâm cụm
    centers: np.ndarray  # (k, D) float64 trong thang 0..255
    feature: str  # 'gray' hoặc 'rgb'
    core: KMeansResult


def image_to_features(image: np.ndarray, feature: str) -> np.ndarray:
    """Chuyển ảnh thành ma trận đặc trưng (H*W, D).

    feature='gray': D = 1 (mức xám); feature='rgb': D = 3 (R, G, B).
    Ảnh 2 chiều chỉ cho phép 'gray'.
    """
    img = np.asarray(image)
    if feature == "gray":
        if img.ndim == 3:
            raise ValueError("Ảnh màu cần được chuyển sang mức xám trước khi dùng đặc trưng 'gray'.")
        return img.reshape(-1, 1).astype(np.float64)
    if feature == "rgb":
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("Đặc trưng 'rgb' yêu cầu ảnh 3 kênh (H, W, 3).")
        return img.reshape(-1, 3).astype(np.float64)
    raise ValueError(f"Loại đặc trưng không hợp lệ: {feature!r}.")


def unique_pixels(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gom các pixel cùng giá trị: trả về (giá trị phân biệt (U, D), chỉ số ngược (N,), số pixel (U,)).

    Mỗi pixel 0..255 với D <= 3 kênh được gộp thành một số nguyên R*65536 + G*256 + B
    để np.unique chạy trên mảng 1 chiều.
    """
    ints = features.astype(np.int64)
    packed = np.zeros(ints.shape[0], dtype=np.int64)
    for c in range(ints.shape[1]):
        packed = packed * 256 + ints[:, c]
    uniq, inverse, counts = np.unique(packed, return_inverse=True, return_counts=True)
    values = np.empty((uniq.size, ints.shape[1]), dtype=np.float64)
    rest = uniq.copy()
    for c in range(ints.shape[1] - 1, -1, -1):
        values[:, c] = rest % 256
        rest //= 256
    return values, inverse.ravel(), counts


def segment_image_kmeans(
    image: np.ndarray,
    k: int,
    feature: str = "rgb",
    max_iter: int = 100,
    tol: float = 1e-4,
    seed: int | None = 0,
    init: str = "kmeans++",
) -> ImageKMeansResult:
    """Phân đoạn ảnh bằng K-means: mỗi pixel là một vector đặc trưng."""
    img = np.asarray(image)
    if img.ndim not in (2, 3):
        raise ValueError("Ảnh phải có dạng (H, W) hoặc (H, W, 3).")
    if img.dtype != np.uint8:
        raise ValueError("Ảnh phải có kiểu uint8 (giá trị 0..255).")
    h, w = img.shape[:2]
    t0 = time.perf_counter()
    values, inverse, counts = unique_pixels(image_to_features(img, feature))
    # K-means có trọng số trên các giá trị phân biệt == K-means trên toàn bộ pixel
    core = kmeans(values, k, max_iter=max_iter, tol=tol, seed=seed, init=init, weights=counts)
    core.labels = core.labels[inverse]  # trả nhãn về cho từng pixel
    core.elapsed_seconds = time.perf_counter() - t0
    labels = core.labels.reshape(h, w)
    palette = np.clip(np.round(core.centers), 0, 255).astype(np.uint8)
    segmented = palette[labels]
    if feature == "gray":
        segmented = segmented[..., 0]
    return ImageKMeansResult(labels=labels, segmented=segmented, centers=core.centers, feature=feature, core=core)
