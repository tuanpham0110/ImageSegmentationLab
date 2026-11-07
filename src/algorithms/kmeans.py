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
from functools import wraps

import numpy as np

# Số điểm xử lý mỗi lần khi tính ma trận khoảng cách, để bộ nhớ tạm luôn nhỏ:
# Mỗi khối chỉ giữ hiệu N×D và khoảng cách tới một tâm, không tạo N×K×D.
DEFAULT_CHUNK_SIZE = 262_144

MIN_K = 1
MAX_K = 8
NUMERIC_RANGE_ERROR = "Dữ liệu hoặc trọng số vượt phạm vi tính toán an toàn của float64. Hãy giảm biên độ hoặc trọng số."


def _checked_arithmetic(function):
    """Chuyển lỗi số học thành lỗi dữ liệu rõ ràng thay vì trả NaN/Inf."""
    @wraps(function)
    def checked(*args, **kwargs):
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                return function(*args, **kwargs)
        except FloatingPointError as exc:
            raise ValueError(NUMERIC_RANGE_ERROR) from exc
    return checked


def _weighted_sse(weights: np.ndarray, squared_distances: np.ndarray) -> float:
    value = float(np.dot(weights, squared_distances))
    if not np.isfinite(value):
        raise ValueError(NUMERIC_RANGE_ERROR)
    return value


@dataclass
class KMeansResult:
    """Kết quả phân cụm trên dữ liệu dạng (N, D)."""

    labels: np.ndarray  # (N,) int32, nhãn cụm 0..k_effective-1
    centers: np.ndarray  # (k_effective, D) float64
    counts: np.ndarray  # int64 cho số pixel; float64 cho trọng số lẻ hoặc tổng ngoài miền int64
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


def _validate_chunk_size(chunk_size: int) -> None:
    if isinstance(chunk_size, (bool, np.bool_)) or not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
        raise ValueError("Kích thước khối phải là số nguyên >= 1.")


def _validate_params(k: int, max_iter: int, tol: float, init: str,
                     chunk_size: int = DEFAULT_CHUNK_SIZE, seed: int | None = 0) -> None:
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or not MIN_K <= int(k) <= MAX_K:
        raise ValueError(f"K phải là số nguyên trong khoảng [{MIN_K}, {MAX_K}], nhận được {k!r}.")
    if isinstance(max_iter, (bool, np.bool_)) or not isinstance(max_iter, (int, np.integer)) or int(max_iter) < 1:
        raise ValueError(f"Số vòng lặp tối đa phải là số nguyên >= 1, nhận được {max_iter!r}.")
    if (isinstance(tol, (bool, np.bool_)) or not isinstance(tol, (int, float, np.integer, np.floating))
            or not np.isfinite(tol) or tol < 0):
        raise ValueError(f"Ngưỡng hội tụ phải là số thực >= 0, nhận được {tol!r}.")
    if init not in ("kmeans++", "random"):
        raise ValueError(f"Phương pháp khởi tạo không hợp lệ: {init!r} (chỉ hỗ trợ 'kmeans++' hoặc 'random').")
    _validate_chunk_size(chunk_size)
    if seed is not None and (isinstance(seed, (bool, np.bool_))
                             or not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError("Random seed phải là số nguyên >= 0 hoặc None.")


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


@_checked_arithmetic
def assign_labels(
    data: np.ndarray, centers: np.ndarray, chunk_size: int = DEFAULT_CHUNK_SIZE
) -> tuple[np.ndarray, np.ndarray]:
    """Gán mỗi điểm vào tâm gần nhất (khoảng cách Euclid).

    Tính trực tiếp ||x - c||^2 theo từng tâm và khối `chunk_size` điểm.
    Phép trừ trước khi bình phương tránh triệt tiêu số khi các vector có độ lệch lớn;
    không tạo ma trận khoảng cách N x K x D.

    Returns:
        labels: (N,) int32 — chỉ số tâm gần nhất (hoà thì lấy chỉ số nhỏ nhất).
        min_sq_dist: (N,) float64 — bình phương khoảng cách tới tâm đó.
    """
    _validate_chunk_size(chunk_size)
    data, centers = np.asarray(data), np.asarray(centers)
    if (data.ndim != 2 or centers.ndim != 2 or data.shape[0] == 0 or centers.shape[0] == 0
            or data.shape[1] == 0 or data.shape[1] != centers.shape[1]):
        raise ValueError("Dữ liệu và tâm phải là các ma trận không rỗng, có cùng số chiều đặc trưng.")
    if data.dtype.kind not in "buif" or centers.dtype.kind not in "buif":
        raise ValueError("Dữ liệu và tâm phải chứa số thực hữu hạn.")
    data, centers = data.astype(np.float64, copy=False), centers.astype(np.float64, copy=False)
    if not np.all(np.isfinite(data)) or not np.all(np.isfinite(centers)):
        raise ValueError("Dữ liệu và tâm phải chứa số thực hữu hạn.")
    labels, min_sq = _assign_labels(data, centers, chunk_size)
    if not np.all(np.isfinite(min_sq)):
        raise ValueError(NUMERIC_RANGE_ERROR)
    return labels, min_sq


def _assign_labels(data: np.ndarray, centers: np.ndarray, chunk_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Đường nội bộ cho float64 đã kiểm tra; không lặp lại kiểm tra N×D ở mỗi vòng Lloyd."""
    n = data.shape[0]
    labels = np.empty(n, dtype=np.int32)
    min_sq = np.empty(n, dtype=np.float64)
    for start in range(0, n, chunk_size):
        block = data[start : start + chunk_size]
        diff = block - centers[0]
        best = np.einsum("nd,nd->n", diff, diff)
        block_labels = np.zeros(block.shape[0], dtype=np.int32)
        for j in range(1, centers.shape[0]):
            center = centers[j]
            diff = block - center
            d2 = np.einsum("nd,nd->n", diff, diff)
            np.copyto(block_labels, j, where=d2 < best)  # Hoà thì giữ chỉ số nhỏ hơn.
            np.minimum(best, d2, out=best)
        labels[start : start + chunk_size] = block_labels
        min_sq[start : start + chunk_size] = best
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
        total_mass = mass.sum()
        if not np.isfinite(total_mass) or total_mass <= 0:
            # k_eff đã được chặn bởi số điểm phân biệt. Khối lượng bằng 0/Inf
            # ở đây là mất khả năng biểu diễn số học, không phải thiếu điểm.
            raise ValueError(NUMERIC_RANGE_ERROR)
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


@_checked_arithmetic
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
    _validate_params(k, max_iter, tol, init, chunk_size, seed)
    k = int(k)
    max_iter = int(max_iter)
    x = np.asarray(data)
    if x.dtype.kind not in "buif":
        raise ValueError("Dữ liệu phải chứa số thực hữu hạn.")
    x = x.astype(np.float64, copy=False)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("Dữ liệu phải là mảng 2 chiều (N, D) với N >= 1 và D >= 1.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Dữ liệu chứa giá trị NaN hoặc vô cực.")
    if weights is None:
        w = np.ones(x.shape[0], dtype=np.float64)
    else:
        w = np.asarray(weights)
        if w.dtype.kind not in "buif":
            raise ValueError("Trọng số phải là mảng (N,) các số dương hữu hạn.")
        w = w.astype(np.float64, copy=False)
        if w.shape != (x.shape[0],) or not np.all(np.isfinite(w)) or np.any(w <= 0):
            raise ValueError("Trọng số phải là mảng (N,) các số dương hữu hạn.")
    total_weight = float(w.sum())
    if not np.isfinite(total_weight):
        raise ValueError(NUMERIC_RANGE_ERROR)

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
        # Bước 1: gán nhãn theo tâm gần nhất. Hoà khoảng cách được phá theo thứ tự độ sáng
        # của tâm — đúng quy tắc của bước gán nhãn cuối — nên tâm tính ra ở vòng này khớp
        # với nhãn trả về. Chỉ số tâm nội bộ không đổi để lịch sử tâm theo dõi đúng từng cụm.
        rank = _relabel_by_brightness(centers)
        sorted_labels, min_sq = _assign_labels(x, centers[rank], chunk_size)
        labels = rank.astype(np.int32)[sorted_labels]
        sse_history.append(_weighted_sse(w, min_sq))

        # Bước 2: cập nhật tâm = trung bình (có trọng số) các điểm trong cụm, dùng np.bincount theo từng chiều
        counts = np.bincount(labels, weights=w, minlength=k_eff)
        new_centers = np.empty_like(centers)
        for d in range(x.shape[1]):
            sums = np.bincount(labels, weights=w * x[:, d], minlength=k_eff)
            if not np.all(np.isfinite(sums)):
                raise ValueError(NUMERIC_RANGE_ERROR)
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

        if not np.all(np.isfinite(new_centers)):
            raise ValueError(NUMERIC_RANGE_ERROR)
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

    # Sắp xếp TRƯỚC khi gán nhãn cuối để quy tắc hoà (chỉ số nhỏ nhất) dùng
    # đúng thứ tự tâm được báo cáo, kể cả khi dừng sớm do max_iter/tol.
    order = _relabel_by_brightness(centers)
    centers = centers[order]
    centers_history = [c[order] for c in centers_history]
    labels, min_sq = _assign_labels(x, centers, chunk_size)
    counts_f = np.bincount(labels, weights=w, minlength=k_eff)
    fits_integer_counts = (np.all(w == np.rint(w)) and total_weight < 2**63
                           and np.all(counts_f < 2**63))
    counts = np.rint(counts_f).astype(np.int64) if fits_integer_counts else counts_f
    # Gán nhãn cuối có thể làm một cụm rỗng nếu dừng sớm do tol/max_iter.
    # Không chạy lén thêm vòng vượt giới hạn, cũng không báo hội tụ trong trường hợp này.
    if np.any(counts_f == 0):
        converged = False
        warnings.append(f"Kết quả cuối còn {np.count_nonzero(counts_f == 0)} cụm rỗng sau bước gán nhãn; "
                        "chưa hội tụ. Hãy tăng max_iter hoặc giảm tol.")
    elif converged:
        # Bảo đảm "hội tụ" nghĩa là tâm báo cáo đúng là trung bình của các cụm theo nhãn trả về.
        final_means = np.empty_like(centers)
        for d in range(x.shape[1]):
            final_means[:, d] = np.bincount(labels, weights=w * x[:, d], minlength=k_eff) / counts_f
        gap = float(np.max(np.sqrt(np.sum((final_means - centers) ** 2, axis=1))))
        if not gap <= tol:
            converged = False
            warnings.append(f"Tâm cuối lệch {gap:.4g} so với trung bình cụm theo nhãn cuối (> tol); "
                            "chưa hội tụ. Hãy tăng max_iter hoặc giảm tol.")

    return KMeansResult(
        labels=labels,
        centers=centers,
        counts=counts,
        sse=_weighted_sse(w, min_sq),
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


def image_to_features(image: np.ndarray, feature: str, *, dtype=np.float64) -> np.ndarray:
    """Chuyển ảnh thành ma trận đặc trưng (H*W, D).

    feature='gray': D = 1 (mức xám); feature='rgb': D = 3 (R, G, B).
    Ảnh 2 chiều chỉ cho phép 'gray'. dtype mặc định float64; đường xử lý ảnh
    giữ uint8 ở bước gom màu để tránh bản sao float64 trên toàn bộ pixel.
    """
    img = np.asarray(image)
    if feature == "gray":
        if img.ndim != 2:
            raise ValueError("Đặc trưng 'gray' yêu cầu ảnh xám 2 chiều; ảnh màu cần được chuyển xám trước.")
        return img.reshape(-1, 1).astype(dtype, copy=False)
    if feature == "rgb":
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("Đặc trưng 'rgb' yêu cầu ảnh 3 kênh (H, W, 3).")
        return img.reshape(-1, 3).astype(dtype, copy=False)
    raise ValueError(f"Loại đặc trưng không hợp lệ: {feature!r}.")


def unique_pixels(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gom các pixel cùng giá trị: trả về (giá trị phân biệt (U, D), chỉ số ngược (N,), số pixel (U,)).

    Mỗi pixel 0..255 với D <= 3 kênh được gộp thành một số nguyên R*65536 + G*256 + B
    để np.unique chạy trên mảng 1 chiều.
    """
    features = np.asarray(features)
    if features.ndim != 2 or not 1 <= features.shape[1] <= 3:
        raise ValueError("Pixel phải có 1 đến 3 kênh.")
    if features.dtype != np.uint8:
        if (features.dtype.kind not in "buif" or not np.all(np.isfinite(features))
                or np.any(features < 0) or np.any(features > 255)
                or np.any(features != np.round(features))):
            raise ValueError("Pixel phải có giá trị nguyên trong [0, 255].")
    # Tối đa 24 bit RGB, vừa int32. Không tạo thêm ma trận int64 N×D.
    packed = np.zeros(features.shape[0], dtype=np.int32)
    for c in range(features.shape[1]):
        packed *= 256
        np.add(packed, features[:, c], out=packed, casting="unsafe")
    uniq, inverse, counts = np.unique(packed, return_inverse=True, return_counts=True)
    values = np.empty((uniq.size, features.shape[1]), dtype=np.float64)
    rest = uniq.copy()
    for c in range(features.shape[1] - 1, -1, -1):
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
    if h == 0 or w == 0:
        raise ValueError("Ảnh rỗng: chiều rộng và chiều cao phải >= 1.")
    _validate_params(k, max_iter, tol, init, seed=seed)
    t0 = time.perf_counter()
    values, inverse, counts = unique_pixels(image_to_features(img, feature, dtype=np.uint8))
    # K-means có trọng số trên các giá trị phân biệt == K-means trên toàn bộ pixel
    core = kmeans(values, k, max_iter=max_iter, tol=tol, seed=seed, init=init, weights=counts)
    core.labels = core.labels[inverse]  # trả nhãn về cho từng pixel
    labels = core.labels.reshape(h, w)
    palette = np.clip(np.round(core.centers), 0, 255).astype(np.uint8)
    segmented = palette[labels]
    if feature == "gray":
        segmented = segmented[..., 0]
    # Đo tới kết quả trực tiếp của thuật toán, gồm cả phục hồi nhãn và ảnh tâm.
    core.elapsed_seconds = time.perf_counter() - t0
    return ImageKMeansResult(labels=labels, segmented=segmented, centers=core.centers, feature=feature, core=core)
