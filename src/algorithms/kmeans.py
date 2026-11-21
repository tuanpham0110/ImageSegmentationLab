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

from src.utils.color_space import lab_to_rgb, rgb_to_lab

# Số điểm xử lý mỗi lần khi tính ma trận khoảng cách, để bộ nhớ tạm luôn nhỏ:
# Mỗi khối chỉ giữ hiệu N×D và khoảng cách tới một tâm, không tạo N×K×D.
DEFAULT_CHUNK_SIZE = 65_536

# K-means tổng quát chạy với mọi K >= 1 (K thực tế bị chặn bởi số giá trị phân biệt của dữ liệu).
# Không có chặn trên trong thuật toán; giao diện tự đặt giới hạn thanh trượt cho tiện dùng.
MIN_K = 1
FEATURES = ("gray", "rgb", "lab")
# Trọng số "độ sáng" để sắp thứ tự cụm: BT.601 cho RGB, chỉ kênh L* cho Lab.
LUMA_ORDER = np.array([0.299, 0.587, 0.114])
LIGHTNESS_ORDER = np.array([1.0, 0.0, 0.0])
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
    initial_sse: float = 0.0  # SSE với tâm khởi tạo; lịch sử lưu SSE SAU từng vòng Lloyd


def _validate_chunk_size(chunk_size: int) -> None:
    if isinstance(chunk_size, (bool, np.bool_)) or not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
        raise ValueError("Kích thước khối phải là số nguyên >= 1.")


def _validate_params(k: int, max_iter: int, tol: float, init: str,
                     chunk_size: int = DEFAULT_CHUNK_SIZE, seed: int | None = 0) -> None:
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or int(k) < MIN_K:
        raise ValueError(f"K phải là số nguyên >= {MIN_K}, nhận được {k!r}.")
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
    is_byte_valued = data.shape[1] <= 3
    if is_byte_valued:
        # Không tạo ma trận round/int64 N×D chỉ để nhận diện dữ liệu ảnh.
        for start in range(0, data.shape[0], DEFAULT_CHUNK_SIZE):
            block = data[start:start + DEFAULT_CHUNK_SIZE]
            if np.any(block < 0) or np.any(block > 255) or not np.array_equal(block, np.round(block)):
                is_byte_valued = False
                break
    if is_byte_valued:
        packed = np.zeros(data.shape[0], dtype=np.int32)
        for c in range(data.shape[1]):
            packed *= 256
            np.add(packed, data[:, c], out=packed, casting="unsafe")
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


def _squared_distance(block: np.ndarray, center: np.ndarray, diff: np.ndarray, out: np.ndarray) -> None:
    np.subtract(block, center, out=diff)
    np.einsum("nd,nd->n", diff, diff, out=out)


def _assign_labels(data: np.ndarray, centers: np.ndarray, chunk_size: int,
                   out: tuple[np.ndarray, np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Đường nội bộ cho float64 đã kiểm tra; không lặp lại kiểm tra N×D ở mỗi vòng Lloyd."""
    n = data.shape[0]
    if out is None:
        out = (np.empty(n, dtype=np.int32), np.empty(n, dtype=np.float64))
    labels, min_sq = out
    size = min(n, chunk_size)
    diff = np.empty_like(data[:size])
    distance = np.empty(size, dtype=np.float64)
    better = np.empty(size, dtype=bool)
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        length = stop - start
        block, best, block_labels = data[start:stop], min_sq[start:stop], labels[start:stop]
        scratch, d2, closer = diff[:length], distance[:length], better[:length]
        block_labels.fill(0)
        _squared_distance(block, centers[0], scratch, best)
        for j in range(1, centers.shape[0]):
            _squared_distance(block, centers[j], scratch, d2)
            np.less(d2, best, out=closer)
            np.copyto(block_labels, j, where=closer)  # Hoà thì giữ chỉ số nhỏ hơn.
            np.minimum(best, d2, out=best)
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


def _init_kmeans_pp(data: np.ndarray, weights: np.ndarray, k: int, rng: np.random.Generator,
                    chunk_size: int) -> np.ndarray:
    """Khởi tạo K-means++ (Arthur & Vassilvitskii, 2007) — kiến thức bổ sung ngoài giáo trình.

    Tâm đầu tiên chọn ngẫu nhiên; mỗi tâm tiếp theo được chọn với xác suất tỉ lệ với
    w(x) * D(x)^2 — D(x) là khoảng cách từ x tới tâm gần nhất đã chọn, w(x) là trọng số
    (số pixel có cùng giá trị x). Điểm trùng tâm đã chọn có xác suất 0 nên các tâm luôn
    phân biệt (khi số giá trị phân biệt >= k).
    """
    centers = np.empty((k, data.shape[1]), dtype=np.float64)
    centers[0] = data[_sample_index(weights, rng)]
    if k == 1:
        return centers
    n = data.shape[0]
    size = min(n, chunk_size)
    closest = np.empty(n, dtype=np.float64)
    mass = np.empty_like(closest)
    diff = np.empty_like(data[:size])
    distance = np.empty(size, dtype=np.float64)
    for start in range(0, n, size):
        block = data[start:start + size]
        _squared_distance(block, centers[0], diff[:len(block)], closest[start:start + size])
    for j in range(1, k):
        np.multiply(weights, closest, out=mass)
        total_mass = mass.sum()
        if not np.isfinite(total_mass) or total_mass <= 0:
            # k_eff đã được chặn bởi số điểm phân biệt. Khối lượng bằng 0/Inf
            # ở đây là mất khả năng biểu diễn số học, không phải thiếu điểm.
            raise ValueError(NUMERIC_RANGE_ERROR)
        centers[j] = data[_sample_index(mass, rng)]
        if j + 1 < k:
            for start in range(0, n, size):
                block = data[start:start + size]
                best = closest[start:start + size]
                d2 = distance[:len(block)]
                _squared_distance(block, centers[j], diff[:len(block)], d2)
                np.minimum(best, d2, out=best)
    return centers


def _weighted_means(x: np.ndarray, labels: np.ndarray, w: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """(trung bình có trọng số mỗi cụm (k, D), tổng trọng số mỗi cụm (k,)); cụm rỗng cho NaN."""
    # Mỗi màu đại diện cho w pixel giống nhau. Mỗi chiều tâm dùng tổng w[i] * x[i, d]
    # chia tổng w[i] của cụm, tương đương cập nhật tâm trên toàn bộ pixel của ảnh.
    indices = labels.astype(np.intp, copy=False)
    counts = np.bincount(indices, weights=w, minlength=k)
    means = np.empty((k, x.shape[1]), dtype=np.float64)
    weighted = np.empty_like(w)
    for d in range(x.shape[1]):
        np.multiply(w, x[:, d], out=weighted)
        sums = np.bincount(indices, weights=weighted, minlength=k)
        if not np.all(np.isfinite(sums)):
            raise ValueError(NUMERIC_RANGE_ERROR)
        with np.errstate(invalid="ignore", divide="ignore"):
            means[:, d] = sums / counts
    return means, counts


def _assign_by_brightness(x: np.ndarray, centers: np.ndarray, chunk_size: int, order_weights,
                          out: tuple[np.ndarray, np.ndarray]):
    """Gán nhãn, phá hoà khoảng cách theo thứ tự độ sáng của tâm (quy tắc của kết quả báo cáo).

    Trả về (thứ tự tâm, nhãn theo thứ tự độ sáng, khoảng cách bình phương).
    """
    rank = _relabel_by_brightness(centers, order_weights)
    sorted_labels, min_sq = _assign_labels(x, centers[rank], chunk_size, out)
    return rank, sorted_labels, min_sq


def _max_shift(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.sqrt(np.sum((a - b) ** 2, axis=1))))


def _relabel_by_brightness(centers: np.ndarray, order_weights: np.ndarray | None = None) -> np.ndarray:
    """Thứ tự cụm tăng dần theo độ sáng của tâm — để kết quả ổn định và dễ đọc.

    order_weights: trọng số tổ hợp tuyến tính các kênh làm khoá sắp xếp. Mặc định: 3 kênh dùng
    độ chói BT.601 (0.299R + 0.587G + 0.114B); số kênh khác dùng trung bình. Với Lab truyền (1, 0, 0).
    """
    if order_weights is not None:
        key = centers @ np.asarray(order_weights, dtype=np.float64)
    elif centers.shape[1] == 3:
        key = centers @ LUMA_ORDER
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
    order_weights: np.ndarray | None = None,
) -> KMeansResult:
    """Phân cụm K-means (thuật toán Lloyd) cho dữ liệu (N, D).

    Có trọng số: điểm i được tính như w_i điểm trùng nhau, tức hàm mục tiêu là
    J = sum_i w_i * ||x_i - Z_c(i)||^2 và tâm Z_k = sum w_i x_i / sum w_i. Với ảnh, truyền
    các giá trị pixel PHÂN BIỆT và w = số pixel của mỗi giá trị cho kết quả tương đương
    K-means trên toàn bộ pixel nhưng nhanh hơn nhiều (ảnh xám chỉ còn tối đa 256 điểm).

    Args:
        data: mảng (N, D) hoặc (N,) các vector đặc trưng.
        weights: (N,) trọng số dương; None = mọi điểm trọng số 1.
        k: số cụm yêu cầu (>= 1). Nếu dữ liệu có ít giá trị phân biệt hơn k,
           k được giảm xuống bằng số giá trị phân biệt (có cảnh báo).
        max_iter: số vòng lặp tối đa.
        tol: ngưỡng hội tụ — dừng khi độ dịch chuyển lớn nhất của tâm (chuẩn Euclid) <= tol.
        seed: hạt giống ngẫu nhiên để tái lập kết quả.
        init: 'kmeans++' hoặc 'random'.
        chunk_size: số điểm mỗi khối khi tính khoảng cách.
        order_weights: (D,) trọng số tạo khoá "độ sáng" để đánh số cụm; None = mặc định
            (BT.601 với 3 kênh, trung bình với số kênh khác).

    Returns:
        KMeansResult — nhãn được sắp xếp sao cho cụm 0 có tâm tối nhất.
    """
    return _kmeans(data, k, max_iter, tol, seed, init, chunk_size, weights, order_weights)


@_checked_arithmetic
def _kmeans(data: np.ndarray, k: int, max_iter: int, tol: float, seed: int | None, init: str, chunk_size: int,
            weights: np.ndarray | None, order_weights: np.ndarray | None, *, already_unique: bool = False) -> KMeansResult:
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
    if order_weights is not None:
        order_weights = np.asarray(order_weights, dtype=np.float64)
        if order_weights.shape != (x.shape[1],) or not np.all(np.isfinite(order_weights)):
            raise ValueError("order_weights phải là mảng (D,) các số hữu hạn.")

    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    warnings: list[str] = []

    distinct = x if already_unique else unique_rows(x)
    k_eff = min(k, distinct.shape[0])
    if k_eff < k:
        warnings.append(
            f"Dữ liệu chỉ có {distinct.shape[0]} giá trị phân biệt nên K được giảm từ {k} xuống {k_eff}."
        )

    centers = (_init_kmeans_pp(x, w, k_eff, rng, chunk_size) if init == "kmeans++"
               else _init_random(distinct, k_eff, rng))
    del distinct  # Chỉ cần các giá trị phân biệt để chặn K và khởi tạo.

    sse_history: list[float] = []
    shift_history: list[float] = []
    centers_history: list[np.ndarray] = [centers.copy()]
    converged = False
    n_empty_events = 0

    # Giữ nhãn nội bộ ở intp để bincount dùng trực tiếp, và ghi lại hai vùng đệm
    # sau khi bước cập nhật tâm đã dùng xong nhãn và khoảng cách của vòng trước.
    assignment = (np.empty(x.shape[0], dtype=np.intp), np.empty(x.shape[0], dtype=np.float64))
    rank, labels, min_sq = _assign_by_brightness(x, centers, chunk_size, order_weights, assignment)
    current_sse = initial_sse = _weighted_sse(w, min_sq)
    for n_iter in range(1, max_iter + 1):
        # Gán nhãn theo thứ tự độ sáng. Chỉ đổi thứ tự K tâm và số đếm sau
        # khi tính trung bình để lịch sử giữ đúng định danh tâm qua các vòng.
        sorted_centers, sorted_counts = _weighted_means(x, labels, w, k_eff)
        inverse_rank = np.argsort(rank)
        new_centers, counts = sorted_centers[inverse_rank], sorted_counts[inverse_rank]

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
        shift = _max_shift(new_centers, centers)
        changed = shift != 0.0 or not np.array_equal(new_centers, centers)
        shift_history.append(shift)
        centers = new_centers
        centers_history.append(centers.copy())
        # SSE của vòng i ứng với tâm ĐÃ cập nhật và nhãn gần nhất theo tâm đó.
        # Tâm không đổi cho cùng phép gán. Các vòng khác dùng phép gán mới để
        # ghi SSE, kiểm tra hội tụ và làm đầu vào cho vòng kế tiếp.
        if changed:
            rank, labels, min_sq = _assign_by_brightness(x, centers, chunk_size, order_weights, assignment)
            current_sse = _weighted_sse(w, min_sq)
        sse_history.append(current_sse)
        if shift <= tol and empty.size == 0:
            # Chỉ dừng khi tâm đúng là trung bình cụm theo phép gán nhãn trả về (với tol > 0, điểm
            # sát biên có thể còn đổi cụm). Tâm không đổi đã thỏa điều đó.
            if changed:
                means, check_counts = _weighted_means(x, labels, w, k_eff)
                converged = bool(np.all(check_counts > 0) and _max_shift(means, centers[rank]) <= tol)
            else:
                converged = True
            if converged:
                break

    if n_empty_events:
        warnings.append(f"Đã xử lý {n_empty_events} lần cụm rỗng (khởi tạo lại tâm tại điểm xa nhất).")
    if not converged:
        warnings.append(f"Chưa hội tụ sau {max_iter} vòng lặp (độ dịch tâm cuối = {shift_history[-1]:.4g}).")

    # Nhãn ở phép gán cuối đã dùng đúng thứ tự tâm được báo cáo.
    centers = centers[rank]
    centers_history = [c[rank] for c in centers_history]
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

    return KMeansResult(
        labels=labels.astype(np.int32, copy=False),
        centers=centers,
        counts=counts,
        sse=current_sse,
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
        initial_sse=initial_sse,
    )


@dataclass
class ImageKMeansResult:
    """Kết quả K-means trên ảnh."""

    labels: np.ndarray  # (H, W) int32
    segmented: np.ndarray  # (H, W) hoặc (H, W, 3) uint8 — mỗi pixel thay bằng màu tâm cụm
    centers: np.ndarray  # (k, D) float64 trong không gian đặc trưng (mức xám/RGB 0..255, hoặc L*a*b*)
    feature: str  # 'gray', 'rgb' hoặc 'lab'
    core: KMeansResult
    centers_rgb: np.ndarray | None = None  # (k, 3) màu hiển thị của tâm (thang 0..255); None với ảnh xám


def image_to_features(image: np.ndarray, feature: str, *, dtype=np.float64) -> np.ndarray:
    """Chuyển ảnh thành ma trận đặc trưng (H*W, D).

    feature='gray': D = 1 (mức xám); feature='rgb': D = 3 (R, G, B);
    feature='lab': D = 3 (L*, a*, b* — CIE 1976, D65; luôn float64).
    Ảnh 2 chiều chỉ cho phép 'gray'. dtype mặc định float64; đường xử lý ảnh
    giữ uint8 ở bước gom màu để tránh bản sao float64 trên toàn bộ pixel.
    """
    img = np.asarray(image)
    if feature == "gray":
        if img.ndim != 2:
            raise ValueError("Đặc trưng 'gray' yêu cầu ảnh xám 2 chiều; ảnh màu cần được chuyển xám trước.")
        return img.reshape(-1, 1).astype(dtype, copy=False)
    if feature in ("rgb", "lab"):
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Đặc trưng '{feature}' yêu cầu ảnh 3 kênh (H, W, 3).")
        rgb = img.reshape(-1, 3)
        return rgb.astype(dtype, copy=False) if feature == "rgb" else rgb_to_lab(rgb)
    raise ValueError(f"Loại đặc trưng không hợp lệ: {feature!r} (chỉ hỗ trợ {', '.join(FEATURES)}).")


def unique_pixels(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gom các pixel cùng giá trị: trả về (giá trị phân biệt (U, D), chỉ số ngược (N,), số pixel (U,)).

    Ảnh xám dùng histogram 256 mức O(N); ảnh nhiều kênh gộp pixel 0..255 thành
    một số nguyên R*65536 + G*256 + B để np.unique chạy trên mảng 1 chiều.
    """
    features = np.asarray(features)
    if features.ndim != 2 or not 1 <= features.shape[1] <= 3:
        raise ValueError("Pixel phải có 1 đến 3 kênh.")
    if features.dtype != np.uint8:
        if (features.dtype.kind not in "buif" or not np.all(np.isfinite(features))
                or np.any(features < 0) or np.any(features > 255)
                or np.any(features != np.round(features))):
            raise ValueError("Pixel phải có giá trị nguyên trong [0, 255].")
    if features.shape[1] == 1:
        # Chỉ 256 mức xám: đếm trực tiếp và ánh xạ mức xám -> chỉ số cụm màu.
        # Giữ đúng thứ tự tăng dần, counts và inverse như np.unique, không cần
        # sắp xếp N pixel hay tạo mảng packed int32 cho ảnh xám lớn.
        pixels = features[:, 0].astype(np.uint8, copy=False)
        # bincount chuyển chỉ số sang intp; giới hạn bản chuyển kiểu theo khối.
        histogram = np.bincount(pixels[:DEFAULT_CHUNK_SIZE], minlength=256)
        for start in range(DEFAULT_CHUNK_SIZE, pixels.size, DEFAULT_CHUNK_SIZE):
            histogram += np.bincount(pixels[start:start + DEFAULT_CHUNK_SIZE], minlength=256)
        levels = np.flatnonzero(histogram)
        lookup = np.zeros(256, dtype=np.uint8)
        lookup[levels] = np.arange(levels.size)
        return levels.astype(np.float64)[:, None], lookup[pixels], histogram[levels]
    # Tối đa 24 bit RGB, vừa int32. Không tạo thêm ma trận int64 N×D.
    packed = np.zeros(features.shape[0], dtype=np.int32)
    for c in range(features.shape[1]):
        packed *= 256
        np.add(packed, features[:, c], out=packed, casting="unsafe")
    uniq, inverse, counts = np.unique(packed, return_inverse=True, return_counts=True)
    # Chỉ số màu phân biệt nằm trong 24 bit, kể cả khi ảnh có nhiều pixel lặp.
    inverse = inverse.ravel().astype(np.int32, copy=False)
    values = np.empty((uniq.size, features.shape[1]), dtype=np.float64)
    channel = np.empty_like(uniq)
    for c in range(features.shape[1]):
        np.right_shift(uniq, 8 * (features.shape[1] - c - 1), out=channel)
        np.bitwise_and(channel, 255, out=channel)
        values[:, c] = channel
    return values, inverse, counts




def segment_image_kmeans(
    image: np.ndarray,
    k: int,
    feature: str = "rgb",
    max_iter: int = 100,
    tol: float = 1e-4,
    seed: int | None = 0,
    init: str = "kmeans++",
) -> ImageKMeansResult:
    """Phân đoạn ảnh bằng K-means: mỗi pixel là một vector đặc trưng.

    Với feature='lab': gom các màu RGB phân biệt trước (mỗi màu RGB ứng với đúng một điểm Lab),
    đổi các màu phân biệt sang Lab rồi chạy K-means có trọng số trong Lab — khoảng cách
    Euclid là ΔE*ab. Tâm cụm ở thang Lab, được đổi ngược về sRGB để tô ảnh phân đoạn.
    """
    img = np.asarray(image)
    if img.ndim not in (2, 3):
        raise ValueError("Ảnh phải có dạng (H, W) hoặc (H, W, 3).")
    if img.dtype != np.uint8:
        raise ValueError("Ảnh phải có kiểu uint8 (giá trị 0..255).")
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("Ảnh rỗng: chiều rộng và chiều cao phải >= 1.")
    if feature not in FEATURES:
        raise ValueError(f"Loại đặc trưng không hợp lệ: {feature!r} (chỉ hỗ trợ {', '.join(FEATURES)}).")
    _validate_params(k, max_iter, tol, init, seed=seed)
    t0 = time.perf_counter()
    pixel_feature = "rgb" if feature == "lab" else feature
    values, inverse, counts = unique_pixels(image_to_features(img, pixel_feature, dtype=np.uint8))
    order_weights = None
    if feature == "lab":
        values = rgb_to_lab(values)
        order_weights = LIGHTNESS_ORDER
    # K-means có trọng số trên các giá trị phân biệt == K-means trên toàn bộ pixel
    # unique_pixels đã trả các dòng xám/RGB phân biệt theo thứ tự chuẩn.
    # Lab vẫn sắp các vector chuyển đổi như kmeans tổng quát khi khởi tạo ngẫu nhiên.
    core = _kmeans(values, k, max_iter, tol, seed, init, DEFAULT_CHUNK_SIZE, counts, order_weights,
                   already_unique=feature != "lab")
    display = lab_to_rgb(core.centers) if feature == "lab" else core.centers
    palette = np.clip(np.round(display), 0, 255).astype(np.uint8)
    # Tra màu trên tập mức xám trước khi mở rộng nhãn cho toàn ảnh.
    gray_values = palette[core.labels, 0] if feature == "gray" else None
    core.labels = core.labels[inverse]  # trả nhãn về cho từng pixel
    labels = core.labels.reshape(h, w)
    segmented = gray_values[inverse].reshape(h, w) if gray_values is not None else palette[labels]
    # Đo tới kết quả trực tiếp của thuật toán, gồm cả phục hồi nhãn và ảnh tâm.
    core.elapsed_seconds = time.perf_counter() - t0
    return ImageKMeansResult(labels=labels, segmented=segmented, centers=core.centers, feature=feature, core=core,
                             centers_rgb=None if feature == "gray" else np.asarray(display, dtype=np.float64))
