"""Phân ngưỡng Otsu tự cài đặt từ histogram 256 mức xám (NumPy, không dùng thư viện xử lý ảnh).

Kiến thức chuẩn (N. Otsu, 1979, "A Threshold Selection Method from Gray-Level Histograms"):
với ngưỡng t, chia pixel thành C0 = {I <= t} và C1 = {I > t}. Gọi
    p_i     = h(i) / N                        xác suất mức xám i
    w0(t)   = sum_{i<=t} p_i,   w1 = 1 - w0   trọng số hai lớp
    mu(t)   = sum_{i<=t} i * p_i              mômen bậc nhất tích luỹ
    mu_T    = mu(255)                         trung bình toàn ảnh
    mu0     = mu(t) / w0,  mu1 = (mu_T - mu(t)) / w1
Phương sai giữa hai lớp:
    sigma_B^2(t) = w0 * w1 * (mu0 - mu1)^2 = (mu_T * w0 - mu(t))^2 / (w0 * w1)
Ngưỡng Otsu t* = argmax_t sigma_B^2(t).

Liên hệ giáo trình: hàm f(g) ở mục 3.1.6 "Kỹ thuật tìm tách ngưỡng tự động",
    f(g) = t(g) / (m*n - t(g)) * [m(g) - m(G-1)]^2,
biến đổi đại số được đúng sigma_B^2(g) = w0/w1 * (mu0 - mu_T)^2 (xem docs/ALGORITHMS.md),
nên hai cách cho cùng ngưỡng. Giáo trình không gọi tên "Otsu".

Quy ước của module:
    * Ngưỡng t chỉ được xét khi cả hai lớp khác rỗng; các t làm một lớp rỗng có sigma_B^2 = 0
      và bị đánh dấu không hợp lệ (tránh chia cho 0).
    * Nếu nhiều t cùng đạt cực đại, chọn ngưỡng TRONG tập cực đại gần floor(trung bình) nhất;
      nếu cách đều thì chọn ngưỡng nhỏ hơn. Với một đoạn liên tục, kết quả là floor(trung bình).
    * Ảnh nhị phân: pixel > t -> 255, pixel <= t -> 0.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

N_LEVELS = 256


@dataclass
class OtsuResult:
    threshold: int  # ngưỡng t*: lớp C0 = {I <= t*}, lớp C1 = {I > t*}
    valid: bool  # False nếu ảnh chỉ có một mức xám (không thể chia hai lớp)
    histogram: np.ndarray  # (256,) int64
    between_class_variance: np.ndarray  # (256,) float64, = 0 tại các t không hợp lệ
    valid_thresholds: np.ndarray  # (256,) bool, cả hai lớp khác rỗng
    max_variance: float
    total_variance: float  # sigma_T^2
    separability: float  # eta = sigma_B^2(t*) / sigma_T^2, trong [0, 1]
    weight0: float  # w0(t*)
    weight1: float
    mean0: float  # mu0(t*)
    mean1: float
    tied_thresholds: list[int] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)


def _validate_gray(gray: np.ndarray) -> np.ndarray:
    """Chuẩn hoá đầu vào ảnh xám, không ép kiểu âm thầm các pixel không hợp lệ."""
    arr = np.asarray(gray)
    if arr.ndim != 2:
        raise ValueError("Otsu yêu cầu ảnh xám 2 chiều (H, W).")
    if arr.size == 0:
        raise ValueError("Ảnh rỗng.")
    if arr.dtype != np.uint8:
        if (arr.dtype.kind not in "buif" or not np.all(np.isfinite(arr))
                or np.any(arr < 0) or np.any(arr > 255) or np.any(arr != np.round(arr))):
            raise ValueError("Ảnh xám phải chứa giá trị nguyên trong [0, 255].")
        arr = arr.astype(np.uint8)
    return arr


def compute_histogram(gray: np.ndarray) -> np.ndarray:
    """Histogram 256 mức của ảnh xám uint8 (đếm số pixel có mức xám g — mục 3.1.1 giáo trình)."""
    arr = _validate_gray(gray)
    return np.bincount(arr.ravel(), minlength=N_LEVELS).astype(np.int64)


def otsu_from_histogram(hist: np.ndarray) -> OtsuResult:
    """Tính ngưỡng Otsu từ histogram 256 mức."""
    t0 = time.perf_counter()
    h = np.asarray(hist)
    if h.shape != (N_LEVELS,):
        raise ValueError("Histogram phải có đúng 256 phần tử.")
    if h.dtype.kind not in "buif" or not np.all(np.isfinite(h)):
        raise ValueError("Histogram phải chứa số đếm hữu hạn, không có NaN hoặc vô cực.")
    if np.any(h < 0):
        raise ValueError("Histogram không được chứa giá trị âm.")
    if np.any(h != np.floor(h)):
        raise ValueError("Histogram phải chứa số đếm nguyên.")
    # float64 biểu diễn chính xác số đếm nguyên tới 2**53. Giới hạn này lớn
    # hơn nhiều giới hạn ảnh đầu vào, đồng thời tránh tràn số khi đổi int64.
    if np.any(h >= 2**53) or h.sum() >= 2**53:
        raise ValueError("Tổng số pixel trong histogram quá lớn để tính chính xác bằng float64.")
    counts = h.astype(np.int64)
    n = int(counts.sum())
    if n <= 0:
        raise ValueError("Histogram rỗng (tổng số pixel = 0).")

    levels = np.arange(N_LEVELS, dtype=np.int64)
    p = counts / n  # xác suất từng mức xám
    c0 = np.cumsum(counts)
    c1 = n - c0
    # Với N < 2**53, tổng i*h(i) < 255*2**53 < 2**61, vẫn vừa int64.
    # Trừ các mômen NGUYÊN trước khi chia để không mất lớp thiểu số do triệt tiêu float.
    moment0 = np.cumsum(levels * counts)
    moment1 = moment0[-1] - moment0
    mu_t = float(moment0[-1] / n)
    w0, w1 = c0 / n, c1 / n
    valid = (c0 > 0) & (c1 > 0)

    sigma_b = np.zeros(N_LEVELS, dtype=np.float64)
    mean0_valid = moment0[valid] / c0[valid]
    mean1_valid = moment1[valid] / c1[valid]
    sigma_b[valid] = w0[valid] * w1[valid] * (mean0_valid - mean1_valid) ** 2

    total_var = float(np.sum(((levels - mu_t) ** 2) * p))
    warnings: list[str] = []

    if not np.any(valid):
        # Chỉ có một mức xám: không tồn tại ngưỡng chia ảnh thành hai lớp khác rỗng.
        level = int(np.flatnonzero(h)[0])
        warnings.append(
            f"Ảnh chỉ có một mức xám ({level}); không thể tách hai lớp. Ngưỡng được đặt bằng {level}, "
            "ảnh nhị phân trực tiếp có toàn giá trị 0."
        )
        return OtsuResult(
            threshold=level,
            valid=False,
            histogram=counts,
            between_class_variance=sigma_b,
            valid_thresholds=valid,
            max_variance=0.0,
            total_variance=total_var,
            separability=0.0,
            weight0=1.0,
            weight1=0.0,
            mean0=float(level),
            mean1=float("nan"),
            tied_thresholds=[level],
            elapsed_seconds=time.perf_counter() - t0,
            warnings=warnings,
        )

    # Xác định cực đại và các ngưỡng đồng hạng bằng số hữu tỉ chính xác, không dùng == số thực:
    # sigma_b(t) = (n*m0 - M*c0)^2 / (n^2 * c0 * c1), M = tổng mômen. Các ngưỡng có cùng phương sai
    # đúng về mặt toán (ví dụ ảnh [0, 1, 1, 2]) có thể lệch nhau 1 ulp khi tính bằng float64.
    # Chỉ so chính xác các ứng viên gần cực đại float (sai số float64 ~1e-15 << 1e-9). Các ngưỡng
    # nằm trong cùng khoảng trống của histogram có cùng c0 nên cùng giá trị: chỉ tính một lần mỗi c0.
    total_moment = int(moment0[-1])
    float_max = float(sigma_b[valid].max())
    candidates = np.flatnonzero(valid & (sigma_b >= float_max * (1 - 1e-9)))
    cand_c0 = c0[candidates]
    # Mỗi c0 phân biệt ứng với một ngưỡng đại diện (cùng c0 => cùng m0 vì các mức giữa có h = 0).
    distinct_c0, first = np.unique(cand_c0, return_index=True)
    exact = [Fraction((n * int(moment0[t]) - total_moment * int(a)) ** 2, int(a) * (n - int(a)))
             for a, t in zip(distinct_c0, candidates[first])]
    best = max(exact)
    best_c0 = [a for a, v in zip(distinct_c0, exact) if v == best]
    tied = candidates[np.isin(cand_c0, best_c0)].astype(np.int64)
    max_var = float(best / (n * n))
    midpoint = int(np.floor(tied.mean()))
    # Các cực đại có thể nằm trên nhiều khoảng rời nhau. Trung bình của chúng
    # không nhất thiết là cực đại: phải chọn một phần tử thực sự thuộc argmax.
    threshold = int(tied[np.argmin(np.abs(tied - midpoint))])
    if tied.size > 1:
        warnings.append(
            f"Có {tied.size} ngưỡng cùng đạt cực đại (từ {tied.min()} đến {tied.max()}); "
            f"chọn ngưỡng đạt cực đại gần floor(trung bình) nhất = {threshold}."
        )

    w0_t = float(w0[threshold])
    w1_t = float(w1[threshold])
    mean0 = float(moment0[threshold] / c0[threshold])
    mean1 = float(moment1[threshold] / c1[threshold])
    separability = max_var / total_var if total_var > 0 else 0.0

    return OtsuResult(
        threshold=threshold,
        valid=True,
        histogram=counts,
        between_class_variance=sigma_b,
        valid_thresholds=valid,
        max_variance=max_var,
        total_variance=total_var,
        separability=float(min(max(separability, 0.0), 1.0)),
        weight0=w0_t,
        weight1=w1_t,
        mean0=mean0,
        mean1=mean1,
        tied_thresholds=[int(t) for t in tied],
        elapsed_seconds=time.perf_counter() - t0,
        warnings=warnings,
    )


def otsu_threshold(gray: np.ndarray) -> OtsuResult:
    """Tính ngưỡng Otsu trực tiếp từ ảnh xám; thời gian bao gồm cả bước tính histogram."""
    t0 = time.perf_counter()
    result = otsu_from_histogram(compute_histogram(gray))
    result.elapsed_seconds = time.perf_counter() - t0
    return result


def apply_threshold(gray: np.ndarray, threshold: int) -> np.ndarray:
    """Phân ngưỡng (mục 3.1.3 giáo trình): pixel > threshold -> 255, ngược lại -> 0."""
    if (isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, (int, np.integer))
            or not 0 <= int(threshold) <= 255):
        raise ValueError("Ngưỡng phải là số nguyên trong [0, 255].")
    return np.where(_validate_gray(gray) > int(threshold), 255, 0).astype(np.uint8)


def segment_image_otsu(gray: np.ndarray, manual_threshold: int | None = None) -> tuple[OtsuResult, np.ndarray]:
    """Histogram + tìm ngưỡng + tạo ảnh nhị phân, cùng phạm vi đo với K-means ảnh.

    Ở chế độ thủ công vẫn tính Otsu để đối chiếu. Không tính chuyển xám,
    lựa chọn foreground, ảnh tách nền, hiển thị hoặc mã hoá tệp xuất.
    """
    t0 = time.perf_counter()
    result = otsu_threshold(gray)
    threshold = result.threshold if manual_threshold is None else manual_threshold
    binary = apply_threshold(gray, threshold)
    result.elapsed_seconds = time.perf_counter() - t0
    return result, binary


# ----------------------------- Multi-Otsu (3 lớp) -----------------------------
# Mở rộng tự nhiên của Otsu (Liao, Chen & Chung 2001, "A Fast Algorithm for Multilevel
# Thresholding"): với hai ngưỡng t1 < t2 chia histogram thành C0 = [0, t1], C1 = (t1, t2],
# C2 = (t2, 255]. Phương sai giữa các lớp
#     sigma_B^2(t1, t2) = sum_k w_k (mu_k - mu_T)^2 = (1/N) * sum_k M_k^2 / c_k - mu_T^2
# (c_k: số pixel, M_k: tổng mức xám của lớp k). Vì mu_T cố định, cực đại sigma_B^2 tương đương
# cực đại sum_k M_k^2 / c_k. Tìm kiếm vét cạn trên mọi cặp (t1, t2) bằng tổng tích luỹ:
# mỗi cặp O(1), tổng O(256^2) = 65 536 phép tính — vài mili giây, không phụ thuộc kích thước ảnh.

MULTI_OTSU_CLASSES = 3


@dataclass
class MultiOtsuResult:
    thresholds: tuple[int, int]  # (t1, t2): C0 = {I <= t1}, C1 = {t1 < I <= t2}, C2 = {I > t2}
    valid: bool  # False nếu ảnh có ít hơn 3 mức xám (không thể chia 3 lớp khác rỗng)
    histogram: np.ndarray  # (256,) int64
    between_class_variance: np.ndarray  # (256, 256) float64 theo [t1, t2]; NaN tại cặp không hợp lệ
    max_variance: float
    total_variance: float
    separability: float  # eta = sigma_B^2 / sigma_T^2
    class_weights: list[float]
    class_means: list[float]
    class_counts: list[int]
    n_tied: int = 1
    elapsed_seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def n_classes(self) -> int:
        return MULTI_OTSU_CLASSES


def _closest_to_floor_mean(values: np.ndarray) -> int:
    mid = int(np.floor(values.mean()))
    return int(values[np.argmin(np.abs(values - mid))])


def multi_otsu_from_histogram(hist: np.ndarray) -> MultiOtsuResult:
    """Hai ngưỡng Otsu (3 lớp) từ histogram 256 mức, vét cạn O(256^2)."""
    t0 = time.perf_counter()
    base = otsu_from_histogram(hist)  # dùng lại toàn bộ kiểm tra đầu vào của Otsu 2 lớp
    counts = base.histogram
    n = int(counts.sum())
    levels = np.arange(N_LEVELS, dtype=np.int64)
    c = np.cumsum(counts)
    m = np.cumsum(levels * counts)
    total_moment = int(m[-1])
    mu_t = total_moment / n

    t1 = np.arange(N_LEVELS)[:, None]
    t2 = np.arange(N_LEVELS)[None, :]
    c0, m0 = c[t1], m[t1]
    c1, m1 = c[t2] - c[t1], m[t2] - m[t1]
    c2, m2 = n - c[t2], total_moment - m[t2]
    valid = (t1 < t2) & (c0 > 0) & (c1 > 0) & (c2 > 0)
    score = np.full((N_LEVELS, N_LEVELS), -np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (m0.astype(np.float64) ** 2 / c0 + m1.astype(np.float64) ** 2 / c1
             + m2.astype(np.float64) ** 2 / c2)
    score[valid] = s[valid]
    sigma_b = np.where(valid, score / n - mu_t ** 2, np.nan)
    total_var = base.total_variance
    warnings: list[str] = []

    if not valid.any():
        distinct = int(np.count_nonzero(counts))
        t = base.threshold
        warnings.append(f"Ảnh chỉ có {distinct} mức xám nên không chia được 3 lớp khác rỗng; "
                        f"dùng hai ngưỡng trùng nhau t1 = t2 = {t} (lớp C1 rỗng).")
        lo, hi = int(counts[: t + 1].sum()), int(counts[t + 1 :].sum())
        return MultiOtsuResult(
            thresholds=(t, t), valid=False, histogram=counts, between_class_variance=sigma_b,
            max_variance=base.max_variance, total_variance=total_var, separability=base.separability,
            class_weights=[lo / n, 0.0, hi / n],
            class_means=[base.mean0, float("nan"), base.mean1],
            class_counts=[lo, 0, hi], elapsed_seconds=time.perf_counter() - t0, warnings=warnings,
        )

    # Xác định cực đại chính xác bằng số hữu tỉ trên các ứng viên gần cực đại float (như Otsu 2 lớp).
    # Các cặp trong cùng khoảng trống histogram có cùng (c0, c0 + c1) nên chỉ tính một lần mỗi cặp số đếm.
    float_max = float(score[valid].max())
    cand = np.argwhere(valid & (score >= float_max * (1 - 1e-9)))
    keys = np.stack([c[cand[:, 0]], c[cand[:, 1]]], axis=1)
    distinct, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)

    def exact(i: int, j: int) -> Fraction:
        a, b, ma, mb = int(c[i]), int(c[j]), int(m[i]), int(m[j])
        return (Fraction(ma * ma, a) + Fraction((mb - ma) ** 2, b - a)
                + Fraction((total_moment - mb) ** 2, n - b))

    values = [exact(int(cand[f, 0]), int(cand[f, 1])) for f in first]
    best = max(values)
    best_ids = [i for i, v in enumerate(values) if v == best]
    tied = cand[np.isin(inverse.ravel(), best_ids)]
    # Phá hoà: t1 gần floor(trung bình các t1 đồng hạng) nhất, rồi t2 tương tự trong các cặp có t1 đó.
    th1 = _closest_to_floor_mean(tied[:, 0])
    th2 = _closest_to_floor_mean(tied[tied[:, 0] == th1, 1])
    if len(tied) > 1:
        warnings.append(f"Có {len(tied)} cặp ngưỡng cùng đạt cực đại; chọn (t1, t2) = ({th1}, {th2}) "
                        "gần trung bình các cặp đồng hạng nhất.")
    max_var = float(best / n - Fraction(total_moment, n) ** 2)
    cls_counts = [int(c[th1]), int(c[th2] - c[th1]), int(n - c[th2])]
    cls_moments = [int(m[th1]), int(m[th2] - m[th1]), int(total_moment - m[th2])]
    return MultiOtsuResult(
        thresholds=(th1, th2),
        valid=True,
        histogram=counts,
        between_class_variance=sigma_b,
        max_variance=max_var,
        total_variance=total_var,
        separability=float(min(max(max_var / total_var if total_var > 0 else 0.0, 0.0), 1.0)),
        class_weights=[k / n for k in cls_counts],
        class_means=[mk / k for mk, k in zip(cls_moments, cls_counts)],
        class_counts=cls_counts,
        n_tied=int(len(tied)),
        elapsed_seconds=time.perf_counter() - t0,
        warnings=warnings,
    )


def apply_multi_threshold(gray: np.ndarray, thresholds: tuple[int, int]) -> np.ndarray:
    """Ảnh nhãn 3 lớp (H, W) uint8: 0 nếu I <= t1, 1 nếu t1 < I <= t2, 2 nếu I > t2."""
    if len(thresholds) != 2:
        raise ValueError("Cần đúng hai ngưỡng (t1, t2).")
    for t in thresholds:
        if isinstance(t, (bool, np.bool_)) or not isinstance(t, (int, np.integer)) or not 0 <= int(t) <= 255:
            raise ValueError("Ngưỡng phải là số nguyên trong [0, 255].")
    t1, t2 = int(thresholds[0]), int(thresholds[1])
    if t1 > t2:
        raise ValueError("Ngưỡng phải thoả t1 <= t2.")
    g = _validate_gray(gray)
    return (g > t1).astype(np.uint8) + (g > t2).astype(np.uint8)


def segment_image_multi_otsu(gray: np.ndarray) -> tuple[MultiOtsuResult, np.ndarray]:
    """Histogram + hai ngưỡng Otsu + ảnh nhãn 3 lớp; cùng phạm vi đo thời gian với Otsu 2 lớp."""
    t0 = time.perf_counter()
    result = multi_otsu_from_histogram(compute_histogram(gray))
    labels = apply_multi_threshold(gray, result.thresholds)
    result.elapsed_seconds = time.perf_counter() - t0
    return result, labels
