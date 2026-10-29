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
    * Nếu nhiều t cùng đạt cực đại, chọn floor(trung bình cộng các t đó) (giống MATLAB graythresh).
    * Ảnh nhị phân: pixel > t -> 255, pixel <= t -> 0.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

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


def compute_histogram(gray: np.ndarray) -> np.ndarray:
    """Histogram 256 mức của ảnh xám uint8 (đếm số pixel có mức xám g — mục 3.1.1 giáo trình)."""
    arr = np.asarray(gray)
    if arr.ndim != 2:
        raise ValueError("Otsu yêu cầu ảnh xám 2 chiều (H, W).")
    if arr.size == 0:
        raise ValueError("Ảnh rỗng.")
    if arr.dtype != np.uint8:
        if np.any(arr < 0) or np.any(arr > 255) or np.any(arr != np.round(arr)):
            raise ValueError("Ảnh xám phải chứa giá trị nguyên trong [0, 255].")
        arr = arr.astype(np.uint8)
    return np.bincount(arr.ravel(), minlength=N_LEVELS).astype(np.int64)


def otsu_from_histogram(hist: np.ndarray) -> OtsuResult:
    """Tính ngưỡng Otsu từ histogram 256 mức."""
    t0 = time.perf_counter()
    h = np.asarray(hist, dtype=np.float64)
    if h.shape != (N_LEVELS,):
        raise ValueError("Histogram phải có đúng 256 phần tử.")
    if np.any(h < 0):
        raise ValueError("Histogram không được chứa giá trị âm.")
    n = h.sum()
    if n <= 0:
        raise ValueError("Histogram rỗng (tổng số pixel = 0).")

    levels = np.arange(N_LEVELS, dtype=np.float64)
    p = h / n  # xác suất từng mức xám
    w0 = np.cumsum(p)  # w0(t)
    mu = np.cumsum(levels * p)  # mu(t)
    mu_t = mu[-1]
    w1 = 1.0 - w0

    # Số pixel tích luỹ (số nguyên chính xác) để xác định lớp rỗng, không phụ thuộc sai số float
    c0 = np.cumsum(np.asarray(hist, dtype=np.int64))
    valid = (c0 > 0) & (c0 < int(n))

    sigma_b = np.zeros(N_LEVELS, dtype=np.float64)
    denom = w0[valid] * w1[valid]
    sigma_b[valid] = (mu_t * w0[valid] - mu[valid]) ** 2 / denom

    total_var = float(np.sum(((levels - mu_t) ** 2) * p))
    warnings: list[str] = []

    if not np.any(valid):
        # Chỉ có một mức xám: không tồn tại ngưỡng chia ảnh thành hai lớp khác rỗng.
        level = int(np.flatnonzero(h)[0])
        warnings.append(
            f"Ảnh chỉ có một mức xám ({level}); không thể tách hai lớp. Ngưỡng được đặt bằng {level}, "
            "toàn bộ ảnh thuộc lớp nền (giá trị 0)."
        )
        return OtsuResult(
            threshold=level,
            valid=False,
            histogram=np.asarray(hist, dtype=np.int64),
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

    max_var = float(sigma_b[valid].max())
    tied = np.flatnonzero(valid & (sigma_b == max_var))
    threshold = int(np.floor(tied.mean()))
    if tied.size > 1:
        warnings.append(
            f"Có {tied.size} ngưỡng cùng đạt cực đại (từ {tied.min()} đến {tied.max()}); "
            f"chọn floor(trung bình) = {threshold}."
        )

    w0_t = float(w0[threshold])
    w1_t = 1.0 - w0_t
    mean0 = float(mu[threshold] / w0_t)
    mean1 = float((mu_t - mu[threshold]) / w1_t)
    separability = max_var / total_var if total_var > 0 else 0.0

    return OtsuResult(
        threshold=threshold,
        valid=True,
        histogram=np.asarray(hist, dtype=np.int64),
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
    if not 0 <= int(threshold) <= 255:
        raise ValueError("Ngưỡng phải nằm trong [0, 255].")
    return np.where(np.asarray(gray) > int(threshold), 255, 0).astype(np.uint8)
