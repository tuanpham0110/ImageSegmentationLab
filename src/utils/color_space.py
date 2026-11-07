"""Chuyển đổi sRGB <-> CIE L*a*b* (chuẩn CIE 1976, điểm trắng D65), tự cài đặt bằng NumPy.

Lý do dùng Lab cho K-means: khoảng cách Euclid trong RGB không tương ứng với khác biệt
màu mà mắt người cảm nhận (RGB không đồng nhất về cảm nhận). Trong Lab, khoảng cách
Euclid chính là độ lệch màu ΔE*ab (CIE76), nên "gần nhau" theo K-means gần với
"giống màu" theo thị giác hơn.

Các bước (IEC 61966-2-1 cho sRGB, CIE 15 cho Lab):
    1. Gamma sRGB -> tuyến tính:  c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4
    2. RGB tuyến tính -> XYZ bằng ma trận sRGB/D65
    3. XYZ -> Lab:  L = 116 f(Y/Yn) - 16,  a = 500 [f(X/Xn) - f(Y/Yn)],  b = 200 [f(Y/Yn) - f(Z/Zn)]
       với f(t) = t^(1/3) nếu t > (6/29)^3, ngược lại t / (3 (6/29)^2) + 4/29.
Thang giá trị: L* trong [0, 100]; a*, b* xấp xỉ [-128, 127] với gam màu sRGB.
"""

from __future__ import annotations

import numpy as np

_RGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
_XYZ_TO_RGB = np.linalg.inv(_RGB_TO_XYZ)
WHITE_D65 = np.array([0.95047, 1.0, 1.08883])
_DELTA = 6.0 / 29.0


def _check_last_axis(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim < 1 or a.shape[-1] != 3:
        raise ValueError("Dữ liệu màu phải có 3 kênh ở chiều cuối (..., 3).")
    if a.dtype.kind not in "buif":
        raise ValueError("Dữ liệu màu phải là số thực.")
    a = a.astype(np.float64, copy=False)
    if not np.all(np.isfinite(a)):
        raise ValueError("Dữ liệu màu chứa NaN hoặc vô cực.")
    return a


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB (..., 3) thang 0..255 -> CIE L*a*b* (..., 3) float64."""
    c = _check_last_axis(rgb)
    if np.any(c < 0) or np.any(c > 255):
        raise ValueError("Giá trị RGB phải nằm trong [0, 255].")
    c = c / 255.0
    linear = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    t = (linear @ _RGB_TO_XYZ.T) / WHITE_D65
    f = np.where(t > _DELTA ** 3, np.cbrt(t), t / (3 * _DELTA ** 2) + 4.0 / 29.0)
    lab = np.empty_like(f)
    lab[..., 0] = 116.0 * f[..., 1] - 16.0
    lab[..., 1] = 500.0 * (f[..., 0] - f[..., 1])
    lab[..., 2] = 200.0 * (f[..., 1] - f[..., 2])
    return lab


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """CIE L*a*b* (..., 3) -> sRGB (..., 3) float64 thang 0..255 (màu ngoài gam sRGB bị cắt)."""
    v = _check_last_axis(lab)
    fy = (v[..., 0] + 16.0) / 116.0
    f = np.stack([fy + v[..., 1] / 500.0, fy, fy - v[..., 2] / 200.0], axis=-1)
    t = np.where(f > _DELTA, f ** 3, 3 * _DELTA ** 2 * (f - 4.0 / 29.0))
    linear = np.clip((t * WHITE_D65) @ _XYZ_TO_RGB.T, 0.0, 1.0)
    c = np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055)
    return np.clip(c * 255.0, 0.0, 255.0)
