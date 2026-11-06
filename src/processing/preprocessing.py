"""Tiền xử lý: chuyển mức xám và giới hạn kích thước xử lý."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

# Trọng số độ chói ITU-R BT.601 (chuẩn phổ biến, kiến thức bổ sung)
LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114])

GRAY_METHODS = {
    "luminance": "Độ chói BT.601: 0.299R + 0.587G + 0.114B",
    "average": "Trung bình cộng (R + G + B) / 3 — giáo trình mục 4.2.1",
}


def _validate_image(image: np.ndarray) -> np.ndarray:
    """Chỉ chuyển kiểu sau khi xác nhận hình dạng và giá trị pixel hợp lệ."""
    img = np.asarray(image)
    if img.ndim not in (2, 3) or (img.ndim == 3 and img.shape[2] != 3):
        raise ValueError("Ảnh phải có dạng (H, W) hoặc (H, W, 3).")
    if img.size == 0:
        raise ValueError("Ảnh rỗng: chiều rộng và chiều cao phải >= 1.")
    if img.dtype != np.uint8:
        if (img.dtype.kind not in "buif" or not np.all(np.isfinite(img))
                or np.any(img < 0) or np.any(img > 255) or np.any(img != np.round(img))):
            raise ValueError("Pixel phải có giá trị nguyên trong [0, 255].")
        img = img.astype(np.uint8)
    return img


def _positive_int(value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError("Kích thước phải là số nguyên >= 1.")
    return int(value)


def _resampling_method(resample: str):
    if resample not in ("area", "nearest"):
        raise ValueError(f"Phương pháp đổi kích thước không hợp lệ: {resample!r}.")
    return Image.Resampling.BOX if resample == "area" else Image.Resampling.NEAREST


def to_grayscale(image: np.ndarray, method: str = "luminance") -> np.ndarray:
    """Chuyển ảnh (H, W, 3) uint8 sang ảnh xám (H, W) uint8; ảnh xám giữ nguyên."""
    if method not in GRAY_METHODS:
        raise ValueError(f"Phương pháp chuyển xám không hợp lệ: {method!r}.")
    img = _validate_image(image)
    if img.ndim == 2:
        return img
    rgb = img.astype(np.float64)
    if method == "luminance":
        gray = rgb @ LUMA_WEIGHTS
    elif method == "average":
        gray = rgb.mean(axis=2)
    return np.clip(np.round(gray), 0, 255).astype(np.uint8)


@dataclass
class ResizeInfo:
    original_size: tuple[int, int]  # (W, H)
    processed_size: tuple[int, int]  # (W, H)

    @property
    def resized(self) -> bool:
        return self.original_size != self.processed_size


def limit_size(image: np.ndarray, max_side: int, resample: str = "area") -> tuple[np.ndarray, ResizeInfo]:
    """Thu nhỏ ảnh (giữ tỉ lệ) để cạnh dài nhất <= max_side; ảnh nhỏ hơn giữ nguyên.

    resample='area' dùng Image.BOX (trung bình vùng, phù hợp thu nhỏ ảnh),
    resample='nearest' dùng cho mask/ground truth để không sinh giá trị trung gian.
    """
    max_side = _positive_int(max_side)
    _resampling_method(resample)
    img = _validate_image(image)
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img, ResizeInfo((w, h), (w, h))
    scale = max_side / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    return resize_to(img, (new_w, new_h), resample), ResizeInfo((w, h), (new_w, new_h))


def resize_to(image: np.ndarray, size: tuple[int, int], resample: str = "area") -> np.ndarray:
    """Đổi kích thước ảnh uint8 về (W, H) bằng Pillow."""
    if not isinstance(size, (tuple, list)) or len(size) != 2:
        raise ValueError("Kích thước đích phải là cặp (W, H) số nguyên >= 1.")
    size = (_positive_int(size[0]), _positive_int(size[1]))
    method = _resampling_method(resample)
    return np.asarray(Image.fromarray(_validate_image(image)).resize(size, method))
