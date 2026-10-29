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


def to_grayscale(image: np.ndarray, method: str = "luminance") -> np.ndarray:
    """Chuyển ảnh (H, W, 3) uint8 sang ảnh xám (H, W) uint8; ảnh xám giữ nguyên."""
    img = np.asarray(image)
    if img.ndim == 2:
        return img.astype(np.uint8, copy=False)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Ảnh màu phải có dạng (H, W, 3).")
    rgb = img.astype(np.float64)
    if method == "luminance":
        gray = rgb @ LUMA_WEIGHTS
    elif method == "average":
        gray = rgb.mean(axis=2)
    else:
        raise ValueError(f"Phương pháp chuyển xám không hợp lệ: {method!r}.")
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
    img = np.asarray(image)
    h, w = img.shape[:2]
    if max_side < 1:
        raise ValueError("Giới hạn kích thước phải >= 1.")
    if max(h, w) <= max_side:
        return img, ResizeInfo((w, h), (w, h))
    scale = max_side / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    return resize_to(img, (new_w, new_h), resample), ResizeInfo((w, h), (new_w, new_h))


def resize_to(image: np.ndarray, size: tuple[int, int], resample: str = "area") -> np.ndarray:
    """Đổi kích thước ảnh uint8 về (W, H) bằng Pillow."""
    method = {"area": Image.Resampling.BOX, "nearest": Image.Resampling.NEAREST}[resample]
    return np.asarray(Image.fromarray(np.asarray(image, dtype=np.uint8)).resize(size, method))
