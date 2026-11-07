"""Lọc không gian tự cài đặt bằng NumPy (không dùng OpenCV/SciPy).

* Lọc trung vị (giáo trình Chương 4 — lọc phi tuyến): thay mỗi pixel bằng trung vị của cửa sổ
  k×k quanh nó. Khử nhiễu muối tiêu/hạt mà vẫn giữ biên tốt hơn lọc trung bình. Dùng ở bước
  TIỀN XỬ LÝ, trước Otsu/K-means — vốn là thuật toán theo từng điểm, không biết pixel lân cận.
* Hình thái học nhị phân (giáo trình Chương 7) với phần tử cấu trúc vuông k×k:
    giãn (dilation)  D(A) = {x : (B)_x ∩ A ≠ ∅}  -> max trong cửa sổ
    co   (erosion)   E(A) = {x : (B)_x ⊆ A}      -> min trong cửa sổ
    mở   (opening)   = co rồi giãn  : xoá đốm nhiễu nhỏ hơn B, làm trơn biên lồi
    đóng (closing)   = giãn rồi co  : lấp lỗ thủng nhỏ hơn B, nối khe hẹp
  Dùng ở bước HẬU XỬ LÝ mask.

Biên ảnh được xử lý bằng cách lặp lại pixel mép (padding 'edge'), nên đối tượng chạm mép
ảnh không bị co lẹm vào từ viền. Phần tử cấu trúc vuông là tách được (separable):
max/min trên k×k = max/min theo hàng rồi theo cột, chi phí O(N·k) thay vì O(N·k²).
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from src.utils.masks import mask_to_bool

MEDIAN_SIZES = (3, 5, 7)
MORPH_OPERATIONS = ("none", "open", "close", "open_close", "close_open", "erode", "dilate")
MORPH_LABELS = {
    "none": "Không",
    "open": "Mở (xoá đốm nhiễu)",
    "close": "Đóng (lấp lỗ thủng)",
    "open_close": "Mở rồi đóng",
    "close_open": "Đóng rồi mở",
    "erode": "Co (erosion)",
    "dilate": "Giãn (dilation)",
}
# Số hàng xử lý mỗi khối khi lọc trung vị để bộ nhớ tạm (hàng × W × k²) luôn nhỏ.
_MEDIAN_ROW_BLOCK = 128


def _check_size(size: int) -> int:
    if (isinstance(size, (bool, np.bool_)) or not isinstance(size, (int, np.integer))
            or size < 1 or size % 2 == 0):
        raise ValueError("Kích thước cửa sổ phải là số nguyên lẻ >= 1.")
    return int(size)


def median_filter(image: np.ndarray, size: int = 3) -> np.ndarray:
    """Lọc trung vị k×k cho ảnh uint8 (H, W) hoặc (H, W, 3); mỗi kênh lọc độc lập."""
    size = _check_size(size)
    img = np.asarray(image)
    if img.ndim not in (2, 3) or (img.ndim == 3 and img.shape[2] != 3) or img.size == 0:
        raise ValueError("Ảnh phải có dạng (H, W) hoặc (H, W, 3), không rỗng.")
    if img.dtype != np.uint8:
        raise ValueError("Lọc trung vị yêu cầu ảnh uint8.")
    if size == 1:
        return img.copy()
    r = size // 2
    pad = ((r, r), (r, r)) + (((0, 0),) if img.ndim == 3 else ())
    padded = np.pad(img, pad, mode="edge")
    out = np.empty_like(img)
    mid = size * size // 2
    h = img.shape[0]
    for start in range(0, h, _MEDIAN_ROW_BLOCK):
        stop = min(start + _MEDIAN_ROW_BLOCK, h)
        block = padded[start : stop + 2 * r]
        win = sliding_window_view(block, (size, size), axis=(0, 1))  # (rows, W, [C,] k, k)
        flat = win.reshape(win.shape[:-2] + (size * size,))
        # Trung vị của k² giá trị (k² lẻ) = phần tử thứ k²//2 sau khi sắp: np.partition O(k²).
        out[start:stop] = np.partition(flat, mid, axis=-1)[..., mid]
    return out


def _window_reduce(mask: np.ndarray, size: int, reducer) -> np.ndarray:
    """max/min trên cửa sổ vuông size×size, tách thành hai lượt 1 chiều (hàng rồi cột)."""
    r = size // 2
    out = mask
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (r, r)
        out = reducer(sliding_window_view(np.pad(out, pad, mode="edge"), size, axis=axis), axis=-1)
    return out


def _as_bool(mask: np.ndarray) -> np.ndarray:
    m = mask_to_bool(mask)
    if m.ndim != 2:
        raise ValueError("Mask phải là ảnh 2 chiều (H, W).")
    return m


def dilate(mask: np.ndarray, size: int = 3) -> np.ndarray:
    """Giãn nhị phân với phần tử cấu trúc vuông size×size. Trả mask bool."""
    size = _check_size(size)
    m = _as_bool(mask)
    return m.copy() if size == 1 else _window_reduce(m, size, np.max)


def erode(mask: np.ndarray, size: int = 3) -> np.ndarray:
    """Co nhị phân với phần tử cấu trúc vuông size×size. Trả mask bool."""
    size = _check_size(size)
    m = _as_bool(mask)
    return m.copy() if size == 1 else _window_reduce(m, size, np.min)


def opening(mask: np.ndarray, size: int = 3) -> np.ndarray:
    """Mở = co rồi giãn."""
    return dilate(erode(mask, size), size)


def closing(mask: np.ndarray, size: int = 3) -> np.ndarray:
    """Đóng = giãn rồi co."""
    return erode(dilate(mask, size), size)


def apply_morphology(mask: np.ndarray, operation: str = "none", size: int = 3) -> np.ndarray:
    """Áp dụng phép hình thái học lên mask; trả mask uint8 0/255 cùng quy ước với các mask khác."""
    if operation not in MORPH_OPERATIONS:
        raise ValueError(f"Phép hình thái học không hợp lệ: {operation!r}.")
    size = _check_size(size)
    m = _as_bool(mask)
    if operation == "open":
        m = opening(m, size)
    elif operation == "close":
        m = closing(m, size)
    elif operation == "open_close":
        m = closing(opening(m, size), size)
    elif operation == "close_open":
        m = opening(closing(m, size), size)
    elif operation == "erode":
        m = erode(m, size)
    elif operation == "dilate":
        m = dilate(m, size)
    return np.where(m, 255, 0).astype(np.uint8)
