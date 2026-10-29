"""Hậu xử lý: tạo mask từ nhãn cụm, ảnh tách nền, ảnh nhãn tô màu."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

# Bảng màu phân loại cố định (8 màu, thứ tự không đổi) — dùng cho bản đồ nhãn và biểu đồ cụm.
CLUSTER_PALETTE_HEX = [
    "#2a78d6",  # xanh dương
    "#eb6834",  # cam
    "#1baf7a",  # xanh ngọc
    "#eda100",  # vàng
    "#e87ba4",  # hồng
    "#008300",  # xanh lá
    "#6250d6",  # tím
    "#e34948",  # đỏ
]
CLUSTER_PALETTE = np.array(
    [[int(h[i : i + 2], 16) for i in (1, 3, 5)] for h in CLUSTER_PALETTE_HEX], dtype=np.uint8
)


def colorize_labels(labels: np.ndarray) -> np.ndarray:
    """Tô màu bản đồ nhãn (H, W) -> ảnh RGB (H, W, 3) theo bảng màu cố định."""
    lab = np.asarray(labels)
    if lab.size and (lab.min() < 0 or lab.max() >= len(CLUSTER_PALETTE)):
        raise ValueError("Nhãn cụm phải nằm trong [0, 7].")
    return CLUSTER_PALETTE[lab]


def mask_from_labels(labels: np.ndarray, foreground_clusters: Iterable[int]) -> np.ndarray:
    """Mask nhị phân uint8: 255 nếu pixel thuộc một trong các cụm được chọn làm đối tượng, ngược lại 0."""
    selected = np.array(sorted(set(int(c) for c in foreground_clusters)), dtype=np.int64)
    return np.where(np.isin(labels, selected), 255, 0).astype(np.uint8)


def invert_mask(mask: np.ndarray) -> np.ndarray:
    return (255 - np.asarray(mask, dtype=np.uint8)).astype(np.uint8)


def to_rgb(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image, dtype=np.uint8)
    return np.stack([img] * 3, axis=-1) if img.ndim == 2 else img


def cutout_rgba(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Ảnh tách nền RGBA: giữ pixel đối tượng, nền trong suốt (alpha = 0)."""
    rgb = to_rgb(image)
    m = np.asarray(mask)
    if m.shape != rgb.shape[:2]:
        raise ValueError("Kích thước mask không khớp với ảnh.")
    alpha = np.where(m > 0, 255, 0).astype(np.uint8)
    return np.dstack([rgb, alpha])


def cutout_on_color(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """Ảnh tách nền trên nền màu đặc (dùng để xem trước)."""
    rgb = to_rgb(image).copy()
    rgb[np.asarray(mask) == 0] = color
    return rgb


def overlay_mask(image: np.ndarray, mask: np.ndarray, color=(235, 104, 52), alpha: float = 0.45) -> np.ndarray:
    """Phủ màu bán trong suốt lên vùng đối tượng để kiểm tra trực quan."""
    rgb = to_rgb(image).astype(np.float64)
    m = np.asarray(mask) > 0
    rgb[m] = (1 - alpha) * rgb[m] + alpha * np.array(color, dtype=np.float64)
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8)


def border_share(labels: np.ndarray, n_clusters: int) -> np.ndarray:
    """Tỷ lệ pixel viền ảnh thuộc mỗi cụm (dùng cho gợi ý nền)."""
    lab = np.asarray(labels)
    if lab.shape[0] < 2 or lab.shape[1] < 2:
        border = lab.ravel()
    else:
        border = np.concatenate([lab[0, :], lab[-1, :], lab[1:-1, 0], lab[1:-1, -1]])
    counts = np.bincount(border, minlength=n_clusters).astype(np.float64)
    return counts / counts.sum()


def suggest_foreground_clusters(labels: np.ndarray, n_clusters: int) -> list[int]:
    """GỢI Ý (heuristic, không phải kết luận của K-means): cụm chiếm nhiều pixel viền ảnh nhất
    được coi là nền, các cụm còn lại là đối tượng. Người dùng có thể chọn lại.
    """
    if n_clusters <= 1:
        return []
    background = int(np.argmax(border_share(labels, n_clusters)))
    return [c for c in range(n_clusters) if c != background]
