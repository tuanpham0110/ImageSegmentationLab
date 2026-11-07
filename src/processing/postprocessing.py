"""Hậu xử lý: tạo mask từ nhãn cụm, ảnh tách nền, ảnh nhãn tô màu."""

from __future__ import annotations

from collections.abc import Iterable

import colorsys

import numpy as np

from src.utils.masks import mask_to_bool

# 8 màu phân loại đầu tiên (thứ tự không đổi) — dùng cho bản đồ nhãn và biểu đồ cụm.
# Với K > 8, các màu tiếp theo được sinh động (xem cluster_palette_hex), nên K không bị giới hạn.
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
_GOLDEN_RATIO_CONJUGATE = 0.618033988749895


def cluster_palette_hex(n: int) -> list[str]:
    """n màu phân biệt cho n cụm: 8 màu cố định, sau đó trải góc pha (hue) trong HSV.

    Màu thứ 9 trở đi dùng bước góc pha theo tỉ lệ vàng (golden-angle) để các màu liên tiếp luôn
    cách xa nhau trên vòng màu, xen kẽ 3 mức sáng/bão hoà để tăng độ phân biệt khi n lớn.
    """
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError("Số màu phải là số nguyên >= 0.")
    colors = CLUSTER_PALETTE_HEX[: int(n)]
    for i in range(len(CLUSTER_PALETTE_HEX), int(n)):
        j = i - len(CLUSTER_PALETTE_HEX)
        hue = (0.11 + j * _GOLDEN_RATIO_CONJUGATE) % 1.0
        sat, val = ((0.65, 0.90), (0.85, 0.70), (0.45, 0.95))[j % 3]
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        colors.append(f"#{round(r * 255):02x}{round(g * 255):02x}{round(b * 255):02x}")
    return colors


def cluster_palette(n: int) -> np.ndarray:
    """Bảng màu (n, 3) uint8 tương ứng cluster_palette_hex(n)."""
    return np.array([[int(h[i : i + 2], 16) for i in (1, 3, 5)] for h in cluster_palette_hex(n)],
                    dtype=np.uint8).reshape(-1, 3)


def colorize_labels(labels: np.ndarray) -> np.ndarray:
    """Tô màu bản đồ nhãn (H, W) -> ảnh RGB (H, W, 3); bảng màu mở rộng theo số cụm."""
    lab = np.asarray(labels)
    if lab.size and lab.dtype.kind not in "iu":
        raise ValueError("Nhãn cụm phải là số nguyên.")
    if lab.size and lab.min() < 0:
        raise ValueError("Nhãn cụm phải là số nguyên >= 0.")
    n = int(lab.max()) + 1 if lab.size else 0
    return cluster_palette(max(n, 1))[lab]


def mask_from_labels(labels: np.ndarray, foreground_clusters: Iterable[int]) -> np.ndarray:
    """Mask nhị phân uint8: 255 nếu pixel thuộc một trong các cụm được chọn làm đối tượng, ngược lại 0."""
    selected = np.array(sorted(set(int(c) for c in foreground_clusters)), dtype=np.int64)
    return np.where(np.isin(labels, selected), 255, 0).astype(np.uint8)


def invert_mask(mask: np.ndarray) -> np.ndarray:
    return np.where(mask_to_bool(mask), 0, 255).astype(np.uint8)


def to_rgb(image: np.ndarray) -> np.ndarray:
    """Ảnh xám (H, W) hoặc RGB (H, W, 3) với giá trị nguyên trong [0, 255] -> RGB uint8.

    Kiểm tra trước khi ép kiểu: ép thẳng sang uint8 sẽ quấn số (-1 -> 255, 256 -> 0)
    và ảnh 4 kênh sẽ tạo ảnh tách nền 5 kênh.
    """
    img = np.asarray(image)
    if img.ndim not in (2, 3) or (img.ndim == 3 and img.shape[2] != 3):
        raise ValueError(f"Ảnh phải có dạng (H, W) hoặc (H, W, 3); nhận được {img.shape}.")
    if img.dtype != np.uint8:
        if img.dtype == np.bool_ or img.dtype.kind not in "uif":
            raise ValueError(f"Kiểu dữ liệu ảnh {img.dtype} không được hỗ trợ.")
        if img.size and (not np.all(np.isfinite(img)) or img.min() < 0 or img.max() > 255
                         or not np.array_equal(img, np.round(img))):
            raise ValueError("Giá trị ảnh phải là số nguyên trong [0, 255].")
        img = img.astype(np.uint8)
    return np.stack([img] * 3, axis=-1) if img.ndim == 2 else img


def _mask_for_image(mask: np.ndarray, image: np.ndarray) -> np.ndarray:
    m = mask_to_bool(mask)
    if m.shape != image.shape[:2]:
        raise ValueError("Kích thước mask không khớp với ảnh.")
    return m


def cutout_rgba(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Ảnh tách nền RGBA: giữ pixel đối tượng, nền trong suốt (alpha = 0)."""
    rgb = to_rgb(image)
    m = _mask_for_image(mask, rgb)
    alpha = np.where(m, 255, 0).astype(np.uint8)
    return np.dstack([rgb, alpha])


def cutout_on_color(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """Ảnh tách nền trên nền màu đặc (dùng để xem trước)."""
    rgb = to_rgb(image).copy()
    rgb[~_mask_for_image(mask, rgb)] = color
    return rgb


def overlay_mask(image: np.ndarray, mask: np.ndarray, color=(235, 104, 52), alpha: float = 0.45) -> np.ndarray:
    """Phủ màu bán trong suốt lên vùng đối tượng để kiểm tra trực quan."""
    rgb = to_rgb(image).astype(np.float64)
    m = _mask_for_image(mask, rgb)
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


def center_share(labels: np.ndarray, n_clusters: int) -> np.ndarray:
    """Tỷ lệ pixel của mỗi cụm trong vùng trung tâm (nửa giữa theo mỗi chiều = 25% diện tích)."""
    lab = np.asarray(labels)
    h, w = lab.shape
    center = lab[h // 4 : h - h // 4, w // 4 : w - w // 4]
    counts = np.bincount(center.ravel(), minlength=n_clusters).astype(np.float64)
    return counts / counts.sum()


def suggest_foreground_clusters(labels: np.ndarray, n_clusters: int) -> list[int]:
    """GỢI Ý (heuristic, không phải kết luận của K-means) cụm đối tượng. Người dùng có thể chọn lại.

    Một cụm được coi là NỀN nếu nó xuất hiện ở viền ảnh nhiều hơn ở vùng trung tâm
    (border_share > center_share); các cụm còn lại là đối tượng. So với quy tắc cũ
    "cụm chiếm viền nhiều nhất là nền", cách so sánh tương đối này xử lý được:
      * ảnh cận cảnh: đối tượng tràn ra viền nhưng gần như phủ kín vùng trung tâm -> vẫn là đối tượng;
      * nền nhiều màu (trời ở trên, cỏ ở dưới): mỗi cụm nền đều trội ở viền -> đều là nền.
    Khi không phân biệt được (mọi cụm có tỷ lệ viền = tỷ lệ trung tâm, ví dụ ảnh có cấu trúc
    lặp đều), quay về quy tắc cũ: cụm chiếm viền nhiều nhất là nền.
    Giới hạn còn lại: đối tượng lệch hẳn ra mép, không chạm vùng trung tâm, sẽ bị coi là nền.
    """
    if n_clusters <= 1:
        return []
    border = border_share(labels, n_clusters)
    center = center_share(labels, n_clusters)
    background = border > center + 1e-12
    if not background.any():
        background = np.zeros(n_clusters, dtype=bool)
        background[int(np.argmax(border))] = True
    return [c for c in range(n_clusters) if not background[c]]
