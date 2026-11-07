"""Chỉ số đánh giá mask nhị phân. Chỉ có ý nghĩa là "độ chính xác" khi so với ground truth hợp lệ."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.utils.masks import mask_to_bool


@dataclass
class MaskScores:
    iou: float
    dice: float
    pixel_accuracy: float
    tp: int
    fp: int
    fn: int
    tn: int


def _as_bool(mask: np.ndarray) -> np.ndarray:
    return mask_to_bool(mask)


def compare_masks(pred: np.ndarray, reference: np.ndarray) -> MaskScores:
    """So sánh mask dự đoán với mask tham chiếu (cùng kích thước).

    IoU  = |A ∩ B| / |A ∪ B|,  Dice = 2|A ∩ B| / (|A| + |B|).
    Quy ước: nếu cả hai mask đều rỗng thì IoU = Dice = 1 (hai mask trùng nhau hoàn toàn).
    """
    a, b = _as_bool(pred), _as_bool(reference)
    if a.shape != b.shape:
        raise ValueError(f"Kích thước mask không khớp: {a.shape} và {b.shape}.")
    tp = int(np.count_nonzero(a & b))
    fp = int(np.count_nonzero(a & ~b))
    fn = int(np.count_nonzero(~a & b))
    tn = int(a.size - tp - fp - fn)
    union = tp + fp + fn
    iou = 1.0 if union == 0 else tp / union
    dice = 1.0 if (2 * tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn)
    return MaskScores(iou=iou, dice=dice, pixel_accuracy=(tp + tn) / a.size, tp=tp, fp=fp, fn=fn, tn=tn)


def foreground_ratio(mask: np.ndarray) -> float:
    m = _as_bool(mask)
    return float(np.count_nonzero(m) / m.size)


def compare_trimap(pred: np.ndarray, reference: np.ndarray) -> MaskScores:
    """Oxford iSeg: 0 nền, 255 đối tượng; bỏ qua biên không chắc chắn (128)."""
    a, truth = _as_bool(pred), np.asarray(reference)
    if truth.ndim != 2 or a.shape != truth.shape:
        raise ValueError("Trimap và mask dự đoán phải là ảnh 2 chiều cùng kích thước.")
    if not np.all(np.isin(truth, (0, 128, 255))):
        raise ValueError("Trimap chỉ được chứa 0, 128 hoặc 255.")
    valid = truth != 128
    if not valid.any():
        raise ValueError("Trimap không có pixel chắc chắn để đánh giá.")
    return compare_masks(a[valid].reshape(1, -1), (truth[valid] == 255).reshape(1, -1))
