"""Chỉ số đánh giá mask nhị phân. Chỉ có ý nghĩa là "độ chính xác" khi so với ground truth hợp lệ."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
    return np.asarray(mask) > 0


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
