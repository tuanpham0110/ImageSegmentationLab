"""Quy ước mask dùng chung cho đánh giá, resize và hậu xử lý."""

from __future__ import annotations

import numpy as np


def mask_to_bool(mask: np.ndarray) -> np.ndarray:
    """Mask 2D không rỗng, giá trị không âm/hữu hạn; mọi giá trị >0 là đối tượng."""
    m = np.asarray(mask)
    if m.ndim != 2 or m.size == 0:
        raise ValueError("Mask phải là ma trận 2 chiều không rỗng.")
    if m.dtype.kind not in "buif":
        raise ValueError("Mask phải chứa giá trị số không âm, hữu hạn.")
    if m.dtype.kind not in "bu" and (not np.all(np.isfinite(m)) or np.any(m < 0)):
        raise ValueError("Mask phải chứa giá trị số không âm, hữu hạn.")
    return m if m.dtype == np.bool_ else m > 0
