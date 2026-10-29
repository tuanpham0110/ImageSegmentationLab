"""Luồng xử lý hoàn chỉnh (đầu vào -> thuật toán -> kết quả -> tệp xuất).

Giao diện Streamlit, script thực nghiệm và kiểm thử tích hợp đều gọi các hàm ở đây,
nên ảnh hiển thị và tệp tải xuống luôn được tạo từ cùng một dữ liệu.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.algorithms.kmeans import ImageKMeansResult, segment_image_kmeans
from src.algorithms.otsu import OtsuResult, apply_threshold, otsu_threshold
from src.processing.postprocessing import (
    colorize_labels,
    cutout_on_color,
    cutout_rgba,
    invert_mask,
    mask_from_labels,
    overlay_mask,
    suggest_foreground_clusters,
)
from src.processing.preprocessing import ResizeInfo, limit_size, resize_to, to_grayscale
from src.utils.image_io import LoadedImage, matrix_to_csv_bytes, table_to_csv_bytes, to_json_bytes, to_png_bytes
from src.utils.metrics import compare_masks, foreground_ratio


@dataclass
class PreparedInput:
    image: np.ndarray  # ảnh dùng để xử lý (đã giới hạn kích thước), (H,W) hoặc (H,W,3)
    gray: np.ndarray  # ảnh xám tương ứng (H,W)
    resize: ResizeInfo
    gray_method: str
    name: str

    @property
    def is_color(self) -> bool:
        return self.image.ndim == 3


def prepare_input(loaded: LoadedImage, max_side: int, gray_method: str = "luminance") -> PreparedInput:
    """Giới hạn kích thước xử lý và tính ảnh xám."""
    image, info = limit_size(loaded.array, max_side, resample="area")
    return PreparedInput(
        image=np.ascontiguousarray(image),
        gray=to_grayscale(image, gray_method),
        resize=info,
        gray_method=gray_method,
        name=loaded.source_name,
    )


def fit_ground_truth(gt: np.ndarray, prepared: PreparedInput) -> np.ndarray:
    """Đưa mask ground truth về đúng kích thước xử lý (nội suy láng giềng gần nhất)."""
    w, h = prepared.resize.processed_size
    if gt.shape == (h, w):
        return gt
    ow, oh = prepared.resize.original_size
    if gt.shape != (oh, ow):
        raise ValueError(
            f"Kích thước ground truth {gt.shape[1]}x{gt.shape[0]} không khớp ảnh gốc {ow}x{oh}."
        )
    return np.where(resize_to(gt, (w, h), "nearest") > 127, 255, 0).astype(np.uint8)


# ------------------------------- K-means -------------------------------

@dataclass
class KMeansParams:
    k: int = 3
    feature: str = "rgb"  # 'rgb' hoặc 'gray'
    max_iter: int = 100
    tol: float = 1e-4
    seed: int = 0
    init: str = "kmeans++"

    def as_dict(self) -> dict:
        return dict(k=self.k, feature=self.feature, max_iter=self.max_iter, tol=self.tol, seed=self.seed, init=self.init)


def run_kmeans(prepared: PreparedInput, params: KMeansParams) -> ImageKMeansResult:
    feature = params.feature if prepared.is_color else "gray"
    data = prepared.image if feature == "rgb" else prepared.gray
    return segment_image_kmeans(
        data, params.k, feature=feature, max_iter=params.max_iter, tol=params.tol, seed=params.seed, init=params.init
    )


@dataclass
class KMeansOutputs:
    segmented: np.ndarray
    label_color: np.ndarray
    mask: np.ndarray
    cutout_rgba: np.ndarray
    cutout_preview: np.ndarray
    overlay: np.ndarray
    foreground_clusters: list[int]


def default_foreground(result: ImageKMeansResult) -> list[int]:
    return suggest_foreground_clusters(result.labels, result.core.k_effective)


def kmeans_outputs(prepared: PreparedInput, result: ImageKMeansResult, foreground_clusters) -> KMeansOutputs:
    fg = sorted(set(int(c) for c in foreground_clusters))
    mask = mask_from_labels(result.labels, fg)
    return KMeansOutputs(
        segmented=result.segmented,
        label_color=colorize_labels(result.labels),
        mask=mask,
        cutout_rgba=cutout_rgba(prepared.image, mask),
        cutout_preview=cutout_on_color(prepared.image, mask),
        overlay=overlay_mask(prepared.image, mask),
        foreground_clusters=fg,
    )


def cluster_table(result: ImageKMeansResult) -> list[dict]:
    core = result.core
    total = int(core.counts.sum())
    rows = []
    for i in range(core.k_effective):
        c = core.centers[i]
        row = {"Cụm": i}
        if c.size == 1:
            row["Tâm (mức xám)"] = round(float(c[0]), 2)
        else:
            row.update({"Tâm R": round(float(c[0]), 2), "Tâm G": round(float(c[1]), 2), "Tâm B": round(float(c[2]), 2)})
        row["Số pixel"] = int(core.counts[i])
        row["Tỷ lệ (%)"] = round(100.0 * core.counts[i] / total, 3)
        rows.append(row)
    return rows


def kmeans_stats(prepared: PreparedInput, params: KMeansParams, result: ImageKMeansResult, fg: list[int]) -> dict:
    core = result.core
    return {
        "thuat_toan": "K-means (tự cài đặt, Lloyd)",
        "anh": prepared.name,
        "kich_thuoc_goc_WxH": list(prepared.resize.original_size),
        "kich_thuoc_xu_ly_WxH": list(prepared.resize.processed_size),
        "tham_so": {**params.as_dict(), "feature_thuc_te": result.feature},
        "k_yeu_cau": core.k_requested,
        "k_thuc_te": core.k_effective,
        "so_vong_lap": core.n_iter,
        "hoi_tu": core.converged,
        "sse": core.sse,
        "sse_theo_vong_lap": core.sse_history,
        "do_dich_tam_theo_vong_lap": core.shift_history,
        "tam_cum": core.centers.tolist(),
        "so_pixel_moi_cum": core.counts.tolist(),
        "thoi_gian_giay": core.elapsed_seconds,
        "cum_doi_tuong": fg,
        "canh_bao": core.warnings,
    }


def kmeans_export_files(prepared: PreparedInput, params: KMeansParams, result: ImageKMeansResult,
                        out: KMeansOutputs) -> dict[str, tuple[bytes, str]]:
    """Tên tệp -> (nội dung, MIME)."""
    return {
        "kmeans_segmented.png": (to_png_bytes(out.segmented), "image/png"),
        "kmeans_label_map.png": (to_png_bytes(out.label_color), "image/png"),
        "kmeans_mask.png": (to_png_bytes(out.mask), "image/png"),
        "kmeans_cutout.png": (to_png_bytes(out.cutout_rgba), "image/png"),
        "kmeans_labels.csv": (matrix_to_csv_bytes(result.labels), "text/csv"),
        "kmeans_clusters.csv": (table_to_csv_bytes(cluster_table(result)), "text/csv"),
        "kmeans_stats.json": (to_json_bytes(kmeans_stats(prepared, params, result, out.foreground_clusters)),
                              "application/json"),
    }


# -------------------------------- Otsu ---------------------------------

POLARITIES = ("bright", "dark", "auto")


def suggest_dark_foreground(binary: np.ndarray) -> bool:
    """GỢI Ý (heuristic): nếu đa số pixel viền ảnh thuộc lớp sáng thì nền sáng -> đối tượng là lớp tối.
    Cùng quy tắc "viền ảnh là nền" như gợi ý cụm nền của K-means."""
    b = np.asarray(binary) > 0
    if b.shape[0] < 2 or b.shape[1] < 2:
        border = b.ravel()
    else:
        border = np.concatenate([b[0, :], b[-1, :], b[1:-1, 0], b[1:-1, -1]])
    return bool(border.mean() > 0.5)


@dataclass
class OtsuOutputs:
    result: OtsuResult
    threshold_used: int
    mode: str  # 'otsu' hoặc 'manual'
    invert: bool  # True = đối tượng là lớp tối (<= t)
    polarity: str  # 'bright' | 'dark' | 'auto' (lựa chọn của người dùng)
    binary: np.ndarray  # pixel > t -> 255 (chưa đảo)
    mask: np.ndarray  # mask đối tượng (đã áp dụng tuỳ chọn đảo)
    cutout_rgba: np.ndarray
    cutout_preview: np.ndarray
    overlay: np.ndarray
    foreground_ratio: float
    warnings: list[str] = field(default_factory=list)


def run_otsu(prepared: PreparedInput, polarity: str = "bright", manual_threshold: int | None = None) -> OtsuOutputs:
    """Otsu trên ảnh xám.

    polarity: 'bright' = đối tượng là lớp sáng (> t); 'dark' = lớp tối (<= t);
              'auto' = gợi ý theo viền ảnh (suggest_dark_foreground).
    manual_threshold: nếu khác None thì dùng ngưỡng thủ công để phân ngưỡng (ngưỡng Otsu vẫn được tính để đối chiếu).
    """
    if polarity not in POLARITIES:
        raise ValueError(f"polarity phải là một trong {POLARITIES}.")
    res = otsu_threshold(prepared.gray)
    t = res.threshold if manual_threshold is None else int(manual_threshold)
    binary = apply_threshold(prepared.gray, t)
    invert = polarity == "dark" or (polarity == "auto" and suggest_dark_foreground(binary))
    mask = invert_mask(binary) if invert else binary
    warnings = list(res.warnings) if manual_threshold is None else []
    return OtsuOutputs(
        result=res,
        threshold_used=t,
        mode="otsu" if manual_threshold is None else "manual",
        invert=invert,
        polarity=polarity,
        binary=binary,
        mask=mask,
        cutout_rgba=cutout_rgba(prepared.image, mask),
        cutout_preview=cutout_on_color(prepared.image, mask),
        overlay=overlay_mask(prepared.image, mask),
        foreground_ratio=foreground_ratio(mask),
        warnings=warnings,
    )


def otsu_stats(prepared: PreparedInput, out: OtsuOutputs) -> dict:
    r = out.result
    return {
        "thuat_toan": "Otsu (tự cài đặt từ histogram)",
        "anh": prepared.name,
        "kich_thuoc_goc_WxH": list(prepared.resize.original_size),
        "kich_thuoc_xu_ly_WxH": list(prepared.resize.processed_size),
        "chuyen_xam": prepared.gray_method,
        "che_do": out.mode,
        "nguong_otsu": r.threshold,
        "nguong_su_dung": out.threshold_used,
        "otsu_hop_le": r.valid,
        "quy_uoc": "pixel > nguong -> 255 (lớp C1)",
        "doi_tuong_la_lop": "tối (<= ngưỡng)" if out.invert else "sáng (> ngưỡng)",
        "lua_chon_cuc_tinh": out.polarity,
        "phuong_sai_giua_lop_max": r.max_variance,
        "phuong_sai_toan_phan": r.total_variance,
        "do_tach_eta": r.separability,
        "w0": r.weight0, "w1": r.weight1, "mu0": r.mean0,
        "mu1": None if np.isnan(r.mean1) else r.mean1,
        "nguong_dong_hang": r.tied_thresholds,
        "ty_le_foreground": out.foreground_ratio,
        "thoi_gian_giay": r.elapsed_seconds,
        "canh_bao": r.warnings,
    }


def otsu_export_files(prepared: PreparedInput, out: OtsuOutputs) -> dict[str, tuple[bytes, str]]:
    hist_rows = [
        {"muc_xam": g, "so_pixel": int(out.result.histogram[g]),
         "sigma_B2": float(out.result.between_class_variance[g]),
         "hop_le": bool(out.result.valid_thresholds[g])}
        for g in range(256)
    ]
    return {
        "otsu_binary.png": (to_png_bytes(out.binary), "image/png"),
        "otsu_mask.png": (to_png_bytes(out.mask), "image/png"),
        "otsu_cutout.png": (to_png_bytes(out.cutout_rgba), "image/png"),
        "otsu_gray.png": (to_png_bytes(prepared.gray), "image/png"),
        "otsu_binary_matrix.csv": (matrix_to_csv_bytes(out.mask // 255), "text/csv"),
        "otsu_histogram.csv": (table_to_csv_bytes(hist_rows), "text/csv"),
        "otsu_stats.json": (to_json_bytes(otsu_stats(prepared, out)), "application/json"),
    }


def mask_scores(mask: np.ndarray, gt: np.ndarray | None):
    return None if gt is None else compare_masks(mask, gt)
