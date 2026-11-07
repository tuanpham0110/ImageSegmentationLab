"""Luồng xử lý hoàn chỉnh (đầu vào -> thuật toán -> kết quả -> tệp xuất).

Giao diện Streamlit, script thực nghiệm và kiểm thử tích hợp đều gọi các hàm ở đây,
nên ảnh hiển thị và tệp tải xuống luôn được tạo từ cùng một dữ liệu.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.algorithms.kmeans import ImageKMeansResult, segment_image_kmeans
from src.algorithms.otsu import MultiOtsuResult, OtsuResult, segment_image_multi_otsu, segment_image_otsu
from src.processing.filters import MORPH_OPERATIONS, apply_morphology, median_filter
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
from src.utils.masks import mask_to_bool

TIMING_SCOPE = ("Từ ảnh đã tiền xử lý đến kết quả trực tiếp của thuật toán (nhãn/ảnh tâm hoặc ảnh nhị phân); "
                "không gồm đọc tệp, resize/chuyển xám, chọn foreground, ảnh tách nền, biểu đồ và xuất tệp.")


@dataclass
class PreparedInput:
    image: np.ndarray  # ảnh dùng để xử lý (đã giới hạn kích thước), (H,W) hoặc (H,W,3)
    gray: np.ndarray  # ảnh xám tương ứng (H,W)
    resize: ResizeInfo
    gray_method: str
    name: str
    median_size: int = 1  # cửa sổ lọc trung vị tiền xử lý (1 = không lọc)

    @property
    def is_color(self) -> bool:
        return self.image.ndim == 3


def prepare_input(loaded: LoadedImage, max_side: int, gray_method: str = "luminance",
                  median_size: int = 1) -> PreparedInput:
    """Giới hạn kích thước xử lý, lọc trung vị khử nhiễu (tuỳ chọn) và tính ảnh xám.

    Lọc trung vị được áp dụng lên ảnh đã thu nhỏ, trước cả hai thuật toán, nên Otsu và
    K-means luôn nhận cùng một ảnh đã khử nhiễu.
    """
    image, info = limit_size(loaded.array, max_side, resample="area")
    if median_size != 1:
        image = median_filter(image, median_size)
    return PreparedInput(
        image=np.ascontiguousarray(image),
        gray=to_grayscale(image, gray_method),
        resize=info,
        gray_method=gray_method,
        name=loaded.source_name,
        median_size=int(median_size),
    )


@dataclass(frozen=True)
class MaskPostprocess:
    """Hậu xử lý hình thái học cho mask đối tượng (mở/đóng/co/giãn, phần tử cấu trúc vuông)."""

    operation: str = "none"
    size: int = 3

    def __post_init__(self):
        if self.operation not in MORPH_OPERATIONS:
            raise ValueError(f"Phép hình thái học không hợp lệ: {self.operation!r}.")

    def apply(self, mask: np.ndarray) -> np.ndarray:
        return mask if self.operation == "none" else apply_morphology(mask, self.operation, self.size)

    def as_dict(self) -> dict:
        return {"phep_hinh_thai": self.operation, "kich_thuoc": self.size}


NO_POSTPROCESS = MaskPostprocess()


def fit_ground_truth(gt: np.ndarray, prepared: PreparedInput) -> np.ndarray:
    """Đưa mask ground truth về đúng kích thước xử lý (nội suy láng giềng gần nhất)."""
    gt = np.where(mask_to_bool(gt), 255, 0).astype(np.uint8)
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
    feature: str = "rgb"  # 'rgb', 'lab' hoặc 'gray'
    max_iter: int = 100
    tol: float = 1e-4
    seed: int = 0
    init: str = "kmeans++"

    def as_dict(self) -> dict:
        return dict(k=self.k, feature=self.feature, max_iter=self.max_iter, tol=self.tol, seed=self.seed, init=self.init)


def run_kmeans(prepared: PreparedInput, params: KMeansParams) -> ImageKMeansResult:
    feature = params.feature if prepared.is_color else "gray"
    data = prepared.image if feature in ("rgb", "lab") else prepared.gray
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
    raw_mask: np.ndarray | None = None  # mask trước hậu xử lý hình thái học
    postprocess: MaskPostprocess = NO_POSTPROCESS


def default_foreground(result: ImageKMeansResult) -> list[int]:
    return suggest_foreground_clusters(result.labels, result.core.k_effective)


def kmeans_outputs(prepared: PreparedInput, result: ImageKMeansResult, foreground_clusters,
                   postprocess: MaskPostprocess = NO_POSTPROCESS) -> KMeansOutputs:
    fg = sorted(set(int(c) for c in foreground_clusters))
    raw_mask = mask_from_labels(result.labels, fg)
    mask = postprocess.apply(raw_mask)
    return KMeansOutputs(
        segmented=result.segmented,
        label_color=colorize_labels(result.labels),
        mask=mask,
        cutout_rgba=cutout_rgba(prepared.image, mask),
        cutout_preview=cutout_on_color(prepared.image, mask),
        overlay=overlay_mask(prepared.image, mask),
        foreground_clusters=fg,
        raw_mask=raw_mask,
        postprocess=postprocess,
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
        elif result.feature == "lab":
            rgb = result.centers_rgb[i]
            row.update({"Tâm L*": round(float(c[0]), 2), "Tâm a*": round(float(c[1]), 2), "Tâm b*": round(float(c[2]), 2),
                        "Màu R": round(float(rgb[0]), 2), "Màu G": round(float(rgb[1]), 2), "Màu B": round(float(rgb[2]), 2)})
        else:
            row.update({"Tâm R": round(float(c[0]), 2), "Tâm G": round(float(c[1]), 2), "Tâm B": round(float(c[2]), 2)})
        row["Số pixel"] = int(core.counts[i])
        row["Tỷ lệ (%)"] = round(100.0 * core.counts[i] / total, 3)
        rows.append(row)
    return rows


def _preprocess_stats(prepared: PreparedInput) -> dict:
    return {"loc_trung_vi": (f"{prepared.median_size}x{prepared.median_size}" if prepared.median_size > 1
                             else "không")}


def kmeans_stats(prepared: PreparedInput, params: KMeansParams, result: ImageKMeansResult, fg: list[int],
                 postprocess: MaskPostprocess = NO_POSTPROCESS) -> dict:
    core = result.core
    return {
        "thuat_toan": "K-means (tự cài đặt, Lloyd)",
        "anh": prepared.name,
        "kich_thuoc_goc_WxH": list(prepared.resize.original_size),
        "kich_thuoc_xu_ly_WxH": list(prepared.resize.processed_size),
        "chuyen_xam": prepared.gray_method,
        "tien_xu_ly": _preprocess_stats(prepared),
        "hau_xu_ly_mask": postprocess.as_dict(),
        "tham_so": {**params.as_dict(), "feature_thuc_te": result.feature},
        "k_yeu_cau": core.k_requested,
        "k_thuc_te": core.k_effective,
        "so_vong_lap": core.n_iter,
        "hoi_tu": core.converged,
        "sse": core.sse,
        "sse_theo_vong_lap": core.sse_history,
        "do_dich_tam_theo_vong_lap": core.shift_history,
        "khong_gian_tam_cum": {"gray": "mức xám", "rgb": "RGB", "lab": "CIE L*a*b* (D65)"}[result.feature],
        "tam_cum": core.centers.tolist(),
        "mau_hien_thi_tam_cum_rgb": None if result.centers_rgb is None else result.centers_rgb.tolist(),
        "so_pixel_moi_cum": core.counts.tolist(),
        "thoi_gian_giay": core.elapsed_seconds,
        "pham_vi_thoi_gian": TIMING_SCOPE,
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
        "kmeans_stats.json": (to_json_bytes(kmeans_stats(prepared, params, result, out.foreground_clusters,
                                                         out.postprocess)),
                              "application/json"),
    }


# -------------------------------- Otsu ---------------------------------

POLARITIES = ("bright", "dark", "auto")


def suggest_dark_foreground(binary: np.ndarray) -> bool:
    """GỢI Ý (heuristic): True nếu lớp tối nên là đối tượng.

    Cùng quy tắc với gợi ý cụm của K-means (suggest_foreground_clusters): lớp xuất hiện ở viền ảnh
    nhiều hơn ở vùng trung tâm là nền. Nếu lớp sáng là nền thì đối tượng là lớp tối.
    """
    labels = (np.asarray(binary) > 0).astype(np.int64)
    return suggest_foreground_clusters(labels, 2) == [0]


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
    raw_mask: np.ndarray | None = None  # mask trước hậu xử lý hình thái học
    postprocess: MaskPostprocess = NO_POSTPROCESS


def run_otsu(prepared: PreparedInput, polarity: str = "bright", manual_threshold: int | None = None,
             postprocess: MaskPostprocess = NO_POSTPROCESS) -> OtsuOutputs:
    """Otsu trên ảnh xám.

    polarity: 'bright' = đối tượng là lớp sáng (> t); 'dark' = lớp tối (<= t);
              'auto' = gợi ý theo viền ảnh (suggest_dark_foreground).
    manual_threshold: nếu khác None thì dùng ngưỡng thủ công để phân ngưỡng (ngưỡng Otsu vẫn được tính để đối chiếu).
    """
    if polarity not in POLARITIES:
        raise ValueError(f"polarity phải là một trong {POLARITIES}.")
    res, binary = segment_image_otsu(prepared.gray, manual_threshold)
    t = res.threshold if manual_threshold is None else int(manual_threshold)
    invert = polarity == "dark" or (polarity == "auto" and suggest_dark_foreground(binary))
    mask = invert_mask(binary) if invert else binary
    warnings = list(res.warnings) if manual_threshold is None else []
    if manual_threshold is None and not res.valid:
        # Không suy diễn đối tượng từ một ảnh chỉ có một mức xám. Lựa chọn
        # sáng/tối vẫn được lưu, nhưng chỉ áp dụng khi có hai lớp hợp lệ.
        invert = False
        mask = np.zeros_like(binary)
        warnings.append("Otsu không chia được hai lớp: mask tự động được để trống (toàn nền), "
                        "không áp dụng lựa chọn lớp sáng/tối. Có thể dùng ngưỡng thủ công để đối chiếu.")
    raw_mask = mask
    mask = postprocess.apply(raw_mask)
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
        raw_mask=raw_mask,
        postprocess=postprocess,
    )


def otsu_stats(prepared: PreparedInput, out: OtsuOutputs) -> dict:
    r = out.result
    return {
        "thuat_toan": "Otsu (tự cài đặt từ histogram)",
        "anh": prepared.name,
        "kich_thuoc_goc_WxH": list(prepared.resize.original_size),
        "kich_thuoc_xu_ly_WxH": list(prepared.resize.processed_size),
        "chuyen_xam": prepared.gray_method,
        "tien_xu_ly": _preprocess_stats(prepared),
        "hau_xu_ly_mask": out.postprocess.as_dict(),
        "che_do": out.mode,
        "nguong_otsu": r.threshold,
        "nguong_su_dung": out.threshold_used,
        "otsu_hop_le": r.valid,
        "quy_uoc": "pixel > nguong -> 255 (lớp C1)",
        "doi_tuong_la_lop": ("không xác định (mask tự động rỗng)" if out.mode == "otsu" and not r.valid
                            else "tối (<= ngưỡng)" if out.invert else "sáng (> ngưỡng)"),
        "lua_chon_cuc_tinh": out.polarity,
        "phuong_sai_giua_lop_max": r.max_variance,
        "phuong_sai_toan_phan": r.total_variance,
        "do_tach_eta": r.separability,
        "w0": r.weight0, "w1": r.weight1, "mu0": r.mean0,
        "mu1": None if np.isnan(r.mean1) else r.mean1,
        "nguong_dong_hang": r.tied_thresholds,
        "ty_le_foreground": out.foreground_ratio,
        "thoi_gian_giay": r.elapsed_seconds,
        "pham_vi_thoi_gian": TIMING_SCOPE,
        "canh_bao": out.warnings,
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


# ----------------------------- Multi-Otsu ------------------------------

@dataclass
class MultiOtsuOutputs:
    result: MultiOtsuResult
    labels: np.ndarray  # (H, W) uint8, lớp 0/1/2
    label_color: np.ndarray
    foreground_classes: list[int]
    mask: np.ndarray
    cutout_rgba: np.ndarray
    cutout_preview: np.ndarray
    overlay: np.ndarray
    foreground_ratio: float
    warnings: list[str] = field(default_factory=list)
    raw_mask: np.ndarray | None = None
    postprocess: MaskPostprocess = NO_POSTPROCESS


def run_multi_otsu(prepared: PreparedInput) -> tuple[MultiOtsuResult, np.ndarray]:
    """Hai ngưỡng Otsu (3 lớp) trên ảnh xám; trả (kết quả, ảnh nhãn 0/1/2)."""
    return segment_image_multi_otsu(prepared.gray)


def default_multi_otsu_foreground(labels: np.ndarray) -> list[int]:
    """Gợi ý lớp đối tượng theo cùng heuristic viền/trung tâm với K-means."""
    return suggest_foreground_clusters(labels, 3)


def multi_otsu_outputs(prepared: PreparedInput, result: MultiOtsuResult, labels: np.ndarray, foreground_classes,
                       postprocess: MaskPostprocess = NO_POSTPROCESS) -> MultiOtsuOutputs:
    fg = sorted(set(int(c) for c in foreground_classes))
    if any(c not in (0, 1, 2) for c in fg):
        raise ValueError("Lớp đối tượng phải nằm trong {0, 1, 2}.")
    raw_mask = mask_from_labels(labels, fg)
    mask = postprocess.apply(raw_mask)
    return MultiOtsuOutputs(
        result=result,
        labels=labels,
        label_color=colorize_labels(labels),
        foreground_classes=fg,
        mask=mask,
        cutout_rgba=cutout_rgba(prepared.image, mask),
        cutout_preview=cutout_on_color(prepared.image, mask),
        overlay=overlay_mask(prepared.image, mask),
        foreground_ratio=foreground_ratio(mask),
        warnings=list(result.warnings),
        raw_mask=raw_mask,
        postprocess=postprocess,
    )


def multi_otsu_stats(prepared: PreparedInput, out: MultiOtsuOutputs) -> dict:
    r = out.result
    return {
        "thuat_toan": "Multi-Otsu 3 lớp (tự cài đặt, vét cạn O(256^2))",
        "anh": prepared.name,
        "kich_thuoc_goc_WxH": list(prepared.resize.original_size),
        "kich_thuoc_xu_ly_WxH": list(prepared.resize.processed_size),
        "chuyen_xam": prepared.gray_method,
        "tien_xu_ly": _preprocess_stats(prepared),
        "hau_xu_ly_mask": out.postprocess.as_dict(),
        "nguong": list(r.thresholds),
        "hop_le": r.valid,
        "quy_uoc": "lớp 0: I <= t1; lớp 1: t1 < I <= t2; lớp 2: I > t2",
        "phuong_sai_giua_lop_max": r.max_variance,
        "phuong_sai_toan_phan": r.total_variance,
        "do_tach_eta": r.separability,
        "trong_so_lop": r.class_weights,
        "trung_binh_lop": [None if np.isnan(m) else m for m in r.class_means],
        "so_pixel_lop": r.class_counts,
        "so_cap_nguong_dong_hang": r.n_tied,
        "lop_doi_tuong": out.foreground_classes,
        "ty_le_foreground": out.foreground_ratio,
        "thoi_gian_giay": r.elapsed_seconds,
        "pham_vi_thoi_gian": TIMING_SCOPE,
        "canh_bao": out.warnings,
    }


def multi_otsu_export_files(prepared: PreparedInput, out: MultiOtsuOutputs) -> dict[str, tuple[bytes, str]]:
    return {
        "multiotsu_label_map.png": (to_png_bytes(out.label_color), "image/png"),
        "multiotsu_mask.png": (to_png_bytes(out.mask), "image/png"),
        "multiotsu_cutout.png": (to_png_bytes(out.cutout_rgba), "image/png"),
        "multiotsu_labels.csv": (matrix_to_csv_bytes(out.labels), "text/csv"),
        "multiotsu_stats.json": (to_json_bytes(multi_otsu_stats(prepared, out)), "application/json"),
    }


def mask_scores(mask: np.ndarray, gt: np.ndarray | None):
    return None if gt is None else compare_masks(mask, gt)
