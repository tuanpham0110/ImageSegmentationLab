"""Trang Otsu Thresholding: histogram, ngưỡng tối ưu, so với ngưỡng thủ công, tách nền, tải xuống."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.processing.pipeline import otsu_export_files, run_otsu
from src.processing.preprocessing import GRAY_METHODS
from src.ui import components as ui
from src.ui.state import get_result, has_stale_result, store_result
from src.utils import visualization as viz
from src.utils.metrics import compare_masks

RESULT_KEY = "otsu_result"
POLARITY_OPTIONS = {
    "Gợi ý theo viền ảnh": "auto",
    "Lớp sáng (> t)": "bright",
    "Lớp tối (≤ t)": "dark",
}
DEFAULTS = {"otsu_polarity": "Gợi ý theo viền ảnh", "otsu_mode": "Tự động (Otsu)", "otsu_manual_t": 128}
MODES = ["Tự động (Otsu)", "Thủ công (đối chiếu)"]
POLARITY_HELP = ("Otsu chỉ chia 2 lớp sáng/tối, không biết lớp nào là đối tượng. "
                 "'Gợi ý theo viền ảnh' (heuristic): lớp chiếm đa số pixel viền ảnh được coi là nền.")


def polarity_caption(out) -> str:
    side = "lớp tối (≤ t)" if out.invert else "lớp sáng (> t)"
    how = " — theo gợi ý viền ảnh" if out.polarity == "auto" else ""
    return f"Đối tượng = {side}{how}."

DOWNLOAD_LABELS = {
    "otsu_binary.png": "⬇ Ảnh nhị phân (PNG)",
    "otsu_mask.png": "⬇ Mask đối tượng (PNG)",
    "otsu_cutout.png": "⬇ Ảnh tách nền RGBA (PNG)",
    "otsu_gray.png": "⬇ Ảnh xám (PNG)",
    "otsu_binary_matrix.csv": "⬇ Ma trận mask 0/1 (CSV)",
    "otsu_histogram.csv": "⬇ Histogram & σ²_B (CSV)",
    "otsu_stats.json": "⬇ Thông số & kết quả (JSON)",
}


def _reset_defaults() -> None:
    for k, v in DEFAULTS.items():
        st.session_state[k] = v


def render() -> None:
    ui.page_header("Phân ngưỡng Otsu (Otsu Thresholding)",
                   "Chọn ngưỡng t* làm cực đại phương sai giữa hai lớp trên histogram 256 mức xám (tự cài đặt).")
    ctx = ui.require_input()
    if ctx is None:
        return
    ui.input_summary(ctx)
    prep = ctx.prepared
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(k, v)

    left, right = st.columns([1, 2.6], gap="large")
    with left:
        with st.form("otsu_form", border=True):
            st.markdown("**Tham số phân ngưỡng**")
            mode = st.radio("Ngưỡng dùng để tạo kết quả", MODES, key="otsu_mode")
            manual_t = st.slider("Ngưỡng thủ công T (chỉ dùng ở chế độ thủ công)", 0, 255, key="otsu_manual_t")
            polarity = st.radio("Đối tượng (foreground) là", list(POLARITY_OPTIONS), key="otsu_polarity",
                                help=POLARITY_HELP)
            submitted = st.form_submit_button("Chạy Otsu", type="primary", width="stretch")
        st.button("Khôi phục mặc định", on_click=_reset_defaults, key="otsu_reset", width="stretch")
        st.caption(f"Ảnh màu được chuyển xám bằng: {GRAY_METHODS[prep.gray_method]} (đổi ở thanh bên).")

    if submitted:
        manual = int(manual_t) if mode == MODES[1] else None
        out = run_otsu(prep, polarity=POLARITY_OPTIONS[polarity], manual_threshold=manual)
        store_result(RESULT_KEY, ctx.signature, {"out": out})
        st.toast(f"Otsu: ngưỡng t* = {out.result.threshold} ({out.result.elapsed_seconds * 1000:.1f} ms).", icon="✅")

    run = get_result(RESULT_KEY, ctx.signature)
    with right:
        if run is None:
            if has_stale_result(RESULT_KEY, ctx.signature):
                st.warning("Ảnh đầu vào đã thay đổi — kết quả cũ đã bị ẩn. Hãy bấm **Chạy Otsu** lại.")
            ui.show_images([(prep.image, "Ảnh gốc"), (prep.gray, "Ảnh xám")])
            return
        _render_result(ctx, run["out"])


def _render_result(ctx, out) -> None:
    prep, r = ctx.prepared, out.result
    m = st.columns(5)
    m[0].metric("Ngưỡng Otsu t*", r.threshold if r.valid else f"{r.threshold} (không hợp lệ)")
    m[1].metric("Ngưỡng dùng", f"{out.threshold_used}" + (" (thủ công)" if out.mode == "manual" else ""))
    m[2].metric("Độ tách η", f"{r.separability:.3f}", help="η = σ²_B(t*) / σ²_T, trong [0, 1]; càng gần 1 hai lớp càng tách biệt.")
    m[3].metric("Đối tượng", f"{out.foreground_ratio * 100:.1f}%", help="Tỷ lệ pixel đối tượng; phần còn lại là nền.")
    m[4].metric("Thời gian", f"{r.elapsed_seconds * 1000:.2f} ms")
    st.caption(f"Quy ước: pixel > {out.threshold_used} → 255 (lớp C1, sáng). " + polarity_caption(out))
    for w in r.warnings:
        st.warning(w)

    tab_res, tab_hist, tab_dl = st.tabs(["Kết quả phân ngưỡng", "Histogram & phương sai", "Tải xuống"])
    with tab_res:
        ui.show_images([(prep.image, "Ảnh gốc"), (prep.gray, "Ảnh xám"), (out.binary, f"Ảnh nhị phân (> {out.threshold_used})")])
        ui.show_images([(out.mask, "Mask đối tượng"), (out.cutout_preview, "Ảnh tách nền (nền trắng)"),
                        (out.overlay, "Đối tượng tô màu trên ảnh gốc")])
        if ctx.ground_truth is not None:
            s = compare_masks(out.mask, ctx.ground_truth)
            st.success(f"So với ground truth của ảnh mẫu: IoU = {s.iou:.4f} · Dice = {s.dice:.4f} · "
                       f"độ chính xác pixel = {s.pixel_accuracy:.4f}")

    with tab_hist:
        manual = out.threshold_used if out.mode == "manual" else None
        ui.show_figure(viz.histogram_figure(r.histogram, r.threshold if r.valid else None, manual))
        ui.show_figure(viz.variance_figure(r.between_class_variance, r.valid_thresholds, r.threshold if r.valid else None))
        if r.valid:
            stats = pd.DataFrame([
                {"Lớp": f"C0 (≤ {r.threshold})", "Trọng số ω": round(r.weight0, 5), "Trung bình μ": round(r.mean0, 3),
                 "Số pixel": int(r.histogram[: r.threshold + 1].sum())},
                {"Lớp": f"C1 (> {r.threshold})", "Trọng số ω": round(r.weight1, 5), "Trung bình μ": round(r.mean1, 3),
                 "Số pixel": int(r.histogram[r.threshold + 1 :].sum())},
            ])
            st.dataframe(stats, hide_index=True, width="stretch")
            st.caption(f"σ²_B(t*) = {r.max_variance:.3f} · σ²_T = {r.total_variance:.3f}. "
                       + (f"Có {len(r.tied_thresholds)} ngưỡng đồng hạng ({min(r.tied_thresholds)}–{max(r.tied_thresholds)}), "
                          "chọn floor(trung bình)." if len(r.tied_thresholds) > 1 else "Ngưỡng cực đại là duy nhất."))
        with st.expander("Bảng histogram đầy đủ (256 mức)"):
            nz = np.flatnonzero(r.histogram)
            st.dataframe(pd.DataFrame({"Mức xám g": nz, "h(g)": r.histogram[nz],
                                       "σ²_B(g)": np.round(r.between_class_variance[nz], 4)}),
                         hide_index=True, width="stretch", height=260)

    with tab_dl:
        st.caption("Các tệp được tạo từ đúng kết quả đang hiển thị.")
        ui.download_buttons(otsu_export_files(prep, out), DOWNLOAD_LABELS, "otsu_dl")
