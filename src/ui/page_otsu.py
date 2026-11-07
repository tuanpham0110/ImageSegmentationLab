"""Trang Otsu Thresholding: histogram, ngưỡng tối ưu (2 lớp hoặc Multi-Otsu 3 lớp), tách nền, tải xuống."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.processing.pipeline import (
    TIMING_SCOPE,
    default_multi_otsu_foreground,
    multi_otsu_export_files,
    multi_otsu_outputs,
    otsu_export_files,
    run_multi_otsu,
    run_otsu,
)
from src.processing.postprocessing import cluster_palette_hex
from src.processing.preprocessing import GRAY_METHODS
from src.ui import components as ui
from src.ui.state import (
    get_result,
    has_stale_result,
    persistent_foreground,
    remember_widget,
    restore_widget,
    save_foreground,
    store_result,
)
from src.utils import visualization as viz
from src.utils.metrics import compare_masks

RESULT_KEY = "otsu_result"
POLARITY_OPTIONS = {
    "Gợi ý theo viền ảnh": "auto",
    "Lớp sáng (> t)": "bright",
    "Lớp tối (≤ t)": "dark",
}
CLASS_MODES = ["2 lớp — Otsu (1 ngưỡng)", "3 lớp — Multi-Otsu (2 ngưỡng)"]
DEFAULTS = {"otsu_classes": CLASS_MODES[0], "otsu_polarity": "Gợi ý theo viền ảnh", "otsu_mode": "Tự động (Otsu)",
            "otsu_manual_t": 128}
MODES = ["Tự động (Otsu)", "Thủ công (đối chiếu)"]
POLARITY_HELP = ("Otsu chỉ chia 2 lớp sáng/tối, không biết lớp nào là đối tượng. "
                 "'Gợi ý theo viền ảnh' (heuristic): lớp xuất hiện ở viền ảnh nhiều hơn ở vùng trung tâm là nền.")
CLASSES_HELP = ("Multi-Otsu tìm hai ngưỡng t1 < t2 cực đại phương sai giữa 3 lớp (vét cạn 256² cặp ngưỡng). "
                "Dùng khi ảnh có nhiều đối tượng ở các mức xám khác nhau — Otsu 2 lớp sẽ gộp lớp giữa vào nền "
                "hoặc vào đối tượng.")


def polarity_caption(out) -> str:
    if out.mode == "otsu" and not out.result.valid:
        return "Không có ngưỡng chia hai lớp; mask tự động được để trống."
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
MULTI_DOWNLOAD_LABELS = {
    "multiotsu_label_map.png": "⬇ Bản đồ 3 lớp (PNG)",
    "multiotsu_mask.png": "⬇ Mask đối tượng (PNG)",
    "multiotsu_cutout.png": "⬇ Ảnh tách nền RGBA (PNG)",
    "multiotsu_labels.csv": "⬇ Ma trận nhãn lớp 0/1/2 (CSV)",
    "multiotsu_stats.json": "⬇ Thông số & kết quả (JSON)",
}


def _reset_defaults() -> None:
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.session_state["otsu_morph"] = "none"
    st.session_state["otsu_morph_size"] = 3


def render() -> None:
    ui.page_header("Phân ngưỡng Otsu (Otsu Thresholding)",
                   "Chọn ngưỡng t* làm cực đại phương sai giữa các lớp trên histogram 256 mức xám (tự cài đặt).")
    ctx = ui.require_input()
    if ctx is None:
        return
    ui.input_summary(ctx)
    prep = ctx.prepared
    for k, v in DEFAULTS.items():
        restore_widget(k, v)

    left, right = st.columns([1, 2.6], gap="large")
    with left:
        with st.form("otsu_form", border=True):
            st.markdown("**Tham số phân ngưỡng**")
            classes = st.radio("Số lớp", CLASS_MODES, key="otsu_classes", help=CLASSES_HELP)
            mode = st.radio("Ngưỡng dùng để tạo kết quả (2 lớp)", MODES, key="otsu_mode")
            manual_t = st.slider("Ngưỡng thủ công T (chỉ dùng ở chế độ thủ công)", 0, 255, key="otsu_manual_t")
            polarity = st.radio("Đối tượng (foreground) là (2 lớp)", list(POLARITY_OPTIONS), key="otsu_polarity",
                                help=POLARITY_HELP)
            post = ui.morphology_controls("otsu")
            submitted = st.form_submit_button("Chạy Otsu", type="primary", width="stretch")
        for k in DEFAULTS:
            remember_widget(k)
        st.button("Khôi phục mặc định", on_click=_reset_defaults, key="otsu_reset", width="stretch")
        st.caption(f"Ảnh màu được chuyển xám bằng: {GRAY_METHODS[prep.gray_method]} (đổi ở thanh bên).")

    if submitted:
        try:
            with st.spinner("Đang phân ngưỡng…"):
                if classes == CLASS_MODES[1]:
                    result, labels = run_multi_otsu(prep)
                    run_id = st.session_state.get("otsu_run_id", 0) + 1
                    st.session_state["otsu_run_id"] = run_id
                    payload = {"kind": "multi", "result": result, "labels": labels, "post": post, "run_id": run_id}
                else:
                    manual = int(manual_t) if mode == MODES[1] else None
                    out = run_otsu(prep, polarity=POLARITY_OPTIONS[polarity], manual_threshold=manual, postprocess=post)
                    payload = {"kind": "binary", "out": out}
        except (ValueError, MemoryError) as exc:
            st.session_state.pop(RESULT_KEY, None)
            st.error("Không phân ngưỡng được ảnh. Hãy kiểm tra dữ liệu hoặc giảm giới hạn cạnh dài. " + str(exc))
            return
        store_result(RESULT_KEY, ctx.signature, payload)
        if payload["kind"] == "multi":
            r = payload["result"]
            st.toast(f"Multi-Otsu: (t1, t2) = {r.thresholds} ({r.elapsed_seconds * 1000:.1f} ms).",
                     icon="✅" if r.valid else "⚠️")
        elif out.mode == "otsu" and not out.result.valid:
            st.toast("Otsu không chia được hai lớp; mask tự động được để trống.", icon="⚠️")
        else:
            st.toast(f"Ngưỡng sử dụng = {out.threshold_used} ({out.result.elapsed_seconds * 1000:.1f} ms).", icon="✅")

    run = get_result(RESULT_KEY, ctx.signature)
    with right:
        if run is None:
            if has_stale_result(RESULT_KEY, ctx.signature):
                st.warning("Ảnh đầu vào đã thay đổi — kết quả cũ đã bị ẩn. Hãy bấm **Chạy Otsu** lại.")
            ui.show_images([(prep.image, "Ảnh gốc"), (prep.gray, "Ảnh xám")])
            return
        if run.get("kind") == "multi":
            _render_multi(ctx, run)
        else:
            _render_result(ctx, run["out"], run)


def _render_result(ctx, out, run: dict | None = None) -> None:
    prep, r = ctx.prepared, out.result
    m = st.columns(5)
    m[0].metric("Ngưỡng Otsu t*", r.threshold if r.valid else f"{r.threshold} (không hợp lệ)")
    m[1].metric("Ngưỡng dùng", f"{out.threshold_used}" + (" (thủ công)" if out.mode == "manual" else ""))
    m[2].metric("Độ tách η", f"{r.separability:.3f}", help="η = σ²_B(t*) / σ²_T, trong [0, 1]; càng gần 1 hai lớp càng tách biệt.")
    m[3].metric("Đối tượng", f"{out.foreground_ratio * 100:.1f}%", help="Tỷ lệ pixel đối tượng; phần còn lại là nền.")
    m[4].metric("Thời gian", f"{r.elapsed_seconds * 1000:.2f} ms", help=TIMING_SCOPE)
    st.caption(f"Quy ước: pixel > {out.threshold_used} → 255 (lớp C1, sáng). " + polarity_caption(out))
    for w in out.warnings:
        st.warning(w)

    tab_res, tab_hist, tab_dl = st.tabs(["Kết quả phân ngưỡng", "Histogram & phương sai", "Tải xuống"])
    with tab_res:
        ui.show_images([(prep.image, "Ảnh gốc"), (prep.gray, "Ảnh xám"), (out.binary, f"Ảnh nhị phân (> {out.threshold_used})")])
        if out.postprocess.operation != "none":
            ui.show_images([(out.raw_mask, "Mask trước hình thái học"), (out.mask, "Mask sau hình thái học"),
                            (out.overlay, "Đối tượng tô màu trên ảnh gốc")])
            ui.show_image(out.cutout_preview, "Ảnh tách nền (nền trắng)")
        else:
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
                          "chọn một cực đại gần floor(trung bình) nhất; các khoảng có thể rời nhau."
                          if len(r.tied_thresholds) > 1 else "Ngưỡng cực đại là duy nhất."))
        with st.expander("Bảng histogram đầy đủ (256 mức)"):
            levels = np.arange(256)
            st.caption(f"Đủ 256 mức xám, gồm cả mức có h(g) = 0; ảnh có {np.count_nonzero(r.histogram)} mức xuất hiện. "
                       "σ²_B(g) = 0 tại ngưỡng làm một lớp rỗng (không hợp lệ).")
            st.dataframe(pd.DataFrame({"Mức xám g": levels, "h(g)": r.histogram,
                                       "σ²_B(g)": np.round(r.between_class_variance, 4)}),
                         hide_index=True, width="stretch", height=260)

    with tab_dl:
        st.caption("Các tệp được tạo từ đúng kết quả đang hiển thị.")
        files = ui.cached_export_files({} if run is None else run,
                                       (out.mode, out.threshold_used, out.polarity, out.postprocess),
                                       lambda: otsu_export_files(prep, out))
        if files is not None:
            ui.download_buttons(files, DOWNLOAD_LABELS, "otsu_dl")


def _render_multi(ctx, run: dict) -> None:
    prep, r, labels, post = ctx.prepared, run["result"], run["labels"], run["post"]
    t1, t2 = r.thresholds
    options = [0, 1, 2]
    fg_key = f"otsu_multi_fg_{run['run_id']}"
    persistent_foreground(run, fg_key, options, default_multi_otsu_foreground(labels))

    m = st.columns(5)
    m[0].metric("Ngưỡng t1", t1)
    m[1].metric("Ngưỡng t2", t2 if r.valid else f"{t2} (không hợp lệ)")
    m[2].metric("Độ tách η", f"{r.separability:.3f}", help="η = σ²_B(t1, t2) / σ²_T, trong [0, 1].")
    m[4].metric("Thời gian", f"{r.elapsed_seconds * 1000:.2f} ms", help=TIMING_SCOPE)
    st.caption(f"Quy ước: lớp 0 = {{g ≤ {t1}}}, lớp 1 = {{{t1} < g ≤ {t2}}}, lớp 2 = {{g > {t2}}}.")
    for w in r.warnings:
        st.warning(w)

    st.markdown("Multi-Otsu chia ảnh thành **3 lớp** theo mức xám nhưng không biết lớp nào là đối tượng. "
                "Lựa chọn mặc định là **gợi ý** theo viền ảnh/vùng trung tâm.")
    legend = " ".join(f"<span class='isl-chip'><span class='isl-swatch' style='background:{c}'></span>Lớp {i}</span>"
                      for i, c in enumerate(cluster_palette_hex(3)))
    fg = st.multiselect("Lớp làm đối tượng (foreground)", options, format_func=lambda c: f"Lớp {c}", key=fg_key)
    save_foreground(run, fg)
    try:
        out = multi_otsu_outputs(prep, r, labels, fg, post)
    except (ValueError, MemoryError) as exc:
        run.pop("_export_cache", None)
        st.error("Không tạo ảnh kết quả được. " + str(exc))
        return
    m[3].metric("Đối tượng", f"{out.foreground_ratio * 100:.1f}%")

    tab_res, tab_hist, tab_dl = st.tabs(["Kết quả 3 lớp", "Histogram", "Tải xuống"])
    with tab_res:
        ui.show_images([(prep.image, "Ảnh gốc"), (prep.gray, "Ảnh xám"), (out.label_color, "Bản đồ 3 lớp")])
        st.markdown("Chú giải: " + legend, unsafe_allow_html=True)
        if not fg:
            st.info("Chưa chọn lớp nào: mask rỗng (toàn nền).")
        if post.operation != "none":
            ui.show_images([(out.raw_mask, "Mask trước hình thái học"), (out.mask, "Mask sau hình thái học"),
                            (out.overlay, "Đối tượng tô màu trên ảnh gốc")])
            ui.show_image(out.cutout_preview, "Ảnh tách nền (nền trắng)")
        else:
            ui.show_images([(out.mask, "Mask đối tượng"), (out.cutout_preview, "Ảnh tách nền (nền trắng)"),
                            (out.overlay, "Đối tượng tô màu trên ảnh gốc")])
        if ctx.ground_truth is not None:
            s = compare_masks(out.mask, ctx.ground_truth)
            st.success(f"So với ground truth của ảnh mẫu: IoU = {s.iou:.4f} · Dice = {s.dice:.4f} · "
                       f"độ chính xác pixel = {s.pixel_accuracy:.4f}")

    with tab_hist:
        ui.show_figure(viz.multi_histogram_figure(r.histogram, r.thresholds))
        st.dataframe(pd.DataFrame([
            {"Lớp": f"Lớp {k}", "Trọng số ω": round(r.class_weights[k], 5),
             "Trung bình μ": None if np.isnan(r.class_means[k]) else round(r.class_means[k], 3),
             "Số pixel": r.class_counts[k]} for k in range(3)
        ]), hide_index=True, width="stretch")
        st.caption(f"σ²_B(t1, t2) = {r.max_variance:.3f} · σ²_T = {r.total_variance:.3f}. "
                   "Tìm kiếm vét cạn trên mọi cặp t1 < t2 bằng tổng tích luỹ: O(256²), không phụ thuộc kích thước ảnh."
                   + (f" Có {r.n_tied} cặp ngưỡng đồng hạng." if r.n_tied > 1 else ""))

    with tab_dl:
        st.caption("Các tệp được tạo từ đúng kết quả đang hiển thị.")
        files = ui.cached_export_files(run, (tuple(sorted(fg)), post), lambda: multi_otsu_export_files(prep, out))
        if files is not None:
            ui.download_buttons(files, MULTI_DOWNLOAD_LABELS, "otsu_multi_dl")
