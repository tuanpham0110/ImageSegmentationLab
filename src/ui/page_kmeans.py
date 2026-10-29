"""Trang K-means: cấu hình, chạy, xem kết quả, chọn cụm đối tượng, tải xuống."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.processing.pipeline import (
    KMeansParams,
    cluster_table,
    default_foreground,
    kmeans_export_files,
    kmeans_outputs,
    run_kmeans,
)
from src.ui import components as ui
from src.ui.state import get_result, has_stale_result, store_result
from src.utils import visualization as viz
from src.utils.metrics import compare_masks

RESULT_KEY = "km_result"
DEFAULTS = {"km_k": 3, "km_feature": "RGB (3 kênh)", "km_init": "K-means++", "km_max_iter": 100,
            "km_tol": 1e-4, "km_seed": 0}
FEATURE_OPTIONS = {"RGB (3 kênh)": "rgb", "Mức xám (1 kênh)": "gray"}
INIT_OPTIONS = {"K-means++": "kmeans++", "Ngẫu nhiên (random)": "random"}

DOWNLOAD_LABELS = {
    "kmeans_segmented.png": "⬇ Ảnh phân đoạn (PNG)",
    "kmeans_label_map.png": "⬇ Bản đồ nhãn màu (PNG)",
    "kmeans_mask.png": "⬇ Mask nhị phân (PNG)",
    "kmeans_cutout.png": "⬇ Ảnh tách nền RGBA (PNG)",
    "kmeans_labels.csv": "⬇ Ma trận nhãn cụm (CSV)",
    "kmeans_clusters.csv": "⬇ Bảng thống kê cụm (CSV)",
    "kmeans_stats.json": "⬇ Thông số & kết quả (JSON)",
}


def _reset_defaults() -> None:
    for k, v in DEFAULTS.items():
        st.session_state[k] = v


def params_form(is_color: bool, key_prefix: str = "km") -> KMeansParams | None:
    """Form tham số K-means. Trả về KMeansParams khi người dùng bấm chạy, ngược lại None."""
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(k.replace("km", key_prefix, 1), v)
    with st.form(f"{key_prefix}_form", border=True):
        st.markdown("**Tham số K-means**")
        k = st.slider("Số cụm K", 2, 8, key=f"{key_prefix}_k",
                      help="Số nhóm màu/mức xám cần chia. K-means không tự xác định K.")
        feature_label = st.radio(
            "Vector đặc trưng của pixel", list(FEATURE_OPTIONS), key=f"{key_prefix}_feature",
            disabled=not is_color,
            help="Ảnh xám luôn dùng 1 kênh mức xám." if not is_color else "RGB: 3 chiều (R,G,B); mức xám: 1 chiều.",
        )
        init_label = st.selectbox("Khởi tạo tâm", list(INIT_OPTIONS), key=f"{key_prefix}_init")
        c1, c2 = st.columns(2)
        max_iter = c1.number_input("Vòng lặp tối đa", 1, 1000, step=10, key=f"{key_prefix}_max_iter")
        seed = c2.number_input("Random seed", 0, 2**31 - 1, step=1, key=f"{key_prefix}_seed")
        tol = st.number_input("Ngưỡng hội tụ (độ dịch tâm)", 0.0, 50.0, step=1e-4, format="%.4f",
                              key=f"{key_prefix}_tol",
                              help="Dừng khi tâm cụm dịch chuyển (chuẩn Euclid, thang 0–255) ≤ ngưỡng này.")
        submitted = st.form_submit_button("Chạy K-means", type="primary", width="stretch")
    if not submitted:
        return None
    return KMeansParams(k=int(k), feature=FEATURE_OPTIONS[feature_label] if is_color else "gray",
                        max_iter=int(max_iter), tol=float(tol), seed=int(seed), init=INIT_OPTIONS[init_label])


def render() -> None:
    ui.page_header("Phân đoạn bằng K-means", "Phân cụm pixel theo khoảng cách Euclid tới tâm cụm (tự cài đặt, NumPy).")
    ctx = ui.require_input()
    if ctx is None:
        return
    ui.input_summary(ctx)
    prep = ctx.prepared

    left, right = st.columns([1, 2.6], gap="large")
    with left:
        params = params_form(prep.is_color)
        st.button("Khôi phục mặc định", on_click=_reset_defaults, width="stretch")
    if params is not None:
        with right, st.spinner("Đang phân cụm K-means…"):
            result = run_kmeans(prep, params)
        store_result(RESULT_KEY, ctx.signature, {"params": params, "result": result,
                                                 "run_id": st.session_state.get("km_run_id", 0) + 1})
        st.session_state["km_run_id"] = st.session_state.get("km_run_id", 0) + 1
        st.toast(f"K-means hoàn thành sau {result.core.n_iter} vòng lặp ({result.core.elapsed_seconds:.2f} s).", icon="✅")

    run = get_result(RESULT_KEY, ctx.signature)
    with right:
        if run is None:
            if has_stale_result(RESULT_KEY, ctx.signature):
                st.warning("Ảnh đầu vào đã thay đổi — kết quả cũ đã bị ẩn. Hãy bấm **Chạy K-means** lại.")
            ui.show_image(prep.image, "Ảnh gốc (chưa chạy thuật toán)")
            return
        _render_result(ctx, run)


def _render_result(ctx, run: dict) -> None:
    prep, params, result = ctx.prepared, run["params"], run["result"]
    core = result.core
    m = st.columns(5)
    m[0].metric("Số cụm K", f"{core.k_effective}" + ("" if core.k_effective == core.k_requested else f" (yêu cầu {core.k_requested})"))
    m[1].metric("Số vòng lặp", core.n_iter)
    m[2].metric("Hội tụ", "Có" if core.converged else "Chưa")
    m[3].metric("SSE (inertia)", ui.compact_number(core.sse), help=f"SSE = {core.sse:,.2f}")
    m[4].metric("Thời gian", f"{core.elapsed_seconds * 1000:.0f} ms")
    st.caption(f"Đặc trưng: {'RGB' if result.feature == 'rgb' else 'mức xám'} · khởi tạo {params.init} · "
               f"seed {params.seed} · tối đa {params.max_iter} vòng · tol {params.tol:g}")
    for w in core.warnings:
        st.warning(w)

    tab_seg, tab_fg, tab_stats, tab_dl = st.tabs(["Kết quả phân cụm", "Tách nền (chọn đối tượng)", "Thống kê", "Tải xuống"])

    with tab_fg:
        st.markdown(
            "K-means chỉ chia pixel thành các **cụm** — thuật toán **không biết** cụm nào là đối tượng. "
            "Hãy chọn một hoặc nhiều cụm làm đối tượng (foreground). Lựa chọn mặc định chỉ là **gợi ý**: "
            "cụm chiếm nhiều pixel ở viền ảnh nhất được coi là nền."
        )
        options = list(range(core.k_effective))
        fg = st.multiselect("Cụm làm đối tượng (foreground)", options, default=default_foreground(result),
                            format_func=lambda c: f"Cụm {c}", key=f"km_fg_{run['run_id']}")
    out = kmeans_outputs(prep, result, fg)

    with tab_seg:
        ui.show_images([(prep.image, "Ảnh gốc"), (out.segmented, "Ảnh K-means (màu tâm cụm)"),
                        (out.label_color, "Bản đồ nhãn cụm")])
        legend = " ".join(
            f"<span class='isl-chip'><span class='isl-swatch' style='background:{c}'></span>Cụm {i}</span>"
            for i, c in enumerate(viz.CLUSTER_PALETTE_HEX[: core.k_effective])
        )
        st.markdown("Chú giải bản đồ nhãn: " + legend, unsafe_allow_html=True)

    with tab_fg:
        if not fg:
            st.info("Chưa chọn cụm nào: mask rỗng (toàn nền).")
        ui.show_images([(out.mask, "Mask nhị phân (trắng = đối tượng)"), (out.cutout_preview, "Ảnh tách nền (nền trắng)"),
                        (out.overlay, "Đối tượng tô màu trên ảnh gốc")])
        ratio = out.mask.mean() / 255 * 100
        st.caption(f"Đối tượng chiếm {ratio:.2f}% pixel · nền {100 - ratio:.2f}%.")
        if ctx.ground_truth is not None:
            s = compare_masks(out.mask, ctx.ground_truth)
            st.success(f"So với ground truth của ảnh mẫu: IoU = {s.iou:.4f} · Dice = {s.dice:.4f} · "
                       f"độ chính xác pixel = {s.pixel_accuracy:.4f}")

    with tab_stats:
        c1, c2 = st.columns([1.1, 1])
        with c1:
            st.markdown("**Bảng thống kê cụm**")
            st.dataframe(pd.DataFrame(cluster_table(result)), hide_index=True, width="stretch")
            ui.show_figure(viz.cluster_share_figure(core.counts, core.centers))
        with c2:
            ui.show_figure(viz.sse_figure(core.sse_history))
            st.caption("SSE = Σ‖x − tâm cụm của x‖², tính sau bước gán nhãn của mỗi vòng. "
                       "Theo lý thuyết, SSE không tăng qua các vòng lặp.")
            hist = pd.DataFrame({"Vòng lặp": range(1, core.n_iter + 1), "SSE": core.sse_history,
                                 "Độ dịch tâm lớn nhất": core.shift_history})
            st.dataframe(hist, hide_index=True, width="stretch")

    with tab_dl:
        st.caption("Các tệp được tạo từ đúng kết quả đang hiển thị (cùng ảnh, tham số và cụm đối tượng đã chọn).")
        ui.download_buttons(kmeans_export_files(prep, params, result, out), DOWNLOAD_LABELS, "km_dl")
