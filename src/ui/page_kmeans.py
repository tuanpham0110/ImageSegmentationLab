"""Trang K-means: cấu hình, chạy, xem kết quả, chọn cụm đối tượng, tải xuống."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.processing.pipeline import (
    TIMING_SCOPE,
    KMeansParams,
    cluster_table,
    default_foreground,
    kmeans_export_files,
    kmeans_outputs,
    run_kmeans,
)
from src.processing.postprocessing import cluster_palette_hex, colorize_labels
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

RESULT_KEY = "km_result"
DEFAULTS = {"k": 3, "feature": "RGB (3 kênh)", "init": "K-means++", "max_iter": 100, "tol": 1e-4, "seed": 0}
FEATURE_OPTIONS = {"RGB (3 kênh)": "rgb", "CIE L*a*b* (3 kênh, theo cảm nhận)": "lab", "Mức xám (1 kênh)": "gray"}
FEATURE_NAMES = {"rgb": "RGB", "lab": "CIE L*a*b*", "gray": "mức xám"}
UI_MAX_K = 16  # giới hạn thanh trượt cho tiện dùng; thuật toán không giới hạn K
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


def reset_kmeans_params(key_prefix: str) -> None:
    for name, value in DEFAULTS.items():
        key = f"{key_prefix}_{name}"
        st.session_state[key] = value
        remember_widget(key)


def _reset_defaults() -> None:
    reset_kmeans_params("km")
    ui.reset_morphology("km")
    st.session_state["km_invert_foreground"] = False
    remember_widget("km_invert_foreground")


def params_controls(is_color: bool, key_prefix: str = "km", *, submit_label: str = "Chạy K-means") -> tuple[KMeansParams, bool]:
    """Đọc thiết lập hiện tại để phát hiện kết quả cũ. Nút chạy mới kích hoạt thuật toán."""
    defaults = {f"{key_prefix}_{name}": value for name, value in DEFAULTS.items() if name != "feature" or is_color}
    for key, value in defaults.items():
        restore_widget(key, value)
    with st.container(border=True, key=f"{key_prefix}-params"):
        st.markdown("**Thiết lập K-means**")
        k = st.slider("Số cụm K", 1, UI_MAX_K, key=f"{key_prefix}_k",
                      help="K = 1 tạo một cụm để quan sát tâm trung bình. Muốn tách hai nhóm, hãy chọn K từ 2. "
                           "Bảng màu bản đồ nhãn tự mở rộng theo K.")
        if is_color:
            feature_label = st.radio(
                "Đặc trưng phân cụm", list(FEATURE_OPTIONS), key=f"{key_prefix}_feature",
                format_func=lambda label: {"RGB (3 kênh)": "Màu RGB", "CIE L*a*b* (3 kênh, theo cảm nhận)": "Màu Lab (theo cảm nhận)",
                                           "Mức xám (1 kênh)": "Mức xám"}[label],
                help="RGB và Lab dùng ba kênh. Mức xám dùng độ sáng của pixel.",
            )
            feature = FEATURE_OPTIONS[feature_label]
        else:
            feature = "gray"
            st.caption("Đặc trưng: mức xám (1 kênh).")
        with st.expander("Thiết lập nâng cao", icon=":material/tune:"):
            init_label = st.selectbox("Khởi tạo tâm", list(INIT_OPTIONS), key=f"{key_prefix}_init")
            max_iter = st.number_input("Vòng lặp tối đa", 1, 1000, step=10, key=f"{key_prefix}_max_iter")
            seed = st.number_input("Hạt giống ngẫu nhiên (seed)", 0, 2**31 - 1, step=1, key=f"{key_prefix}_seed")
            tol = st.number_input("Ngưỡng hội tụ", 0.0, 50.0, step=1e-4, format="%.4f",
                                  key=f"{key_prefix}_tol",
                                  help="Ngưỡng Euclid theo đặc trưng đang dùng (mức xám, RGB hoặc Lab).")
        submitted = st.button(submit_label, type="primary", width="stretch", key=f"{key_prefix}_run")
    for key in defaults:
        remember_widget(key)
    params = KMeansParams(k=int(k), feature=feature, max_iter=int(max_iter), tol=float(tol), seed=int(seed),
                          init=INIT_OPTIONS[init_label])
    return params, submitted


def render() -> None:
    ui.page_header("Phân đoạn bằng K-means", "Gom các pixel theo màu hoặc độ sáng rồi chọn cụm để tách đối tượng.")
    ctx = ui.require_input()
    if ctx is None:
        return
    ui.input_summary(ctx)
    prep = ctx.prepared

    left, right = ui.workspace("kmeans")
    with left:
        params, submitted = params_controls(prep.is_color)
        st.button("Khôi phục mặc định", on_click=_reset_defaults, key="km_reset", type="tertiary", width="stretch")
    if submitted:
        try:
            with right, st.spinner("Đang phân cụm K-means…", show_time=True):
                result = run_kmeans(prep, params)
        except (ValueError, MemoryError) as exc:
            st.session_state.pop(RESULT_KEY, None)
            with right:
                st.error("Không chạy được K-means. Hãy kiểm tra tham số hoặc giảm giới hạn cạnh dài. " + str(exc))
            return
        store_result(RESULT_KEY, ctx.signature, {"params": params, "result": result,
                                                 "run_id": st.session_state.get("km_run_id", 0) + 1})
        st.session_state["km_run_id"] = st.session_state.get("km_run_id", 0) + 1
        status = "đã hội tụ" if result.core.converged else "đã dừng, chưa hội tụ"
        st.toast(f"K-means {status} sau {result.core.n_iter} vòng lặp ({result.core.elapsed_seconds:.2f} s).",
                 icon="✅" if result.core.converged else "⚠️")

    run = get_result(RESULT_KEY, ctx.signature)
    with right:
        if run is None:
            ui.result_empty(prep, "Chạy K-means", stale=has_stale_result(RESULT_KEY, ctx.signature))
            return
        pending = params != run["params"]
        if pending:
            ui.show_pending_settings("Chạy K-means")
        _render_result(ctx, run, pending)


def _render_result(ctx, run: dict, pending: bool) -> None:
    prep, params, result = ctx.prepared, run["params"], run["result"]
    core = result.core
    m = ui.metric_columns(5, "kmeans")
    m[0].metric("Số cụm K", f"{core.k_effective}" + ("" if core.k_effective == core.k_requested else f" (yêu cầu {core.k_requested})"))
    m[1].metric("Số vòng lặp", core.n_iter)
    m[2].metric("Hội tụ", "Có" if core.converged else "Chưa")
    m[3].metric("SSE (inertia)", ui.compact_number(core.sse), help=f"SSE = {core.sse:,.2f}")
    m[4].metric("Thời gian", f"{core.elapsed_seconds * 1000:.0f} ms", help=TIMING_SCOPE)
    st.caption(f"Lần chạy đã áp dụng · K={params.k} · {FEATURE_NAMES[result.feature]} · khởi tạo {params.init} · "
               f"seed {params.seed} · tối đa {params.max_iter} vòng · tol {params.tol:g}")
    for w in core.warnings:
        st.warning(w)

    tab_seg, tab_fg, tab_stats, tab_dl = st.tabs(
        ["Kết quả phân cụm", "Tách nền (chọn đối tượng)", "Thống kê", "Tải xuống"],
        key="km_result_tab", on_change="rerun",
    )

    with tab_fg:
        if tab_fg.open:
            st.caption("Chọn một cụm để xem mask riêng, hoặc chọn nhiều cụm để tạo một mask chung. "
                       "Lựa chọn ban đầu là gợi ý theo viền ảnh.")
        options = list(range(core.k_effective))
        fg_key = f"km_fg_{run['run_id']}"
        if fg_key not in st.session_state:
            initial = run["foreground"] if "foreground" in run else default_foreground(result)
            persistent_foreground(run, fg_key, options, initial)
        fg = st.multiselect("Cụm làm đối tượng (foreground)", options,
                            format_func=lambda c: f"Cụm {c}", key=fg_key)
        save_foreground(run, fg)
        restore_widget("km_invert_foreground", False)
        invert_foreground = st.checkbox("Đảo vùng đối tượng", key="km_invert_foreground",
                                       help="Dùng các cụm còn lại làm đối tượng. Không thay đổi kết quả phân cụm.")
        remember_widget("km_invert_foreground")
        if invert_foreground:
            fg = sorted(set(options) - set(fg))
        post = ui.morphology_controls("km")
    try:
        out = kmeans_outputs(prep, result, fg, post) if tab_fg.open or (tab_dl.open and not pending) else None
    except (ValueError, MemoryError) as exc:
        run.pop("_export_cache", None)
        run.pop("_original_export_cache", None)
        st.error("Không tạo ảnh kết quả được. Hãy kiểm tra lựa chọn cụm hoặc giảm giới hạn cạnh dài. " + str(exc))
        return

    if tab_seg.open:
        with tab_seg:
            _show_segmentation(prep, result)

    if tab_fg.open:
        with tab_fg:
            _show_foreground(ctx, out, fg, post)

    if tab_stats.open:
        with tab_stats:
            _show_statistics(result)

    if tab_dl.open and not pending:
        with tab_dl:
            st.caption("Các tệp dùng tham số lần chạy ở trên và cụm đối tượng đã chọn.")
            export_key = (tuple(sorted(fg)), post)
            files = ui.cached_export_files(run, export_key, lambda: kmeans_export_files(prep, params, result, out))
            if files is not None:
                ui.download_buttons(files, DOWNLOAD_LABELS, "km_dl")
                ui.offer_original_size_pngs(ctx, run, export_key, "kmeans", out.mask,
                                            {"segmented": out.segmented, "label_map": out.label_color})


def _show_segmentation(prep, result) -> None:
    original, *processed = ui.source_panels(prep)
    ui.show_images([original, (result.segmented, "Ảnh K-means (màu tâm cụm)")])
    st.markdown("**Màu tâm cụm**")
    colors = result.centers_rgb if result.centers_rgb is not None else np.repeat(result.centers, 3, axis=1)
    colors = np.clip(np.round(colors), 0, 255).astype(np.uint8)
    counts = result.core.counts
    total = counts.sum()
    swatches = "".join(
        f"<span class='isl-chip'><span class='isl-swatch' aria-hidden='true' style='background:#{r:02x}{g:02x}{b:02x}'></span>"
        f"Cụm {i} · {count / total:.1%}</span>"
        for i, ((r, g, b), count) in enumerate(zip(colors, counts))
    )
    st.html(f"<div class='isl-palette' aria-label='Màu tâm cụm và tỷ lệ pixel'>{swatches}</div>")

    details = st.expander("Bản đồ nhãn cụm", key="km_label_preview", icon=":material/layers:", on_change="rerun")
    if details.open:
        with details:
            try:
                label_color = colorize_labels(result.labels)
            except (ValueError, MemoryError) as exc:
                st.error("Không tạo ảnh kết quả được. " + str(exc))
                return
            ui.show_images([(label_color, "Nhãn cụm · màu giả"), *processed])
            legend = "".join(
                f"<span class='isl-chip'><span class='isl-swatch' aria-hidden='true' style='background:{c}'></span>Cụm {i}</span>"
                for i, c in enumerate(cluster_palette_hex(result.core.k_effective))
            )
            st.html(f"<div class='isl-palette' aria-label='Chú giải bản đồ nhãn'>{legend}</div>")


def _show_foreground(ctx, out, fg, post) -> None:
    if not fg:
        st.info("Chưa chọn cụm nào: mask rỗng (toàn nền).")
    mask_caption = "Mask sau hình thái học" if post.operation != "none" else "Mask đối tượng"
    st.caption("Mask: trắng là đối tượng, đen là nền. Lớp phủ cam đánh dấu đối tượng trên ảnh gốc.")
    ui.show_images([(out.mask, mask_caption), (out.cutout_preview, "Ảnh tách nền (nền trắng)"),
                    (out.overlay, "Đối tượng tô màu trên ảnh gốc")], columns=3)
    if post.operation != "none":
        ui.show_images([(out.raw_mask, "Mask trước hình thái học")])
    ratio = out.mask.mean() / 255 * 100
    st.caption(f"Đối tượng chiếm {ratio:.2f}% pixel · nền {100 - ratio:.2f}%.")
    if ctx.ground_truth is not None:
        s = compare_masks(out.mask, ctx.ground_truth)
        st.success(f"So với ground truth của ảnh mẫu: IoU = {s.iou:.4f} · Dice = {s.dice:.4f} · "
                   f"độ chính xác pixel = {s.pixel_accuracy:.4f}")


def _show_statistics(result) -> None:
    core = result.core
    st.markdown("**Bảng thống kê cụm**")
    st.dataframe(pd.DataFrame(cluster_table(result)), hide_index=True, width="stretch")
    ui.show_figure(viz.cluster_share_figure(core.counts, core.centers))
    ui.show_figure(viz.sse_figure(core.sse_history))
    st.caption("SSE = Σ‖x − tâm cụm của x‖², tính sau cập nhật tâm và gán lại nhãn của mỗi vòng Lloyd. "
               "Điểm cuối trên biểu đồ bằng SSE của kết quả trả về. "
               "Theo lý thuyết, SSE không tăng qua các vòng lặp.")
    hist = pd.DataFrame({"Vòng lặp": range(1, core.n_iter + 1), "SSE": core.sse_history,
                         "Độ dịch tâm lớn nhất": core.shift_history})
    st.dataframe(hist, hide_index=True, width="stretch")
