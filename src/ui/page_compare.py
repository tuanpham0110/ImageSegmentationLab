"""Trang So sánh thuật toán: cùng một ảnh, K-means và Otsu đặt cạnh nhau."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.processing.pipeline import (
    TIMING_SCOPE,
    comparison_export_files,
    default_foreground,
    fit_ground_truth,
    kmeans_outputs,
    run_kmeans,
    run_otsu,
)
from src.ui import components as ui
from src.ui.page_kmeans import params_form
from src.ui.page_otsu import POLARITY_HELP, POLARITY_OPTIONS, polarity_caption, show_threshold_warnings
from src.ui.state import (
    clear_uploaded_gt,
    get_result,
    get_uploaded_gt,
    has_stale_result,
    persistent_foreground,
    remember_widget,
    restore_widget,
    save_foreground,
    store_result,
    store_uploaded_gt,
    store_uploaded_gt_error,
)
from src.utils.image_io import InputError, load_mask_bytes
from src.utils.metrics import compare_masks

RESULT_KEY = "cmp_result"


def _on_gt_upload_change(uploader_key: str) -> None:
    """Chỉ khi người dùng chủ động bỏ tệp (bấm ×) mới xoá ground truth đã lưu.

    Khi chuyển trang, Streamlit dựng lại widget rỗng nhưng không gọi on_change,
    nên ground truth đã tải vẫn được giữ.
    """
    if st.session_state.get(uploader_key) is None:
        clear_uploaded_gt()


def _ground_truth_uploader(ctx) -> tuple[np.ndarray | None, str | None, bool]:
    """Widget tải ground truth + lưu mask bền theo chữ ký đầu vào.

    Trả về (mask, tên tệp, có_lỗi). Mask (hoặc lỗi) chỉ được giải mã lại khi tệp đổi.
    """
    key = f"cmp_gt_upload_{ctx.signature}"
    gt_file = st.file_uploader("Ground-truth mask (tuỳ chọn, PNG/BMP/CSV)", type=["png", "bmp", "csv"],
                               key=key, on_change=_on_gt_upload_change, args=(key,),
                               help="Nhận mask 0/1 (1 = đối tượng) hoặc 0/255 (trắng = đối tượng). "
                                    "CSV là ma trận không có tiêu đề, không phải ma trận nhãn nhiều cụm. "
                                    "Mask gắn với ảnh đầu vào hiện tại. Khi đổi ảnh hoặc tiền xử lý, hãy tải lại mask. "
                                    "Mask phải đục (không có vùng trong suốt).")
    stored = get_uploaded_gt(ctx.signature)
    if gt_file is not None and (stored is None or stored["file_id"] != gt_file.file_id):
        clear_uploaded_gt()
        try:
            mask = fit_ground_truth(load_mask_bytes(gt_file.getvalue(), gt_file.name), ctx.prepared)
        except (InputError, ValueError, MemoryError) as exc:
            # Lưu cả trạng thái lỗi: quay lại trang không được lặng lẽ dùng GT mẫu.
            reason = str(exc) or ("không đủ bộ nhớ để xử lý mask" if isinstance(exc, MemoryError)
                                  else type(exc).__name__)
            store_uploaded_gt_error(ctx.signature, gt_file.name, gt_file.file_id, reason)
        else:
            store_uploaded_gt(ctx.signature, gt_file.name, gt_file.file_id, mask)
        stored = get_uploaded_gt(ctx.signature)
    if stored is None:
        return None, None, False
    failed = stored.get("error") is not None
    if failed:
        st.error(f"Không dùng được ground-truth mask: {stored['error']}")
    if gt_file is None:
        # Widget rỗng do vừa chuyển trang: vẫn giữ tệp đã tải (kể cả tệp lỗi), cho phép bỏ chủ động.
        if failed:
            st.caption(f"Tệp ground truth đã tải lên bị lỗi: **{stored['name']}**.")
        else:
            st.caption(f"Đang dùng ground truth đã tải lên: **{stored['name']}**.")
        st.button("Bỏ ground truth tải lên", on_click=clear_uploaded_gt, key="cmp_gt_clear", width="stretch")
    if failed:
        return None, stored["name"], True
    return stored["mask"], stored["name"], False


def _agreement_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Bản đồ khác biệt: trắng = cả hai là đối tượng, cam = chỉ K-means, xanh = chỉ Otsu, đen = cả hai là nền."""
    a, b = a > 0, b > 0
    out = np.zeros(a.shape + (3,), np.uint8)
    out[a & b] = (255, 255, 255)
    out[a & ~b] = (235, 104, 52)
    out[~a & b] = (42, 120, 214)
    return out


QUALITATIVE = [
    ("Bản chất", "Phân cụm không giám sát: tối thiểu tổng bình phương khoảng cách tới tâm cụm (SSE).",
     "Phân ngưỡng toàn cục: chọn t làm cực đại phương sai giữa hai lớp trên histogram."),
    ("Dữ liệu đầu vào", "Vector đặc trưng mỗi pixel: mức xám (1D), RGB hoặc CIE L*a*b* (3D).",
     "Chỉ ảnh xám (histogram 256 mức)."),
    ("Số vùng đầu ra", "K cụm (người dùng chọn K); cần chọn cụm nào là đối tượng.",
     "Hai lớp sáng/tối (trang Otsu có thêm Multi-Otsu 3 lớp); cần chọn lớp đối tượng."),
    ("Tham số", "K, khởi tạo, seed, số vòng lặp, ngưỡng hội tụ.", "Không có tham số (chỉ chọn lớp sáng/tối là đối tượng)."),
    ("Phụ thuộc tham số", "Cao: K và khởi tạo ảnh hưởng kết quả; có thể rơi vào cực tiểu địa phương.",
     "Quy tắc chọn ngưỡng cho kết quả tái lập với cùng histogram."),
    ("Ưu điểm", "Dùng được màu sắc; tách được nhiều vùng; tổng quát.", "Rất nhanh O(N + 256); tối ưu toàn cục; không cần tham số."),
    ("Hạn chế", "Phải chọn K; vòng lặp có chi phí O(U·K·D·số vòng) với U là số giá trị pixel phân biệt; "
     "bản thân thuật toán không dùng thông tin không gian "
                "(bù bằng lọc trung vị trước và hình thái học sau).",
     "Kém khi histogram không có đỉnh rõ, đối tượng nhỏ, chiếu sáng không đều; không dùng màu; "
     "không dùng thông tin không gian."),
    ("Phù hợp khi", "Ảnh màu, nhiều vùng màu khác nhau, cần lượng tử hoá màu.",
     "Ảnh xám hai đỉnh rõ: văn bản quét, đối tượng tương phản với nền."),
]


def render() -> None:
    ui.page_header("So sánh thuật toán", "Đặt ảnh gốc cạnh kết quả K-means và Otsu trên cùng một đầu vào.")
    ctx = ui.require_input()
    if ctx is None:
        return
    ui.input_summary(ctx)
    prep = ctx.prepared

    left, right = ui.workspace("compare")
    with left:
        params = params_form(prep.is_color, key_prefix="cmp", submit_label="Chạy K-means & Otsu")
        with st.container(border=True, key="cmp-otsu-controls"):
            restore_widget("cmp_otsu_polarity", "Gợi ý theo viền ảnh")
            st.radio("Otsu: đối tượng là", list(POLARITY_OPTIONS), key="cmp_otsu_polarity", help=POLARITY_HELP)
            remember_widget("cmp_otsu_polarity")
        with st.expander("Ảnh tham chiếu", icon=":material/fact_check:"):
            uploaded_gt = _ground_truth_uploader(ctx)
    if params is not None:
        try:
            with right, st.spinner("Đang chạy K-means…", show_time=True):
                km = run_kmeans(prep, params)
        except (ValueError, MemoryError) as exc:
            st.session_state.pop(RESULT_KEY, None)
            right.error("Không chạy được K-means. Hãy kiểm tra tham số hoặc giảm giới hạn cạnh dài. " + str(exc))
            return
        else:
            store_result(RESULT_KEY, ctx.signature, {"params": params, "km": km,
                                                     "run_id": st.session_state.get("cmp_run_id", 0) + 1})
            st.session_state["cmp_run_id"] = st.session_state.get("cmp_run_id", 0) + 1

    run = get_result(RESULT_KEY, ctx.signature)
    with right:
        if run is None:
            ui.result_empty(prep, "Chạy K-means & Otsu", stale=has_stale_result(RESULT_KEY, ctx.signature))
            return
        _render(ctx, run, uploaded_gt)


def _render(ctx, run, uploaded_gt) -> None:
    prep, km = ctx.prepared, run["km"]
    # Payload đã gắn với chữ ký ảnh. Chỉ tính lại Otsu khi đổi cực tính hoặc chạy mới.
    polarity = POLARITY_OPTIONS[st.session_state.get("cmp_otsu_polarity", "Gợi ý theo viền ảnh")]
    try:
        otsu = run.get("otsu")
        if otsu is None or otsu.polarity != polarity:
            run.pop("otsu", None)
            run.pop("_export_cache", None)
            with st.spinner("Đang phân ngưỡng Otsu…", show_time=True):
                otsu = run_otsu(prep, polarity=polarity)
            run["otsu"] = otsu
    except (ValueError, MemoryError) as exc:
        st.error("Không phân ngưỡng được ảnh. Hãy giảm giới hạn cạnh dài hoặc kiểm tra dữ liệu. " + str(exc))
        return
    for warning in km.core.warnings:
        st.warning("K-means: " + warning)
    show_threshold_warnings(otsu.warnings, prefix="Otsu: ")
    metrics = ui.metric_columns(4, "compare")
    metrics[0].metric("K-means · số cụm", km.core.k_effective)
    metrics[1].metric("Otsu · ngưỡng", otsu.result.threshold if otsu.result.valid else "Không hợp lệ")
    metrics[2].metric("K-means · thời gian", f"{km.core.elapsed_seconds * 1000:.1f} ms", help=TIMING_SCOPE)
    metrics[3].metric("Otsu · thời gian", f"{otsu.result.elapsed_seconds * 1000:.2f} ms", help=TIMING_SCOPE)
    convergence = "hội tụ" if km.core.converged else "chưa hội tụ"
    st.caption(f"K-means · {km.feature.upper()} · {km.core.n_iter} vòng · {convergence} · seed {run['params'].seed}. "
               + polarity_caption(otsu))
    _result_tabs(ctx, run, otsu, uploaded_gt)


def _result_tabs(ctx, run, otsu, uploaded_gt) -> None:
    prep, km = ctx.prepared, run["km"]
    images, masks, scores, downloads = st.tabs(
        ["Kết quả phân đoạn", "Đối chiếu mask", "Đánh giá", "Tải xuống"],
        key="cmp_result_tab", on_change="rerun",
    )
    with masks:
        options = list(range(km.core.k_effective))
        fg_key = f"cmp_fg_{run['run_id']}"
        if fg_key not in st.session_state:
            initial = run["foreground"] if "foreground" in run else default_foreground(km)
            persistent_foreground(run, fg_key, options, initial)
        fg = st.multiselect("Cụm K-means làm đối tượng", options,
                            format_func=lambda c: f"Cụm {c}", key=fg_key,
                            help="Mặc định là gợi ý theo viền ảnh. Đổi cụm tại đây để cập nhật mask và tệp tải xuống.")
        save_foreground(run, fg)
    if images.open:
        with images:
            _segmented_images(prep, km, otsu)
        return
    try:
        out = kmeans_outputs(prep, km, fg)
    except (ValueError, MemoryError) as exc:
        run.pop("_export_cache", None)
        st.error("Không tạo ảnh kết quả được. Hãy kiểm tra lựa chọn cụm hoặc giảm giới hạn cạnh dài. " + str(exc))
        return
    if masks.open:
        with masks:
            _mask_comparison(out, otsu)
        return
    if scores.open:
        with scores:
            _scores(ctx, run, out, otsu, uploaded_gt)
            _current_image_analysis(prep, km, out, otsu)
            with st.expander("Đặc điểm và tham số hai phương pháp", icon=":material/tune:"):
                _comparison_table(run)
        return
    with downloads:
        st.caption("Tệp ZIP chứa toàn bộ ảnh PNG, ma trận CSV và tham số JSON của hai kết quả đang hiển thị.")
        files = ui.cached_export_files(
            run, (tuple(out.foreground_clusters), otsu.polarity),
            lambda: comparison_export_files(prep, run["params"], km, out, otsu),
        )
        if files is not None:
            ui.download_buttons(files, {
                "comparison_results.zip": "⬇ Toàn bộ kết quả hai thuật toán (ZIP)",
                "kmeans_mask.png": "⬇ Mask K-means (PNG)",
                "otsu_mask.png": "⬇ Mask Otsu (PNG)",
                "kmeans_cutout.png": "⬇ Ảnh tách nền K-means (PNG)",
                "otsu_cutout.png": "⬇ Ảnh tách nền Otsu (PNG)",
            }, "cmp_dl")


def _segmented_images(prep, km, otsu) -> None:
    original, *processed = ui.source_panels(prep)
    threshold = f"t* = {otsu.result.threshold}" if otsu.result.valid else "không có ngưỡng hợp lệ"
    ui.show_images([original, (km.segmented, f"K-means · {km.core.k_effective} cụm"),
                    (otsu.binary, f"Otsu · nhị phân ({threshold})")], columns=3)
    st.caption("Mở **Đối chiếu mask** để so sánh hai vùng đối tượng trên cùng quy ước trắng là đối tượng.")
    if processed:
        details = st.expander("Ảnh đầu vào sau lọc", key="cmp_input_details", icon=":material/tune:", on_change="rerun")
        if details.open:
            with details:
                ui.show_images(processed)


def _mask_comparison(out, otsu) -> None:
    ui.show_images([(out.mask, "Mask K-means"), (otsu.mask, "Mask Otsu"),
                    (_agreement_map(out.mask, otsu.mask), "Khác biệt hai mask")], columns=3)
    legend = [("#ffffff", "cả hai là đối tượng"), ("#eb6834", "chỉ K-means"), ("#2a78d6", "chỉ Otsu"), ("#000000", "cả hai là nền")]
    swatches = "".join(f"<span class='isl-chip'><span class='isl-swatch' aria-hidden='true' style='background:{c}'></span>"
                       f"{t}</span>" for c, t in legend)
    st.html(f"<div class='isl-palette' aria-label='Chú giải khác biệt hai mask'>{swatches}</div>")
    agree = compare_masks(out.mask, otsu.mask)
    st.caption(f"Mức trùng khớp giữa hai mask (không phải độ chính xác): IoU = {agree.iou:.4f}, "
               f"tỷ lệ pixel cùng nhãn = {agree.pixel_accuracy:.4f}.")


def _scores(ctx, run, out, otsu, uploaded_gt) -> None:
    km = run["km"]
    st.caption("Phạm vi đo thời gian: " + TIMING_SCOPE)
    uploaded_mask, uploaded_name, upload_failed = uploaded_gt
    if upload_failed:
        # Tệp tải lên lỗi: không lặng lẽ thay bằng ground truth của ảnh mẫu.
        gt, gt_source = None, None
    elif uploaded_mask is not None:
        gt, gt_source = uploaded_mask, f"ground truth tải lên ({uploaded_name})"
    else:
        gt, gt_source = ctx.ground_truth, "ground truth của ảnh mẫu"

    st.markdown("##### Đánh giá định lượng")
    rows = []
    if gt is not None:
        for name, mask, t in (("K-means", out.mask, km.core.elapsed_seconds), ("Otsu", otsu.mask, otsu.result.elapsed_seconds)):
            s = compare_masks(mask, gt)
            rows.append({"Phương pháp": name, "IoU": round(s.iou, 4), "Dice": round(s.dice, 4),
                         "Độ chính xác pixel": round(s.pixel_accuracy, 4), "FP (pixel)": s.fp, "FN (pixel)": s.fn,
                         "Thời gian (ms)": round(t * 1000, 2)})
        st.caption(f"Tính trên {gt_source}.")
    else:
        for name, t in (("K-means", km.core.elapsed_seconds), ("Otsu", otsu.result.elapsed_seconds)):
            rows.append({"Phương pháp": name, "Thời gian (ms)": round(t * 1000, 2)})
        st.info("Không có ground truth nên **không tính** IoU/Dice/độ chính xác. Tải mask trong mục **Ảnh tham chiếu** để tính điểm.")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _comparison_table(run) -> None:
    """Đối chiếu tham số và bản chất hai phương pháp trong tab Đánh giá."""
    km, otsu = run["km"], run.get("otsu")
    if otsu is None:
        return
    threshold = f"t* = {otsu.result.threshold}. " if otsu.result.valid else ""
    qual = [("Tham số lần chạy này",
             f"K = {km.core.k_effective}, đặc trưng {km.feature}, {run['params'].init}, seed {run['params'].seed}, "
             f"{km.core.n_iter} vòng ({'hội tụ' if km.core.converged else 'chưa hội tụ'})",
             threshold + polarity_caption(otsu)),
            ("Thời gian đo được", f"{km.core.elapsed_seconds * 1000:.1f} ms", f"{otsu.result.elapsed_seconds * 1000:.2f} ms")]
    st.table(pd.DataFrame(qual + QUALITATIVE, columns=["Tiêu chí", "K-means", "Otsu"]).set_index("Tiêu chí"))


def _current_image_analysis(prep, km, out, otsu) -> None:
    """Diễn giải thống kê đo được, không suy diễn độ chính xác khi thiếu ground truth."""
    st.markdown("##### Phân tích ảnh hiện tại")
    r = otsu.result
    if prep.is_color and km.feature in ("rgb", "lab"):
        st.write(f"K-means đang dùng đặc trưng {km.feature.upper()} ({km.core.k_effective} cụm), "
                 "còn Otsu chỉ dùng mức xám. Các vùng khác màu nhưng gần độ sáng có thể bị Otsu gộp lại.")
    elif km.core.k_effective > 2:
        st.write(f"K-means chia {km.core.k_effective} cụm mức xám; Otsu chỉ chia hai lớp. "
                 "Khi đối tượng có nhiều mức sáng, cần kiểm tra xem lớp Otsu có bỏ sót vùng nào của đối tượng.")
    if r.valid:
        st.write(f"Hai lớp Otsu có độ sáng trung bình {r.mean0:.2f} và {r.mean1:.2f} "
                 f"(chênh {r.mean1 - r.mean0:.2f}/255), chiếm {r.weight0:.1%} và {r.weight1:.1%} pixel. "
                 f"Độ tách histogram η = {r.separability:.3f}; đây không phải xác suất phân đoạn đúng.")
        if len(r.tied_thresholds) > 1:
            st.caption(f"Có {len(r.tied_thresholds)} ngưỡng đồng hạng. Otsu chọn t = {r.threshold} "
                       "trong nhóm ngưỡng cực đại theo quy tắc gần phần nguyên của trung bình nhất.")
        # Phân vị rời rạc từ histogram: không cần sắp xếp lại toàn bộ ảnh.
        lo, hi = np.searchsorted(np.cumsum(r.histogram), np.array([0.05, 0.95]) * r.histogram.sum())
        if hi - lo <= 32 and r.mean1 - r.mean0 <= 32:
            st.info(f"Dải mức xám chứa ít nhất 90% pixel là [{lo}, {hi}], chỉ rộng {hi - lo}/255. "
                    "Với dải sáng hẹp, nhiễu dễ làm thay đổi nhãn. Có thể thử lọc trung vị rồi đối chiếu lại mask.")
    else:
        st.write("Ảnh không có hai lớp mức xám khác rỗng; Otsu tự động không xác định được đối tượng.")
    disagree = int(np.count_nonzero(out.mask != otsu.mask))
    st.caption(f"Hai mask bất đồng tại {disagree:,}/{out.mask.size:,} pixel ({disagree / out.mask.size:.2%}). "
               "Mức bất đồng chỉ cho biết hai phương pháp khác nhau; cần ground truth độc lập để kết luận phương pháp nào đúng.")
