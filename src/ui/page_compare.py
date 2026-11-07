"""Trang So sánh thuật toán: cùng một ảnh, K-means và Otsu đặt cạnh nhau."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.processing.pipeline import TIMING_SCOPE, default_foreground, fit_ground_truth, kmeans_outputs, run_kmeans, run_otsu
from src.ui import components as ui
from src.ui.page_kmeans import params_form
from src.ui.page_otsu import POLARITY_HELP, POLARITY_OPTIONS, polarity_caption
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
    gt_file = st.file_uploader("Ground-truth mask (tuỳ chọn, PNG trắng = đối tượng)", type=["png", "bmp"],
                               key=key, on_change=_on_gt_upload_change, args=(key,),
                               help="Mask gắn với ảnh đầu vào hiện tại. Khi đổi ảnh hoặc tiền xử lý, hãy tải lại mask. "
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
    ("Dữ liệu đầu vào", "Vector đặc trưng mỗi pixel: mức xám (1D) hoặc RGB (3D).", "Chỉ ảnh xám (histogram 256 mức)."),
    ("Số vùng đầu ra", "K cụm (người dùng chọn K); cần chọn cụm nào là đối tượng.",
     "Hai lớp sáng/tối nếu có ngưỡng hợp lệ; cần chọn lớp đối tượng."),
    ("Tham số", "K, khởi tạo, seed, số vòng lặp, ngưỡng hội tụ.", "Không có tham số (chỉ chọn lớp sáng/tối là đối tượng)."),
    ("Phụ thuộc tham số", "Cao: K và khởi tạo ảnh hưởng kết quả; có thể rơi vào cực tiểu địa phương.",
     "Thấp: kết quả xác định duy nhất từ histogram."),
    ("Ưu điểm", "Dùng được màu sắc; tách được nhiều vùng; tổng quát.", "Rất nhanh O(N + 256); tối ưu toàn cục; không cần tham số."),
    ("Hạn chế", "Phải chọn K; chậm hơn O(N·K·D·số vòng); không dùng thông tin không gian.",
     "Chỉ 2 lớp; kém khi histogram không hai đỉnh, đối tượng nhỏ, chiếu sáng không đều; không dùng màu."),
    ("Phù hợp khi", "Ảnh màu, nhiều vùng màu khác nhau, cần lượng tử hoá màu.",
     "Ảnh xám hai đỉnh rõ: văn bản quét, đối tượng tương phản với nền."),
]


def render() -> None:
    ui.page_header("So sánh thuật toán", "Cùng một ảnh đầu vào được xử lý bằng K-means và Otsu.")
    ctx = ui.require_input()
    if ctx is None:
        return
    ui.input_summary(ctx)
    prep = ctx.prepared

    left, right = st.columns([1, 2.6], gap="large")
    with left:
        params = params_form(prep.is_color, key_prefix="cmp")
        restore_widget("cmp_otsu_polarity", "Gợi ý theo viền ảnh")
        st.radio("Otsu: đối tượng là", list(POLARITY_OPTIONS), key="cmp_otsu_polarity", help=POLARITY_HELP)
        remember_widget("cmp_otsu_polarity")
        uploaded_gt = _ground_truth_uploader(ctx)
    if params is not None:
        try:
            with st.spinner("Đang chạy K-means…"):
                km = run_kmeans(prep, params)
        except (ValueError, MemoryError) as exc:
            st.session_state.pop(RESULT_KEY, None)
            st.error("Không chạy được K-means. Hãy kiểm tra tham số hoặc giảm giới hạn cạnh dài. " + str(exc))
            return
        else:
            store_result(RESULT_KEY, ctx.signature, {"params": params, "km": km,
                                                     "run_id": st.session_state.get("cmp_run_id", 0) + 1})
            st.session_state["cmp_run_id"] = st.session_state.get("cmp_run_id", 0) + 1

    run = get_result(RESULT_KEY, ctx.signature)
    with right:
        if run is None:
            if has_stale_result(RESULT_KEY, ctx.signature):
                st.warning("Ảnh đầu vào đã thay đổi — hãy bấm **Chạy K-means** để so sánh lại.")
            st.info("Chọn tham số rồi bấm **Chạy K-means** — Otsu được chạy cùng lúc trên cùng ảnh.")
            ui.show_image(prep.image, "Ảnh gốc")
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
            otsu = run_otsu(prep, polarity=polarity)
            run["otsu"] = otsu
    except (ValueError, MemoryError) as exc:
        st.error("Không phân ngưỡng được ảnh. Hãy giảm giới hạn cạnh dài hoặc kiểm tra dữ liệu. " + str(exc))
        return
    for warning in km.core.warnings:
        st.warning("K-means: " + warning)
    for warning in otsu.warnings:
        st.warning("Otsu: " + warning)
    st.caption("Phạm vi đo thời gian: " + TIMING_SCOPE)

    options = list(range(km.core.k_effective))
    fg_key = f"cmp_fg_{run['run_id']}"
    persistent_foreground(run, fg_key, options, default_foreground(km))
    fg = st.multiselect("Cụm K-means làm đối tượng", options,
                        format_func=lambda c: f"Cụm {c}", key=fg_key,
                        help="Mặc định là gợi ý theo viền ảnh; hãy chỉnh cho đúng đối tượng.")
    save_foreground(run, fg)
    try:
        out = kmeans_outputs(prep, km, fg)
    except (ValueError, MemoryError) as exc:
        st.error("Không tạo ảnh kết quả được. Hãy kiểm tra lựa chọn cụm hoặc giảm giới hạn cạnh dài. " + str(exc))
        return

    st.markdown("##### Kết quả trực tiếp của từng thuật toán")
    ui.show_images([(prep.image, "Ảnh gốc"), (out.segmented, f"K-means: {km.core.k_effective} cụm"),
                    (otsu.binary, f"Otsu: nhị phân (t* = {otsu.result.threshold})")])
    st.caption("Lưu ý: K-means cho ảnh nhiều cụm, Otsu cho ảnh 2 lớp — hai ảnh trên không cùng loại nên chỉ so sánh trực quan. "
               "Phép so sánh định lượng dưới đây dùng **mask nhị phân** của cả hai.")

    st.markdown("##### So sánh mask đối tượng")
    ui.show_images([(out.mask, "Mask K-means"), (otsu.mask, "Mask Otsu"),
                    (_agreement_map(out.mask, otsu.mask), "Khác biệt hai mask")])
    legend = [("#ffffff", "cả hai là đối tượng"), ("#eb6834", "chỉ K-means"), ("#2a78d6", "chỉ Otsu"), ("#000000", "cả hai là nền")]
    st.markdown("".join(f"<span class='isl-chip'><span class='isl-swatch' style='background:{c};border:1px solid #8c959f'>"
                        f"</span>{t}</span>" for c, t in legend), unsafe_allow_html=True)
    agree = compare_masks(out.mask, otsu.mask)
    st.caption(f"Mức trùng khớp giữa hai mask (không phải độ chính xác): IoU = {agree.iou:.4f}, "
               f"tỷ lệ pixel cùng nhãn = {agree.pixel_accuracy:.4f}.")

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
        st.info("Không có ground truth nên **không tính** IoU/Dice/độ chính xác. Có thể tải lên ground-truth mask ở cột trái.")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.markdown("##### Bảng so sánh")
    qual = [("Tham số lần chạy này",
             f"K = {km.core.k_effective}, đặc trưng {km.feature}, {run['params'].init}, seed {run['params'].seed}, "
             f"{km.core.n_iter} vòng ({'hội tụ' if km.core.converged else 'chưa hội tụ'})",
             f"t* = {otsu.result.threshold}; {polarity_caption(otsu)}"),
            ("Thời gian đo được", f"{km.core.elapsed_seconds * 1000:.1f} ms", f"{otsu.result.elapsed_seconds * 1000:.2f} ms")]
    st.table(pd.DataFrame(qual + QUALITATIVE, columns=["Tiêu chí", "K-means", "Otsu"]).set_index("Tiêu chí"))
