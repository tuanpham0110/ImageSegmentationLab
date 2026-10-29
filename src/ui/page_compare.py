"""Trang So sánh thuật toán: cùng một ảnh, K-means và Otsu đặt cạnh nhau."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.processing.pipeline import default_foreground, fit_ground_truth, kmeans_outputs, run_kmeans, run_otsu
from src.ui import components as ui
from src.ui.page_kmeans import params_form
from src.ui.page_otsu import POLARITY_HELP, POLARITY_OPTIONS, polarity_caption
from src.ui.state import get_result, has_stale_result, store_result
from src.utils.image_io import InputError, load_mask_bytes
from src.utils.metrics import compare_masks

RESULT_KEY = "cmp_result"


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
    ("Số vùng đầu ra", "K cụm (người dùng chọn K); cần chọn cụm nào là đối tượng.", "Luôn 2 lớp: nền / đối tượng."),
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
        st.session_state.setdefault("cmp_otsu_polarity", "Gợi ý theo viền ảnh")
        st.radio("Otsu: đối tượng là", list(POLARITY_OPTIONS), key="cmp_otsu_polarity", help=POLARITY_HELP)
        gt_file = st.file_uploader("Ground-truth mask (tuỳ chọn, PNG trắng = đối tượng)", type=["png", "bmp"],
                                   key="cmp_gt_upload")
    if params is not None:
        with st.spinner("Đang chạy K-means và Otsu…"):
            km = run_kmeans(prep, params)
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
        _render(ctx, run, gt_file)


def _render(ctx, run, gt_file) -> None:
    prep, km = ctx.prepared, run["km"]
    # Otsu rất nhanh (< 10 ms) nên được tính lại theo lựa chọn cực tính hiện tại
    otsu = run_otsu(prep, polarity=POLARITY_OPTIONS[st.session_state.get("cmp_otsu_polarity", "Gợi ý theo viền ảnh")])

    fg = st.multiselect("Cụm K-means làm đối tượng", list(range(km.core.k_effective)), default=default_foreground(km),
                        format_func=lambda c: f"Cụm {c}", key=f"cmp_fg_{run['run_id']}",
                        help="Mặc định là gợi ý theo viền ảnh; hãy chỉnh cho đúng đối tượng.")
    out = kmeans_outputs(prep, km, fg)

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

    gt, gt_source = ctx.ground_truth, "ground truth của ảnh mẫu"
    if gt_file is not None:
        try:
            gt = fit_ground_truth(load_mask_bytes(gt_file.getvalue(), gt_file.name), prep)
            gt_source = f"ground truth tải lên ({gt_file.name})"
        except (InputError, ValueError) as exc:
            st.error(f"Không dùng được ground-truth mask: {exc}")
            gt = None

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
