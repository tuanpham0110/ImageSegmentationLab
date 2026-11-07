"""Thanh bên: chọn nguồn ảnh (mẫu / tải lên), giới hạn kích thước xử lý, cách chuyển xám, lọc trung vị."""

from __future__ import annotations

from html import escape

import streamlit as st

from src.processing.preprocessing import GRAY_METHODS
from src.ui.state import (
    DATA_DIR,
    DEFAULT_MAX_SIDE,
    MAX_SIDE_OPTIONS,
    MEDIAN_OPTIONS,
    InputError,
    build_context,
    load_manifest,
    set_context,
)

SOURCES = ["Ảnh mẫu", "Tải tệp lên"]


def render_input_sidebar() -> None:
    """Vẽ phần chọn đầu vào ở thanh bên và lưu InputContext vào session_state."""
    manifest_error = None
    try:
        samples = load_manifest()
    except InputError as exc:
        samples = []
        manifest_error = str(exc)
    sample_ids = [s["id"] for s in samples]
    st.session_state.setdefault("input_source", SOURCES[0])
    st.session_state.setdefault("max_side", DEFAULT_MAX_SIDE)
    st.session_state.setdefault("gray_method", "luminance")
    st.session_state.setdefault("median_size", 1)
    if sample_ids:
        default_sample = "multi_shapes_color" if "multi_shapes_color" in sample_ids else sample_ids[0]
        if st.session_state.get("sample_id") not in sample_ids:
            st.session_state["sample_id"] = default_sample

    with st.sidebar:
        if manifest_error:
            st.error(manifest_error)
        st.html("<div class='isl-sidebar-label'>Ảnh đầu vào</div>")
        source = st.radio("Nguồn dữ liệu", SOURCES, key="input_source", horizontal=True, label_visibility="collapsed")

        data = name = sample = gt_bytes = None
        if source == "Ảnh mẫu":
            if not samples:
                set_context(None, "Không có ảnh mẫu. Chạy `python scripts/generate_samples.py` để tạo.")
                return
            titles = {s["id"]: s["title"] for s in samples}
            sid = st.selectbox("Chọn ảnh mẫu", sample_ids, key="sample_id", format_func=lambda i: titles[i])
            sample = next(s for s in samples if s["id"] == sid)
            path = DATA_DIR / sample["file"]
            try:
                data, name = path.read_bytes(), path.name
                if sample["ground_truth"]:
                    gt_bytes = (DATA_DIR / sample["ground_truth"]).read_bytes()
            except OSError as exc:
                message = f"Không đọc được dữ liệu ảnh mẫu: {exc}. Hãy kiểm tra thư mục data/ hoặc tải tệp lên."
                set_context(None, message)
                st.error(message)
                return
        else:
            uploaded = st.file_uploader("Ảnh PNG/JPG/BMP hoặc ma trận CSV", type=["png", "jpg", "jpeg", "bmp", "csv"],
                                        key="uploader")
            injected = st.session_state.get("_test_upload")  # chỉ dùng trong kiểm thử tự động (AppTest)
            if uploaded is not None:
                data, name = uploaded.getvalue(), uploaded.name
            elif injected is not None:
                name, data = injected

        st.html("<div class='isl-sidebar-label'>Chuẩn bị dữ liệu</div>")
        st.select_slider("Cạnh dài tối đa (pixel)", MAX_SIDE_OPTIONS, key="max_side",
                         help="Ảnh lớn hơn được thu nhỏ (giữ tỉ lệ) trước khi xử lý để tránh treo máy.")
        st.selectbox("Chuyển ảnh màu sang xám", list(GRAY_METHODS), key="gray_method",
                     format_func=lambda m: GRAY_METHODS[m])
        st.selectbox("Lọc trung vị khử nhiễu", list(MEDIAN_OPTIONS), key="median_size",
                     format_func=lambda k: MEDIAN_OPTIONS[k],
                     help="Lọc phi tuyến (giáo trình Chương 4): thay mỗi pixel bằng trung vị cửa sổ k×k. "
                          "Khử nhiễu hạt trước Otsu/K-means — hai thuật toán theo từng điểm, không dùng pixel lân cận.")

        if data is None:
            set_context(None, None)
            st.info("Chưa có tệp. Kéo thả ảnh hoặc CSV vào ô trên.")
            return
        try:
            ctx = build_context(data, name, int(st.session_state["max_side"]), st.session_state["gray_method"],
                                sample, gt_bytes, median_size=int(st.session_state["median_size"]))
        except MemoryError:
            message = "Không đủ bộ nhớ để đọc ảnh. Hãy thu nhỏ ảnh trước khi tải lên hoặc giải phóng RAM rồi thử lại."
            set_context(None, message)
            st.error(message)
            return
        except (InputError, ValueError) as exc:
            set_context(None, str(exc))
            st.error(str(exc))
            return
        set_context(ctx)
        w, h = ctx.prepared.resize.processed_size
        st.html("<div class='isl-input-status'><span class='isl-status-dot'></span>"
                f"<div><strong>{escape(name)}</strong><p>Sẵn sàng xử lý · {w} × {h} pixel</p></div></div>")
