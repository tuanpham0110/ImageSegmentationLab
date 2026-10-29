"""Thanh bên: chọn nguồn ảnh (mẫu / tải lên), giới hạn kích thước xử lý, cách chuyển xám."""

from __future__ import annotations

import streamlit as st

from src.processing.preprocessing import GRAY_METHODS
from src.ui.state import (
    DATA_DIR,
    DEFAULT_MAX_SIDE,
    MAX_SIDE_OPTIONS,
    InputError,
    build_context,
    load_manifest,
    set_context,
)

SOURCES = ["Ảnh mẫu", "Tải tệp lên"]


def render_input_sidebar() -> None:
    """Vẽ phần chọn đầu vào ở thanh bên và lưu InputContext vào session_state."""
    samples = load_manifest()
    sample_ids = [s["id"] for s in samples]
    st.session_state.setdefault("input_source", SOURCES[0])
    st.session_state.setdefault("max_side", DEFAULT_MAX_SIDE)
    st.session_state.setdefault("gray_method", "luminance")
    if sample_ids:
        default_sample = "multi_shapes_color" if "multi_shapes_color" in sample_ids else sample_ids[0]
        if st.session_state.get("sample_id") not in sample_ids:
            st.session_state["sample_id"] = default_sample

    with st.sidebar:
        st.markdown("### 🧩 Image Segmentation Lab")
        st.caption("K-means & Otsu · Đề tài 3 · Xử lý ảnh PTIT")
        st.markdown("**Ảnh đầu vào**")
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
            data, name = path.read_bytes(), path.name
            if sample["ground_truth"]:
                gt_bytes = (DATA_DIR / sample["ground_truth"]).read_bytes()
        else:
            uploaded = st.file_uploader("Ảnh PNG/JPG/BMP hoặc ma trận CSV", type=["png", "jpg", "jpeg", "bmp", "csv"],
                                        key="uploader")
            injected = st.session_state.get("_test_upload")  # chỉ dùng trong kiểm thử tự động (AppTest)
            if uploaded is not None:
                data, name = uploaded.getvalue(), uploaded.name
            elif injected is not None:
                name, data = injected

        st.select_slider("Giới hạn cạnh dài khi xử lý (pixel)", MAX_SIDE_OPTIONS, key="max_side",
                         help="Ảnh lớn hơn được thu nhỏ (giữ tỉ lệ) trước khi xử lý để tránh treo máy.")
        st.selectbox("Chuyển ảnh màu sang xám", list(GRAY_METHODS), key="gray_method",
                     format_func=lambda m: GRAY_METHODS[m])

        if data is None:
            set_context(None, None)
            st.info("Chưa có tệp. Kéo thả ảnh hoặc CSV vào ô trên.")
            return
        try:
            ctx = build_context(data, name, int(st.session_state["max_side"]), st.session_state["gray_method"],
                                sample, gt_bytes)
        except (InputError, ValueError) as exc:
            set_context(None, str(exc))
            st.error(str(exc))
            return
        set_context(ctx)
        w, h = ctx.prepared.resize.processed_size
        st.success(f"Đang dùng: **{name}** · xử lý ở {w}×{h}")
