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
from src.utils.image_io import read_input_bytes

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

    with st.sidebar, st.container(key="sidebar-input", gap="small"):
        st.html("<h2 class='isl-sidebar-label'>Ảnh đầu vào</h2>")
        if manifest_error:
            st.error(manifest_error)
        source = st.radio("Nguồn dữ liệu", SOURCES, key="input_source", horizontal=True, label_visibility="collapsed")

        data = name = sample = gt_path = None
        if source == "Ảnh mẫu":
            if not samples:
                message = "Không có ảnh mẫu. Chọn Tải tệp lên để dùng ảnh hoặc CSV trên máy."
                set_context(None, message)
                if not manifest_error:
                    st.warning(message)
                return
            titles = {s["id"]: s["title"] for s in samples}
            sid = st.selectbox("Chọn ảnh mẫu", sample_ids, key="sample_id", format_func=lambda i: titles[i])
            sample = next(s for s in samples if s["id"] == sid)
            path = DATA_DIR / sample["file"]
            try:
                data, name = read_input_bytes(path), path.name
            except (InputError, MemoryError) as exc:
                reason = str(exc) or "Không đủ bộ nhớ để đọc ảnh."
                message = f"Không đọc được dữ liệu ảnh mẫu. {reason} Hãy kiểm tra thư mục data/ hoặc tải tệp lên."
                set_context(None, message)
                st.error(message)
                return
            if sample["ground_truth"]:
                gt_path = DATA_DIR / sample["ground_truth"]
        else:
            uploaded = st.file_uploader(
                "Ảnh PNG/JPG/BMP hoặc ma trận CSV",
                type=["png", "jpg", "jpeg", "bmp", "csv"],
                key="uploader",
                help="Kéo thả hoặc chọn một tệp. Tối đa 30 MiB mỗi tệp và 40 triệu pixel với ảnh.",
            )
            injected = st.session_state.get("_test_upload")  # chỉ dùng trong kiểm thử tự động (AppTest)
            if uploaded is not None:
                data, name = uploaded.getvalue(), uploaded.name
            elif injected is not None:
                name, data = injected

        status = st.empty()
        with st.expander("Tiền xử lý ảnh", icon=":material/tune:"):
            st.select_slider("Cạnh dài tối đa (pixel)", MAX_SIDE_OPTIONS, key="max_side",
                             help="Ảnh lớn hơn được thu nhỏ (giữ tỉ lệ) trước khi xử lý để tránh treo máy.")
            st.selectbox("Chuyển ảnh màu sang xám", list(GRAY_METHODS), key="gray_method",
                         format_func=lambda m: GRAY_METHODS[m])
            st.selectbox("Lọc trung vị khử nhiễu", list(MEDIAN_OPTIONS), key="median_size",
                         format_func=lambda k: MEDIAN_OPTIONS[k],
                         help="Thay mỗi pixel bằng trung vị trong vùng k×k để giảm nhiễu trước khi phân đoạn.")

        if data is None:
            set_context(None, None)
            status.caption("Kéo thả ảnh hoặc CSV vào ô tải tệp để bắt đầu.")
            return
        try:
            with status, st.spinner("Đang chuẩn bị ảnh…", show_time=True):
                ctx = build_context(data, name, int(st.session_state["max_side"]), st.session_state["gray_method"],
                                    sample, gt_path=gt_path, median_size=int(st.session_state["median_size"]))
        except MemoryError:
            message = "Không đủ bộ nhớ để đọc ảnh. Hãy thu nhỏ ảnh trước khi tải lên hoặc giải phóng RAM rồi thử lại."
            set_context(None, message)
            status.error(message)
            return
        except ValueError as exc:
            set_context(None, str(exc))
            status.error(str(exc))
            return
        set_context(ctx)
        resize = ctx.prepared.resize
        width, height = resize.original_size
        processed_width, processed_height = resize.processed_size
        details = f"<p>Đã thu nhỏ để xử lý · {processed_width} × {processed_height} px</p>" if resize.resized else ""
        status.html(
            "<div class='isl-input-status' role='status'><span class='isl-status-dot' aria-hidden='true'></span>"
            f"<div><strong title='{escape(name, quote=True)}'>{escape(name)}</strong>"
            f"<p>Sẵn sàng · {escape(ctx.loaded.file_format)} · {width} × {height} px</p>{details}</div></div>"
        )
