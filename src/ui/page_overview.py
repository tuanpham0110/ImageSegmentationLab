"""Điểm bắt đầu cho các luồng phân đoạn và so sánh."""

from __future__ import annotations

import streamlit as st

from src.ui import components as ui
from src.ui.state import get_context

METHODS = (
    ("kmeans", "K-means", ":material/palette:",
     "Phân cụm theo màu hoặc mức xám. Đổi K, xem tâm cụm và chọn các cụm đối tượng."),
    ("otsu", "Otsu", ":material/contrast:",
     "Tách các lớp sáng và tối theo ngưỡng. Đối chiếu ảnh nhị phân với histogram."),
    ("compare", "So sánh hai thuật toán", ":material/compare:",
     "Đặt kết quả cạnh nhau trên cùng ảnh. Đối chiếu mask và chấm IoU/Dice khi có ảnh tham chiếu."),
)


def render() -> None:
    pages = st.session_state["_nav_pages"]
    ctx = get_context()
    with st.container(key="overview-hero"):
        intro, preview = st.columns([1.13, 1], gap="large", vertical_alignment="center")
        with intro:
            st.html(
                "<section class='isl-hero'>"
                "<p class='isl-hero-course'>PTIT · Xử lý ảnh INT13146</p>"
                "<h1>Image Segmentation<br>Studio</h1>"
                "<p class='isl-hero-description'>Phân đoạn ảnh bằng K-means và Otsu. "
                "Quan sát mặt nạ đối tượng, so sánh hai phương pháp và lưu kết quả.</p>"
                "</section>"
            )
            with st.container(horizontal=True, gap="small", key="overview-start"):
                st.page_link(pages["kmeans"], label="Bắt đầu phân đoạn", icon=":material/arrow_forward:",
                             icon_position="right")
                st.page_link(pages["library"], label="Chọn ảnh mẫu", icon=":material/photo_library:")
        with preview, st.container(key="overview-preview"):
            if ctx is None:
                st.info("Chọn ảnh mẫu hoặc tải ảnh/CSV lên ở thanh bên để xem trước.")
            else:
                image, _ = ui.source_panels(ctx.prepared)[0]
                ui.show_image(image, ctx.loaded.source_name)

    st.subheader("Chọn phương pháp")
    with st.container(key="overview-methods"):
        for col, (key, title, icon, description) in zip(st.columns(len(METHODS), gap="small"), METHODS):
            with col, st.container(border=True, height="stretch", key=f"method-{key}"):
                st.page_link(pages[key], label=title, icon=icon, width="stretch")
                st.caption(description)

    with st.container(border=True, key="overview-guide"):
        st.markdown("Chọn **Ảnh mẫu** hoặc **Tải tệp lên** trong thanh bên. Ứng dụng nhận PNG, JPG, BMP và CSV.")
        st.page_link(pages["guide"], label="Hướng dẫn sử dụng", icon=":material/menu_book:")

    with st.expander("Về đề tài và nguyên tắc cài đặt", icon=":material/info:"):
        st.markdown(
            "Bài tập lớn môn **Xử lý ảnh (INT13146)**, đề tài 3, PTIT. "
            "K-means đối chiếu giáo trình mục **6.2.4.2**. Otsu liên hệ tiêu chí phân ngưỡng mục **3.1.6**.\n\n"
            "- Hai thuật toán tự cài đặt bằng **NumPy** và có kiểm thử Pytest.\n"
            "- Pillow đọc, ghi và đổi kích thước ảnh. Matplotlib vẽ biểu đồ. Streamlit xây dựng giao diện.\n"
            "- Ứng dụng chạy cục bộ và hỗ trợ xuất PNG, CSV, JSON."
        )
