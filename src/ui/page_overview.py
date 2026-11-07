"""Trang tổng quan: chọn phương pháp và theo dõi quy trình thực hành."""

from __future__ import annotations

import streamlit as st

from src.ui import components as ui
from src.ui.state import get_context

STEPS = [
    ("Chọn ảnh đầu vào", "Dùng ảnh mẫu hoặc tải ảnh PNG, JPG, BMP hay ma trận CSV từ máy tính."),
    ("Chuẩn bị dữ liệu", "Giới hạn kích thước, chuyển mức xám và lọc trung vị nếu cần khử nhiễu."),
    ("Chạy thuật toán", "Phân cụm màu với K-means hoặc tìm ngưỡng mức xám bằng Otsu."),
    ("Chọn đối tượng, tách nền", "Tạo mask nhị phân, ảnh tô màu đối tượng và ảnh nền trong suốt."),
    ("Đánh giá và lưu kết quả", "Xem thống kê, đối chiếu ground truth nếu có; tải PNG, CSV hoặc JSON."),
]


def render() -> None:
    pages = st.session_state["_nav_pages"]
    with st.container(border=True, key="overview-hero"):
        intro, preview = st.columns([1.5, 1], gap="medium", vertical_alignment="center")
        with intro:
            st.html(
                "<section class='isl-hero'>"
                "<div class='isl-eyebrow'>Xử lý ảnh · PTIT</div>"
                "<h1>Phân đoạn ảnh.<br>Hiểu rõ từng kết quả.</h1>"
                "<p>Thực hành K-means và Otsu trên ảnh của bạn. "
                "Điều chỉnh tham số, tách đối tượng và so sánh hai phương pháp ngay trong một không gian làm việc.</p>"
                "</section>"
            )
            with st.container(horizontal=True, gap="small", key="overview-start"):
                st.page_link(pages["kmeans"], label="Thực hành K-means", icon=":material/play_arrow:")
                st.page_link(pages["otsu"], label="Khám phá Otsu", icon=":material/arrow_forward:")
        with preview, st.container(key="overview-preview"):
            ctx = get_context()
            if ctx is not None:
                ui.show_image(ctx.prepared.image, "Ảnh đang chọn · sẵn sàng thực hành")
            else:
                st.info("Chọn ảnh mẫu hoặc tải ảnh lên ở thanh bên để bắt đầu.")
        st.html("<div class='isl-hero-footer'><span>02 thuật toán</span>"
                "<span>Ảnh màu &amp; ảnh mức xám</span><span>Xuất PNG · CSV · JSON</span>"
                "<span>Xử lý cục bộ trên máy</span></div>")

    st.html("<div class='isl-section-heading'><h2>Chọn phương pháp</h2>"
            "<span>Hai cách tiếp cận, cùng một ảnh đầu vào</span></div>")
    with st.container(key="overview-methods"):
        left, right = st.columns(2, gap="medium")
        with left, st.container(border=True, height="stretch", key="method-kmeans"):
            st.html(
                "<div class='isl-method-top'><span class='isl-method-number'>01</span>"
                "<span class='isl-method-tag'>PHÂN CỤM PIXEL</span></div>"
                "<h3 class='isl-method-title'>K-means</h3>"
                "<p class='isl-method-copy'>Chia ảnh thành các nhóm màu tương đồng. "
                "Chọn số cụm K, quan sát bản đồ nhãn và tách vùng đối tượng bạn quan tâm.</p>"
                "<p class='isl-method-note'><strong>Phù hợp:</strong> ảnh màu, nhiều vùng màu khác nhau "
                "và thực hành lượng tử hoá màu.</p>"
            )
            st.page_link(pages["kmeans"], label="Bắt đầu với K-means", icon=":material/arrow_forward:", width="stretch")
        with right, st.container(border=True, height="stretch", key="method-otsu"):
            st.html(
                "<div class='isl-method-top'><span class='isl-method-number'>02</span>"
                "<span class='isl-method-tag'>PHÂN NGƯỠNG MỨC XÁM</span></div>"
                "<h3 class='isl-method-title'>Otsu</h3>"
                "<p class='isl-method-copy'>Tự tìm ngưỡng phân chia đối tượng và nền theo độ sáng. "
                "Xem histogram, điều chỉnh ngưỡng và thử Multi-Otsu với ba lớp.</p>"
                "<p class='isl-method-note'><strong>Phù hợp:</strong> ảnh có đối tượng tương phản với nền, "
                "văn bản quét và histogram hai đỉnh rõ.</p>"
            )
            st.page_link(pages["otsu"], label="Bắt đầu với Otsu", icon=":material/arrow_forward:", width="stretch")

    with st.container(key="overview-bottom"):
        left, right = st.columns([1.1, 1], gap="medium")
        with left, st.container(border=True, height="stretch", key="overview-workflow"):
            st.html("<div class='isl-section-heading'><h2>Quy trình xử lý</h2><span>05 bước thực hành</span></div>")
            steps = "".join(
                f"<li><span class='isl-step-number'>{i:02d}</span><div><h3>{title}</h3><p>{body}</p></div></li>"
                for i, (title, body) in enumerate(STEPS, 1)
            )
            st.html(f"<ol class='isl-workflow'>{steps}</ol>")
        with right, st.container(border=True, height="stretch", key="overview-quickstart"):
            st.html(
                "<div class='isl-section-heading'><h2>Lần đầu sử dụng?</h2></div>"
                "<ol class='isl-quickstart'>"
                "<li>Chọn <strong>Ảnh mẫu</strong> ở thanh bên trái. "
                "Bạn có thể bắt đầu với mẫu <em>Nhiều hình màu khác nhau</em>.</li>"
                "<li>Mở <strong>K-means</strong> hoặc <strong>Otsu</strong>, "
                "điều chỉnh tham số rồi bấm nút chạy.</li>"
                "<li>Xem ảnh kết quả, chọn đối tượng trong phần <strong>Tách nền</strong> "
                "và lưu tệp ở tab <strong>Tải xuống</strong>.</li>"
                "<li>Vào <strong>So sánh thuật toán</strong> để đối chiếu hai phương pháp trên cùng một ảnh.</li>"
                "</ol>"
            )
            st.page_link(pages["library"], label="Khám phá thư viện ảnh mẫu", icon=":material/photo_library:", width="stretch")
            st.page_link(pages["guide"], label="Đọc hướng dẫn sử dụng", icon=":material/menu_book:", width="stretch")

    with st.expander("Về đề tài và nguyên tắc cài đặt", icon=":material/info:"):
        st.markdown(
            "Bài tập lớn môn **Xử lý ảnh (INT13146)**, đề tài 3, PTIT. "
            "K-means đối chiếu giáo trình mục **6.2.4.2**; Otsu liên hệ tiêu chí phân ngưỡng mục **3.1.6**.\n\n"
            "- K-means và Otsu tự cài đặt bằng **NumPy**, tách biệt với giao diện và có kiểm thử Pytest.\n"
            "- Không dùng OpenCV, `sklearn.cluster.KMeans` hay `skimage.filters.threshold_otsu`.\n"
            "- Pillow đọc, ghi và đổi kích thước ảnh; Matplotlib vẽ biểu đồ; Streamlit xây dựng giao diện.\n"
            "- Ứng dụng chạy cục bộ, không cần Internet hoặc API."
        )
    st.html("<p class='isl-footnote'>Image Segmentation Lab · PTIT · "
            "IoU và Dice chỉ được tính khi có ground truth để đối chiếu.</p>")
