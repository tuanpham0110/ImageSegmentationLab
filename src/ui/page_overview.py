"""Trang tổng quan: chọn phương pháp và theo dõi quy trình thực hành."""

from __future__ import annotations

import streamlit as st

STEPS = [
    ("Chọn ảnh đầu vào", "Dùng ảnh mẫu hoặc tải ảnh PNG, JPG, BMP hay ma trận CSV từ máy tính."),
    ("Chuẩn bị dữ liệu", "Giới hạn kích thước, chuyển mức xám và lọc trung vị nếu cần khử nhiễu."),
    ("Chạy thuật toán", "Phân cụm màu với K-means hoặc tìm ngưỡng mức xám bằng Otsu."),
    ("Chọn đối tượng, tách nền", "Tạo mask nhị phân, ảnh tô màu đối tượng và ảnh nền trong suốt."),
    ("Đánh giá và lưu kết quả", "Xem thống kê, đối chiếu ground truth nếu có; tải PNG, CSV hoặc JSON."),
]


def render() -> None:
    st.html(
        "<section class='isl-hero'>"
        "<div class='isl-eyebrow'>PTIT · XỬ LÝ ẢNH · ĐỀ TÀI 03</div>"
        "<h1>Phân đoạn ảnh<br>với K-means và Otsu</h1>"
        "<p>Khám phá cách thuật toán phân chia một bức ảnh thành các vùng. "
        "Chọn dữ liệu, điều chỉnh tham số và quan sát kết quả qua từng bước xử lý.</p>"
        "<div class='isl-hero-footer'><span>02 phương pháp phân đoạn</span>"
        "<span>Ảnh màu &amp; ảnh mức xám</span><span>Xuất kết quả PNG · CSV · JSON</span></div>"
        "</section>"
    )

    st.html("<div class='isl-section-heading'><h2>Chọn phương pháp thực hành</h2>"
            "<span>Cùng một ảnh, hai cách tiếp cận</span></div>")
    pages = st.session_state["_nav_pages"]
    with st.container(key="overview-methods"):
        left, right = st.columns(2, gap="medium")
        with left, st.container(border=True, height="stretch", key="method-kmeans"):
            st.html(
                "<div class='isl-method-top'><span class='isl-method-number'>01</span>"
                "<span class='isl-method-tag'>PHÂN CỤM PIXEL</span></div>"
                "<h3 class='isl-method-title'>K-means</h3>"
                "<p class='isl-method-copy'>Gom những pixel có màu sắc hoặc mức xám gần nhau thành K cụm. "
                "Quan sát màu tâm cụm, bản đồ nhãn và chọn các cụm thuộc đối tượng cần tách.</p>"
                "<p class='isl-method-note'><strong>Phù hợp:</strong> ảnh màu, nhiều vùng màu khác nhau "
                "và thực hành lượng tử hoá màu.</p>"
            )
            st.page_link(pages["kmeans"], label="Bắt đầu với K-means", icon=":material/arrow_forward:", width="stretch")
        with right, st.container(border=True, height="stretch", key="method-otsu"):
            st.html(
                "<div class='isl-method-top'><span class='isl-method-number'>02</span>"
                "<span class='isl-method-tag'>PHÂN NGƯỠNG MỨC XÁM</span></div>"
                "<h3 class='isl-method-title'>Otsu</h3>"
                "<p class='isl-method-copy'>Tự tìm ngưỡng phân chia các lớp sáng và tối dựa trên histogram. "
                "Theo dõi ngưỡng tối ưu, ảnh nhị phân và mở rộng với Multi-Otsu ba lớp.</p>"
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
    st.html("<p class='isl-footnote'>Image Segmentation Lab · Học viện Công nghệ Bưu chính Viễn thông "
            "<br>IoU và Dice chỉ được tính khi có ảnh ground truth để đối chiếu.</p>")
