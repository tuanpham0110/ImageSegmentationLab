"""Trang Tổng quan."""

from __future__ import annotations

import streamlit as st

STEPS = [
    ("1 · Đầu vào", "Ảnh PNG/JPG/BMP, ma trận CSV hoặc ảnh mẫu. Kiểm tra định dạng, EXIF, alpha, kích thước."),
    ("2 · Tiền xử lý", "Giới hạn kích thước xử lý; chuyển xám (BT.601 hoặc trung bình cộng)."),
    ("3 · Phân đoạn", "K-means: phân cụm pixel. Otsu: ngưỡng cực đại phương sai giữa lớp."),
    ("4 · Tách nền", "Tạo mask nhị phân, ảnh tách nền trong suốt, ảnh tô màu đối tượng."),
    ("5 · Đánh giá & lưu", "Thống kê, biểu đồ, IoU/Dice khi có ground truth; tải PNG/CSV/JSON."),
]


def render() -> None:
    st.markdown(
        "<div class='isl-hero'><h1>IMAGE SEGMENTATION LAB</h1>"
        "<p>Hệ thống phân đoạn ảnh bằng K-means và Otsu — Bài tập lớn môn Xử lý ảnh (INT13146), Đề tài 3, PTIT</p></div>",
        unsafe_allow_html=True,
    )
    st.markdown("#### Quy trình xử lý")
    cols = st.columns(len(STEPS))
    for col, (title, body) in zip(cols, STEPS):
        col.markdown(f"<div class='isl-step'><b>{title}</b>{body}</div>", unsafe_allow_html=True)

    st.markdown("#### Hai phương pháp")
    c1, c2 = st.columns(2, gap="large")
    c1.markdown(
        "<div class='isl-card'><h4>K-means (phân cụm)</h4><p>Mỗi pixel là một vector (mức xám hoặc R,G,B). "
        "Lặp: gán pixel vào tâm gần nhất theo khoảng cách Euclid, cập nhật tâm bằng trung bình cụm, tới khi hội tụ. "
        "Giáo trình mục 6.2.4.2. Người dùng chọn cụm nào là đối tượng.</p></div>",
        unsafe_allow_html=True,
    )
    c2.markdown(
        "<div class='isl-card'><h4>Otsu (phân ngưỡng tự động)</h4><p>Từ histogram 256 mức, chọn ngưỡng t* làm cực đại "
        "phương sai giữa hai lớp σ²_B(t). Tương đương hàm f(g) ở giáo trình mục 3.1.6; phân ngưỡng theo mục 3.1.3 và 5.2.</p></div>",
        unsafe_allow_html=True,
    )

    st.markdown("#### Bắt đầu nhanh")
    st.markdown(
        "1. Ở **thanh bên trái**, chọn *Ảnh mẫu* (ví dụ “Nhiều hình màu khác nhau”) hoặc *Tải tệp lên*.\n"
        "2. Mở trang **K-means**, chọn K rồi bấm **Chạy K-means**; sang tab *Tách nền* để chọn cụm đối tượng.\n"
        "3. Mở trang **Otsu Thresholding**, bấm **Chạy Otsu**, xem histogram và ngưỡng t*.\n"
        "4. Mở trang **So sánh** để đặt hai kết quả cạnh nhau; tải kết quả ở tab **Tải xuống**."
    )
    st.markdown("#### Nguyên tắc cài đặt")
    st.markdown(
        "- **Không dùng OpenCV** (`cv2`), không dùng `sklearn.cluster.KMeans` hay `skimage.filters.threshold_otsu`.\n"
        "- K-means và Otsu **tự cài đặt** bằng NumPy (`src/algorithms/`), tách biệt với giao diện, có kiểm thử Pytest.\n"
        "- Pillow chỉ dùng để đọc/ghi/đổi kích thước ảnh; Matplotlib để vẽ biểu đồ; Streamlit cho giao diện.\n"
        "- Toàn bộ chạy cục bộ, không cần Internet, không dùng API."
    )
