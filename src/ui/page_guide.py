"""Trang Hướng dẫn sử dụng."""

from __future__ import annotations

import streamlit as st

from src.ui.components import page_header


def render() -> None:
    page_header("Hướng dẫn sử dụng", "Các thao tác chính, định dạng dữ liệu và cách lưu kết quả.")
    st.markdown("""
#### 1. Chọn dữ liệu đầu vào (thanh bên trái)
- **Ảnh mẫu**: chọn trong danh sách, hoặc vào trang *Thư viện ảnh mẫu* và bấm **Dùng ảnh này**.
- **Tải tệp lên**: PNG, JPG/JPEG, BMP hoặc CSV, tối đa 30 MB và 40 triệu pixel.
  - Ảnh có kênh trong suốt (alpha) được ghép lên nền trắng; ảnh có thẻ EXIF hướng chụp được xoay đúng chiều.
  - Ảnh RGB có R = G = B ở mọi pixel được xử lý như ảnh xám.
- **Giới hạn cạnh dài khi xử lý**: ảnh lớn hơn được thu nhỏ (giữ tỉ lệ) để chạy nhanh; kích thước thật được ghi rõ trên mỗi trang.
- **Cách chuyển ảnh xám** (dùng cho Otsu và K-means 1 kênh): BT.601 hoặc trung bình cộng (R+G+B)/3.

#### 2. Định dạng CSV
- Ma trận **2 chiều** mức xám: mỗi dòng là một hàng ảnh, mỗi giá trị là **số nguyên 0–255**.
- Phân cách bằng dấu phẩy `,` (cũng nhận `;`, tab hoặc khoảng trắng); **không có dòng tiêu đề**; mọi dòng cùng số cột.
- Ví dụ 3×4:
```
0,0,255,255
0,128,255,255
0,0,0,255
```
- Lỗi được báo kèm vị trí dòng/cột: ký tự không phải số, số thập phân, giá trị ngoài [0, 255], dòng thiếu cột, ô trống.

#### 3. K-means
1. Chọn **K** (2–8), vector đặc trưng (RGB hoặc mức xám), cách khởi tạo, số vòng lặp tối đa, ngưỡng hội tụ, seed.
2. Bấm **Chạy K-means**. Thay đổi tham số không tự chạy lại — chỉ chạy khi bấm nút.
3. Tab **Tách nền**: chọn cụm làm đối tượng (mặc định là gợi ý theo viền ảnh, không phải kết luận của thuật toán).
4. Tab **Thống kê**: tâm cụm, số pixel, tỷ lệ, SSE theo vòng lặp.

#### 4. Otsu
1. Chọn chế độ **Tự động (Otsu)** hoặc **Thủ công** (để đối chiếu với ngưỡng tự chọn).
2. Bật **Đảo foreground/background** nếu đối tượng tối trên nền sáng.
3. Bấm **Chạy Otsu**; xem histogram (đường đỏ là t*) và đồ thị σ²_B(t).

#### 5. So sánh
- Chạy K-means và Otsu trên cùng ảnh; so sánh ảnh kết quả và **mask nhị phân**.
- IoU/Dice chỉ được tính khi có **ground truth** (ảnh mẫu có sẵn, hoặc tải mask PNG trắng = đối tượng, cùng kích thước ảnh gốc).

#### 6. Lưu kết quả
| Tệp | Nội dung |
|---|---|
| `kmeans_segmented.png` | Ảnh mỗi pixel mang màu tâm cụm |
| `kmeans_label_map.png` | Bản đồ nhãn tô màu theo bảng màu cố định |
| `*_mask.png` | Mask nhị phân 0/255 (255 = đối tượng) |
| `*_cutout.png` | Ảnh RGBA, nền trong suốt |
| `kmeans_labels.csv` | Ma trận nhãn cụm (0..K−1) |
| `kmeans_clusters.csv` | Bảng tâm cụm, số pixel, tỷ lệ |
| `otsu_binary_matrix.csv` | Ma trận mask 0/1 |
| `otsu_histogram.csv` | h(g) và σ²_B(g) cho 256 mức |
| `*_stats.json` | Tham số và toàn bộ thống kê của lần chạy |

#### 7. Xử lý sự cố
- *Kết quả biến mất sau khi đổi ảnh*: đúng thiết kế — kết quả cũ bị ẩn để không tải nhầm; bấm chạy lại.
- *Chạy chậm*: giảm “Giới hạn cạnh dài khi xử lý” hoặc giảm K.
- *Ảnh lỗi*: ứng dụng hiển thị thông báo đỏ ở đầu trang, không bị dừng.
""")
