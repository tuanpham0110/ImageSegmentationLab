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
- Tối đa 4.194.304 ô. Có thể có một dấu phân cách thừa ở cuối dòng. Ô rỗng có dấu nháy hoặc chứa khoảng trắng bị báo lỗi.
- Ví dụ 3×4:
```
0,0,255,255
0,128,255,255
0,0,0,255
```
- Lỗi được báo kèm vị trí dòng/cột: ký tự không phải số, số thập phân, giá trị ngoài [0, 255], dòng thiếu cột, ô trống.

#### 3. K-means
1. Chọn **K** (1–16), vector đặc trưng (RGB, Lab hoặc mức xám), cách khởi tạo, số vòng lặp tối đa, ngưỡng hội tụ, seed.
2. Bấm **Chạy K-means**. Thay đổi tham số không tự chạy lại — chỉ chạy khi bấm nút.
3. Tab **Tách nền**: chọn cụm làm đối tượng (mặc định là gợi ý theo viền ảnh, không phải kết luận của thuật toán).
4. Tab **Thống kê**: tâm cụm, số pixel, tỷ lệ, SSE theo vòng lặp.

#### 4. Otsu
1. Chọn chế độ **Tự động (Otsu)** hoặc **Thủ công** (để đối chiếu với ngưỡng tự chọn).
2. Ở **Đối tượng (foreground) là**, chọn *Lớp tối (≤ t)* nếu đối tượng tối trên nền sáng, *Lớp sáng (> t)*
   cho đối tượng sáng, hoặc *Gợi ý theo viền ảnh*. Gợi ý có thể sai khi đối tượng phủ viền ảnh.
3. Bấm **Chạy Otsu**; xem histogram (đường đỏ là t*) và đồ thị σ²_B(t).
   Đổi phương pháp, cực tính, ngưỡng đang dùng hoặc hậu xử lý thì cần chạy lại để tải kết quả.
4. Nếu ảnh chỉ có một mức xám, Otsu báo không hợp lệ và mask tự động rỗng; có thể dùng chế độ thủ công để đối chiếu.

#### 5. So sánh
- Chạy K-means và Otsu trên cùng ảnh; so sánh ảnh kết quả và **mask nhị phân**.
- IoU/Dice chỉ được tính khi có **ground truth** (ảnh mẫu có sẵn, hoặc tải mask PNG/BMP/CSV, cùng kích thước ảnh gốc
  hoặc kích thước xử lý). Mask nhị phân **0/1**: 1 = đối tượng; mask **0/255**: trắng = đối tượng.
- Có thể tải lại `otsu_binary_matrix.csv` làm mask tham chiếu để kiểm tra dữ liệu xuất. Mask dự đoán của thuật toán
  không phải ground truth độc lập, nên độ trùng khớp khi so lại chính mask đó không đánh giá chất lượng phân đoạn.
- Mask tải lên gắn với đầu vào hiện tại. Khi đổi ảnh hoặc tiền xử lý, hãy tải lại mask để tránh dùng nhầm ground truth.
- Tab **Tải xuống** trên trang So sánh cho tải mask, ảnh tách nền hoặc toàn bộ PNG/CSV/JSON của cả hai thuật toán trong một tệp ZIP.

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

Trình duyệt xử lý nơi lưu và xác nhận ghi đè. Bấm tải không chạy lại thuật toán hoặc thay đổi kết quả.
Kiểm tra tệp trong mục tải xuống của trình duyệt. Nếu tạo tệp thất bại, trang hiển thị lỗi.
Với ảnh đã thu nhỏ, tab Tải xuống của từng thuật toán có **Thêm PNG kích thước gốc**.
Nhãn và mask được ánh xạ bằng láng giềng gần nhất. Ảnh tách nền lấy màu gốc theo mask đã ánh xạ.
Các tệp CSV và ZIP so sánh dùng kích thước xử lý.

Thời gian hiển thị tính từ ảnh đã tiền xử lý đến nhãn/ảnh tâm hoặc ảnh nhị phân; loại trừ đọc tệp,
chuyển xám/resize, chọn đối tượng, biểu đồ và xuất tệp.

#### 7. Xử lý sự cố
- *Kết quả biến mất sau khi đổi ảnh*: đúng thiết kế — kết quả cũ bị ẩn để không tải nhầm; bấm chạy lại.
- *Chạy chậm*: giảm “Giới hạn cạnh dài khi xử lý” hoặc giảm K.
- *Ảnh lỗi*: ứng dụng hiển thị thông báo đỏ ở đầu trang, không bị dừng.
""")
