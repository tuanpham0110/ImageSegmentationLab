# Kết quả thực nghiệm (sinh tự động bởi scripts/run_experiments.py)

Môi trường: Python 3.11.7, NumPy 2.4.6, Windows-10-10.0.19045-SP0, CPU: Intel64 Family 6 Model 165 Stepping 2, GenuineIntel.
Thời điểm đo: 2026-11-06T15:15:29+07:00 (Asia/Ho_Chi_Minh).
Bảng mẫu: trung vị 5 lần chạy; quét K và kích thước: trung vị 3 lần. Ảnh mẫu đều ≤ 800 px.
Phạm vi đo: Từ ảnh đã tiền xử lý đến kết quả trực tiếp của thuật toán (nhãn/ảnh tâm hoặc ảnh nhị phân); không gồm đọc tệp, resize/chuyển xám, chọn foreground, ảnh tách nền, biểu đồ và xuất tệp.

| Mẫu | Kích thước | K (thực tế) | Vòng lặp | K-means (ms) | t* Otsu | Otsu (ms) | IoU K-means | IoU Otsu | Dice K-means | Dice Otsu |
|---|---|---|---|---|---|---|---|---|---|---|
| circle_black_on_white | 256x256 | 2 (gray) | 1 | 1.9 | 127 đảo | 0.57 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| bright_rect_on_dark | 256x192 | 2 (gray) | 1 | 2.5 | 124 | 0.39 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| multi_shapes_color | 320x240 | 5 (rgb) | 1 | 5.0 | 148 đảo | 0.67 | 1.0000 | 0.7964 | 1.0000 | 0.8867 |
| gray_levels_objects | 300x200 | 4 (gray) | 1 | 2.9 | 119 | 0.41 | 1.0000 | 0.6667 | 1.0000 | 0.8000 |
| noisy_circle | 256x256 | 2 (gray) | 3 | 2.9 | 126 đảo | 0.36 | 0.8397 | 0.8397 | 0.9129 | 0.9129 |
| low_contrast | 256x256 | 2 (gray) | 6 | 2.4 | 139 | 0.33 | 0.9532 | 0.9532 | 0.9761 | 0.9761 |
| uniform_gray | 128x128 | 1 (gray) | 1 | 0.5 | 128 (không hợp lệ) | 0.15 | — | — | — | — |
| rgb_regions | 360x240 | 6 (rgb) | 8 | 35.9 | 139 đảo | 0.71 | — | — | — | — |
| tiny_10x10 | 10x10 | 2 (gray) | 2 | 0.4 | 125 | 0.14 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| csv_rectangle_32x32 | 32x32 | 2 (gray) | 2 | 0.6 | 119 | 0.15 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| textbook_otsu_5x6 | 6x5 | 2 (gray) | 2 | 0.4 | 1 | 0.12 | — | — | — | — |

Cảnh báo phát sinh:

- `circle_black_on_white`: Có 255 ngưỡng cùng đạt cực đại (từ 0 đến 254); chọn ngưỡng đạt cực đại gần floor(trung bình) nhất = 127.
- `bright_rect_on_dark`: Có 190 ngưỡng cùng đạt cực đại (từ 30 đến 219); chọn ngưỡng đạt cực đại gần floor(trung bình) nhất = 124.
- `multi_shapes_color`: Có 76 ngưỡng cùng đạt cực đại (từ 111 đến 186); chọn ngưỡng đạt cực đại gần floor(trung bình) nhất = 148.
- `gray_levels_objects`: Có 60 ngưỡng cùng đạt cực đại (từ 90 đến 149); chọn ngưỡng đạt cực đại gần floor(trung bình) nhất = 119.
- `uniform_gray`: Dữ liệu chỉ có 1 giá trị phân biệt nên K được giảm từ 2 xuống 1.
- `uniform_gray`: Ảnh chỉ có một mức xám (128); không thể tách hai lớp. Ngưỡng được đặt bằng 128, ảnh nhị phân trực tiếp có toàn giá trị 0.
- `uniform_gray`: Otsu không chia được hai lớp: mask tự động được để trống (toàn nền), không áp dụng lựa chọn lớp sáng/tối. Có thể dùng ngưỡng thủ công để đối chiếu.
- `rgb_regions`: Có 32 ngưỡng cùng đạt cực đại (từ 124 đến 155); chọn ngưỡng đạt cực đại gần floor(trung bình) nhất = 139.
- `tiny_10x10`: Có 108 ngưỡng cùng đạt cực đại (từ 72 đến 179); chọn ngưỡng đạt cực đại gần floor(trung bình) nhất = 125.
- `csv_rectangle_32x32`: Có 130 ngưỡng cùng đạt cực đại (từ 55 đến 184); chọn ngưỡng đạt cực đại gần floor(trung bình) nhất = 119.

## Ảnh hưởng của K (rgb_regions, RGB, K-means++, seed 0)

| K | SSE | Vòng lặp | Thời gian (ms) |
|---|---|---|---|
| 2 | 235,777,059 | 3 | 17.7 |
| 3 | 133,726,637 | 4 | 27.8 |
| 4 | 103,020,593 | 2 | 19.4 |
| 5 | 11,011,447 | 2 | 19.4 |
| 6 | 6,507,222 | 8 | 34.8 |
| 7 | 3,797,605 | 8 | 36.0 |
| 8 | 2,878,251 | 19 | 75.7 |

## Ảnh hưởng của seed (rgb_regions, K = 6, khởi tạo ngẫu nhiên)

| Seed | SSE | Vòng lặp |
|---|---|---|
| 0 | 6,507,252 | 10 |
| 1 | 70,408,242 | 20 |
| 2 | 6,507,222 | 8 |
| 3 | 6,507,252 | 7 |
| 4 | 10,995,159 | 34 |

## Thời gian theo kích thước ảnh (ảnh RGB ngẫu nhiên; K-means K = 4, tối đa 20 vòng)

| Cạnh | Số pixel | K-means (ms) | Vòng lặp | Hội tụ | Otsu (ms) |
|---|---|---|---|---|---|
| 128 | 16,384 | 33.3 | 20 | Chưa | 0.20 |
| 256 | 65,536 | 181.5 | 20 | Chưa | 0.35 |
| 512 | 262,144 | 751.3 | 20 | Chưa | 1.80 |
| 1024 | 1,048,576 | 2996.1 | 20 | Chưa | 6.80 |
