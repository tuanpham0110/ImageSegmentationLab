# Kết quả thực nghiệm (sinh tự động bởi scripts/run_experiments.py)

Môi trường: Python 3.11.7, NumPy 2.4.6, Windows-10-10.0.19045-SP0, CPU: Intel64 Family 6 Model 165 Stepping 2, GenuineIntel.
Thời gian là trung vị của 5 lần chạy, ảnh xử lý ở kích thước gốc (đều ≤ 800 px).

| Mẫu | Kích thước | K (thực tế) | Vòng lặp | K-means (ms) | t* Otsu | Otsu (ms) | IoU K-means | IoU Otsu | Dice K-means | Dice Otsu |
|---|---|---|---|---|---|---|---|---|---|---|
| circle_black_on_white | 256x256 | 2 (gray) | 1 | 3.3 | 127 đảo | 0.38 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| bright_rect_on_dark | 256x192 | 2 (gray) | 1 | 3.6 | 124 | 0.31 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| multi_shapes_color | 320x240 | 5 (rgb) | 1 | 7.6 | 148 đảo | 0.43 | 1.0000 | 0.7964 | 1.0000 | 0.8867 |
| gray_levels_objects | 300x200 | 4 (gray) | 1 | 4.3 | 119 | 0.38 | 1.0000 | 0.6667 | 1.0000 | 0.8000 |
| noisy_circle | 256x256 | 2 (gray) | 3 | 5.0 | 126 đảo | 0.30 | 0.8397 | 0.8397 | 0.9129 | 0.9129 |
| low_contrast | 256x256 | 2 (gray) | 6 | 4.4 | 139 | 0.30 | 0.9532 | 0.9532 | 0.9761 | 0.9761 |
| uniform_gray | 128x128 | 1 (gray) | 1 | 0.7 | 128 (không hợp lệ) | 0.13 | — | — | — | — |
| rgb_regions | 360x240 | 6 (rgb) | 8 | 47.2 | 139 đảo | 0.36 | — | — | — | — |
| tiny_10x10 | 10x10 | 2 (gray) | 2 | 0.5 | 125 | 0.11 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| csv_rectangle_32x32 | 32x32 | 2 (gray) | 2 | 0.5 | 119 | 0.12 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| textbook_otsu_5x6 | 6x5 | 2 (gray) | 2 | 0.4 | 1 | 0.08 | — | — | — | — |

Cảnh báo phát sinh:

- `circle_black_on_white`: Có 255 ngưỡng cùng đạt cực đại (từ 0 đến 254); chọn floor(trung bình) = 127.
- `bright_rect_on_dark`: Có 190 ngưỡng cùng đạt cực đại (từ 30 đến 219); chọn floor(trung bình) = 124.
- `multi_shapes_color`: Có 76 ngưỡng cùng đạt cực đại (từ 111 đến 186); chọn floor(trung bình) = 148.
- `gray_levels_objects`: Có 60 ngưỡng cùng đạt cực đại (từ 90 đến 149); chọn floor(trung bình) = 119.
- `uniform_gray`: Dữ liệu chỉ có 1 giá trị phân biệt nên K được giảm từ 2 xuống 1.
- `uniform_gray`: Ảnh chỉ có một mức xám (128); không thể tách hai lớp. Ngưỡng được đặt bằng 128, toàn bộ ảnh thuộc lớp nền (giá trị 0).
- `rgb_regions`: Có 32 ngưỡng cùng đạt cực đại (từ 124 đến 155); chọn floor(trung bình) = 139.
- `tiny_10x10`: Có 108 ngưỡng cùng đạt cực đại (từ 72 đến 179); chọn floor(trung bình) = 125.
- `csv_rectangle_32x32`: Có 130 ngưỡng cùng đạt cực đại (từ 55 đến 184); chọn floor(trung bình) = 119.

## Ảnh hưởng của K (rgb_regions, RGB, K-means++, seed 0)

| K | SSE | Vòng lặp | Thời gian (ms) |
|---|---|---|---|
| 2 | 235,777,059 | 3 | 21.6 |
| 3 | 133,726,637 | 4 | 24.8 |
| 4 | 103,020,593 | 2 | 22.2 |
| 5 | 11,011,447 | 2 | 25.7 |
| 6 | 6,507,222 | 8 | 46.8 |
| 7 | 3,797,605 | 8 | 50.6 |
| 8 | 2,878,251 | 19 | 101.4 |

## Ảnh hưởng của seed (rgb_regions, K = 6, khởi tạo ngẫu nhiên)

| Seed | SSE | Vòng lặp |
|---|---|---|
| 0 | 6,507,252 | 10 |
| 1 | 70,408,242 | 20 |
| 2 | 6,507,222 | 8 |
| 3 | 6,507,252 | 7 |
| 4 | 10,995,159 | 34 |

## Thời gian theo kích thước ảnh (ảnh RGB ngẫu nhiên; K-means K = 4, đúng 20 vòng)

| Cạnh | Số pixel | K-means 20 vòng (ms) | Otsu (ms) |
|---|---|---|---|
| 128 | 16,384 | 39.3 | 0.13 |
| 256 | 65,536 | 261.3 | 0.28 |
| 512 | 262,144 | 1047.2 | 1.83 |
| 1024 | 1,048,576 | 4001.2 | 5.89 |
