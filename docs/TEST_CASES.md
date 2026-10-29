# Test case và kết quả kiểm thử

Hai phần: **(A)** test case trình diễn trên dữ liệu mẫu — kết quả thực tế lấy từ `scripts/run_experiments.py`
(chạy ngày 30/10/2026, Python 3.11.7, NumPy 2.4.6, Windows 10, CPU Intel Family 6 Model 165); **(B)** kiểm tra
đầu vào/giao diện; **(C)** danh sách kiểm thử tự động Pytest.

Quy tắc cố định cho phần A (không chỉnh theo kết quả): K-means dùng K và đặc trưng gợi ý trong `data/samples.json`,
K-means++, seed 0, tol 1e-4, tối đa 100 vòng, cụm đối tượng = gợi ý theo viền ảnh. Otsu tự động, lớp đối tượng =
gợi ý theo viền ảnh. Thời gian là trung vị 5 lần chạy, chỉ tính phần thuật toán. IoU/Dice chỉ tính khi có ground truth.

---

## A. Test case trên dữ liệu mẫu

| Mã | Ảnh mẫu (`data/`) | Mục đích | Tham số | Kết quả kỳ vọng | Kết quả thực tế | Nhận xét |
|---|---|---|---|---|---|---|
| TC01 | `sample_images/circle_black_on_white.png` 256×256 | Trường hợp lý tưởng 2 mức xám | K=2 xám; Otsu | Cả hai tách đúng hình tròn | K-means: 1 vòng, SSE = 0, cụm đối tượng gợi ý = [0]. Otsu: t* = 127 (255 ngưỡng đồng hạng 0–254), đối tượng = lớp tối. **IoU = 1.0000 cả hai** | Đạt. Đồng hạng là do histogram chỉ có 0 và 255 |
| TC02 | `bright_rect_on_dark.png` 256×192 | Đối tượng sáng/nền tối, quy ước foreground | K=2 xám; Otsu | Ngưỡng giữa 30 và 220; đối tượng là lớp sáng | t* = 124 (đồng hạng 30–219), đối tượng = lớp sáng. **IoU = 1.0000 cả hai** | Đạt |
| TC03 | `multi_shapes_color.png` 320×240 RGB | Nhiều đối tượng màu khác nhau | K=5 RGB; Otsu | K-means tách mỗi màu một cụm; Otsu 2 lớp có thể gộp sai | K-means: 1 vòng, SSE = 0, **IoU = 1.0000**. Otsu: t* = 148, **IoU = 0.7964**, Dice 0.8867 — hình tròn vàng (xám ≈ 187) bị xếp cùng lớp nền (≈ 234) | Thể hiện hạn chế của Otsu: 2 lớp, mất thông tin màu |
| TC04 | `gray_levels_objects.png` 300×200 | Ảnh xám nhiều mức (35/90/150/215) | K=4 xám; Otsu | K-means tách 4 mức; Otsu gộp đối tượng tối nhất vào nền | K-means: **IoU = 1.0000**. Otsu: t* = 119, **IoU = 0.6667** — khối 90 bị coi là nền | Otsu không xử lý được nhiều đối tượng khác mức xám |
| TC05 | `noisy_circle.png` 256×256 | Nhiễu Gauss σ=45 + muối tiêu 6% | K=2 xám; Otsu | Hình dạng đúng nhưng có điểm lốm đốm, IoU < 1 | t* = 126, η = 0.922. **IoU = 0.8397 cả hai**, Dice 0.9129. K-means 3 vòng | Trường hợp hạn chế: không dùng thông tin lân cận. Hướng khắc phục: lọc trung vị trước (giáo trình 3.2.3) |
| TC06 | `low_contrast.png` 256×256 | Đối tượng 150 / nền 130, nhiễu σ=4 | K=2 xám; Otsu | Tách được nhưng nhạy nhiễu | t* = 139, η = 0.778 (thấp nhất trong các ảnh có đối tượng). **IoU = 0.9532 cả hai**. K-means 6 vòng | Đạt một phần. Biên độ tương phản nhỏ ⇒ nhiễu làm lẫn pixel hai lớp |
| TC07 | `uniform_gray.png` 128×128 (toàn 128) | Ngoại lệ: không có gì để tách | K=2; Otsu | Không lỗi; có cảnh báo | K-means: cảnh báo “K được giảm từ 2 xuống 1”, mask rỗng. Otsu: `valid = False`, cảnh báo “chỉ có một mức xám”, toàn ảnh là nền | Đạt — không crash, không bịa kết quả |
| TC08 | `rgb_regions.png` 360×240 RGB | Cảnh tổng hợp nhiều vùng màu (không có GT) | K=6 RGB; Otsu | Tách các vùng màu; không tính IoU | K-means: 8 vòng, hội tụ, SSE = 6 507 222. Otsu t* = 139. Không tính IoU (không có GT) | Xem thêm thí nghiệm K và seed ở dưới |
| TC09 | `tiny_10x10.png` 10×10 | Minh hoạ từng bước | K=2 xám; Otsu | Hội tụ sau vài vòng, tách đúng | K-means 2 vòng (K-means++); trang *Kiến thức thuật toán* hiển thị tâm từng vòng với khởi tạo ngẫu nhiên. Otsu t* = 125. **IoU = 1.0000 cả hai** | Đạt |
| TC10 | `sample_csv/csv_rectangle_32x32.csv` | Đầu vào CSV | K=2; Otsu | Tách đúng hình chữ nhật | t* = 119. **IoU = 1.0000 cả hai** | Đạt — CSV đọc và xử lý như ảnh xám |
| TC11 | `sample_csv/textbook_otsu_5x6.csv` | Đối chiếu ví dụ giáo trình mục 3.1.6 | Otsu | θ = 1 như giáo trình | **t* = 1**; f(0)=1.3611, f(1)=1.6806, f(2)=1.5625, f(3)=1.1142, f(4)=0.5067, f(5) không hợp lệ | Khớp giáo trình (giáo trình làm tròn: 1.35, 1.66, 1.54, 1.10, 0.49, ∞) |

### A.1 Ảnh hưởng của K (TC08, RGB, K-means++, seed 0)

| K | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| SSE (triệu) | 235.8 | 133.7 | 103.0 | 11.0 | 6.51 | 3.80 | 2.88 |
| Vòng lặp | 3 | 4 | 2 | 2 | 8 | 8 | 19 |

SSE giảm khi K tăng (luôn đúng với nghiệm tốt); “khuỷu” rõ ở K = 5 — đúng bằng số vùng màu chính của cảnh
(trời, mặt trời, núi, cỏ, hồ; hai dãy núi màu gần nhau).

### A.2 Ảnh hưởng của khởi tạo (TC08, K = 6, khởi tạo ngẫu nhiên)

| Seed | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| SSE | 6 507 252 | **70 408 242** | 6 507 222 | 6 507 252 | 10 995 159 |
| Vòng lặp | 10 | 20 | 8 | 7 | 34 |

Seed 1 rơi vào cực tiểu địa phương xấu (SSE gấp ~10,8 lần) ⇒ minh chứng K-means phụ thuộc khởi tạo; đây là lý do có
random seed và tuỳ chọn K-means++.

### A.3 Thời gian theo kích thước (ảnh RGB ngẫu nhiên — trường hợp xấu nhất, mọi pixel khác màu)

| Cạnh ảnh | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|
| K-means, K=4, đúng 20 vòng (ms) | 34.6 | 177.5 | 858.8 | 3 693.5 |
| Otsu (ms) | 0.12 | 0.22 | 1.17 | 4.44 |

Thời gian tăng gần tuyến tính theo số pixel. Ảnh thực tế có nhiều pixel trùng màu nên K-means nhanh hơn nhiều
(TC08: 43 ms cho 86 400 pixel).

---

## B. Kiểm tra đầu vào và giao diện

| Mã | Tình huống | Cách thực hiện | Kết quả kỳ vọng | Kết quả thực tế | Kiểm chứng bởi |
|---|---|---|---|---|---|
| TB01 | CSV có chữ | Tải `data/sample_csv/invalid_text.csv` | Thông báo lỗi, không crash | “Giá trị không phải số tại dòng 2, cột 2: 'abc'.” | `test_invalid_csv_upload_shows_error_message`, `test_csv_errors` |
| TB02 | CSV ngoài [0,255] | `invalid_range.csv` | Báo lỗi vị trí | “Giá trị 300 tại dòng 2, cột 2 nằm ngoài khoảng [0, 255].” | `test_csv_errors` |
| TB03 | CSV thiếu cột | `invalid_ragged.csv` | Báo lỗi | “Dòng 2 có 2 cột, khác với 3 cột của dòng 1.” | `test_csv_errors` |
| TB04 | Tệp không phải ảnh | Đổi tên tệp văn bản thành `.png` | Báo tệp hỏng | “Không đọc được ảnh …” | `test_invalid_file_rejected` |
| TB05 | Ảnh quá lớn | Ảnh > 40 triệu pixel | Từ chối trước khi giải mã | “Ảnh quá lớn (… triệu pixel)…” | `test_too_large_image_rejected` |
| TB06 | Ảnh lớn hợp lệ | Giảm “Giới hạn cạnh dài” xuống 128 | Thu nhỏ giữ tỉ lệ, thông báo kích thước | Hộp thông tin “Ảnh được thu nhỏ để xử lý: W×H → w×h” | `test_resize_limit_applied_and_reported` |
| TB07 | Ảnh có alpha | PNG RGBA nửa trong suốt | Ghép lên nền trắng, có ghi chú | Pixel trong suốt thành (255,255,255) | `test_alpha_is_composited_on_white` |
| TB08 | Ảnh EXIF xoay | JPEG Orientation = 6 | Xoay đúng chiều | 20×10 → 10×20, có ghi chú | `test_exif_orientation_applied` |
| TB09 | Đổi tham số và chạy lại | K = 3 → 4, bấm *Chạy K-means* | Kết quả cập nhật theo K mới | Thẻ “Số cụm K” = 4 | `test_kmeans_page_run_button_and_downloads` |
| TB10 | Đổi ảnh sau khi chạy | Chọn ảnh mẫu khác | Ẩn kết quả cũ | Cảnh báo “Ảnh đầu vào đã thay đổi — kết quả cũ đã bị ẩn” | Kiểm tra thủ công trên trình duyệt |
| TB11 | Tải xuống | Tab *Tải xuống* | 7 tệp, mở lại đúng dữ liệu | 7 nút; PNG/CSV/JSON đọc lại khớp từng pixel | `test_export_files_reopen_with_same_content` |
| TB12 | Không có GT | So sánh trên `rgb_regions` | Không hiển thị IoU/Dice | Thông báo “Không có ground truth nên không tính…” | `test_compare_page_without_ground_truth_does_not_invent_scores` |
| TB13 | Ảnh chụp thật | Tải ảnh chụp của nhóm | Chạy được, kết quả hợp lý | **Chưa thực hiện — nhóm tự thử với ảnh của mình và ghi kết quả vào đây** | — |

---

## C. Kiểm thử tự động (Pytest)

Lệnh: `python -m pytest -v`. Kết quả lần chạy cuối: **86 passed** (≈ 6 giây).

| Tệp | Số test | Nội dung chính |
|---|---|---|
| `tests/test_kmeans.py` | 23 | Hai nhóm tách biệt; ví dụ 1 chiều tính tay (tâm 2 và 11, SSE = 4); K > số giá trị phân biệt / số mẫu; ảnh đồng nhất; cụm rỗng; max_iter; ngưỡng hội tụ; seed tái lập; SSE không tăng; kích thước đầu ra và tổng pixel; nhãn–tâm–SSE nhất quán; sắp xếp cụm; tính theo khối = tính toàn bộ = brute force; K-means có trọng số = dữ liệu khai triển; kết quả ảnh là nghiệm K-means trên mọi pixel; tham số sai |
| `tests/test_otsu.py` | 16 | Ví dụ giáo trình 3.1.6 (t* = 1); ma trận tính tay (đồng hạng 20–199 ⇒ 109); ảnh hai mức; toàn đen; toàn trắng; lớp rỗng không NaN; 5 ảnh ngẫu nhiên khớp cài đặt duyệt trực tiếp; ảnh hai đỉnh; mask chỉ 0/255; histogram; đầu vào sai; gợi ý cực tính theo viền |
| `tests/test_image_io.py` | 24 | PNG RGB/xám, JPEG, BMP; R=G=B ⇒ xám; alpha; EXIF; tệp hỏng, rỗng, sai định dạng; ảnh quá lớn; CSV đọc–ghi, 4 kiểu phân cách, 7 loại lỗi, không phải UTF-8; chuyển xám; giới hạn kích thước; ghi–đọc PNG RGBA |
| `tests/test_integration.py` | 23 | Mọi ảnh mẫu và GT hợp lệ; IoU = 1 trên hình tròn và CSV; CSV giáo trình; ảnh một màu; RGB nhiều hình; giới hạn kích thước và GT co theo; tệp xuất đọc lại đúng; đổi tham số đổi kết quả; **không import OpenCV/sklearn/skimage/scipy**; 7 trang giao diện chạy không lỗi; nút chạy, đổi K, 7 nút tải xuống; Otsu; So sánh có/không GT; CSV lỗi hiện thông báo |

Ngoài Pytest, giao diện đã được chạy thật bằng `python -m streamlit run app.py` và điều khiển bằng Chrome (headless)
để bấm nút và chụp màn hình — ảnh trong `docs/screenshots/`.
