# Test case và kết quả kiểm thử

Các phần: **(A)** test case trình diễn trên dữ liệu mẫu — kết quả thực tế lấy từ `scripts/run_experiments.py`
(chạy ngày 05/11/2026, Python 3.11.7, NumPy 2.4.6, Windows 10, CPU Intel Family 6 Model 165); **(B)** kiểm tra
đầu vào/giao diện; **(C)** kiểm thử tự động Pytest; **(D)** hồi quy sau review và kiểm tra Chrome.

Quy tắc cố định cho phần A (không chỉnh theo kết quả): K-means dùng K và đặc trưng gợi ý trong `data/samples.json`,
K-means++, seed 0, tol 1e-4, tối đa 100 vòng, cụm đối tượng = gợi ý theo viền ảnh. Otsu tự động, lớp đối tượng =
gợi ý theo viền ảnh. Bảng mẫu lấy trung vị 5 lần; quét K/kích thước lấy 3 lần. Thời gian tính tới nhãn/ảnh tâm
hoặc ảnh nhị phân, loại trừ đọc tệp, resize/chuyển xám, chọn đối tượng, biểu đồ và mã hoá tệp. IoU/Dice cần GT.

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

Trong lần chạy này SSE giảm khi K tăng; “khuỷu” rõ ở K = 5 — số vùng màu chính của cảnh (trời, mặt trời, núi,
cỏ, hồ; hai dãy núi màu gần nhau). Giá trị tối ưu toàn cục không tăng theo K; các nghiệm Lloyd chạy riêng không
được bảo đảm đơn điệu vì phụ thuộc khởi tạo/nghiệm cục bộ.

### A.2 Ảnh hưởng của khởi tạo (TC08, K = 6, khởi tạo ngẫu nhiên)

| Seed | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| SSE | 6 507 252 | **70 408 242** | 6 507 222 | 6 507 252 | 10 995 159 |
| Vòng lặp | 10 | 20 | 8 | 7 | 34 |

Seed 1 rơi vào cực tiểu địa phương xấu (SSE gấp ~10,8 lần) ⇒ minh chứng K-means phụ thuộc khởi tạo; đây là lý do có
random seed và tuỳ chọn K-means++.

### A.3 Thời gian theo kích thước (RGB ngẫu nhiên nhiều màu, K=4, tối đa 20 vòng)

| Cạnh ảnh | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|
| K-means (ms) | 32.3 | 160.7 | 760.7 | 3125.3 |
| Vòng thực tế | 20 | 20 | 20 | 20 |
| Hội tụ | Chưa | Chưa | Chưa | Chưa |
| Otsu (ms) | 0.17 | 0.33 | 1.79 | 6.92 |

Thời điểm đo: 2026-11-05T15:41:08+07:00. Số đo áp dụng cho môi trường và dữ liệu đã ghi.
Dữ liệu ngẫu nhiên có nhiều màu, không bảo đảm mọi pixel khác màu. Xem JSON gốc để tái lập tham số.

---

## B. Kiểm tra đầu vào và giao diện

| Mã | Tình huống | Cách thực hiện | Kết quả kỳ vọng | Kết quả thực tế | Kiểm chứng bởi |
|---|---|---|---|---|---|
| TB01 | CSV có chữ | Tải `data/sample_csv/invalid_text.csv` | Thông báo lỗi, không crash | “Giá trị không phải số tại dòng 2, cột 2: 'abc'.” | `test_invalid_csv_upload_shows_error_message`, `test_csv_errors` |
| TB02 | CSV ngoài [0,255] | `invalid_range.csv` | Báo lỗi vị trí | “Giá trị 300 tại dòng 2, cột 2 nằm ngoài khoảng [0, 255].” | `test_csv_errors` |
| TB03 | CSV thiếu cột | `invalid_ragged.csv` | Báo lỗi | “Dòng 2 có 2 cột, khác với 3 cột của dòng 1.” | `test_csv_errors` |
| TB04 | Tệp không phải ảnh | Đổi tên tệp văn bản thành `.png` | Báo tệp hỏng | “Không đọc được ảnh …” | `test_invalid_file_rejected` |
| TB05 | Kiểm tra giới hạn giải mã | Unit test hạ giới hạn xuống 1000 pixel, nạp ảnh 50×50 | Từ chối trước giải mã | Ném InputError; không cấp phát ảnh >40 triệu pixel để thử trực tiếp | `test_too_large_image_rejected` |
| TB06 | Ảnh lớn hợp lệ | Giảm “Giới hạn cạnh dài” xuống 128 | Thu nhỏ giữ tỉ lệ, thông báo kích thước | Hộp thông tin “Ảnh được thu nhỏ để xử lý: W×H → w×h” | `test_resize_limit_applied_and_reported` |
| TB07 | Ảnh có alpha | PNG RGBA nửa trong suốt | Ghép lên nền trắng, có ghi chú | Pixel trong suốt thành (255,255,255) | `test_alpha_is_composited_on_white` |
| TB08 | Ảnh EXIF xoay | JPEG Orientation = 6 | Xoay đúng chiều | 20×10 → 10×20, có ghi chú | `test_exif_orientation_applied` |
| TB09 | Đổi tham số và chạy lại | K = 3 → 4, bấm *Chạy K-means* | Kết quả cập nhật theo K mới | Thẻ “Số cụm K” = 4 | `test_kmeans_page_run_button_and_downloads` |
| TB10 | Đổi ảnh sau khi chạy | Chọn ảnh mẫu khác | Ẩn kết quả cũ | Cảnh báo “Ảnh đầu vào đã thay đổi — kết quả cũ đã bị ẩn” | `test_input_change_releases_results_and_returning_to_old_image_does_not_restore_downloads` |
| TB11 | Tải xuống | Tab *Tải xuống* | 7 tệp, mở lại đúng dữ liệu | 7 nút; PNG/CSV/JSON đọc lại khớp từng pixel | `test_export_files_reopen_with_same_content` |
| TB12 | Không có GT | So sánh trên `rgb_regions` | Không hiển thị IoU/Dice | Thông báo “Không có ground truth nên không tính…” | `test_compare_page_without_ground_truth_does_not_invent_scores` |
| TB13 | Ảnh chụp thật | Tải ảnh chụp của nhóm | Chạy được, kết quả hợp lý | **NOT RUN — nhóm xác nhận hiện chưa có ảnh chụp thật để nghiệm thu** | — |

---

## C. Kiểm thử tự động (Pytest)

Lệnh: `python -B -m pytest -q -p no:cacheprovider --capture=sys`. Kết quả sau sửa: **170 passed**.
Bằng chứng lần tổng kiểm tra: `docs/results/pytest_results.xml`; môi trường Python 3.11.7, NumPy 2.4.6, Streamlit 1.66.1.

| Tệp | Số test | Nội dung chính |
|---|---|---|
| `tests/test_kmeans.py` | 45 | Hai nhóm tách biệt; ví dụ 1 chiều tính tay (tâm 2 và 11, SSE = 4); K > số giá trị phân biệt / số mẫu; ảnh đồng nhất; cụm rỗng; max_iter; ngưỡng hội tụ; seed tái lập; SSE không tăng; kích thước đầu ra và tổng pixel; nhãn–tâm–SSE nhất quán; sắp xếp cụm; tính theo khối = tính toàn bộ = brute force; K-means có trọng số = dữ liệu khai triển; kết quả ảnh là nghiệm K-means trên mọi pixel; tham số sai |
| `tests/test_otsu.py` | 38 | Ví dụ giáo trình 3.1.6 (t* = 1); ma trận tính tay (đồng hạng 20–199 ⇒ 109); ảnh hai mức; toàn đen; toàn trắng; lớp rỗng không NaN; 5 ảnh ngẫu nhiên khớp cài đặt duyệt trực tiếp; ảnh hai đỉnh; mask chỉ 0/255; histogram; đầu vào sai; gợi ý cực tính theo viền |
| `tests/test_image_io.py` | 31 | PNG RGB/xám, JPEG, BMP; R=G=B ⇒ xám; alpha; EXIF; tệp hỏng, rỗng, sai định dạng; ảnh quá lớn; CSV đọc–ghi, 4 kiểu phân cách, 7 loại lỗi, không phải UTF-8; chuyển xám; giới hạn kích thước; ghi–đọc PNG RGBA |
| `tests/test_integration.py` | 56 | Mọi ảnh mẫu và GT hợp lệ; IoU = 1 trên hình tròn và CSV; CSV giáo trình; ảnh một màu; RGB nhiều hình; giới hạn kích thước và GT co theo; tệp xuất đọc lại đúng; đổi tham số đổi kết quả; **không import OpenCV/sklearn/skimage/scipy**; 7 trang giao diện chạy không lỗi; nút chạy, đổi K, 7 nút tải xuống; Otsu; So sánh có/không GT; CSV lỗi hiện thông báo |

Ngoài Pytest, giao diện đã được chạy thật bằng `python -m streamlit run app.py` và điều khiển bằng Chrome (headless)
để bấm nút và chụp màn hình — ảnh trong `docs/screenshots/`.


## D. Hồi quy sau review — 05/11/2026

PASS ở đây nghĩa là khớp kỳ vọng kiểm thử, không phải mọi phương pháp đều tách đối tượng hoàn hảo.
Các mảng sinh trong test là dữ liệu tổng hợp để kiểm chứng, không được gọi là ảnh chụp thật.

| Mã | Dữ liệu đầu vào | Thao tác | Mong đợi | Thực tế sau sửa | Trạng thái |
|---|---|---|---|---|---|
| R01 | h(0)=h(255)=1, h(127)=h(128)=127 | Otsu và đối chiếu duyệt mọi t | t thuộc argmax; thống kê khớp | t=126, σ²_B=63.75; ngưỡng 127 không bị chọn | PASS |
| R02 | Ảnh toàn 128, cả ba cực tính | Otsu tự động, xuất mask/RGBA/JSON | Invalid, không suy diễn đối tượng | Mask rỗng, alpha=0, cảnh báo khớp | PASS |
| R03 | Ảnh toàn 128, ngưỡng thủ công 200, đối tượng tối | Chạy thủ công | Cho lựa chọn lớp có chủ đích | Mask toàn 255; ảnh gốc giữ nguyên | PASS |
| R04 | CSV 400 chữ số 9 | Parser, AppTest và Chrome upload | Báo lỗi đầu vào, không OverflowError | InputError/thông báo ngoài miền; không ngoại lệ chưa bắt | PASS |
| R05 | CSV 1.00000000000000001 và 255.00000000000000001 | Parser | Không làm tròn để chấp nhận | Từ chối chính xác | PASS |
| R06 | CSV thiếu dấu nháy, ô quá dài hoặc ô số xuống dòng | Parser/AppTest | InputError | Có lỗi được kiểm soát | PASS |
| R07 | K=0,9,2.5,-1,True; seed/tol/max_iter sai kiểu | Gọi core | Từ chối trước tính toán | ValueError | PASS |
| R08 | chunk_size=0,-1,1.5,True,None | Gọi gán nhãn và K-means | Không dùng mảng chưa khởi tạo | ValueError trước cấp phát/gán nhãn | PASS |
| R09 | 80 vector 3D lệch, có trọng số, seed 42 | K=8 random đến hội tụ | Thực sự chạy nhánh cụm rỗng | Có cảnh báo xử lý; hội tụ; counts đều dương | PASS |
| R10 | Cùng dữ liệu R09, max_iter=1, tol=1000 | Dừng sớm | Không vượt vòng và không báo hội tụ sai | Một vòng; cảnh báo cụm rỗng cuối, converged=False | PASS |
| R11 | PNG tổng hợp RGB 4000×3000 | Nạp, giới hạn 800, chạy, xuất | 800×600; mask khớp GT | IoU cả hai=1; mọi PNG đúng cỡ | PASS |
| R12 | Nền sáng/vật tối; nền tối/vật sáng; tương phản cao | Hai thuật toán, so GT | Tách đúng trên mẫu hai mức | IoU=1 | PASS |
| R13 | Ảnh tương phản thấp 130/150 có nhiễu | Hai thuật toán, so GT | Tách được nhưng có sai số | IoU=0.9532 cả hai | PASS |
| R14 | RGB nhiều hình/màu; xám nhiều mức | K=5 RGB/K=4 xám và Otsu | Giữ kết quả đã kiểm chứng | K-means IoU=1; Otsu lần lượt 0.7964/0.6667 | PASS |
| R15 | Ảnh rỗng/sai chiều; NaN/Inf/âm/>255/lẻ | API ảnh và ngưỡng | Từ chối dữ liệu bất thường | ValueError | PASS |
| R16 | Cùng đầu vào, render lại/đổi cách chuyển xám | Theo dõi giải mã | Tái sử dụng ảnh gốc hiện tại | Không giải mã lại; đổi ảnh mới thì đọc lại | PASS |
| R17 | Đổi ảnh rồi chọn lại ảnh cũ; đổi tên tệp cùng bytes | AppTest | Bỏ payload/tệp cũ, metadata không nhầm | Không có nút tải cũ; chạy lại ghi đúng tên mới | PASS |
| R18 | Cùng kết quả, render lại/đổi foreground | Theo dõi xuất PNG/CSV/JSON | Không mã hoá lại khi không đổi; đổi mask phải cập nhật | Đúng số lần xuất; mask và JSON khớp foreground mới | PASS |
| R19 | Lỗi MemoryError/OSError chủ động trong test | Nạp ảnh/chạy/xuất qua UI | Hiện lỗi, không mời tải dữ liệu cũ | Có thông báo; không ngoại lệ chưa bắt/nút tải cũ | PASS |
| R20 | Tệp ảnh không hợp lệ: bytes văn bản mang đuôi PNG | API/AppTest/Chrome upload | Hiện lỗi định dạng | Có thông báo, không dừng app bằng exception | PASS |
| R21 | Các dependency được cài và cây phụ thuộc đang dùng | Kiểm tra metadata + source | Thỏa requirements, không có OpenCV trong cây active | Đạt | PASS |
| R22 | Ảnh chụp thật của nhóm | Nghiệm thu thực tế | Cần ảnh thực của nhóm | Chưa có dữ liệu; không suy đoán chất lượng | NOT RUN |

### D.1 Kiểm tra Chrome thực

Báo cáo máy: [browser_checks_20261105_155516.json](results/browser_checks_20261105_155516.json).

- HTTP health và trang mặc định: PASS.
- Upload PNG từ máy, chỉnh K=5, chạy K-means; tải PNG mask và CSV nhãn, đọc lại so pipeline: PASS.
- Otsu trên cùng ảnh; tải PNG mask và CSV 0/1, đọc lại so pipeline: PASS, t=148.
- Điều hướng và So sánh trên cùng ảnh đã upload: PASS.
- Viewport 1440×1000, 768×1024, 390×844: không tràn ngang toàn trang, không stException: PASS.
- Upload CSV số quá lớn và PNG không hợp lệ: lỗi có kiểm soát, PASS.
- Ngoại lệ JavaScript quan sát được trong lượt kiểm tra: 0.
- Không xác nhận lưu thành công bằng toast: bằng chứng tải là tệp Chrome thực sự ghi vào thư mục kiểm thử tạm,
  được đọc lại và so mảng dữ liệu. Người dùng cuối vẫn chọn nơi lưu/ghi đè trong trình duyệt.

Ảnh chụp tương ứng: `docs/screenshots/*_browser_*_20261105_155516.png`.
