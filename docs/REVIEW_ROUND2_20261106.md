# Review và sửa lỗi tiếp — lượt hai, 06/11/2026

## 1. Tổng quan

Bắt đầu tại HEAD `66f3245`, nhánh `main`, Git sạch. Bộ test nền **249 PASS**. Giữ các sửa đổi đã được commit
từ lượt trước; đọc lại các ranh giới chuyển đổi ảnh/mask, số học, xuất tệp và trạng thái ground truth.

Tìm và sửa thêm **9 nhóm vấn đề: 2 HIGH, 5 MEDIUM, 2 LOW**. Trong đó có lỗi PNG 16-bit phát sinh từ nhánh alpha
của lượt sửa trước: chuyển sang RGBA quá sớm làm mất tương phản. Lượt này thêm kiểm thử trực tiếp cho trường hợp đó.
Không phát hiện vấn đề CRITICAL trong phạm vi kiểm tra. Không kết luận dự án đã hết mọi lỗi.

Kết quả cuối: **295 PASS, 0 FAIL, 0 ERROR, 0 SKIP**, thêm **46 ca**. Các ngưỡng, số cụm, số vòng, SSE và
IoU/Dice trên 11 mẫu được đối chiếu lại, đều giữ nguyên.

## 2. Các lỗi đã sửa

Vị trí theo mã nguồn sau sửa. Những mục dưới đây đã được thực hiện và kiểm chứng.

| STT | File / vị trí | Vấn đề và tác động | Nguyên nhân | Cách sửa | Mức độ |
|---|---|---|---|---|---|
| 1 | `src/utils/image_io.py:109` | PNG 16-bit có tRNS, mức 1000/2000/3000, bị đọc thành toàn trắng; mất thông tin phân đoạn | Nhánh alpha gọi Pillow convert RGBA trực tiếp, các mức >255 bị clip, khóa tRNS không được xử lý trong thang gốc | Xử lý 16-bit trước: chuẩn hóa mức sáng, đối chiếu khóa trong thang gốc rồi ghép pixel trong suốt trắng | HIGH |
| 2 | `src/utils/image_io.py:246` | Ground truth 16-bit toàn trắng 65535 bị biến thành toàn nền, có thể cho IoU/Dice sai | Dùng cùng phép giãn min/max của ảnh cường độ cho mask; ảnh đồng nhất bị đưa về 0 | Dùng chung decoder nhưng tách chính sách chuyển đổi mask: thang tuyệt đối 0–65535 về 0–255; trắng giữ 255 | HIGH |
| 3 | `src/utils/masks.py:8`, `src/processing/pipeline.py:58`, `src/processing/postprocessing.py:41` | Mask bool/0–1 mất đối tượng sau resize; đảo mask tạo 254 thay vì 0; cutout/overlay xử lý mask NaN/âm không nhất quán | Dùng phép trừ 255 và ngưỡng >127 trực tiếp trên các cách biểu diễn mask khác nhau; thiếu validation dùng chung | Thống nhất giá trị >0 là foreground cho mask trong pipeline, đổi về 0/255 trước resize/đảo; dùng chung kiểm tra hình dạng, số hữu hạn/không âm | MEDIUM |
| 4 | `src/algorithms/kmeans.py:28`, `:40`, `:351` | Trọng số lớn có counts âm do tràn int64; đầu vào hữu hạn vẫn có thể trả tâm/SSE/khoảng cách Inf | Ép counts không kiểm tra miền; chỉ kiểm tra đầu vào mà chưa kiểm soát số học/trung gian và đầu ra | Giữ counts float64 nếu ngoài miền int64; kiểm tra SSE/tâm/phân bố khởi tạo; lỗi tràn số thành ValueError có nguyên nhân; không trả kết quả NaN/Inf | MEDIUM |
| 5 | `src/utils/image_io.py:259`, `:275` | Xuất PNG/CSV có thể làm tròn 1.5 thành 1, wrap -1 thành 255 hoặc uint64 lớn thành số âm; dữ liệu xuất sai âm thầm | Ép uint8/int64 trước kiểm tra | Validate hình dạng/giá trị trước ghi; PNG yêu cầu pixel nguyên 0–255; CSV giữ dtype số nguyên gốc và từ chối dữ liệu lẻ/không hữu hạn/rỗng | MEDIUM |
| 6 | `src/utils/image_io.py:297` | JSON có thể chứa NaN/Infinity, gây lỗi ở consumer tuân thủ JSON chuẩn | json.dumps mặc định allow_nan=True | allow_nan=False; báo ValueError để luồng xuất hiển thị lỗi, không tạo tệp không chuẩn | MEDIUM |
| 7 | `src/processing/pipeline.py:148` | JSON K-means mức xám thiếu cách chuyển xám, không đủ thông tin tái lập kết quả từ ảnh RGB | Chỉ lưu tham số clustering, không lưu prepared.gray_method | Thêm trường `chuyen_xam`, cùng quy ước với JSON Otsu; giữ các trường cũ | MEDIUM |
| 8 | `src/utils/image_io.py:301` | Numpy boolean hợp lệ không xuất được JSON | Custom serializer có integer/float/array nhưng thiếu np.bool_ | Chuyển np.bool_ thành bool gốc | LOW |
| 9 | `src/ui/page_compare.py:136` | Thao tác foreground/cực tính giải mã lại cùng GT tải lên, tốn CPU/RAM với mask lớn | Mỗi rerender gọi lại load_mask_bytes và fit_ground_truth | Giữ một mask đã fit trong payload theo file_id; đổi tệp/chạy mới/input sẽ đọc lại; bỏ cache khi xóa tệp hoặc xảy ra lỗi | LOW |

Một ca kiểm chứng kết hợp hai lỗi HIGH sử dụng dữ liệu thật đi qua decoder và hai thuật toán:

| Trạng thái | Mức sáng ảnh đọc được | Pixel foreground của GT trắng | IoU K-means | IoU Otsu |
|---|---|---:|---:|---:|
| Decoder tại HEAD trước sửa | Chỉ 255 | 0 | 1.0000 | 1.0000 |
| Sau sửa | 128 và 255 | 1536 | 0.1250 | 0.1250 |

Trước sửa, ảnh và GT đều sai nên hai mask rỗng tạo điểm 1 giả. Sau sửa, đối tượng 192 pixel được so với GT
trắng 1536 pixel, đúng tỷ lệ 192/1536 = 0.125. Bằng chứng:
[review_round2_16bit_case_20261106.json](results/review_round2_16bit_case_20261106.json).
Một test AppTest độc lập tải các PNG 16-bit qua widget và xác nhận cả hai IoU = 0.125.

## 3. Cải tiến và tương thích

- Thêm `src/utils/masks.py` để metrics, resize ground truth, đảo mask và cutout dùng chung quy ước/validation.
  Đường bool/uint8 không quét tính hữu hạn không cần thiết. Không thay cấu trúc phân lớp dự án.
- Decoder ảnh dùng context manager và xử lý EXIF/chuyển màu trong cùng ranh giới xử lý lỗi.
- Cache GT chỉ giữ mask ở kích thước xử lý, không giữ thêm ảnh GT gốc đã giải mã. Có test đếm số lần đọc và
  xác minh đổi/xóa tệp sẽ vô hiệu cache.
- JSON K-means bổ sung một trường metadata; tên trường hiện có, tên hàm công khai và các tệp xuất được giữ.
- `to_png_bytes`/`matrix_to_csv_bytes` từ chối dữ liệu có thể bị biến đổi âm thầm. CSV tiện ích vẫn ghi được
  uint64 lớn; parser ảnh CSV tiếp tục yêu cầu miền 0–255 như trước.
- Giữ counts int64 trên đường xử lý pixel thông thường. Trọng số ngoài miền int64 dùng float64, theo độ chính
  xác biểu diễn của float64; không hỗ trợ tính toán chính xác tùy ý cho số cực lớn.
- Không thay dependency, dữ liệu mẫu, app.py, cấu hình Streamlit, API hay kiến trúc. Không tạo commit.

Đo K-means trước/sau bằng mã tại HEAD và mã hiện tại, ảnh RGB 512×512, seed 19, K=4, tối đa 20 vòng,
3 lần chạy xen kẽ: trung vị **823.13 → 813.89 ms**. Nhãn, tâm và SSE **giống hệt**. Không thấy suy giảm rõ rệt
trong phép đo này; không coi chênh lệch nhỏ là cam kết tăng tốc. Số liệu:
[review_round2_performance_20261106.json](results/review_round2_performance_20261106.json).

## 4. Kết quả kiểm thử

Môi trường giữ nguyên: Python 3.11.7, NumPy 2.4.6, Pillow 13.0.0, Streamlit 1.66.1, Windows 10.

| Nhóm | Kết quả | Bằng chứng |
|---|---|---|
| Test nền | 249 PASS | Chạy lại trước thay đổi; mã HEAD `66f3245` |
| Tái hiện trước sửa | 38 ca mới FAIL; test cache GT cũng FAIL | Chạy test trước khi sửa các nhóm tương ứng; không coi test cũ PASS là đủ |
| Unit/kiểm thử module | 210 PASS | K-means 64, Otsu 44, I/O 58, metrics 11, preprocessing 24, postprocessing 9 |
| Tích hợp/giao diện | 85 PASS | Pipeline, 7 trang, upload, export, GT, cache và lỗi tài nguyên |
| Tổng cuối | **295 PASS**, khoảng 19.3 giây | [JUnit](results/review_round2_pytest_20261106.xml), 0 fail/error/skip |
| Compile | PASS | compileall app.py/src/scripts/tests |
| Static analysis giới hạn | PASS | AST của 27 tệp ứng dụng/script; không phát hiện bare except hoặc eval/exec trong phép quét |
| Dependency / whitespace | PASS | pip check; git diff --check |
| 11 mẫu hồi quy | PASS | Ngưỡng, số cụm, số vòng, SSE và IoU/Dice khớp artifact lượt một: [kết quả](results/review_round2_samples_20261106.json) |
| Chromium thật | **12 PASS** | Điều hướng, PNG 16-bit, hai thuật toán, upload GT trắng 16-bit, đổi ảnh bỏ GT, viewport 1440/768/390; không JS exception quan sát được |
| Tệp tải thật | **7 PASS** | 4 PNG + 2 CSV đối chiếu từng pixel/nhãn; JSON kiểm tra trường chuyển xám |

Bằng chứng thêm: [compile/static/dependency](results/review_round2_checks_20261106.json),
[trình duyệt/tệp tải](results/review_round2_browser_20261106.json),
[desktop](screenshots/review_round2_compare_1440_20261106.png),
[tablet](screenshots/review_round2_compare_768_20261106.png),
[mobile](screenshots/review_round2_compare_390_20261106.png).

## 5. Phần chưa xác minh

- Chưa có ảnh chụp thật/ground truth của nhóm để nghiệm thu chất lượng phân đoạn theo nghiệp vụ.
- Chưa đo line/branch coverage, chưa chạy linter chuyên dụng hoặc quét CVE; công cụ tương ứng chưa có trong
  môi trường. AST và pip check không thay thế những kiểm tra này.
- Chưa kiểm tra tải nhiều phiên đồng thời, hủy tác vụ hoặc hệ điều hành/Python khác. Ảnh rất lớn vẫn có thể
  cần nhiều RAM; ứng dụng giữ giới hạn đầu vào và thông báo thiếu bộ nhớ.
- Số học dùng float64, không phải số học chính xác tùy ý. Phép tính vượt phạm vi được từ chối thay vì trả
  NaN/Inf hoặc counts bị wrap.

Không có ca test thất bại chưa xử lý trong phạm vi lượt này. Các artifact và báo cáo lượt trước được giữ.

## 6. Đánh giá sau lượt hai

| Tiêu chí | Đánh giá |
|---|---|
| Correctness | Sửa hai lỗi đọc ảnh/GT có thể tạo điểm đánh giá giả; thống nhất cách biểu diễn mask và dữ liệu xuất |
| Code Quality | Quy ước mask được tập trung; decoder và kiểm tra số học có ranh giới rõ ràng |
| Security | Dữ liệu xuất và trung gian được kiểm tra; giữ giới hạn đầu vào; chưa audit CVE/triển khai Internet |
| Performance | Tránh đọc lại GT khi chỉ đổi lựa chọn; benchmark không cho thấy suy giảm rõ rệt |
| Maintainability | Giữ API, thêm metadata tái lập và 46 ca kiểm thử sát các lỗi tìm được |
| Reliability | Không trả counts âm/NaN/Inf do tràn; lỗi xuất không tạo tệp đã bị âm thầm biến đổi |
| Test Coverage | 295 ca PASS; có đối chiếu trình duyệt và dữ liệu tải thật; chưa có số đo phần trăm coverage |
