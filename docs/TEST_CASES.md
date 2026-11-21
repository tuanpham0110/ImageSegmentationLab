# Test case và kết quả kiểm thử

## Nghiệm thu Giai đoạn 6 ngày 21/11/2026

Kết quả hiện tại gồm **616 PASS, 0 FAIL, 0 SKIP** trong 45,35 giây trên Windows, Python 3.11.7, NumPy 2.4.6 và
Streamlit 1.66.1. Ruff và `pip check` đều đạt. Các ca Chrome được thực thi riêng với Chrome headless.
Mỗi dòng dưới đây là một ca nghiệm thu, có thể đại diện cho nhiều tham số trong Pytest. Tên hàm ở cột thao tác dùng
để tìm test bằng `pytest -k`. Bảng hiện tại sử dụng 11 mẫu đã có trong `data/`, ma trận nhỏ cố định và ảnh chụp
`grace_hopper.jpg` có sẵn trong Matplotlib. Các ca tự động chạy ngoại tuyến sau khi đã cài dependency.
Ca đích chỉ đọc sẽ báo SKIP nếu tài khoản chạy test vẫn có quyền ghi vào tệp đã đặt chỉ đọc.

```powershell
.\.venv\Scripts\python.exe -m pytest -q --junitxml=outputs/stage6/pytest.xml
```

Bằng chứng cục bộ nằm ở `outputs/stage6/pytest.xml`, `outputs/stage6/compare_masks.png` và hai tệp tải từ Chrome
`outputs/stage6/otsu_mask.png`, `outputs/stage6/otsu_binary_matrix.csv`. Các bảng A–H bên dưới ghi những lần chạy trước.

### Nhóm 1 Kiểm thử thuật toán

| ID | Mục đích | Dữ liệu đầu vào | Thao tác | Kỳ vọng | Thực tế | Trạng thái |
|---|---|---|---|---|---|---|
| ALG-01 | Đối chiếu tính tay hai thuật toán | Ma trận `[[0,0,2],[8,10,10]]` | Chạy K=2 xám và Otsu qua pipeline trong `test_hand_computed_gray_matrix_agrees_across_both_algorithms` | Tâm `2/3, 28/3`, SSE `16/3`, ngưỡng Otsu 4, mask chỉ chọn hàng dưới | Tâm, SSE, ngưỡng, phương sai giữa lớp `169/9` và từng pixel mask khớp | PASS |
| ALG-02 | Ảnh đồng nhất và ngưỡng suy biến | `uniform_gray.png`, toàn 128 | Chạy K yêu cầu 4 và Otsu với cả ba lựa chọn cực tính trong `test_uniform_image_is_handled_without_crash` và `test_uniform_auto_otsu_always_has_empty_foreground` | K giảm về 1, Otsu không tách hai lớp, mask tự động rỗng | K thực tế 1, SSE 0, Otsu `valid=False`, mức dự phòng 128 và mask toàn 0 | PASS |
| ALG-03 | Vật tối và vật sáng trên nền tương phản | `circle_black_on_white.png`, `bright_rect_on_dark.png` | Chạy K=2 và Otsu theo cấu hình mẫu, so với mask tham chiếu trong `test_reference_samples_keep_expected_segmentation` | Cả hai phương pháp đạt IoU 1 cho cả hai ảnh | Ngưỡng lần lượt 127 và 124, IoU và Dice đều bằng 1 | PASS |
| ALG-04 | Phân đoạn nhiều mức xám | `gray_levels_objects.png`, bốn mức 35/90/150/215 | Chạy K=4, Otsu hai lớp và kiểm tra Multi-Otsu trong `test_reference_samples_keep_expected_segmentation`, `test_multi_otsu_fixes_gray_levels_sample` | K-means giữ bốn mức, Otsu hai lớp bỏ sót mức thấp nhất của đối tượng | SSE K-means bằng 0, IoU/Dice `1/1`. Otsu t=119, IoU/Dice `0,6667/0,8000` | PASS |
| ALG-05 | Ảnh tương phản thấp có nhiễu | `low_contrast.png`, nền 130 và vật 150 | Chạy cả hai phương pháp, so mask trong `test_reference_samples_keep_expected_segmentation` | IoU mỗi phương pháp đạt ít nhất 0,95 | Otsu t=139. Cả hai đạt IoU `0,9532` và Dice `0,9761` | PASS |
| ALG-06 | Gom nhóm màu RGB | `multi_shapes_color.png` và ma trận RGB ba màu | Chạy K=5 và K=3 tương ứng trong `test_reference_samples_keep_expected_segmentation`, `test_three_rgb_colors_recover_exact_partition_and_leave_input_unchanged` | Giữ đủ số cụm, nhãn và màu tâm đúng, ảnh vào giữ nguyên | Mẫu năm màu có K=5, SSE 0, IoU 1. Ma trận ba màu trả đúng từng nhãn | PASS |
| ALG-07 | Biên hợp lệ của K và seed | Ảnh 320×240, hai vector phân biệt và mảng RGB dùng seed cố định | Thử K=1, K=4 và K=8 trên hai điểm, rồi lặp cùng seed trong `test_kmeans_ui_can_run_one_cluster`, `test_output_shapes_and_pixel_counts_rgb`, `test_k_larger_than_number_of_samples`, `test_seed_reproducibility` | K=1 dùng tâm trung bình, K=4 giữ bốn cụm, K=8 giới hạn ở 2, cùng seed tái lập | Các điều kiện về tâm, K thực tế, nhãn và số đếm đều khớp | PASS |
| ALG-08 | Tái khởi tạo cụm rỗng và giới hạn vòng lặp | 80 vector 3D có trọng số, seed 42, K=8 | Chạy đến hội tụ và chạy tối đa một vòng trong `test_empty_cluster_is_reinitialised`, `test_final_empty_cluster_is_reported_without_exceeding_iteration_budget` | Báo cụm rỗng, tái khởi tạo khi còn vòng lặp, dừng đúng giới hạn | Khi đủ vòng, hội tụ và mọi cụm có điểm. Ca một vòng dừng đúng, cảnh báo còn cụm rỗng | PASS |
| ALG-09 | Histogram một đỉnh và tương phản thấp | Ma trận 5×5, mức 100–108, tần suất `1,2,3,4,5,4,3,2,1` | Tính Otsu trong `test_unimodal_low_contrast_histogram_still_has_a_mathematical_optimum` | Cực đại tại 103–104, chọn 103, độ tách bằng `2/3` | t=103, phương sai giữa lớp `8/3`, η=`2/3`, ảnh nhị phân khớp `g > 103` | PASS |
| ALG-10 | Ngưỡng theo ví dụ giáo trình và đối chiếu độc lập | Ma trận CSV 5×6 và năm ảnh sinh theo seed | So với phép tính tay và vét cạn ngưỡng trong `test_textbook_example_section_3_1_6`, `test_random_images_match_brute_force` | Mẫu giáo trình t=1, kết quả ngẫu nhiên thuộc tập cực đại theo vét cạn | t=1, phương sai `1,6806` sau làm tròn. Năm ca đều khớp đối chiếu | PASS |

### Nhóm 2 Kiểm thử dữ liệu

| ID | Mục đích | Dữ liệu đầu vào | Thao tác | Kỳ vọng | Thực tế | Trạng thái |
|---|---|---|---|---|---|---|
| DATA-01 | Ảnh chụp JPEG hợp lệ và ghi thư mục có quyền | `grace_hopper.jpg` từ Matplotlib, RGB 512×600 | Đọc, thu cạnh dài về 120, chạy cả hai, ghi PNG vào `tmp_path` và mở lại trong `test_bundled_real_jpeg_is_segmented_and_saved_to_a_writable_directory` | Hai mask đúng kích thước và miền giá trị, byte ảnh nguồn giữ nguyên | JPEG giải mã đúng RGB 512×600, cả hai mask PNG 102×120, uint8 0/255, mở lại đúng từng pixel | PASS |
| DATA-02 | CSV hợp lệ và sai cấu trúc | `csv_rectangle_32x32.csv`, `invalid_ragged.csv`, `invalid_text.csv` | Đọc và chạy mẫu hợp lệ. Kiểm tra số cột, ô lỗi trong `test_csv_sample_segmented_by_both`, `test_csv_errors`, `test_alternating_algorithms_preserves_only_results_for_the_current_upload` | Ma trận đúng cỡ, lỗi có vị trí | Mẫu 32×32 cho t=119 và IoU 1 cả hai. CSV lệch số cột báo dòng 2, CSV chữ báo dòng 2 cột 2 | PASS |
| DATA-03 | Rỗng, sai định dạng và ảnh PNG hỏng | Bytes rỗng, TIFF mang đuôi PNG, PNG thiếu IEND hoặc có CRC sai | Đọc trong `test_invalid_file_rejected`, `test_unsupported_tiff_content_is_rejected_even_with_a_png_filename`, `test_png_with_damaged_chunk_structure_is_rejected` | Trả `InputError` có lý do, gồm CRC hoặc độ dài IEND sai | Từ chối tệp rỗng, nội dung TIFF và cả năm biến thể PNG hỏng | PASS |
| DATA-04 | Ảnh tối thiểu | PNG 1×1 với mức 0 hoặc 255 | Chạy hai phương pháp và mở lại PNG mask trong `test_one_pixel_image_reaches_both_algorithms_and_exports` | K giảm về 1, Otsu không chia lớp, kết quả giữ kích thước | K=1 với đúng một pixel, Otsu không hợp lệ, hai mask 1×1 đều bằng 0 | PASS |
| DATA-05 | Ảnh lớn và chặn quá giới hạn | RGB tổng hợp 4000×3000 và ảnh 50×50 với giới hạn test 1000 pixel | Chạy và đọc lại PNG trong `test_large_encoded_rgb_image_is_resized_and_exported_at_reported_size`, kiểm tra `test_too_large_image_rejected` | Ảnh lớn xử lý ở 800×600, đầu vào vượt hạn mức bị từ chối | PNG đều 800×600, IoU hai phương pháp bằng 1 trên fixture. Ca vượt giới hạn trả `InputError` | PASS |
| DATA-06 | Ghi vào đích chỉ đọc | PNG trong `tmp_path` có thuộc tính chỉ đọc | Ghi ảnh khác vào cùng đường dẫn trong `test_writing_png_to_a_read_only_path_preserves_existing_file` | Hệ điều hành báo `PermissionError`, giữ nguyên byte đích | Trên Windows, nhận đúng lỗi quyền ghi và đích giữ nguyên | PASS |
| DATA-07 | Từ chối mask tham chiếu sai quy ước | PNG và CSV với các mức `0,128,255`, mask PNG 16 bit | Gọi `test_comparison_rejects_multilevel_mask_without_using_sample_ground_truth`, `test_strict_16bit_ground_truth_checks_native_pixel_values` | Từ chối nhiều mức, chỉ tính điểm với mask nhị phân đúng độ sâu | Cả hai tệp nhiều mức bị từ chối. PNG 16 bit hợp lệ giữ đúng mask, mức trung gian bị từ chối | PASS |

### Nhóm 3 Kiểm thử hệ thống

| ID | Mục đích | Dữ liệu đầu vào | Thao tác | Kỳ vọng | Thực tế | Trạng thái |
|---|---|---|---|---|---|---|
| SYS-01 | Luân phiên hai thuật toán và xử lý nhiều ảnh trong một phiên | PNG RGB 320×240, CSV 32×32, PNG xám 10×10 | AppTest tải lần lượt, chạy K-means và Otsu cho từng tệp rồi quay về K-means trong `test_alternating_algorithms_preserves_only_results_for_the_current_upload` | Sáu lượt chạy, dữ liệu xuất chỉ thuộc ảnh đang dùng, quay lại vẫn xem đúng kết quả hiện tại | Sáu lượt xuất khớp chữ ký, tên tệp và kích thước PNG/JSON. Sau mỗi ảnh mới, nút tải chỉ có lại khi chạy | PASS |
| SYS-02 | Đổi tham số và chạy lại | Mẫu RGB, K=3 → 5, seed 0 → 12, Otsu hai lớp → ba lớp | Chỉnh, kiểm tra cảnh báo, chạy và đặt lại trong `test_kmeans_settings_require_rerun_before_download`, `test_otsu_mode_change_blocks_previous_downloads_until_rerun` | Hiện trạng thái thiết lập mới, ẩn tải kết quả cũ, chạy lại xuất đúng tham số | K và seed mới áp dụng sau nút chạy, khôi phục mặc định tạm ẩn tải. Multi-Otsu xuất đúng bộ tệp | PASS |
| SYS-03 | Lỗi đầu vào và phục hồi | CSV sai hàng, PNG sai CRC IEND, CSV hợp lệ 32×32 | AppTest luân phiên và Chrome nạp tệp PNG lỗi rồi CSV hợp lệ | Báo vị trí hoặc lỗi cấu trúc, bỏ kết quả cũ, tiếp tục với tệp hợp lệ | AppTest báo dòng 2. Chrome báo lỗi IEND và ẩn tab tải cũ, sau đó Otsu chạy lại với t=119 | PASS |
| SYS-04 | Xuất và đọc lại các tệp | Mẫu RGB nhiều hình và CSV 32×32 | Mở ZIP, PNG, RGBA, CSV, JSON trong `test_export_files_reopen_with_same_content`. Tải PNG/CSV thật từ Chrome | Tên, kích thước, nhãn, metadata và nội dung nhất quán, ZIP hợp lệ | ZIP có đủ 14 tệp và CRC hợp lệ. PNG tải từ Chrome 32×32. CSV có 32 hàng, mỗi hàng 32 nhãn và khớp mask PNG | PASS |
| SYS-05 | Lỗi ở bước tạo tệp và khả năng thử lại | `OSError` và `PermissionError` được mô phỏng | Chạy ba trang và phục hồi bộ mã hóa trong `test_export_failure_is_visible_and_recovers_on_retry` | Có lỗi rõ ràng, không cung cấp tệp cũ, thử lại được | Mọi biến thể đều hiện lỗi, ẩn nút tải và khôi phục đúng số nút khi thử lại | PASS |

### Nhóm 4 Kiểm thử kết quả

| ID | Mục đích | Dữ liệu đầu vào | Thao tác | Kỳ vọng | Thực tế | Trạng thái |
|---|---|---|---|---|---|---|
| OUT-01 | Kiểm tra kích thước, dtype và miền giá trị | Ma trận 2×3, PNG 1×1, JPEG thật và ảnh RGB lớn | Kiểm tra các đầu ra trong ALG-01, DATA-01, DATA-04, DATA-05 | Ảnh đầu ra cùng kích thước xử lý, `uint8`, mask 0/255 | Các cỡ 3×2, 1×1, 102×120 và 800×600 đều qua kiểm tra theo từng ca | PASS |
| OUT-02 | IoU/Dice có tham chiếu và chặn suy diễn khi thiếu | Mask tính tay, mẫu hình tròn và mẫu không có ground truth | Chạy `test_mask_scores_match_hand_computed_confusion_matrix`, `test_compare_page_with_ground_truth_reports_iou`, `test_compare_page_without_ground_truth_does_not_invent_scores` | Chỉ số khớp công thức, trang chỉ hiển thị điểm chất lượng khi có tham chiếu hợp lệ | Kết quả tính tay khớp. Trang hiển thị IoU với mask mẫu và bỏ cột IoU/Dice khi thiếu mask | PASS |
| OUT-03 | Màu khác nhau nhưng trùng mức xám | Hình đỏ 4×4 ở giữa nền xám trên ảnh 8×8, mask tạo từ tọa độ hình | Chạy K=2 RGB, Otsu và đối chiếu pixel bản đồ trong `test_equal_luma_object_scores_and_agreement_map_match_ground_truth` | K-means giữ hình đỏ, Otsu gặp ảnh xám đồng nhất, bản đồ đánh dấu đúng vùng chỉ K-means giữ | K-means IoU/Dice `1/1`, Otsu `0/0`, vùng khác biệt chỉ có đúng 16 pixel cam | PASS |
| OUT-04 | Đối chiếu trực quan trên trình duyệt | Mẫu `multi_shapes_color.png`, K=5 RGB, Otsu tự động | Chạy Chrome, mở **Đối chiếu mask** và **Đánh giá**, xem ảnh chụp kết quả | Có hai mask, bản đồ khác biệt, chú giải và điểm tham chiếu. Giải thích được vùng nào bị gộp | Chrome hiện đủ ba ảnh và chú giải. Hình tròn vàng tạo vùng cam vì K-means giữ được còn Otsu gộp vào lớp sáng của nền. Với ground truth, IoU/Dice K-means `1/1`, Otsu `0,7964/0,8867` | PASS |

### Kiểm tra thủ công còn lại

| ID | Mục đích | Dữ liệu đầu vào | Thao tác | Kỳ vọng | Thực tế | Trạng thái |
|---|---|---|---|---|---|---|
| MAN-01 | Khả năng đọc trên máy chiếu thật | Mẫu nhiều hình màu, máy và máy chiếu dùng khi bảo vệ | Mở ba ảnh, xem ngưỡng, chú giải và thử nút tải từ vị trí người xem | Đọc được nội dung trên màn chiếu, thao tác không cần đổi tỉ lệ hiển thị nhiều lần | Chưa kiểm tra thiết bị trình chiếu thực tế | NOT RUN |
| MAN-02 | Hộp thoại lưu và thông báo quyền ghi của trình duyệt | Một PNG đã xuất, thư mục người dùng được ghi và thư mục người dùng không được ghi | Chọn hai đích bằng hộp thoại tải xuống của trình duyệt trên máy demo | Đích có quyền lưu thành công, đích bị giới hạn hiện lỗi và cho chọn lại | Đã chạy phép thử OS ở DATA-06 và tải headless ở SYS-04. Chưa thử hộp thoại lưu trên máy demo | NOT RUN |

### Lỗi phát hiện và đã sửa

`DATA-03` phát hiện Pillow đọc được PNG có `IEND` thiếu CRC, sai CRC hoặc khai báo chiều dài khác 0 mặc dù đã gọi
`verify()`. Ba biến thể này FAIL trên mã cũ. `src/utils/image_io.py` bổ sung xác thực chiều dài và CRC của chunk
`IEND` tại vị trí Pillow dừng. Cả năm biến thể PNG hỏng đạt sau sửa. Chrome cũng báo lỗi khi nạp một PNG có CRC
`IEND` sai và tiếp tục được sau khi nạp CSV hợp lệ.

---

Các phần lịch sử: **(A)** test case trình diễn trên dữ liệu mẫu — kết quả thực tế lấy từ `scripts/run_experiments.py`
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
| SSE | 6 507 252 | **70 408 242** | 6 507 222 | 6 507 252 | 10 995 052 |
| Vòng lặp | 10 | 20 | 8 | 7 | 37 |

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

Lệnh: `python -X utf8 -m pytest -q --junitxml=docs/results/review_round2_pytest_20261106.xml`. Kết quả lượt hai ngày 06/11/2026: **295 passed**.
Bằng chứng lần tổng kiểm tra: `docs/results/review_round2_pytest_20261106.xml`; môi trường Python 3.11.7, NumPy 2.4.6, Streamlit 1.66.1.
Nhật ký ngày 05/11 trong phần D và các artifact cũ được giữ để đối chiếu lịch sử.

| Tệp | Số test | Nội dung chính |
|---|---|---|
| `tests/test_kmeans.py` | 64 | Nghiệm tính tay, trọng số, SSE, cụm rỗng, tham số; khoảng cách/tie-break; counts vượt int64 và từ chối số học tràn float64 |
| `tests/test_otsu.py` | 44 | Ví dụ giáo trình, brute force, ngưỡng đồng hạng rời nhau, lớp rỗng, histogram mất cân bằng, validation và cực tính |
| `tests/test_image_io.py` | 80 | PNG/JPEG/BMP, alpha/EXIF/tRNS gồm 16-bit; mask 16-bit giữ thang sáng; CSV chính xác; validation PNG/CSV/JSON trước xuất |
| `tests/test_integration.py` | 98 | Pipeline/11 mẫu/7 trang; upload/GT/cache; lỗi tài nguyên/dữ liệu; JSON ghi cách chuyển xám; luồng PNG 16-bit + GT trắng 16-bit cho đúng IoU |
| `tests/test_preprocessing.py` | 24 | Pixel sai, NaN/Inf, ảnh rỗng/sai chiều; kích thước/phương pháp sai; bảo toàn pixel nguyên và mask khi resize nearest |
| `tests/test_metrics.py` | 11 | Ma trận nhầm lẫn tính tay, IoU/Dice/accuracy, toàn nền hợp lệ; từ chối mask rỗng/sai chiều/không hữu hạn |
| `tests/test_postprocessing.py` | 9 | Đảo mask bool/0–1/0–255; từ chối mask sai đồng bộ; RGBA/preview/overlay cùng quy ước foreground |

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
| R07 | K=0, 2.5, -1, True hoặc seed/tol/max_iter sai kiểu | Gọi core | Từ chối trước tính toán | ValueError | PASS |
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

## E. Review và kiểm chứng — 06/11/2026

- Trước sửa: 183 test PASS; thêm 66 ca hồi quy, tổng sau sửa: **249 PASS**.
- Trình duyệt: **12 kiểm tra PASS**, thêm **6 tệp PNG/CSV tải thật được đối chiếu từng pixel**; không có ngoại lệ JavaScript quan sát được.
- Chạy lại đủ 11 mẫu, 7 giá trị K, 5 seed, 4 kích thước. Ngưỡng/số cụm/số vòng/IoU/Dice của 11 mẫu giữ nguyên so với artifact cũ.
- Chi tiết từng lỗi và giới hạn: [REVIEW_20261106.md](REVIEW_20261106.md).
- Bằng chứng: [JUnit](results/review_pytest_20261106.xml), [trình duyệt](results/review_browser_20261106.json), [thực nghiệm](results/review_experiments_20261106/experiment_results.md).

## F. Review tiếp — lượt hai ngày 06/11/2026

- Từ 249 lên **295 PASS**, thêm 46 ca; báo cáo: [REVIEW_ROUND2_20261106.md](REVIEW_ROUND2_20261106.md).
- PNG 16-bit có tRNS giữ tương phản pixel đục; GT 16-bit trắng giữ toàn foreground.
- Mask bool/0–1/0–255 giữ đối tượng khi resize/đảo; không chấp nhận mask sai ở cutout/overlay.
- K-means không trả counts âm hoặc NaN/Inf khi gặp tràn số; tệp xuất từ chối dữ liệu bị mất mát/chưa hợp lệ.
- JSON K-means lưu cách chuyển xám; GT tải lên được tái sử dụng đến khi đổi tệp/chạy mới.
- 12 kiểm tra trình duyệt và 7 tệp tải xuống đối chiếu dữ liệu; 11 mẫu giữ nguyên mọi số đo chất lượng đã đối chiếu.
- Bằng chứng: [JUnit](results/review_round2_pytest_20261106.xml), [trình duyệt](results/review_round2_browser_20261106.json), [11 mẫu](results/review_round2_samples_20261106.json).

## G. Sửa lỗi sau review độc lập (06/11/2026, sau commit 08d0b6a)

- Từ 295 lên **317 PASS**, thêm 22 ca hồi quy; 18 ca mới fail trên mã 08d0b6a và pass sau khi sửa.
- BUG-001: PNG màu 16-bit có tRNS bị từ chối kèm thông báo; PNG màu 16-bit không tRNS có ghi chú giảm về 8 bit.
- BUG-002: cụm đối tượng (K-means, So sánh), cực tính Otsu và tham số form được giữ khi chuyển trang rồi quay lại; JSON xuất khớp lựa chọn.
- BUG-003: bảng histogram hiển thị đủ 256 mức, kể cả h(g) = 0.
- BUG-004: lỗi CSV báo số dòng thực trong tệp (tính cả dòng trống và hàng trải nhiều dòng).
- RISK-002: hậu xử lý từ chối ảnh ngoài [0, 255], không nguyên, NaN hoặc sai số kênh thay vì ép kiểu quấn số.

## H. Sửa lỗi sau review commit c1e1b0f (06/11/2026)

- Từ 317 lên **330 PASS**, thêm 13 ca; 9 ca mới fail trên mã c1e1b0f và pass sau khi sửa.
- BUG-005: ground truth tải lên giữ nguyên tên nguồn và IoU/Dice khi chuyển trang rồi quay lại, kể cả khi tải trước lần chạy đầu;
  chỉ bỏ khi xóa tệp, bấm “Bỏ ground truth tải lên” hoặc đổi ảnh.
- RISK-004: ground truth RGBA/LA có alpha < 255, ảnh xám/bảng màu/16-bit có tRNS bị từ chối kèm thông báo;
  qua giao diện không báo IoU/Dice và không lặng lẽ dùng GT mẫu thay thế. Mask RGBA đục vẫn hợp lệ.
