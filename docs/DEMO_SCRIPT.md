# Kịch bản demo (5–7 phút)

**Chuẩn bị trước buổi bảo vệ**

1. Mở PowerShell tại thư mục dự án, chạy `.\.venv\Scripts\python.exe -m pytest -q` để chắc chắn mọi test PASS.
2. Chạy `.\.venv\Scripts\python.exe -m streamlit run app.py`, mở sẵn `http://localhost:8501`, phóng to trình duyệt 100%.
3. Mở [bộ slide](SLIDES.pptx). Chuẩn bị một ảnh chụp thật PNG/JPG và CSV mẫu `data/sample_csv/csv_rectangle_32x32.csv`.
4. Không cần Internet. Nếu cổng 8501 bận: thêm `--server.port 8502`.

**Luồng rút gọn (3–4 phút)**

1. Chọn **Nhiều hình màu khác nhau (RGB)**. Vào **K-means**, chọn K=5 và bấm **Chạy K-means**. Chỉ vào ảnh gốc, ảnh tô theo tâm cụm và số cụm thực tế.
2. Vào **Phân ngưỡng Otsu**, chọn **Tự động** và **Đối tượng tối**. Chạy, chỉ ngưỡng trên histogram, rồi giải thích quy ước trắng là đối tượng trên mask.
3. Vào **So sánh thuật toán**, đặt K=5, bấm chạy và mở **Đối chiếu mask** hoặc **Đánh giá**. Trên máy chiếu hẹp, thu thanh bên sau khi chọn ảnh để mở rộng vùng kết quả.
4. Tải ZIP trên trang So sánh và mở một PNG kết quả. Nạp `csv_rectangle_32x32.csv` rồi chạy lại để minh họa đầu vào CSV. Nếu còn thời gian, thử `invalid_text.csv` để xem thông báo dòng và cột.

**Luồng đầy đủ**

| Thời gian | Thao tác | Lời thuyết minh |
|---|---|---|
| 0:00–0:30 | Trang **Tổng quan** | “Đề tài 3: phân đoạn ảnh, tách nền và đối tượng bằng K-means và Otsu. Hai thuật toán nhóm tự cài đặt bằng NumPy trong `src/algorithms`, không dùng OpenCV hay hàm có sẵn. Ứng dụng chạy cục bộ, nhận ảnh thật và CSV.” |
| 0:30–1:00 | Thanh bên: *Ảnh mẫu* → **Nhiều hình màu khác nhau (RGB)**. Chỉ vào thẻ thông tin | “Ảnh 320×240, 3 kênh. Ứng dụng kiểm tra định dạng, xoay theo EXIF, ghép kênh alpha, và giới hạn kích thước xử lý để không treo máy.” |
| 1:00–1:40 | Trang **K-means**, K = 2, bấm *Chạy K-means* | “Mỗi pixel là vector (R,G,B). Thuật toán gán pixel vào tâm gần nhất theo khoảng cách Euclid rồi cập nhật tâm bằng trung bình cụm — công thức 6.10 trong giáo trình. Với K = 2, hình tròn vàng sáng gần bằng nền nên bị gộp vào cụm nền — IoU chỉ 0.80.” |
| 1:40–2:20 | Đổi K = 4, chạy lại; rồi K = 5 | “K = 4: bốn cụm nhưng đỏ và vàng bị gộp thành một tâm màu cam. K = 5: đúng 5 màu — mỗi hình một cụm, SSE = 0. K-means không tự biết K; tab *Thống kê* cho thấy SSE theo vòng lặp, tâm và số pixel từng cụm.” |
| 2:20–2:50 | Tab **Tách nền (chọn đối tượng)** | “Gợi ý nền so tỷ lệ của từng cụm ở viền và vùng trung tâm. Các cụm đối tượng có thể chọn lại. Mask và IoU cập nhật theo lựa chọn mà không phân cụm lại.” |
| 2:50–3:40 | Trang **Otsu Thresholding**, bấm *Chạy Otsu*, tab *Histogram & phương sai* | “Otsu dựng histogram 256 mức, tính σ²_B(t) cho các ngưỡng hợp lệ và chọn một ngưỡng thực sự đạt cực đại — ở đây 148. Nếu có nhiều cực đại, chọn phần tử gần trung bình nhất; không lấy trung bình ngoài tập cực đại. Hàm f(g) ở giáo trình mục 3.1.6 biến đổi ra đúng σ²_B — trang Kiến thức có chứng minh và ví dụ trang 32.” |
| 3:40–4:00 | Chọn *Thủ công (đối chiếu)*, T = 200, chạy | “Ngưỡng thủ công 200 nằm giữa hình vàng (xám ≈ 187) và nền (≈ 234) nên tách đủ bốn hình, IoU = 1 — nhưng phải tự dò. Otsu tối ưu theo tiêu chí phương sai, không phải theo đối tượng ta muốn.” |
| 4:00–4:50 | Trang **So sánh thuật toán**, K = 5, bấm *Chạy K-means & Otsu*. Mở **Đối chiếu mask**, sau đó **Đánh giá** | “K-means đạt IoU 1,0000, Otsu 0,7964 trên mẫu này. Bản đồ khác biệt cho thấy Otsu gộp hình tròn vàng vào nền sáng. IoU được tính với ground truth của ảnh mẫu.” |
| 4:50–5:20 | Trang K-means → tab **Tải xuống**: bấm *Ảnh tách nền RGBA* và *Ma trận nhãn cụm (CSV)*; mở tệp vừa tải | “Tải được PNG phân đoạn, mask, ảnh nền trong suốt, CSV nhãn, thống kê JSON — đúng kết quả đang hiển thị.” |
| 5:20–6:00 | Thanh bên chọn **Hình tròn có nhiễu**, chạy Otsu và K-means K = 2 | “Ảnh nhiễu: cả hai IoU 0.84 — vì chỉ dùng giá trị pixel, không dùng lân cận. Đây là hạn chế chung; hướng khắc phục là lọc trung vị trước.” |
| 6:00–6:30 | Tải `data/sample_csv/invalid_text.csv`, rồi tải `csv_rectangle_32x32.csv` và chạy lại | “Ứng dụng báo dòng và cột sai. Khi tải CSV hợp lệ, hai thuật toán tiếp tục chạy với ảnh mới.” |
| 6:30–7:00 | Kết luận | “K-means phân cụm theo màu hoặc mức xám. Otsu tìm ngưỡng toàn cục từ histogram. Bộ Pytest kiểm tra thuật toán, I/O và giao diện. Báo cáo trình bày riêng kết quả 11 mẫu tổng hợp và 24 ảnh test Oxford VGG iSeg.” |

**Phương án dự phòng:** nếu trình duyệt lỗi, mở ảnh kết quả có sẵn trong `docs/figures/` và ảnh chụp giao diện trong
`docs/screenshots/`; nếu cần chứng minh thuật toán, chạy `python -m pytest tests/test_otsu.py -v`.
