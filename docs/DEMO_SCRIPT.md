# Kịch bản demo (5–7 phút)

**Chuẩn bị trước buổi bảo vệ (không tính giờ):**

1. Mở PowerShell tại thư mục dự án, chạy `.\.venv\Scripts\python.exe -m pytest -q` để chắc chắn mọi test PASS.
2. Chạy `.\.venv\Scripts\python.exe -m streamlit run app.py`, mở sẵn `http://localhost:8501`, phóng to trình duyệt 100%.
3. Chuẩn bị sẵn một ảnh chụp thật của nhóm (PNG/JPG) trên Desktop để demo phần tải ảnh.
4. Không cần Internet. Nếu cổng 8501 bận: thêm `--server.port 8502`.

| Thời gian | Thao tác | Lời thuyết minh |
|---|---|---|
| 0:00–0:30 | Trang **Tổng quan** | “Đề tài 3: phân đoạn ảnh, tách nền và đối tượng bằng K-means và Otsu. Hai thuật toán nhóm tự cài đặt bằng NumPy trong `src/algorithms`, không dùng OpenCV hay hàm có sẵn. Ứng dụng chạy cục bộ, nhận ảnh thật và CSV.” |
| 0:30–1:00 | Thanh bên: *Ảnh mẫu* → **Nhiều hình màu khác nhau (RGB)**. Chỉ vào thẻ thông tin | “Ảnh 320×240, 3 kênh. Ứng dụng kiểm tra định dạng, xoay theo EXIF, ghép kênh alpha, và giới hạn kích thước xử lý để không treo máy.” |
| 1:00–1:40 | Trang **K-means**, K = 2, bấm *Chạy K-means* | “Mỗi pixel là vector (R,G,B). Thuật toán gán pixel vào tâm gần nhất theo khoảng cách Euclid rồi cập nhật tâm bằng trung bình cụm — công thức 6.10 trong giáo trình. Với K = 2, hình tròn vàng sáng gần bằng nền nên bị gộp vào cụm nền — IoU chỉ 0.80.” |
| 1:40–2:20 | Đổi K = 4, chạy lại; rồi K = 5 | “K = 4: bốn cụm nhưng đỏ và vàng bị gộp thành một tâm màu cam. K = 5: đúng 5 màu — mỗi hình một cụm, SSE = 0. K-means không tự biết K; tab *Thống kê* cho thấy SSE theo vòng lặp, tâm và số pixel từng cụm.” |
| 2:20–2:50 | Tab **Tách nền (chọn đối tượng)** | “K-means không biết cụm nào là đối tượng. Mặc định là gợi ý: cụm chiếm nhiều viền ảnh nhất là nền, ở đây cụm 4. Ta có thể bỏ/thêm cụm — ví dụ chỉ giữ cụm 1 là hình tròn đỏ. Mask, ảnh tách nền và IoU so với ground truth cập nhật ngay, không chạy lại thuật toán.” |
| 2:50–3:40 | Trang **Otsu Thresholding**, bấm *Chạy Otsu*, tab *Histogram & phương sai* | “Otsu dựng histogram 256 mức, tính σ²_B(t) cho các ngưỡng hợp lệ và chọn một ngưỡng thực sự đạt cực đại — ở đây 148. Nếu có nhiều cực đại, chọn phần tử gần trung bình nhất; không lấy trung bình ngoài tập cực đại. Hàm f(g) ở giáo trình mục 3.1.6 biến đổi ra đúng σ²_B — trang Kiến thức có chứng minh và ví dụ trang 32.” |
| 3:40–4:00 | Chọn *Thủ công (đối chiếu)*, T = 200, chạy | “Ngưỡng thủ công 200 nằm giữa hình vàng (xám ≈ 187) và nền (≈ 234) nên tách đủ bốn hình, IoU = 1 — nhưng phải tự dò. Otsu tối ưu theo tiêu chí phương sai, không phải theo đối tượng ta muốn.” |
| 4:00–4:50 | Trang **So sánh thuật toán**, K = 5, bấm *Chạy K-means* | “Cùng ảnh: K-means IoU 1.0, Otsu 0.796. Bản đồ khác biệt cho thấy hình tròn vàng — mức xám gần nền nên Otsu hai lớp gộp nó vào nền. IoU chỉ được tính vì ảnh mẫu có ground truth; ảnh không có ground truth thì ứng dụng không tính.” |
| 4:50–5:20 | Trang K-means → tab **Tải xuống**: bấm *Ảnh tách nền RGBA* và *Ma trận nhãn cụm (CSV)*; mở tệp vừa tải | “Tải được PNG phân đoạn, mask, ảnh nền trong suốt, CSV nhãn, thống kê JSON — đúng kết quả đang hiển thị.” |
| 5:20–6:00 | Thanh bên chọn **Hình tròn có nhiễu**, chạy Otsu và K-means K = 2 | “Ảnh nhiễu: cả hai IoU 0.84 — vì chỉ dùng giá trị pixel, không dùng lân cận. Đây là hạn chế chung; hướng khắc phục là lọc trung vị trước.” |
| 6:00–6:30 | Tải `data/sample_csv/invalid_text.csv`; chỉ demo ảnh chụp thật khi nhóm đã có ảnh và thử trước | “Tệp CSV sai được báo lỗi rõ dòng, cột — ứng dụng không bị dừng. Bộ ảnh đã kiểm chứng trong báo cáo là ảnh tổng hợp; chưa có ảnh chụp thật của nhóm để nghiệm thu.” |
| 6:30–7:00 | Kết luận | “Otsu duyệt ngưỡng tối ưu theo phương sai, chỉ hai lớp xám nếu có ngưỡng hợp lệ. K-means dùng màu/nhiều vùng nhưng cần chọn K và foreground, phụ thuộc khởi tạo. 295 kiểm thử tự động đạt. Bộ ảnh đánh giá hiện tại là tổng hợp; ảnh chụp thật chỉ demo khi nhóm đã chuẩn bị và thử trước.” |

**Phương án dự phòng:** nếu trình duyệt lỗi, mở ảnh kết quả có sẵn trong `docs/figures/` và ảnh chụp giao diện trong
`docs/screenshots/`; nếu cần chứng minh thuật toán, chạy `python -m pytest tests/test_otsu.py -v`.
