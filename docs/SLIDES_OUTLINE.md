# Nội dung slide (15 slide, ~10 phút + demo)

Hình gợi ý lấy từ `docs/screenshots/` (ảnh chụp thật của ứng dụng) và `docs/figures/` (kết quả thực nghiệm).

| # | Tiêu đề | Nội dung chính | Hình |
|---|---|---|---|
| 1 | **Hệ thống phân đoạn ảnh bằng K-means và Otsu** | Bài tập lớn Xử lý ảnh INT13146 — Đề tài 3. Tên nhóm, thành viên, MSSV, **% đóng góp**. GV: Nguyễn Xuân Đức | Logo PTIT |
| 2 | Bài toán | Phân đoạn = tách ảnh thành nền và đối tượng; bước then chốt trước trích chọn đặc trưng, nhận dạng (giáo trình Ch.5). Yêu cầu: K-means + Otsu, tải/lưu ảnh thật và CSV, xem trực quan | Sơ đồ “ảnh → phân đoạn → mask” |
| 3 | Mục tiêu và ràng buộc | Tự cài đặt bằng NumPy; **không OpenCV**, không `sklearn`/`skimage`; test case tự chuẩn bị; chạy offline trên Windows | — |
| 4 | Tổng quan phân đoạn ảnh | Định nghĩa 5.1–5.4 (phủ kín, rời nhau, đồng nhất); ngưỡng (5.2) và phân cụm (6.2.4); cả hai là kỹ thuật toàn thể, chỉ dùng giá trị pixel | — |
| 5 | K-means: ý tưởng | Pixel = vector (xám hoặc RGB); cực tiểu J = ΣΣ‖x − Z_k‖² (6.9); Z_k = trung bình cụm (6.10) | Ảnh gốc / ảnh K-means / bản đồ nhãn (`02_kmeans_ket_qua.png`) |
| 6 | K-means: quy trình và cài đặt | Khởi tạo (K-means++/ngẫu nhiên) → gán nhãn → cập nhật tâm → hội tụ. Xử lý cụm rỗng, K > số màu, tính theo khối, **K-means có trọng số trên giá trị phân biệt** (tương đương, nhanh 3–20×). Người dùng chọn cụm đối tượng | `04_kmeans_thong_ke.png` |
| 7 | Otsu: ý tưởng | Histogram → σ²_B(t) = ω₀ω₁(μ₀ − μ₁)² → t* = argmax. **f(g) ở giáo trình 3.1.6 = σ²_B** (chứng minh 1 dòng) | Công thức + chứng minh |
| 8 | Otsu: quy trình và cài đặt | cumsum, O(N + 256), lớp rỗng, ngưỡng đồng hạng (floor trung bình), ảnh một mức xám; đối chiếu ví dụ giáo trình: t* = 1 ✔ | `05_otsu_histogram.png`, `07_kien_thuc_otsu.png` |
| 9 | Kiến trúc hệ thống | 4 tầng: UI (Streamlit) → pipeline → xử lý ảnh → thuật toán (chỉ NumPy). Chữ ký đầu vào chống tải nhầm kết quả cũ | Sơ đồ khối (REPORT 3.3) |
| 10 | Giao diện phần mềm | 7 trang; thanh bên dùng chung; 7 loại tệp tải xuống mỗi thuật toán | `01_tong_quan.png`, `09_tai_xuong.png` |
| 11 | Demo K-means | K = 2 → 4 → 5 trên ảnh nhiều hình màu; chọn cụm đối tượng | (demo trực tiếp) `03_kmeans_tach_nen.png` |
| 12 | Demo Otsu | Histogram, t* = 148, so với ngưỡng thủ công | (demo trực tiếp) |
| 13 | So sánh kết quả | Bảng IoU: hai mức 1.0 / 1.0; nhiều màu **1.0 / 0.80**; nhiều mức xám **1.0 / 0.67**; nhiễu 0.84 / 0.84; tương phản thấp 0.95 / 0.95. Thời gian Otsu < 1 ms | `06_so_sanh.png`, `docs/figures/multi_shapes_color.png`, `gray_levels_objects.png` |
| 14 | Kiểm thử và đánh giá | 11 test case offline + 13 tình huống đầu vào; **86 kiểm thử Pytest đều PASS** (thuật toán, I/O, tích hợp, giao diện). Phụ thuộc K (khuỷu tại K = 5) và seed (SSE chênh 10.8×) | `docs/figures/noisy_circle.png` |
| 15 | Kết luận | Otsu: nhanh, không tham số, tối ưu toàn cục, chỉ 2 lớp/xám. K-means: dùng màu, nhiều vùng, cần K, cục bộ, cần chọn cụm. Hạn chế chung: không dùng lân cận. Hướng phát triển: lọc trung vị, hình thái, Otsu nhiều ngưỡng/cục bộ, đặc trưng (x, y) | — |
