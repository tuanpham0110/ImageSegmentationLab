# IMAGE SEGMENTATION LAB

**Hệ thống phân đoạn ảnh bằng K-means và Thresholding (Otsu)** — Bài tập lớn môn Xử lý ảnh (INT13146),
Đề tài 3, Học viện Công nghệ Bưu chính Viễn thông.

Ứng dụng Streamlit chạy cục bộ trên Windows: tải ảnh thật (PNG/JPG/BMP) hoặc ma trận CSV, phân đoạn (tách nền và
đối tượng) bằng **K-means** và **Otsu** tự cài đặt bằng NumPy, xem kết quả trực quan và tải kết quả về máy.

- **Không dùng OpenCV**, không dùng `sklearn.cluster.KMeans` hay `skimage.filters.threshold_otsu`.
- K-means và Otsu nằm trong [src/algorithms/](src/algorithms/), tách biệt hoàn toàn với giao diện.
- 86 kiểm thử tự động (Pytest), gồm cả kiểm thử giao diện bằng Streamlit `AppTest`.
- Không cần Internet, không có API, không có cơ sở dữ liệu.

---

## 1. Cài đặt và chạy (Windows PowerShell)

Yêu cầu: **Python 3.10 – 3.12** (đã kiểm thử với Python 3.11.7 trên Windows 10). Mở PowerShell tại thư mục dự án:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Trình duyệt tự mở `http://localhost:8501`. Dừng ứng dụng bằng `Ctrl + C`.

**Nếu PowerShell chặn `Activate.ps1`** (lỗi *running scripts is disabled*), không cần kích hoạt môi trường — gọi thẳng
Python trong `.venv`:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Hoặc chỉ cho phép trong cửa sổ hiện tại: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.

### Chạy kiểm thử

```powershell
python -m pytest -v
# hoặc, không kích hoạt môi trường:
.\.venv\Scripts\python.exe -m pytest -v
```

### Sinh lại dữ liệu mẫu và chạy lại thực nghiệm (tuỳ chọn)

Dữ liệu mẫu đã có sẵn trong `data/`. Nếu muốn tạo lại (kết quả giống hệt vì dùng seed cố định):

```powershell
python scripts\generate_samples.py      # ảnh mẫu, ground truth, CSV -> data/
python scripts\run_experiments.py       # số liệu thật -> docs/results/, hình -> docs/figures/
```

---

## 2. Chức năng

| Trang | Nội dung |
|---|---|
| **Tổng quan** | Giới thiệu đề tài, quy trình xử lý, cách bắt đầu |
| **K-means** | Chọn K (2–8), đặc trưng RGB/mức xám, khởi tạo K-means++/ngẫu nhiên, số vòng lặp, ngưỡng hội tụ, seed. Xem ảnh phân cụm, bản đồ nhãn, **chọn cụm làm đối tượng** để tạo mask và ảnh tách nền, bảng tâm cụm/số pixel/tỷ lệ, biểu đồ tỷ lệ cụm và SSE theo vòng lặp |
| **Otsu Thresholding** | Histogram 256 mức với đường ngưỡng t\*, đồ thị σ²_B(t), ảnh xám, ảnh nhị phân, mask, ảnh tách nền; chế độ ngưỡng thủ công để đối chiếu; chọn đối tượng là lớp sáng/tối/gợi ý theo viền ảnh |
| **So sánh thuật toán** | Cùng một ảnh: ảnh gốc / K-means / Otsu; so sánh hai mask, bản đồ khác biệt; IoU/Dice **chỉ khi có ground truth**; bảng so sánh bản chất, tham số, thời gian, ưu nhược điểm |
| **Thư viện ảnh mẫu** | 11 tình huống kiểm thử tổng hợp chạy offline, có ground truth khi phù hợp |
| **Kiến thức thuật toán** | Công thức, giả mã, chứng minh f(g) ở giáo trình mục 3.1.6 chính là σ²_B của Otsu, minh hoạ từng vòng lặp K-means trên ảnh 10×10, đối chiếu ví dụ giáo trình |
| **Hướng dẫn** | Thao tác, định dạng CSV, danh sách tệp tải xuống |

**Đầu vào được xử lý:** ảnh xám, RGB, có kênh alpha (ghép lên nền trắng), có thẻ EXIF hướng chụp (xoay đúng chiều),
ảnh 16-bit (chuẩn hoá về 0–255), ảnh RGB có R=G=B (xử lý như ảnh xám). Giới hạn: tệp ≤ 30 MB, ≤ 40 triệu pixel;
ảnh lớn được thu nhỏ theo **“Giới hạn cạnh dài khi xử lý”** (mặc định 800 px) và kích thước thật được ghi rõ trên trang.

**Tải xuống:** ảnh phân đoạn, bản đồ nhãn, mask nhị phân, ảnh tách nền RGBA (nền trong suốt) dạng PNG; ma trận nhãn
cụm, ma trận mask 0/1, bảng thống kê cụm, histogram dạng CSV; thông số và kết quả dạng JSON. Tệp được tạo từ đúng kết
quả đang hiển thị; khi đổi ảnh đầu vào, kết quả cũ bị ẩn để không tải nhầm.

## 3. Định dạng CSV đầu vào

- Ma trận **2 chiều** mức xám: mỗi dòng là một hàng ảnh, mỗi giá trị là **số nguyên trong [0, 255]**.
- Phân cách bằng dấu phẩy `,` (cũng nhận `;`, tab hoặc khoảng trắng). **Không có dòng tiêu đề.** Mọi dòng cùng số cột.
- Tệp UTF-8 (có hoặc không có BOM). Tối đa 4 000 000 phần tử.

```
0,0,255,255
0,128,255,255
0,0,0,255
```

Lỗi được báo bằng tiếng Việt kèm vị trí dòng/cột: ký tự không phải số, số thập phân, giá trị ngoài [0, 255],
ô trống, dòng thiếu/thừa cột. Ví dụ tệp lỗi để demo: `data/sample_csv/invalid_*.csv`.

## 4. Cấu trúc dự án

```
app.py                         Điểm khởi chạy Streamlit (điều hướng 7 trang)
requirements.txt  pytest.ini  .streamlit/config.toml
src/algorithms/kmeans.py       K-means tự cài đặt (Lloyd, K-means++, cụm rỗng, có trọng số, tính theo khối)
src/algorithms/otsu.py         Otsu tự cài đặt từ histogram (σ²_B, ngưỡng đồng hạng, lớp rỗng)
src/processing/preprocessing.py  Chuyển xám (BT.601 / trung bình), giới hạn kích thước
src/processing/postprocessing.py Mask từ cụm, ảnh tách nền RGBA, tô màu nhãn, gợi ý nền theo viền
src/processing/pipeline.py     Luồng xử lý dùng chung cho giao diện, thực nghiệm và kiểm thử; tạo tệp xuất
src/utils/image_io.py          Đọc/kiểm tra ảnh và CSV, ghi PNG/CSV/JSON
src/utils/metrics.py           IoU, Dice, độ chính xác pixel (chỉ dùng với ground truth)
src/utils/visualization.py     Biểu đồ Matplotlib (histogram, σ²_B, tỷ lệ cụm, SSE)
src/ui/                        Thanh bên, các trang, thành phần giao diện, quản lý trạng thái
data/sample_images/ data/sample_csv/ data/ground_truth/ data/samples.json   Dữ liệu demo offline
scripts/generate_samples.py    Sinh dữ liệu mẫu
scripts/run_experiments.py     Chạy thực nghiệm, ghi số liệu thật
tests/                         test_kmeans.py, test_otsu.py, test_image_io.py, test_integration.py
docs/                          Báo cáo, thuật toán, test case, kịch bản demo, slide, câu hỏi phản biện,
                               kết quả thực nghiệm (results/), hình (figures/), ảnh chụp giao diện (screenshots/)
```

## 5. Tài liệu

| Tệp | Nội dung |
|---|---|
| [docs/REPORT.md](docs/REPORT.md) | Báo cáo bài tập lớn (6 chương), chuyển sang Word được |
| [docs/ALGORITHMS.md](docs/ALGORITHMS.md) | Lý thuyết, công thức, giả mã, quyết định cài đặt |
| [docs/TEST_CASES.md](docs/TEST_CASES.md) | Test case thủ công với kết quả thực tế + danh sách kiểm thử tự động |
| [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md) | Kịch bản demo 5–7 phút |
| [docs/SLIDES_OUTLINE.md](docs/SLIDES_OUTLINE.md) | Nội dung 15 slide |
| [docs/QUESTIONS_AND_ANSWERS.md](docs/QUESTIONS_AND_ANSWERS.md) | Câu hỏi phản biện và trả lời |
| [docs/results/experiment_results.md](docs/results/experiment_results.md) | Số liệu thực nghiệm sinh tự động |

## 6. Xử lý sự cố

| Hiện tượng | Cách xử lý |
|---|---|
| `py` không có | Cài Python từ python.org (tích *Add to PATH*), hoặc dùng `python -m venv .venv` |
| `streamlit` không được nhận ra | Dùng `python -m streamlit run app.py` (không gọi `streamlit` trực tiếp) |
| Cổng 8501 bận | `python -m streamlit run app.py --server.port 8502` |
| Chạy chậm với ảnh lớn | Giảm “Giới hạn cạnh dài khi xử lý” ở thanh bên hoặc giảm K |
| Kết quả biến mất sau khi đổi ảnh | Đúng thiết kế (tránh tải nhầm kết quả cũ) — bấm chạy lại |

## 7. Checklist trước khi nộp

- [ ] `python -m pytest -v` — tất cả PASS trên máy nộp bài
- [ ] `python -m streamlit run app.py` mở được 7 trang, chạy được K-means và Otsu trên ảnh mẫu và ảnh thật
- [ ] Đã thử tải ảnh chụp thật của nhóm và ghi nhận kết quả vào Chương 5 của báo cáo (mục còn trống)
- [ ] Điền tên thành viên, MSSV và **% đóng góp** (yêu cầu của đề) trong `docs/REPORT.md` và slide bìa
- [ ] Chuyển `docs/REPORT.md` sang Word, chèn hình từ `docs/figures/` và `docs/screenshots/`
- [ ] Xoá thư mục `.venv/` khi nén mã nguồn để nộp (người chấm cài lại bằng `requirements.txt`)
