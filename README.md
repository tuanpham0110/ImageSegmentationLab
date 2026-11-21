# Image Segmentation Studio

**Đề tài 3 · Xử lý ảnh (INT13146) · Học viện Công nghệ Bưu chính Viễn thông**

Ứng dụng Streamlit phân đoạn ảnh và tách nền bằng **K-means** và **Otsu** tự cài đặt bằng NumPy, không dùng OpenCV. Ứng dụng nhận ảnh hoặc ma trận CSV, hiển thị ảnh gốc cùng ảnh kết quả và cho tải mask, ảnh tách nền và số liệu.

## 1. Cài đặt và chạy

Cần **Python 3.11 64-bit**, `venv`, `pip` và trình duyệt. Cài thư viện đúng phiên bản trong `requirements.txt`. Các thư viện chính gồm NumPy, Pillow, Streamlit, Matplotlib và pandas. `requirements-dev.txt` chứa thêm công cụ kiểm thử.

Giải nén đầy đủ, rồi mở terminal tại thư mục chứa `app.py`. Thư mục này cũng chứa `data/` và `.streamlit/`. Dùng lệnh phù hợp với hệ điều hành.

### Windows PowerShell

Kiểm tra phiên bản Python của dòng đầu là 3.11.x.

```powershell
py -3.11 -V
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

Nếu `py` không tìm thấy Python nhưng `python -V` trả về 3.11.x, tạo môi trường bằng `python -m venv .venv`. Nếu cả hai lệnh đều trỏ sai phiên bản, gọi trực tiếp tệp `python.exe` của bản CPython 3.11 64-bit đã cài. Đoạn lệnh dưới dùng vị trí cài mặc định cho người dùng hiện tại. Thay đường dẫn nếu Python nằm ở nơi khác.

```powershell
$python311 = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python311\python.exe'
& $python311 -V
& $python311 -m venv .venv
```

### Ubuntu hoặc Linux có Python 3.11

```bash
python3.11 -m venv .venv
./.venv/bin/python -m pip install --only-binary=:all: -r requirements.txt
./.venv/bin/python -m streamlit run app.py --server.address 127.0.0.1
```

Mở [http://127.0.0.1:8501](http://127.0.0.1:8501). Dừng máy chủ bằng `Ctrl+C` trong terminal. Các lệnh gọi Python trực tiếp trong môi trường ảo nên có thể chạy ngay bằng PowerShell.

### Cài và chạy ngoại tuyến

Các ảnh mẫu, mask tham chiếu và tệp CSV cần cho demo nằm trong `data/`. Sau khi cài thư viện, ứng dụng phân đoạn và xuất tệp qua máy chủ cục bộ, dùng được khi máy ngắt Internet.

Trên Windows x64 đã cài CPython 3.11, dùng gói `PTIT-DeTai3-Wheels-Windows-x64-Python311.zip` đi kèm. Giải nén hai ZIP vào cùng một thư mục cha để `wheelhouse-win-amd64-py311/` nằm cạnh thư mục dự án. Mở PowerShell trong thư mục dự án, tạo `.venv` theo hướng dẫn trên rồi cài và chạy.

```powershell
.\.venv\Scripts\python.exe -m pip install --no-index --find-links '..\wheelhouse-win-amd64-py311' --only-binary=:all: -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

Gói Windows cũng có thư viện kiểm thử. Để chạy test ngoại tuyến, thay `requirements.txt` bằng `requirements-dev.txt` trong lệnh cài trên, rồi chạy các lệnh kiểm tra ở mục 4. Trên nền tảng khác, chuẩn bị một thư mục wheel từ máy có mạng với **cùng hệ điều hành, kiến trúc và Python 3.11**. Dùng `python3.11 -m pip download --only-binary=:all: --dest wheelhouse -r requirements.txt` trên Linux rồi trỏ `--find-links` tới thư mục đã chép sang máy đích.

## 2. Nhập ảnh và chạy demo

Chọn dữ liệu ở thanh bên.

- **Ảnh mẫu** mở 11 tình huống tổng hợp có sẵn. Có thể vào **Thư viện ảnh mẫu** và bấm **Dùng ảnh này**.
- **Tải tệp lên** nhận PNG, JPG/JPEG, BMP và CSV, tối đa 30 MiB mỗi tệp. Ảnh giải mã giới hạn 40 triệu pixel. Mục **Tiền xử lý ảnh** cho chọn cách chuyển xám, lọc trung vị và cạnh dài tối đa. Mặc định ảnh có cạnh dài vượt 800 px được thu nhỏ giữ tỷ lệ. Giới hạn xử lý cao nhất trên giao diện là 2.048 px.

### K-means

1. Chọn mẫu **Nhiều hình màu khác nhau (RGB)**. Mở **K-means**, đặt **Số cụm K** bằng 5, chọn **Màu RGB** và bấm **Chạy K-means**.
2. Xem ảnh tâm cụm trong **Kết quả phân cụm**. Mở **Tách nền (chọn đối tượng)** để chọn cụm cần giữ hoặc đảo vùng đối tượng. Mở **Thống kê** để xem số pixel, tâm cụm và SSE.
3. Với ảnh riêng, chọn K từ 1 đến 16. Ảnh màu có thể dùng RGB, Lab hoặc mức xám. **Thiết lập nâng cao** có cách khởi tạo, seed, ngưỡng hội tụ và số vòng tối đa.

### Otsu

1. Chọn mẫu **Hình tròn đen trên nền trắng**. Mở **Phân ngưỡng Otsu**, chọn **Otsu · 2 lớp**, **Tự động** và **Đối tượng tối**, rồi bấm **Chạy Otsu**. Đối với vật thể sáng trên nền tối, chọn **Đối tượng sáng**.
2. Xem ảnh nhị phân, mask và **Histogram & phương sai**. Ca hình tròn có ngưỡng 127 theo quy tắc chọn ngưỡng đồng hạng. Ở chế độ **Thủ công**, chọn T trong **Ngưỡng thủ công & hậu xử lý**. **Multi-Otsu · 3 lớp** tự tìm hai ngưỡng.
3. Sau khi đổi thiết lập Otsu, bấm **Chạy Otsu** để cập nhật kết quả tải xuống.

Trang **So sánh thuật toán** chạy cả hai phương pháp trên cùng đầu vào. Mục **Đánh giá** tính IoU/Dice với mask tham chiếu hợp lệ của ảnh mẫu hoặc mask PNG, BMP hoặc CSV tải lên. Mask cần khớp kích thước ảnh gốc sau khi áp dụng EXIF hoặc kích thước xử lý. Dùng một cặp giá trị `0/1` hoặc `0/255`. Riêng PNG 16 bit dùng `0/1` hoặc `0/65535` ở độ sâu gốc. Số `0` là nền và giá trị dương là đối tượng. Mask ảnh phải đục, có ba kênh bằng nhau nếu lưu dạng RGB. Khi mask mẫu thiếu hoặc lỗi, vẫn có thể phân đoạn, tải kết quả và dùng mask hợp lệ khác để đánh giá.

### Định dạng CSV

Dùng ma trận mức xám hai chiều, mã hóa UTF-8 có hoặc không BOM, tối đa 4.194.304 ô. Mỗi dòng dữ liệu là một hàng ảnh, mỗi cột là một pixel. Các hàng có cùng số cột và mỗi ô chứa một giá trị nguyên từ 0 đến 255. Bỏ qua dòng trống. Dùng một kiểu dấu phân cách cho cả tệp, chọn dấu phẩy, dấu chấm phẩy, tab hoặc khoảng trắng. Tệp ma trận không có hàng tiêu đề.

```csv
0,0,255,255
0,128,255,255
0,0,0,255
```

Để thử nhập CSV, dùng [ma trận 32×32](data/sample_csv/csv_rectangle_32x32.csv). [CSV có ô không hợp lệ](data/sample_csv/invalid_text.csv) dùng để kiểm tra thông báo vị trí lỗi.

### Lưu kết quả

Vào tab **Tải xuống** sau khi chạy. Các trang thuật toán xuất PNG ảnh phân đoạn, mask 0/255 hoặc ảnh RGBA nền trong suốt, cùng CSV và JSON thống kê. Trang **So sánh thuật toán** có gói ZIP chứa kết quả của cả hai phương pháp. Ứng dụng tạo tệp trong bộ nhớ với tên cố định theo loại kết quả. Trình duyệt chọn đường dẫn lưu, kiểm tra quyền ghi và xử lý trùng tên. Nếu trình duyệt báo lỗi lưu, chọn thư mục có quyền ghi. Tệp nguồn được giữ nguyên.

`kmeans_labels.csv`, `multiotsu_labels.csv` và `otsu_binary_matrix.csv` là ma trận có thể đọc lại. Hai tệp nhãn giữ chỉ số lớp và `otsu_binary_matrix.csv` giữ mask foreground 0/1 sau khi chọn lớp. Các CSV thống kê cụm hoặc histogram có tiêu đề và dùng để đọc dưới dạng bảng.

Ảnh nguồn sau giải mã là ma trận `uint8` mức xám hoặc RGB. Ảnh gốc xem cạnh kết quả dùng cùng kích thước xử lý, có ghi rõ nếu đã thu nhỏ. PNG mask giữ một kênh, ảnh màu dùng RGB và cutout trong suốt dùng RGBA. Nếu muốn ánh xạ nhãn và mask về độ phân giải nguồn, chọn **Thêm PNG kích thước gốc** ở từng trang thuật toán. CSV và ZIP so sánh dùng kích thước xử lý. Khi đổi ảnh đầu vào, chạy lại để có kết quả tương ứng.

## 3. Cấu trúc dự án

```text
app.py                    Điểm khởi chạy Streamlit
.streamlit/config.toml    Cấu hình giao diện và máy chủ
requirements*.in/.txt     Khai báo và khóa phiên bản thư viện
src/algorithms/           K-means và Otsu
src/processing/           Tiền xử lý, tạo mask và xuất kết quả
src/ui/                   Các trang, CSS và tài nguyên giao diện
src/utils/                Đọc/ghi tệp, chỉ số và biểu đồ
data/                     Manifest, ảnh mẫu, CSV và mask tham chiếu
tests/                    Test thuật toán, I/O và tích hợp Streamlit
scripts/                  Tạo mẫu, chạy thực nghiệm và đóng gói
docs/                     Tài liệu thuật toán, test case và demo
```

Xem thêm [thuật toán](docs/ALGORITHMS.md), [test case](docs/TEST_CASES.md) và [kịch bản demo](docs/DEMO_SCRIPT.md).

Khi cần đóng gói lại, chạy `python scripts/package_submission.py --output outputs/PTIT-DeTai3-ImageSegmentation.zip` với Python 3.11. Tệp ZIP giữ mã nguồn, cấu hình, tệp khóa phiên bản thư viện, test, dữ liệu mẫu và tài liệu. Môi trường ảo, cache, thư mục `outputs/` và cache dữ liệu nghiên cứu được bỏ qua khi tạo gói.

## 4. Chạy test

Trên Windows, chạy các lệnh sau khi tạo `.venv`.

```powershell
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m ruff check app.py src scripts tests
.\.venv\Scripts\python.exe -m pytest -q
```

Trên Linux, thay `.\.venv\Scripts\python.exe` bằng `./.venv/bin/python`. Pytest có cả test thuật toán, kiểm tra đọc/ghi tệp và kiểm thử giao diện bằng Streamlit `AppTest`. CI được cấu hình cho Windows và Ubuntu.

## 5. Hạn chế và xử lý sự cố

- **Chất lượng tách nền.** Gợi ý theo viền ảnh có thể chọn sai đối tượng. Otsu toàn cục kém hiệu quả khi nền và vật thể có mức xám chồng lấn. Ảnh xám đồng nhất không có ngưỡng Otsu hợp lệ. K-means phụ thuộc K, khởi tạo và giới hạn vòng lặp.
- **Ảnh lớn.** Xử lý diễn ra đồng bộ, nên K lớn hoặc ảnh nhiều màu có thể cần chờ. Giảm **Cạnh dài tối đa (pixel)** trong thanh bên hoặc giảm K khi cần demo nhanh. PNG kích thước gốc ánh xạ kết quả đã xử lý, nên các chi tiết mất khi thu nhỏ không được khôi phục.
- **Cổng bận hoặc trình duyệt chưa mở.** Bổ sung `--server.port 8502` vào lệnh chạy và mở `http://127.0.0.1:8502`. Nếu Streamlit báo thiếu package, cài lại theo lockfile bằng chính Python trong `.venv`.
- **Thiếu ảnh mẫu.** Kiểm tra đã giải nén cả thư mục `data/` cạnh `app.py`. Tệp `data/samples.json` và các ảnh, CSV, mask đi kèm phải giữ nguyên bố cục thư mục.
- **Dữ liệu nghiên cứu.** Cache ảnh Oxford và kết quả benchmark phục vụ thực nghiệm riêng. Các ca demo đi kèm sử dụng dữ liệu tổng hợp.
