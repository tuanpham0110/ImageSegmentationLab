# Nghiệm thu tích hợp · Giai đoạn 8

Kiểm tra ngày 21/11/2026 trên Windows, Python 3.11.7, Streamlit 1.66.1 và Chrome 157 ở khung nhìn 1440×1000. Playwright thao tác qua giao diện đang chạy, lưu tệp do trình duyệt tải xuống. Trình kiểm tra Python mở lại PNG, CSV, JSON và ZIP, đối chiếu pixel, nhãn, ngưỡng, kích thước và metadata. [Kết quả từng luồng, tên tệp và SHA-256](results/stage8_acceptance.json).

`PASS` nghĩa là đã thực thi và đạt kỳ vọng. `FAIL` là đã thực thi và thấy sai lệch. `BLOCKED` áp dụng khi thiếu điều kiện để chạy một ca đã lên lịch. `NOT TESTED` là ca nằm ngoài phạm vi chạy này.

| ID | Luồng và kỳ vọng | Trạng thái | Bằng chứng |
| --- | --- | --- | --- |
| A | Từ Tổng quan, mở hộp chọn tệp, nạp `multi_shapes_color.png`, vào K-means, đặt K=5, chỉnh tham số, chạy, xem và lưu | PASS | Ảnh gốc và ảnh tâm cụm cùng 320×240. K=5, khởi tạo ngẫu nhiên, seed 9, tối đa 8 vòng, tolerance 0. SSE bằng 0 trên ảnh 5 màu. Cả 7 nút tải xuất tệp hợp lệ. Đổi K ẩn tải xuống của lần cũ. Sau khi chạy K=4 rồi chạy lại K=5, PNG và CSV nhãn trùng từng byte với lần đầu. Nút khôi phục đưa K về 3, seed về 0 và số vòng tối đa về 100. |
| B | Nạp `circle_black_on_white.png`, vào Otsu, chọn đối tượng tối, chạy tự động, xem ngưỡng, ảnh nhị phân và lưu | PASS | Ngưỡng tối ưu 127. Ảnh nhị phân 256×256 đúng quy tắc `I > 127`; mask lớp tối trùng mask tham chiếu từng pixel. Histogram có đủ 256 dòng. Cả 7 tệp tải xuống mở lại được. Chế độ thủ công chạy với T=0 và ẩn tệp cũ cho đến khi chạy lại. Khôi phục tắt thanh ngưỡng thủ công. |
| C | Tải CSV 5 hàng × 6 cột, phân đoạn bằng cả hai thuật toán, xuất và nhập lại | PASS | Cả hai trang hiển thị 6×5 px. Otsu cho ngưỡng 1, PNG và ma trận 0/1 khớp kỳ vọng. CSV nhãn K-means và ảnh xám đối chiếu đúng với các tâm xuất ra. Nhập lại CSV 0/1 bằng trình duyệt, chạy với ngưỡng 0 và kết quả cũ được dọn khi đổi đầu vào. |
| D | Đang có kết quả, tải CSV có ô sai, PNG hỏng và CSV rỗng, sau đó chọn lại ảnh tốt trong cùng phiên | PASS | CSV sai báo dòng 2, cột 2. PNG hỏng và CSV rỗng có thông báo riêng. Nút chạy và tải kết quả cũ biến mất. Chọn lại `bright_rect_on_dark.png`, Otsu cho ngưỡng 124, lưu được ảnh và JSON. Nút xóa tệp đưa ứng dụng về trạng thái chờ. |
| E | Multi-Otsu, điều khiển theo chế độ, hậu xử lý, histogram và tải xuống | PASS | Thanh ngưỡng thủ công bị vô hiệu hóa trong chế độ ba lớp. Áp dụng phép mở 5×5, hiển thị mask trước và sau hậu xử lý. CSV nhãn khớp phép so sánh hai ngưỡng và cả 5 tệp tải xuống mở lại được. |
| F | So sánh hai phương pháp, kiểm tra mask, tải và bỏ ground truth, đổi cực tính, lưu ZIP | PASS | Hai mask ảnh tròn trùng tham chiếu, IoU = 1 cho mỗi phương pháp. Ground truth giữ qua chuyển trang và có thể bỏ. ZIP đủ 14 tệp, kiểm tra CRC thành công, 4 PNG tải trực tiếp trùng tệp trong ZIP. Khôi phục tham số ẩn tải xuống của lần chạy cũ. |
| G | Điều hướng, thư viện và trang lý thuyết | PASS | Truy cập đủ 7 trang. Bấm đủ 11 nút **Dùng ảnh này**, đối chiếu tên tệp và trạng thái ảnh đang dùng. Các liên kết nhanh từ Tổng quan và Thư viện điều hướng đúng. Đổi seed minh họa và mở các tab lý thuyết. |
| H | JPEG thật, tiền xử lý, Lab, đảo vùng, hình thái học và xuất ảnh kích thước nguồn | PASS | `grace_hopper.jpg` 512×600, cạnh xử lý 128 px, chọn chuyển xám trung bình và lọc trung vị 3×3. Chạy K-means Lab, đảo vùng, chọn phép co. Nút xuất ảnh kích thước gốc hoạt động ở K-means, Otsu và Multi-Otsu. Ảnh RGBA khớp alpha của mask, pixel vùng giữ lại khớp ảnh JPEG đã giải mã. SHA-256 của ảnh nguồn giữ nguyên. |
| I | Các đường tích hợp bổ sung và ràng buộc dữ liệu | PASS | Pytest kiểm tra PNG/JPEG/BMP, giới hạn kích thước, giá trị và kiểu tham số, mask tham chiếu lỗi, đổi cụm/lớp đối tượng, kết quả lỗi bộ nhớ và phục hồi sau `OSError` hoặc `PermissionError` mô phỏng. Trình duyệt lưu tệp thật vào thư mục thử có quyền ghi. |
| J | Tuân thủ quy định về OpenCV và thuật toán tự cài đặt | PASS | Kiểm tra 39 tệp Python và 4 tệp khai báo hoặc khóa phiên bản. Truy vết 53 gói phụ thuộc đang hoạt động trong môi trường gồm runtime và dev. Không có gói OpenCV và không tìm thấy module `cv2`. Pytest kiểm tra thuật toán đóng gói và `pip check` đạt. |
| K | Máy chiếu và chính sách ghi tệp của tài khoản trên máy trình chiếu | NOT TESTED | Ca chạy này dùng Chrome trên máy phát triển. Quyền lưu vào thư mục bị khóa được kiểm tra bằng lỗi mô phỏng. |

## Lỗi phát hiện và kết luận

Caption của K-means và trang So sánh đưa `CIE L*a*b*` vào Markdown, nên hai dấu sao được dùng làm dấu định dạng chữ nghiêng. Đã escape dấu sao trong nhãn dùng cho caption. Ca H kiểm tra ký hiệu hiển thị đúng sau khi chạy Lab.

Đủ điều kiện trình diễn bốn luồng chính trên cấu hình đã kiểm tra. Các ca chạy trình duyệt đi qua widget tải tệp và nút tải xuống thật. Kiểm tra quyền ghi phụ thuộc chính sách trình duyệt và thư mục lưu trên máy chiếu. Chất lượng tách nền vẫn phụ thuộc đặc trưng ảnh, số cụm và cách chọn lớp đối tượng.

## Chạy lại

Ứng dụng dùng các dependency hiện có. Bộ kiểm thử trình duyệt dùng Node.js, `playwright-core` và Chrome hoặc Chromium có sẵn, tách khỏi dependency runtime của ứng dụng. Mở terminal thứ nhất trong thư mục dự án.

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8522 --server.headless true
```

Trong terminal thứ hai, đặt hai biến môi trường theo vị trí Playwright và Chrome trên máy. Có thể đặt `ACCEPTANCE_URL` nếu đổi cổng. Runner mặc định dùng Python trong `.venv` để tìm ảnh JPEG của Matplotlib, có thể thay qua `ACCEPTANCE_PYTHON`.

```powershell
$env:ACCEPTANCE_PLAYWRIGHT = '<thư mục playwright-core>'
$env:ACCEPTANCE_CHROME = '<đường dẫn tới chrome.exe>'
node tests/browser_acceptance.cjs
.\.venv\Scripts\python.exe tests/verify_acceptance_exports.py
.\.venv\Scripts\python.exe -m pytest -q --junitxml=outputs/stage8/pytest.xml
```

`outputs/stage8/` chứa các tệp đã tải, ảnh chụp từng luồng chính và `verified.json`. Có thể chạy riêng ca đang chẩn đoán bằng cách truyền ID vào runner. Trình đối chiếu tệp cần một lượt chạy đủ A–H.
