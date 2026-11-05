# Báo cáo sửa lỗi và kiểm chứng — 05/11/2026

Đã hoàn thành các thay đổi P0/P1/P2 trong phạm vi source hiện có. Giữ kiến trúc, tên hàm/class chính và bố cục
giao diện. K-means/Otsu vẫn tự cài đặt bằng NumPy; không thêm dependency ngoài, không sử dụng OpenCV.

Kết quả tổng kiểm tra: **170 PASS, 0 FAIL, 0 ERROR, 0 SKIP**, khoảng **15.51 giây**. Chrome thực: **10 kiểm tra PASS**,
gồm upload, chạy thuật toán, tải PNG/CSV, đối chiếu dữ liệu và viewport. Đây không phải tuyên bố đạt mọi ảnh thực tế.

## 1. Đối chiếu review và kế hoạch

Trước sửa, 86 tệp dự án được đối chiếu hash và không đổi so với lượt review. Bộ test nền: 86 PASS.
Tái hiện được ngưỡng Otsu 127 không thuộc argmax, CSV 400 chữ số gây OverflowError, ảnh đồng nhất với foreground
tối có mask/cảnh báo mâu thuẫn. Những điểm này vẫn còn trong source khi bắt đầu sửa.

Phân loại và thứ tự thực hiện:

1. **P0:** ngưỡng Otsu tối ưu và mask tự động cho ảnh không có hai lớp hợp lệ; test hồi quy.
2. **P1:** dữ liệu/tham số, cụm rỗng, lỗi chạy/xuất, thời gian đo, cảnh báo và phiên bản Streamlit.
3. **P2:** cache và bản sao ảnh, tái sử dụng tệp xuất, tài liệu giải thích và bằng chứng trình duyệt.

Không coi năm giáo trình 2010 là vi phạm: người dùng xác nhận nhóm được phép sử dụng bản này. Chưa có PDF giáo
trình 2010 trong dự án để kiểm chứng trực tiếp toàn bộ số trang/trích dẫn. Không suy đoán thông tin thành viên,
phần trăm đóng góp hoặc kết quả trên ảnh chụp thật chưa được cung cấp.

## 2. Nhật ký kiểm thử theo nhóm

| Nhóm | Kết quả | Nội dung đã xác minh |
|---|---:|---|
| Trước sửa | 86 PASS | Các chức năng nền; lỗi mới được tái hiện riêng |
| P0 | 91 PASS | Cực đại rời nhau; ảnh đồng nhất ở ba cực tính; chế độ thủ công; PNG/RGBA/JSON |
| P1 — đầu vào | 141 PASS | CSV cực trị, cú pháp, kiểu dữ liệu; tham số K-means; histogram/ngưỡng; nhánh cụm rỗng thực sự |
| P1 — UI/thời gian | 158 PASS | Cùng phạm vi đo; cảnh báo so sánh; lỗi thiếu bộ nhớ; ảnh PNG tổng hợp 4000×3000; dependency |
| P2 — tài nguyên | 167 PASS | Không giải mã/mã hoá lại khi không cần; đổi ảnh/tên/foreground không tải nhầm dữ liệu; lỗi xuất |
| Tổng kiểm tra | 170 PASS | Cụm rỗng sau gán nhãn cuối, CSV có ô xuống dòng, thiếu RAM khi đọc ảnh; toàn bộ regression |

## 3. Quản lý thay đổi

| File đã sửa | Vấn đề cũ | Thay đổi | Lý do | Kiểm thử | Kết quả |
|---|---|---|---|---|---|
| `src/algorithms/otsu.py` | Trung bình các cực đại có thể không tối ưu | Chọn phần tử argmax gần floor trung bình nhất, hoà chọn nhỏ hơn | Giữ đúng tiêu chí Otsu và thống kê tại ngưỡng | `test_disconnected_maxima_return_an_actual_optimum`, brute force | PASS |
| `src/processing/pipeline.py`, `src/ui/page_otsu.py` | Ảnh đồng nhất có mask/cảnh báo trái nhau | Mask Otsu tự động rỗng ở mọi cực tính, cảnh báo nhất quán; thủ công vẫn cho chọn lớp | Không suy diễn đối tượng từ ngưỡng không hợp lệ | Ba cực tính, RGBA, JSON, thủ công | PASS |
| `src/utils/image_io.py` | Float làm tràn số hoặc làm tròn CSV không nguyên | Decimal chuẩn thư viện Python; lỗi CSV thành InputError; giữ xuống dòng trong ô trích dẫn | Không chấp nhận sai pixel và không làm crash UI | Số 400 chữ số; thập phân sát nguyên; dấu nháy; `0,"1↵2"` | PASS |
| `src/algorithms/kmeans.py` | Chunk âm có thể dùng nhãn chưa khởi tạo; tham số sai kiểu | Kiểm tra K, max_iter, tol, seed, chunk_size, ảnh rỗng và D=0 | Từ chối trước xử lý/cấp phát | Các nhóm tham số sai, số cụm, trùng pixel | PASS |
| `src/algorithms/kmeans.py` | Dừng sớm có thể còn cụm rỗng sau gán nhãn cuối | Cảnh báo và converged=False; giữ giới hạn số vòng | Không báo hội tụ sai, không vượt ngân sách lặp | Một vòng, tol=1000, dữ liệu lệch seed 42 | PASS |
| Hai module thuật toán, `pipeline.py` | Phạm vi thời gian K-means/Otsu khác nhau | Cùng đo tới kết quả trực tiếp; thêm `segment_image_otsu` và mô tả phạm vi | So sánh có định nghĩa nhất quán | Đầu ra thuật toán, metadata JSON, chạy thực nghiệm | PASS |
| `scripts/run_experiments.py` | Đo Otsu khác UI; ghi “đúng 20 vòng”; luôn ghi hình | Dùng cùng endpoint/đồng hồ; lưu n_iter/converged; thêm `--skip-figures`, thời điểm Asia/Ho_Chi_Minh | Số liệu tái lập và giữ nguyên ảnh minh hoạ khi cập nhật số đo | Chạy đủ 11 mẫu, quét K/seed/kích thước | PASS |
| `src/ui/page_kmeans.py`, `page_otsu.py`, `page_compare.py` | Lỗi xử lý chưa bắt; cảnh báo so sánh thiếu | Thông báo cụ thể; bỏ kết quả khi lần chạy mới thất bại; hiện cảnh báo hợp lệ/hội tụ | Tránh hiển thị thành công hoặc tải nhầm | AppTest, MemoryError chủ động, ảnh đồng nhất/chưa hội tụ | PASS |
| `src/ui/sidebar.py` | Thiếu RAM khi giải mã có thể để lại kết quả cũ | Bắt MemoryError lúc nạp, xóa context/payload cũ và hướng dẫn thu nhỏ ảnh trước khi tải | Xử lý lỗi ngay trước tiền xử lý | AppTest mô phỏng lỗi giải mã, xác minh không còn nút tải cũ | PASS |
| `requirements.txt` | Cho phép Streamlit 1.45 thiếu API đang dùng | Cố định Streamlit 1.66.1 đã kiểm thử | Mô tả đúng môi trường tương thích | Metadata phiên bản, cây dependency active, UI thật | PASS |
| `src/ui/state.py` | Cache dùng chung giữ tối đa 16 ảnh gốc; giữ payload cũ | Giữ ảnh gốc hiện tại trong phiên; cache ảnh đã xử lý tối đa 4 mục/300 giây; giải phóng payload stale; tên tệp vào chữ ký | Giảm giữ RAM và tránh metadata nhầm tên | Đếm giải mã, đổi ảnh rồi chọn lại, đổi tên cùng bytes | PASS |
| `src/ui/components.py`, `page_kmeans.py`, `page_otsu.py` | Mã hoá bảy tệp trên mỗi lần render | Cache một bộ tệp theo lần chạy/foreground, bắt lỗi tạo tệp, thông báo yêu cầu tải qua trình duyệt | Giảm công việc lặp và giữ kết quả xuất đúng màn hình | Đếm lần xuất; đổi foreground/ngưỡng; OSError; Chrome tải thực | PASS |
| `src/algorithms/kmeans.py` | Bản sao float64/int64 toàn ảnh trước gom màu | View uint8 và đóng gói RGB 24 bit vào int32; core vẫn float64 trên giá trị phân biệt | Giảm bản sao mà giữ nguyên hàm mục tiêu | Đối chiếu unique/counts/inverse, dữ liệu khai triển và mọi mẫu GT | PASS |
| `src/ui/components.py`, `page_guide.py`, `page_theory.py` | Tên dài xuống dòng; hướng dẫn/lý thuyết cũ | Tên rút gọn có tooltip; mô tả đúng cực tính, lưu tệp, tie-break, độ phức tạp và K=2 | Dễ sử dụng và giải thích được | Render 7 trang; kiểm tra nội dung tài liệu | PASS |
| Bốn file `tests/test_*.py` | Thiếu ca biên; test cụm rỗng không chạy nhánh | Thêm hồi quy và thay dữ liệu test cụm rỗng bằng ca thực sự kích hoạt | Kiểm chứng hành vi thay vì tên hàm | Toàn bộ 170 test | PASS |
| `README.md`, `docs/ALGORITHMS.md`, `REPORT.md`, `TEST_CASES.md`, `QUESTIONS_AND_ANSWERS.md`, `SLIDES_OUTLINE.md`, `DEMO_SCRIPT.md` | Tie-break, quan hệ hai thuật toán, số liệu/count/hướng dẫn chưa đồng bộ | Đồng bộ quyết định cài đặt, kết quả đo thật, phạm vi và phần chưa nghiệm thu | Cơ sở trình bày trung thực, nhất quán | Đối chiếu source, JUnit, JSON thực nghiệm và báo cáo Chrome | PASS |
| `docs/results/experiment_results.json`, `.md` | Số liệu của phiên bản trước | Cập nhật bằng script, giữ nguyên dữ liệu và ảnh minh hoạ | Có bằng chứng của phiên bản sau sửa | Script thoát thành công, IoU/ngưỡng mẫu cũ giữ nguyên | PASS |

## 4. Kết quả thuật toán và phạm vi đo

K-means vẫn là Lloyd có trọng số, khởi tạo random/K-means++, khoảng cách Euclid, cập nhật trung bình, xử lý cụm rỗng,
giới hạn max_iter và tol. Ảnh kết quả dùng màu tâm thực sự; không tạo minh hoạ thay cho phân cụm. Otsu vẫn dựng
histogram và cực đại phương sai giữa lớp, không thay bằng ngưỡng cố định hoặc hàm thuật toán thư viện.

Phản ví dụ Otsu đã sửa: h(0)=h(255)=1, h(127)=h(128)=127. Tập cực đại 0..126 và 128..254; t trả về 126,
σ²_B=63.75. Ngưỡng 127 với σ²_B≈2.2266 bị loại khỏi lựa chọn. Ngưỡng, trọng số, trung bình và η khớp nhau.

Với dữ liệu xám và K=2: SSE/N = σ²_W = σ²_T − σ²_B. Khác biệt cần giải thích là cách tìm nghiệm cục bộ của Lloyd
và duyệt ngưỡng của Otsu, cùng lựa chọn đặc trưng/foreground; không khẳng định phương pháp nào tốt cho mọi ảnh.

Số đo lấy từ [experiment_results.json](results/experiment_results.json), ngày 05/11/2026, Python 3.11.7,
NumPy 2.4.6, Windows 10, CPU Intel Family 6 Model 165. Đầu vào đã tiền xử lý; tính tới nhãn/ảnh tâm hoặc nhị phân;
loại trừ đọc tệp, resize/chuyển xám, chọn foreground, ảnh tách nền, biểu đồ và xuất tệp.

| Ca | K-means | Otsu | Nhận xét |
|---|---:|---:|---|
| RGB ngẫu nhiên 512×512, K=4, tối đa 20 vòng | 760.7 ms | 1.79 ms | Trung vị 3 lần; K-means chưa hội tụ |
| RGB ngẫu nhiên 1024×1024, K=4, tối đa 20 vòng | 3125.3 ms | 6.92 ms | Trung vị 3 lần; không suy ra thời gian mọi ảnh |
| Hình tròn tối/nền sáng và chữ nhật sáng/nền tối | IoU=1 | IoU=1 | Mẫu tổng hợp, có GT |
| Nhiều hình RGB | IoU=1 | IoU=0.7964 | K=5 RGB; khác biệt về phân hoạch và đặc trưng |
| Nhiễu | IoU=0.8397 | IoU=0.8397 | Không dùng quan hệ lân cận |
| Tương phản thấp | IoU=0.9532 | IoU=0.9532 | Không suy ra chất lượng ảnh chụp thật |

Các ca đầy đủ có dữ liệu, bước kiểm tra, kỳ vọng/thực tế và trạng thái tại [TEST_CASES.md](TEST_CASES.md).

## 5. Bằng chứng tổng kiểm tra

- [JUnit: 170 test, 0 lỗi, 0 skip](results/pytest_results.xml).
- [Kết quả thực nghiệm và môi trường](results/experiment_results.md).
- [Chrome: kết quả lượt kiểm tra cuối](results/browser_checks_20261105_155516.json).
- [K-means sau upload PNG](screenshots/10_browser_kmeans_20261105_155516.png).
- [Otsu trên cùng ảnh](screenshots/11_browser_otsu_20261105_155516.png).
- [So sánh desktop](screenshots/13_browser_compare_1440_20261105_155516.png).
- [So sánh ở chiều rộng 390 px](screenshots/13_browser_compare_390_20261105_155516.png).

Chrome chạy headless với profile/thư mục tải tạm riêng. Môi trường quản lý yêu cầu cờ `--no-sandbox` cho tiến trình
Chrome kiểm thử; cờ này chỉ nằm trong harness kiểm tra cục bộ, không được thêm vào source hay cấu hình ứng dụng.
Đã đọc lại PNG/CSV thực sự tải xuống và đối chiếu mảng, không dùng toast làm bằng chứng tệp đã lưu. Viewport
1440/768/390 px không tràn ngang toàn trang và không stException; không có ngoại lệ JavaScript quan sát được.

Đối chiếu hash xác nhận `data/`, `docs/figures/` và `.streamlit/config.toml` giữ nguyên. Không cài thêm gói,
không sửa môi trường `.venv`, không xóa/đổi tên tệp dự án, không chạy lệnh commit hoặc push Git.

## 6. Các phần còn cần nhóm bổ sung

| Nội dung | Trạng thái | Cơ sở |
|---|---|---|
| Ảnh chụp thật của nhóm | NOT RUN | Người dùng xác nhận hiện chưa có; chưa kết luận chất lượng thực tế |
| Thành viên/MSSV/% đóng góp | Chờ nhóm điền | Giữ các ô hiện có trong REPORT, không tạo thông tin giả |
| Slide trình bày cuối cùng | Chờ nhóm hoàn thiện | Đề cương và kịch bản đã đồng bộ; chưa tạo PPT/PDF hoàn chỉnh |
| Cài mới trên một máy sạch | NOT RUN | Đã kiểm tra dependency và môi trường hiện tại, chưa cài lại `.venv` |
| Mọi trình duyệt và mọi kích thước ảnh | Chưa bao phủ toàn bộ | Đã kiểm tra Chrome và ảnh hợp lệ 12 triệu pixel thu nhỏ; không thử cạn kiệt RAM |

Giới hạn còn lại của phương pháp: phân bố màu/xám chồng lấn, nhiễu, chiếu sáng không đều, đối tượng phủ viền hoặc
nền nhiều màu có thể gây phân đoạn sai. Gợi ý viền là heuristic và cho phép chỉnh tay. Ảnh lớn nhiều màu với
max_iter cao vẫn có thể chờ lâu; mỗi vòng có chặn trên nhưng ứng dụng chưa có hủy tác vụ đang tính.
