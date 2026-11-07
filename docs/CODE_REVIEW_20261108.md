# Đối chiếu và xử lý CR-01–CR-10 — 08/11/2026

Baseline: nhánh `main`, commit `e891bfb` (`Fix FE`). Đã chạy mã cục bộ bằng Python 3.11.7 trên Windows 10,
không chỉ đọc source. Baseline **471 test đạt**; sau thay đổi **491 test đạt trong môi trường sạch**.
Không thay K-means/Otsu bằng thư viện có sẵn, không thêm database, không thay framework giao diện.

## Trạng thái từng mục

| Mục | Kết quả | Bằng chứng / giới hạn còn lại |
|---|---|---|
| CR-01 | Đã giảm bộ nhớ tạm và đo trước/sau | Bỏ ma trận int64 N×D khi gom màu; kiểm tra pixel theo khối; giải phóng tập distinct sau khởi tạo; bỏ một hiệu N×D bị tính hai lần. Đầu ra chuẩn giữ nguyên. RAM stress giảm khoảng 6–12%; thời gian không luôn giảm. |
| CR-02 | Đã kiểm chứng giới hạn và bổ sung điều khiển | Test đối tượng sát mép chứng minh gợi ý có thể sai; chọn cụm thủ công khôi phục đúng trong ca kiểm soát. Có thêm **Đảo vùng đối tượng**, kiểm tra cả PNG/JSON xuất ra. Heuristic không trở thành nhận diện ngữ nghĩa. |
| CR-03 | Đã có đánh giá ảnh thật công khai | 30 ảnh Oxford VGG, 6 dev/24 test, thêm 30 biến thể giảm sáng; K/seed/foreground cố định; có IoU/Dice, thời gian, ca thất bại và checksum. Đây là đánh giá ban đầu, không đại diện mọi ảnh điện thoại. |
| CR-04 | Đã khóa và cài lại sạch | `.python-version`, `requirements.in/.txt`, `requirements-dev.in/.txt`. Python 3.11 là chuẩn. Môi trường mới `outputs/cr-clean`: cài từ lockfile, `pip check`, Ruff và toàn bộ 491 test đều đạt. |
| CR-05 | Đã bỏ môi trường ảo khỏi chỉ mục Git | 763 tệp `.venv-1` được bỏ theo dõi bằng `git rm --cached`; thư mục trên máy vẫn còn. Thêm ignore và kiểm tra CI chống tái diễn. Thay đổi đang ở local/index, chưa push; không viết lại lịch sử Git. |
| CR-06 | Đã thêm workflow | Push main / PR / chạy thủ công, Windows + Ubuntu, Python 3.11; cài dependency, `pip check`, Ruff, Pytest, benchmark smoke và kiểm tra môi trường ảo bị theo dõi. GitHub chưa chạy workflow cho tới khi thay đổi được push. |
| CR-07 | Giữ hành vi mặc định, ghi rõ giới hạn | 800 px là cân bằng tốc độ/chất lượng của demo; UI và JSON đã ghi kích thước gốc/xử lý. Không thêm xuất mask phóng lớn rồi gọi đó là khôi phục chi tiết. Chế độ xử lý nguyên độ phân giải là hướng tiếp theo. |
| CR-08 | Đã bổ sung benchmark và regression | Stress 800/1600/2048, K=4/16, ảnh thật phóng lớn, CSV đúng 4 triệu phần tử; SSE dừng sớm, độ sáng, sát viền, trimap và đảo foreground/tệp xuất. Bỏ ngưỡng CSV 1 giây phụ thuộc máy. |
| CR-09 | Không refactor diện rộng | Giữ API và cấu trúc module; các công cụ benchmark/đánh giá tách riêng trong scripts. Chỉ thay phần có mục tiêu và regression tương ứng; không tách module chỉ dựa trên số dòng. |
| CR-10 | Đã sửa lỗi báo cáo SSE | `initial_sse` lưu SSE khởi tạo. `sse_history[i]` ứng với tâm đã cập nhật và gán nhãn lại của vòng i+1; phần tử cuối bằng `result.sse`, kể cả max_iter=1/chưa hội tụ. Bảng lý thuyết, biểu đồ và JSON dùng cùng quy ước. |

## SSE và tính bất biến của thuật toán

Review xác định đúng sự lệch giữa lịch sử SSE trước cập nhật tâm và SSE kết quả cuối. Bản sửa dùng lại
phép gán nhãn sau cập nhật ở vòng tiếp theo và ở đầu ra; không chạy thêm một vòng Lloyd vượt `max_iter`.
Không thay thứ tự khởi tạo, trọng số, quy tắc phá hòa, sắp tâm hay tiêu chí hội tụ.

So sánh SHA-256 của **nhãn, tâm float64, số pixel, ảnh phân đoạn, SSE cuối, số vòng và trạng thái hội tụ**
giữa baseline và bản sửa: khớp tuyệt đối ở 9 cấu hình benchmark, mỗi biến thể lặp 3 lần (54 worker).
Lịch sử SSE được thay có chủ đích nên không nằm trong hash bất biến.
Regression còn tính lại SSE độc lập từ từng bộ tâm và trọng số để kiểm tra quy ước mới.

Script thực nghiệm cũ đã đo trung vị 5 lần cho ảnh mẫu và 3 lần khi quét tham số; không phải mọi nhận xét
trong review đều là phần còn thiếu. Lượt này bổ sung phép đo RAM và dữ liệu ảnh thật có protocol riêng.

## Hiệu năng: báo cáo cả trường hợp không nhanh hơn

CPU: Intel64 Family 6 Model 165, 8 logical CPUs; NumPy 2.4.6, float64, seed=0, tol=0,
K-means++ và tối đa 20 vòng. Mỗi phép đo chạy trong process mới, giới hạn BLAS/OMP/MKL một luồng.
Thời gian là trung vị 3 lần; RAM là peak working set lớn nhất của 3 worker, gồm imports và ảnh đầu vào.
Máy không được dành riêng hoàn toàn cho benchmark, vì vậy không suy diễn các chênh lệch thời gian nhỏ.

### Stress RGB nhiều màu, dữ liệu tổng hợp

| Cạnh | K | Thời gian cũ → mới (s) | Peak RSS cũ → mới (MiB) |
|---:|---:|---:|---:|
| 800 | 4 | 2.055 → 1.970 | 127.1 → 115.6 |
| 800 | 16 | 6.238 → 5.979 | 126.9 → 115.9 |
| 1600 | 4 | 8.538 → 8.132 | 395.2 → 349.4 |
| 1600 | 16 | 21.166 → 21.395 | 395.4 → 349.9 |
| 2048 | 4 | 14.289 → 14.072 | 565.7 → 529.9 |
| 2048 | 16 | 36.269 → 31.906 | 565.8 → 530.3 |

Ảnh RGB ngẫu nhiên là stress test số màu, **không phải ảnh chụp thật**.
20 vòng không đảm bảo hội tụ; cờ `converged` được giữ nguyên trong dữ liệu kết quả.

### Ảnh thật `person_2008_001375`, K=4

| Cạnh sau resize | Thời gian cũ → mới (s) | Peak RSS cũ → mới (MiB) |
|---:|---:|---:|
| 800 | 0.531 → 0.522 | 64.3 → 60.4 |
| 1600 | 1.288 → 1.325 | 134.8 → 134.7 |
| 2048 | 1.793 → 2.281 | 199.2 → 199.2 |

Ảnh nguồn chỉ 500×333; worker phóng về hình vuông để stress theo kích thước yêu cầu.
Đây **không** chứng minh chất lượng hay hiệu năng trên ảnh điện thoại 2048 px nguyên bản.
Không thấy cải thiện RAM đáng kể ở hai kích thước ảnh thật lớn; một phép đo còn chậm hơn.
Không thêm chế độ lấy mẫu xấp xỉ vào mặc định.

CSV: 2000×2000 = 4.000.000 giá trị, 14.281.030 byte; trung vị parse **1,509 s**.
Peak RSS toàn worker khoảng **380,6 MiB**, bao gồm cả sinh CSV và mảng đối chiếu; không gọi đó là RAM riêng parser.
Không đặt ngưỡng thời gian này làm điều kiện pass CI.

Số liệu gốc: [stress](results/cr_benchmark.json), [ảnh thật phóng lớn](results/cr_photo_benchmark.json),
[CSV](results/cr_csv_benchmark.json).

## Ảnh thật: độ chính xác thấp là kết quả cần báo cáo

Nguồn chính: [Oxford VGG — Geodesic star convexity for interactive image segmentation](https://www.robots.ox.ac.uk/~vgg/data/iseg/),
Gulshan và cộng sự, CVPR 2010. Trang tác giả cung cấp 151 ảnh và ground truth từ GrabCut, PASCAL VOC, alpha matting.
Liên kết GrabCut cũ được thử nhưng trả 404/406; không dùng các liên kết lỗi làm nguồn dữ liệu đã tải.

`evaluate_real_images.py` kiểm tra SHA-256 hai archive, trích xuất an toàn và lưu dữ liệu ở `data/external/`.
Không đưa ảnh/mask của bên thứ ba vào Git; chỉ đưa protocol, checksum và bảng số liệu.
SHA-256 archive và từng cặp ảnh/mask có trong [JSON](results/cr_real_images.json).

- Sắp ảnh theo SHA-256 tên tệp, lấy 30 ảnh đầu trước khi xem nội dung/GT; 6 dev và 24 test.
- Tham số khóa trước khi đánh giá: K=5, RGB, K-means++, seed=0, tol=1e-4, max_iter=100, không hậu xử lý.
- Foreground tự động theo viền/trung tâm. Không chọn cụm, cực tính hoặc K theo GT.
- Dự đoán xong mới đọc GT để tính điểm. Nhãn 128 là vùng biên chưa chắc chắn và **bị bỏ qua**, không đổi thành foreground.
- Cả hai phương pháp dùng cùng ảnh đã chuẩn bị; đo trung vị 3 lần theo phạm vi thời gian chung của pipeline.
- Gain=0,5 là biến đổi giảm sáng có kiểm soát, không phải một tập ảnh chụp thiếu sáng độc lập.

Kết quả macro trên **24 ảnh test gốc**, không gộp dev hay biến thể giảm sáng:

| Phương pháp | IoU trung bình | Dice trung bình |
|---|---:|---:|
| K-means + foreground tự động | 0.2397 | 0.3406 |
| Otsu + foreground tự động | 0.2357 | 0.3332 |

Quan sát hình trực tiếp, với tham số không đổi:

- `boat_2007_000837`: thuyền chỉ chiếm 1,4% ảnh; mask tự động chọn phần lớn trời/nước thay vì con thuyền tối. IoU KM=0,0040; Otsu=0,0019.
- `cat_2008_001885`: mèo tối sát viền, trước cửa sổ sáng. Hai mask chọn nhiều vùng cửa sổ; gợi ý foreground sai. IoU KM=0,0364; Otsu=0,0146.
- `boat_2008_001135`: thuyền hòa với nền thành phố/nước về màu và độ sáng; mask rải trên nhiều vùng nền. IoU KM=0,0302; Otsu=0,0510.
- `stone2`: đối tượng trung tâm tương phản tốt hơn; vẫn thủng do bề mặt có cấu trúc. IoU KM=0,8177; Otsu=0,7919.

Chạy `python scripts/render_real_examples.py` để xem ba ca tệ nhất và ba ca tốt nhất tại
`outputs/cr-examples/qualitative_test.png`. Việc chọn ca minh họa sau khi chấm điểm không được dùng để chỉnh tham số hay tính lại tập test.
Chi tiết cả 60 lượt ảnh/biến thể và thời gian: [bảng đánh giá](results/cr_real_images.md).

## Kiểm chứng môi trường và CI

Đã tạo môi trường hoàn toàn mới `outputs/cr-clean` bằng Python 3.11.7 rồi cài `requirements-dev.txt`.
`pip check` không có xung đột; **491 test đạt trong 43,58 s**; Ruff đạt.
Kết quả JUnit lưu ở `outputs/cr_clean_pytest.xml`.
Không nâng dependency trong môi trường đang chạy ứng dụng để giả lập một lần cài sạch.

Workflow dùng [actions/checkout](https://github.com/actions/checkout) và
[actions/setup-python](https://github.com/actions/setup-python) theo hướng dẫn chính thức.
Đã chạy các kiểm tra tương ứng cục bộ; **chưa có bằng chứng workflow chạy trên GitHub/Ubuntu** cho thay đổi chưa push.

## Lệnh tái lập

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m ruff check app.py src scripts tests
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/benchmark_kmeans.py --sizes 800 1600 2048 --ks 4 16 --repeats 3
.\.venv\Scripts\python.exe scripts/benchmark_csv.py
.\.venv\Scripts\python.exe scripts/evaluate_real_images.py --fetch --samples 30 --repeats 3
.\.venv\Scripts\python.exe scripts/render_real_examples.py
```

Để so sánh chính xác trước/sau, lấy source baseline ra tệp UTF-8/binary (không dùng redirect mặc định UTF-16 của PowerShell cũ):

```powershell
python -c "import subprocess; from pathlib import Path; p=Path('outputs/baseline-kmeans.py'); p.parent.mkdir(exist_ok=True); p.write_bytes(subprocess.check_output(['git','show','e891bfb:src/algorithms/kmeans.py']))"
python scripts/benchmark_kmeans.py --baseline outputs/baseline-kmeans.py --sizes 800 1600 2048 --ks 4 16 --repeats 3
```

## Đánh giá nghiệm thu

Phần tính toán, SSE, hồi quy và tái lập môi trường có thêm bằng chứng chạy thật.
Ứng dụng phù hợp để thực hành và phân tích giới hạn K-means/Otsu; không nên giới thiệu là công cụ
tự động tách đối tượng đáng tin cậy trên nền phức tạp. Nếu tiêu chí nghiệm thu đòi độ chính xác cao
trên ảnh tự nhiên bất kỳ, CR-02 vẫn là giới hạn sản phẩm cần quyết định phạm vi, không phải một test còn thiếu.
