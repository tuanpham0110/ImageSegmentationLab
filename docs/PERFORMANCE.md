# Hiệu năng phân đoạn ảnh · Giai đoạn 7

## So sánh trước và sau

Bản trước là commit `f378636`. Chạy trên Windows 10, Intel 8 luồng logic, Python 3.11.7 và NumPy 2.4.6. Mỗi ô là trung vị của 5 lượt trên từng phiên bản. Mỗi lượt dùng tiến trình mới và giới hạn các thư viện số học ở một luồng. Hai phiên bản chạy xen kẽ, đảo thứ tự giữa các cặp để giảm ảnh hưởng của thứ tự chạy.

| Đầu vào | Phương pháp | Trước (ms) | Sau (ms) | Đỉnh bộ nhớ trước → sau (MiB) | Kết quả | Thay đổi thời gian |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| 256×256 RGB tổng hợp | K-means | 66,197 | 58,949 | 37,277 → 37,145 | Trùng | +10,9% |
| 256×256 xám tổng hợp | K-means | 2,148 | 2,119 | 31,059 → 31,047 | Trùng | +1,3% |
| 256×256 xám tổng hợp | Otsu | 0,689 | 0,598 | 30,816 → 30,691 | Trùng | +13,2% |
| 256×256 ảnh thật | K-means RGB | 53,959 | 43,502 | 36,660 → 36,453 | Trùng | +19,4% |
| 512×512 RGB tổng hợp | K-means | 253,470 | 234,618 | 51,441 → 50,094 | Trùng | +7,4% |
| 512×512 xám tổng hợp | K-means | 4,508 | 4,150 | 32,480 → 32,207 | Trùng | +7,9% |
| 512×512 xám tổng hợp | Otsu | 1,721 | 1,636 | 31,504 → 31,469 | Trùng | +4,9% |
| 512×512 ảnh thật | K-means RGB | 119,313 | 100,312 | 43,406 → 42,867 | Trùng | +15,9% |
| 1024×1024 RGB tổng hợp | K-means | 933,940 | 882,530 | 107,438 → 102,520 | Trùng | +5,5% |
| 1024×1024 xám tổng hợp | K-means | 13,145 | 14,976 | 37,906 → 37,984 | Trùng | −13,9% |
| 1024×1024 xám tổng hợp | Otsu | 4,078 | 4,324 | 32,383 → 32,242 | Trùng | −6,0% |
| 1024×1024 ảnh thật | K-means RGB | 322,740 | 313,648 | 76,223 → 76,320 | Trùng | +2,8% |

Phần trăm là `(trước − sau) / trước`. Số dương nghĩa là nhanh hơn. Thời gian gồm phân đoạn và tạo đầu ra trực tiếp, bỏ qua đọc tệp, tiền xử lý của ứng dụng, render và xuất tệp. Bộ nhớ là đỉnh working set của cả tiến trình con, gồm Python, thư viện và ảnh đầu vào, lấy trước khi băm kết quả. Chênh lệch nhỏ ở các lượt ngắn cần đọc cùng số đo từng lượt. Ở 1024×1024, hai dải thời gian trước/sau của K-means xám và Otsu giao nhau. Các phép đo này ghi nhận trung vị tăng, với mẫu 5 lượt chưa đủ để kết luận mức chậm lại ổn định.

Ảnh tổng hợp sinh bằng `np.random.default_rng(42)`, kiểu `uint8`. Tiến trình làm nóng một lượt trên 32×32 trước khi đo. K-means dùng `K=4`, `max_iter=8`, `tol=1e-4`, `seed=0`. Ảnh thật là `grace_hopper.jpg` đi kèm Matplotlib, được nội suy BILINEAR thành các ảnh vuông giống nhau cho cả hai bản trước khi bấm giờ. Phép đo này dùng `tol=0` và cùng `K`, `max_iter`, `seed` như trên. Giới hạn 8 vòng giúp so sánh các lượt đo và có thể kết thúc trước hội tụ. Ứng dụng vẫn dùng các tham số người dùng chọn.

Ở các ca tổng hợp, đối chiếu SHA-256 của toàn bộ kết quả thuật toán, bao gồm kiểu, kích thước và byte của các mảng, ngoại trừ số đo thời gian. Ở ảnh thật, đối chiếu nhãn, tâm, kích thước cụm, ảnh phân đoạn, SSE, số vòng và trạng thái hội tụ. Đối chiếu bổ sung trên cả 11 mẫu trong `data/` cho kết quả trùng khi xét hai cách khởi tạo K-means, các cách chọn lớp đối tượng Otsu, Multi-Otsu và các đầu ra pipeline tương ứng. [Số đo từng lượt và dấu vân tay đầu ra](results/stage7_performance.json).

## Điểm nghẽn và thay đổi

- **K-means.** Trong lượt profiling RGB 1024×1024 khoảng 904 ms, gán nhãn chiếm 513 ms, tính trung bình có trọng số 180 ms và gom màu trùng 98 ms. `kmeans.py` dùng lại mảng nhãn và khoảng cách cực tiểu qua các vòng Lloyd, giữ nhãn nội bộ ở kiểu `intp` để `np.bincount` dùng trực tiếp và đóng gói chỉ số ngược RGB bằng `int32`. Chỉ số này nhận tối đa `2^24` giá trị. Đổi phép giải mã kênh màu từ chia/lấy dư sang dịch bit và lấy 8 bit thấp. Trường hợp RGB tổng hợp 1024×1024 giảm 51,410 ms và 4,918 MiB. Khoảng cách vẫn là tổng bình phương trực tiếp trên `float64`, thứ tự phá hòa và phép tính trọng số giữ nguyên. Thử đổi bố trí mảng đã tạo sai khác ở các bit cuối của khoảng cách nên đã loại phương án đó.
- **Otsu.** Một lượt profiling 1024×1024 khoảng 4,38 ms gồm 2,89 ms tính histogram, 0,41 ms tìm ngưỡng và 1,06 ms tạo ảnh nhị phân. `np.bincount` theo khối vẫn là phương án nhanh nhất trong các cách tính histogram đã thử trên ảnh này. Với các ngưỡng ứng viên có cùng số pixel tích lũy, giữa chúng chỉ có mức xám trống, vì vậy mômen tích lũy và phương sai giữa lớp bằng nhau. `otsu.py` tính phân số chính xác một lần cho trường hợp đó và vẫn trả về đầy đủ các ngưỡng đồng hạng. Với các phân hoạch khác nhau, vẫn so sánh từng phân hoạch bằng phân số. Đo riêng bước tìm ngưỡng trên histogram dày cho kết quả 0,167 → 0,115 ms, histogram hai mức 0,218 → 0,156 ms và trường hợp đồng hạng ở nhiều khoảng rời 0,215 → 0,220 ms. Ngưỡng, quy tắc chọn ngưỡng đồng hạng và mask trùng bản trước.
- **Streamlit.** `state.py` dùng lại ảnh đã tiền xử lý trong phiên nếu nội dung, tên và toàn bộ tham số tiền xử lý giống nhau. Lượt đo riêng `build_context` trên PNG 1024×1024 đã có trong phiên giảm từ 13,230 xuống 3,031 ms. Thay đổi bộ lọc, cách chuyển xám hoặc giới hạn cạnh vẫn dùng đúng bộ tham số mới. Test tích hợp kiểm tra dùng lại đối tượng khi rerun và chuẩn bị lại ảnh khi đổi phương pháp chuyển xám.

## Độ phức tạp và ảnh lớn

Gọi `N = H×W`, `U` là số giá trị pixel phân biệt, `D` là số chiều đặc trưng, `K` là số cụm hiệu dụng, `T` là số vòng thực tế, `B = 65.536` là kích thước khối mặc định và `L = 256` là số mức xám.

| Thành phần | Thời gian | Bộ nhớ phụ và đầu ra |
| --- | --- | --- |
| K-means · gom pixel | Ảnh xám `O(N+L)`. RGB/Lab dùng màu RGB đóng gói, bước sắp xếp có thể tới `O(N log N)` | `O(N+UD)` kể cả ánh xạ ngược và giá trị phân biệt |
| K-means · Lloyd | Gán nhãn `O((T+1)UKD)`, cập nhật tâm `O(TUD)`, sắp thứ tự tâm tối đa `O((T+1)K log K)`. K-means++ thêm `O(UKD)`. Mỗi lần xử lý cụm rỗng có thể cần sắp xếp `O(U log U)` | `O(N+UD+BD+KD+T)` gồm nhãn, ảnh kết quả và lịch sử SSE. Tính khoảng cách theo từng tâm và khối giới hạn kích thước mảng sai phân |
| Otsu nhị phân | Histogram và mask tuyến tính theo `N`. Tính tích lũy tuyến tính theo `L`. Nhánh tổng quát gom các phân hoạch ứng viên để so sánh chính xác, với tối đa 255 ngưỡng | `O(B+L)` bộ nhớ phụ và `O(N)` đầu ra |
| Multi-Otsu ba lớp | `O(N+L²)` do xét các cặp ngưỡng | Tối đa `O(L²)` bộ nhớ phụ và `O(N)` đầu ra |

Ảnh RGB nhiều màu có thể khiến `U` gần bằng `N`. Khi đó, tăng kích thước, `K` hoặc `max_iter` dễ kéo dài lần chạy. Ở I/O, giới hạn giải mã hiện tại là 40 triệu pixel. Riêng mảng RGB `uint8` tại giới hạn đã xấp xỉ 114 MiB trước khi thu nhỏ, chưa tính mảng Pillow, mảng RGBA hay bản sao. Overlay tạo vùng pha màu `float64` cho các pixel được chọn. Xuất PNG, CSV hoặc ZIP có thể giữ thêm dữ liệu trong RAM. UI mặc định giới hạn cạnh xử lý ở 800 px, cho phép chọn đến 2048 px. Giai đoạn này giữ kích thước, độ chính xác và thuật toán hiện có, chỉ thay đổi cách cấp phát và tái sử dụng dữ liệu.

## Kiểm tra giao diện

Chrome headless ở 1440×900 tải PNG tổng hợp 1024×1024, đặt cạnh xử lý 1024 px và chạy K-means RGB với `K=4`, tối đa 8 vòng. Spinner xuất hiện, số đo do app hiển thị là 958 ms, và khoảng thời gian trình duyệt từ thao tác bấm đến khi phần kết quả xuất hiện là 1.563 ms. Đổi K sau đó hiển thị thông báo tham số đã đổi và vẫn giữ kết quả lần chạy trước. Playwright ghi nhận 0 `stException`. Đây là một lượt chạy trình duyệt. Xử lý vẫn đồng bộ trong lần rerun của phiên, nên ảnh nhiều màu hoặc tham số lớn có thể kéo dài thời gian chờ. Các ca đồng thời nhiều phiên và cạnh 2048 px chưa được đo ở giai đoạn này.

Bộ kiểm thử đầy đủ sau thay đổi có 616 test pass, 0 fail và 0 skip. Ruff và `pip check` đều pass. [Số liệu kiểm thử và lượt chạy trình duyệt](results/stage7_performance.json).

## Đo lại

Từ bản `f378636`, chép `src/algorithms/kmeans.py` và `src/algorithms/otsu.py` vào `outputs/stage7/baseline-algorithms`. Chạy các lệnh sau ở bản hiện tại. Hai lệnh đầu đo riêng từng phiên bản, còn bảng trên dùng chế độ worker để chạy xen kẽ. Lệnh thứ ba so sánh cả hai phiên bản trên ảnh thật đi kèm Matplotlib.

```powershell
$photo = .\.venv\Scripts\python.exe -c "from matplotlib import get_data_path; from pathlib import Path; print(Path(get_data_path()) / 'sample_data' / 'grace_hopper.jpg')"
.\.venv\Scripts\python.exe scripts/benchmark_segmentation.py --source-dir outputs/stage7/baseline-algorithms --sizes 256x256 512x512 1024x1024 --repeats 5 --output outputs/stage7/recheck-before.json
.\.venv\Scripts\python.exe scripts/benchmark_segmentation.py --sizes 256x256 512x512 1024x1024 --repeats 5 --output outputs/stage7/recheck-after.json
.\.venv\Scripts\python.exe scripts/benchmark_kmeans.py --baseline outputs/stage7/baseline-algorithms/kmeans.py --image $photo --sizes 256 512 1024 --ks 4 --repeats 5 --max-iter 8 --seed 0 --output outputs/stage7/recheck-photo.json
.\.venv\Scripts\python.exe -m pytest -q
```
