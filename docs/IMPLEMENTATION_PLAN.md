# Kế hoạch hoàn thiện Đề tài 3

Kế hoạch áp dụng cho Image Segmentation Studio, bài tập lớn Xử lý ảnh INT13146 tại PTIT. Mốc Giai đoạn 1 là commit `5815d49`, với 561 test đạt trên Windows, Python 3.11.7 và Streamlit 1.66.1. Cả hai thuật toán tự cài đặt, luồng ảnh và CSV đã xuất được kết quả, và kiểm tra source, dependency đang dùng cùng gói wheel Windows không phát hiện OpenCV. Phần cần hoàn thiện tập trung vào hợp đồng mask tham chiếu, kiểm thử bản phát hành và tài liệu bàn giao.

## Mức ưu tiên

| Mức | Phạm vi | Công việc hiện có |
|---|---|---|
| P0 | Vi phạm yêu cầu giảng viên hoặc làm sai kết quả thuật toán | Giai đoạn 1 chưa phát hiện lỗi P0. Kiểm tra lại các bất biến của hai thuật toán và yêu cầu cấm OpenCV trong mỗi lần xác nhận. Nếu phát hiện sai lệch, dừng phần việc phụ thuộc để khoanh vùng và sửa. |
| P1 | Thiếu chức năng bắt buộc, sai lệch nghiêm trọng hoặc cản trở nghiệm thu | Chốt hợp đồng mask để tránh tính điểm từ dữ liệu sai quy ước, xác minh đường demo trên bản cài sạch, kiểm tra tính đầy đủ của gói nộp và hoàn tất báo cáo, slide. |
| P2 | Khả năng sử dụng, hiệu năng, chất lượng mã và hồi quy | Cho phép phân đoạn khi chỉ hỏng mask mẫu, sửa gợi ý lớp rỗng Multi-Otsu, bổ sung test còn thiếu và kiểm tra chi phí xử lý. |
| P3 | Cải tiến ngoài tiêu chí nghiệm thu | Cân nhắc hợp nhất phần tạo đầu ra trùng lặp, xử lý mã không có nơi gọi và mở rộng nghiên cứu sau khi chốt bản nộp. |

## Yêu cầu và module chịu trách nhiệm

| Yêu cầu | Module chính | Điều kiện nghiệm thu |
|---|---|---|
| 1. K-means tự cài đặt | `src/algorithms/kmeans.py`, `run_kmeans()` trong `src/processing/pipeline.py`, `src/ui/page_kmeans.py` | K=2 trên ảnh hai mức cho hai cụm đúng. Cùng ảnh và seed cho cùng nhãn. Nhãn gần nhất, tâm, số đếm và SSE nhất quán. |
| 2. Otsu Thresholding tự cài đặt | `src/algorithms/otsu.py`, `run_otsu()`, `src/ui/page_otsu.py` | Ngưỡng nằm trong tập cực đại hợp lệ. Mẫu hình tròn cho 127, CSV hình chữ nhật cho 119 và ma trận 5×6 cho 1. Ảnh đồng nhất có cảnh báo và mask tự động rỗng. |
| 3. Tách đối tượng và nền | `src/processing/postprocessing.py`, `filters.py`, `pipeline.py` | Chọn cụm hoặc lớp sáng/tối cho mask 0/255 đúng chiều. Kênh alpha của cutout khớp mask. Hai ca chuẩn đạt IoU 1,0000 với cấu hình demo. |
| 4. Nhập ảnh hoặc CSV, lưu kết quả | `src/utils/image_io.py`, `src/ui/sidebar.py`, `src/ui/state.py`, các hàm xuất ở `pipeline.py` | Đọc ảnh PNG/JPEG/BMP và ma trận CSV hợp lệ. Báo vị trí CSV sai. Đọc lại được PNG, RGBA, CSV, JSON và ZIP với kích thước, nhãn và tham số của lần chạy hiện tại. |
| 5. Trực quan hóa so sánh | `src/ui/page_compare.py`, `src/ui/components.py`, `src/utils/visualization.py` | Hai thuật toán dùng cùng ảnh đã chuẩn bị. Xem được ảnh gốc, kết quả, bản đồ khác biệt mask và histogram Otsu. Điểm chất lượng chỉ xuất hiện khi có mask tham chiếu hợp lệ. |
| 6. Không sử dụng OpenCV hoặc thuật toán đóng gói sẵn | `requirements*.in`, `requirements*.txt`, `tests/test_integration.py`, `.github/workflows/ci.yml` | Quét source, lockfile, metadata dependency được cài và wheelhouse chốt. Các điểm chạy chính không gọi thư viện cài sẵn K-means/Otsu. |
| 7. Test case và mô phỏng | `tests/`, `data/samples.json`, `scripts/run_experiments.py`, `docs/TEST_CASES.md` | Bộ pytest đạt trên môi trường chốt, các hồi quy mới được bao phủ, kịch bản trình duyệt có bằng chứng. Kết quả thực nghiệm ghi nguồn dữ liệu, cấu hình và phạm vi đo. |
| 8. Source, báo cáo, slide và phản biện | `scripts/package_submission.py`, `README.md`, `docs/REPORT.md`, `docs/SLIDES.pptx`, `docs/DEMO_SCRIPT.md`, `docs/QUESTIONS_AND_ANSWERS.md` | ZIP giải nén, cài và chạy được. Báo cáo và slide có thông tin nhóm, số liệu truy được về bằng chứng. Kịch bản demo hoàn thành trong 5–7 phút. |

## Kiến trúc và dependency đích

Giữ ứng dụng Streamlit bảy trang. Luồng vẫn đi từ `app.py` và `sidebar.py` qua `image_io.py` và `state.py`, vào `pipeline.py`, sau đó gọi `algorithms/` và `processing/` để trả kết quả cho giao diện, tiện ích xuất và script thực nghiệm. `pipeline.py` tiếp tục là chỗ dùng chung quy ước dữ liệu và đầu ra. Sửa tại ranh giới đọc mask và trạng thái tham chiếu, rồi truyền trạng thái đó đến các trang cần hiển thị.

| Quyết định | Thành phần |
|---|---|
| Giữ nguyên | Lõi Lloyd và Otsu, quy tắc đồng hạng, xử lý `uint8`, lựa chọn foreground, kiến trúc nhiều trang, chữ ký đầu vào, tên và định dạng tệp xuất đã công bố, mẫu và mask đang có. |
| Sửa | Tách lỗi ảnh nguồn khỏi lỗi mask mẫu. Bổ sung chế độ xác thực mask nhị phân cho đường đánh giá. Cập nhật lựa chọn lớp Multi-Otsu theo nhãn thực tế. Kiểm tra danh sách tệp bắt buộc khi đóng gói. |
| Bổ sung | Test hồi quy cho mask và khôi phục sau lỗi, kiểm tra cấu trúc gói nộp bằng thư mục tạm, bằng chứng chạy sạch và diễn tập Chrome, thông tin nhóm và kết quả đúng phiên bản. |
| Có thể loại bỏ sau nghiệm thu | `mask_scores()` và `MultiOtsuResult.n_classes` hiện chưa có nơi gọi trong repo. Xác nhận hợp đồng công khai và người gọi bên ngoài trước khi bỏ. Hai đường tạo mask và cutout có thể dùng chung một đoạn triển khai nếu thay đổi đó bảo toàn đầu ra. |

Giữ NumPy, Pillow, Streamlit, Matplotlib và pandas ở runtime, cùng pytest, Ruff và pip-tools ở môi trường phát triển. Phiên bản tham chiếu tiếp tục theo các lockfile Python 3.11 hiện có. `.venv` cũ có một số bản cài lệch pin, nên phép xác nhận bản nộp thực hiện bằng môi trường mới từ wheelhouse. Hiện chưa có dependency phải thay thế và không đưa OpenCV, scikit-learn hoặc scikit-image vào dự án. [Khai báo runtime](../requirements.in) · [Khai báo dev](../requirements-dev.in)

## Trình tự triển khai

### 1. Thuật toán

| Công việc | Vấn đề hiện tại | Cách giải quyết và file dự kiến sửa | Ưu tiên | Rủi ro | Điều kiện hoàn thành |
|---|---|---|---|---|---|
| Xác nhận lõi và ràng buộc giảng viên | Lõi đã đạt kiểm thử ở Giai đoạn 1. Các bước sửa I/O và pipeline có thể làm đổi đầu vào hoặc quy ước foreground. | Chạy `tests/test_kmeans.py`, `tests/test_otsu.py` và các test cấm import/dependency trong `tests/test_integration.py`. Khi có ca tái hiện sai, sửa đúng nhánh ở `src/algorithms/kmeans.py` hoặc `otsu.py` cùng test tương ứng. | P0 | Thay đổi cách đánh số cụm hoặc phá hòa có thể làm đổi kết quả dù hàm mục tiêu tương đương. | Nhãn gần nhất, SSE và số đếm khớp. Otsu khớp phép vét cạn, giữ các ngưỡng chuẩn trong bảng yêu cầu. Kiểm tra dependency đạt. |
| Sửa gợi ý Multi-Otsu khi lớp rỗng | Ảnh đồng nhất chỉ có lớp 0 nhưng hệ thống gợi ý lớp 1 và 2. | Giới hạn gợi ý mặc định theo các lớp thực sự có pixel trong `src/processing/pipeline.py`. Với ảnh hai mức, tiếp tục cho chọn các lớp có dữ liệu theo kết quả dự phòng. Cập nhật tùy chọn tương ứng ở `src/ui/page_otsu.py`. | P2 | Lọc sai lớp làm đổi mặc định trên ảnh đủ ba lớp hoặc làm sai lựa chọn đã lưu. | Ảnh một mức có mặc định rỗng. Ảnh hai mức chỉ gợi ý lớp có dữ liệu. Ảnh đủ ba lớp giữ hành vi hiện hành. |

### 2. Input và Output

| Công việc | Vấn đề hiện tại | Cách giải quyết và file dự kiến sửa | Ưu tiên | Rủi ro | Điều kiện hoàn thành |
|---|---|---|---|---|---|
| Xác thực mask người dùng trước khi chấm | PNG có dãy `0,128,255` được chấm sau chuyển ngầm, còn CSV cùng dãy bị từ chối. | Thêm tùy chọn xác thực nhị phân rõ ràng trong `src/utils/image_io.py` và gọi từ đường đánh giá trong `src/ui/page_compare.py`. Chấp nhận giá trị nhị phân 0/1 và 0/255 với CSV hoặc ảnh 8 bit, giữ hỗ trợ ảnh 16 bit với nhãn 0/1 và đen/trắng ở thang gốc. Giữ chế độ đọc mức xám hiện có cho người gọi dùng quy ước cũ. Báo lỗi mask nhiều mức, trong suốt hoặc sai kích thước trước khi tính điểm. | P1 | Chuẩn hóa 16 bit quá sớm có thể làm mất nhãn 1 hoặc biến giá trị trung gian thành nhãn nhị phân. | PNG và CSV `0,128,255` đều bị từ chối ở đường chấm. Mask hợp lệ 8/16 bit, mask toàn nền và toàn đối tượng giữ đúng nghĩa. Tệp tải lên lỗi không sinh điểm hoặc tự thay bằng mask mẫu. |
| Cho phép dùng ảnh khi mask mẫu hỏng hoặc thiếu | Lỗi đọc mask mẫu hiện làm mất cả `InputContext`. | Đọc và xác thực ảnh nguồn độc lập với mask. Truyền cảnh báo tham chiếu tùy chọn qua `src/ui/sidebar.py` và `src/ui/state.py`. Giữ nguyên xử lý lỗi chặn khi ảnh nguồn không đọc được. | P2 | Tái sử dụng sai mask hoặc cảnh báo cũ khi đổi ảnh. | Ảnh còn hợp lệ vẫn chạy và xuất được khi mask thiếu, hỏng hoặc sai kích thước. Lỗi ảnh nguồn vẫn xóa kết quả cũ. Đổi sang mẫu có mask hợp lệ khôi phục điểm bình thường. |
| Xác nhận định dạng xuất | Đã có kiểm tra round-trip, cần xác nhận lại sau khi thay đổi đường đọc mask. | Dùng các hàm xuất hiện có trong `src/processing/pipeline.py`. Bổ sung ca test khi có khoảng trống trong `tests/test_image_io.py` hoặc `tests/test_integration.py`. `otsu_binary_matrix.csv` giữ nội dung mask foreground 0/1. | P1 | Dùng nhầm ảnh ngưỡng thô thay cho mask sau chọn lớp hoặc áp phép hình thái học. | PNG mask, alpha RGBA và CSV mask cho cùng vùng đối tượng. ZIP chứa đúng PNG, CSV và JSON thuộc lần chạy hiện tại. |

### 3. Giao diện

| Công việc | Vấn đề hiện tại | Cách giải quyết và file dự kiến sửa | Ưu tiên | Rủi ro | Điều kiện hoàn thành |
|---|---|---|---|---|---|
| Hiển thị trạng thái ảnh và mask rõ ràng | Lỗi mask mẫu đang được hiển thị như lỗi ảnh. Quy ước mask ở uploader và hướng dẫn chưa nhất quán. | Cập nhật `src/ui/page_compare.py`, `components.py` và `page_guide.py` để nêu rõ định dạng, `0 = nền`, giá trị dương hợp lệ là đối tượng và ảnh vẫn phân đoạn được khi thiếu tham chiếu. Tái sử dụng component cảnh báo và luồng kết quả hiện có. | P1 | Hai thông báo trùng nhau hoặc download cũ vẫn hiện sau khi đổi ảnh. | Cảnh báo chỉ đúng đối tượng bị lỗi. Điểm biến mất khi mask lỗi, ảnh kết quả vẫn xem được, và kết quả xuất đổi đúng theo ảnh hoặc tham số đã chạy. |
| Chuẩn bị đường demo trên lớp | Mẫu thuận lợi và ảnh thật cần cấu hình foreground khác nhau. Kiểm tra AppTest chưa xác nhận thao tác trình duyệt. | Dùng luồng trong `docs/DEMO_SCRIPT.md`. Kịch bản bắt buộc gồm chọn mẫu hình tròn hoặc nhiều hình màu, chạy hai thuật toán, chọn foreground, so sánh cạnh nhau, tải CSV lỗi rồi CSV hợp lệ, chạy một JPG thật đã chuẩn bị và mở kết quả tải xuống. Dùng mẫu có mask cho phần chấm điểm. | P1 | JPG phức tạp làm gợi ý tự động sai hoặc K-means chậm. | Hoàn thành luồng trong 5–7 phút. Dùng được demo ngoại tuyến sau cài đặt. Mở được mask PNG, một CSV và ZIP tải từ Chrome. Multi-Otsu, Lab và hậu xử lý dùng khi cần minh họa bổ sung. |

### 4. Kiểm thử

| Công việc | Vấn đề hiện tại | Cách giải quyết và file dự kiến sửa | Ưu tiên | Rủi ro | Điều kiện hoàn thành |
|---|---|---|---|---|---|
| Unit test thuật toán và I/O | Bộ 561 test bao phủ lõi khá rộng. Hợp đồng mask mới có thêm trường hợp biên. | Tái sử dụng `tests/test_kmeans.py` và `test_otsu.py` cho bất biến đã có. Mở rộng `tests/test_image_io.py` và `test_improvements.py` cho PNG/CSV đa mức, ảnh 8/16 bit hợp lệ, mask rỗng và đầy, cùng Multi-Otsu có một, hai hoặc ba mức. | P2 | Fixture chỉ đi qua đường chuẩn hóa thông thường và bỏ sót khác biệt độ sâu bit. | Test mới tái hiện sai lệch của hành vi cũ và xác nhận quy ước đã chốt. Các ca I/O cũ vẫn đạt. |
| Tích hợp, trạng thái và round-trip | Còn thiếu ca chỉ mất tham chiếu trong khi ảnh nguồn hợp lệ, và ca tải mask nhiều mức ở đường chấm. | Bổ sung vào `tests/test_integration.py` hoặc `test_workflows.py`. Mô phỏng lỗi bằng `monkeypatch` hoặc `tmp_path`, xác nhận cảnh báo, điểm, tình trạng nút tải xuống và khôi phục khi đổi sang tệp hợp lệ. | P2 | Test bám vào tên widget hoặc chi tiết nội bộ thay vì hành vi. | K-means và Otsu vẫn chạy khi chỉ mất mask. Đổi ảnh, mask hoặc lựa chọn foreground cho đúng đầu ra. Toàn bộ pytest và Ruff đạt trước đóng gói. |
| Kiểm tra nghiệm thu bằng Chrome | Kết quả trình duyệt hiện có thuộc các lượt trước. | Chạy checklist demo trên máy Windows từ gói đã giải nén ở bước 6. Ghi phiên bản, bước đã chạy và tệp đã mở trong `docs/TEST_CASES.md`. | P1 | Gói chạy được trong `.venv` phát triển nhưng lỗi tại máy trình chiếu. | Các bước chọn tệp, chạy, xem cảnh báo, chuyển trang, so sánh và tải xuống có kết quả quan sát được trên bản chốt. |

### 5. Tối ưu

| Công việc | Vấn đề hiện tại | Cách giải quyết và file dự kiến sửa | Ưu tiên | Rủi ro | Điều kiện hoàn thành |
|---|---|---|---|---|---|
| Đo trước khi chỉnh hiệu năng | K-means RGB/Lab trên ảnh nhiều màu chạy đồng bộ. Giai đoạn 1 chưa đo lại hiệu năng. | Dùng `scripts/benchmark_kmeans.py` và một JPG thật có sẵn. Ghi cấu hình, phạm vi thời gian và trung vị ít nhất ba lượt trên máy demo. Nếu độ trễ cản trở kịch bản, ưu tiên chọn cấu hình demo trong giới hạn cạnh dài hiện có. Chỉ sửa `kmeans.py` hoặc `pipeline.py` khi phép đo chỉ ra điểm nghẽn cụ thể. | P2 | So lệch cấu hình hoặc giảm độ phân giải có thể làm thay đổi kết quả. | Các phép đo trước và sau dùng cùng cấu hình khi đánh giá thay đổi mã. Demo đạt khung thời gian. Mọi tối ưu mã giữ quy ước nhãn, mask và SSE trong bộ test. |
| Thu gọn mã dư | Phần tạo mask và cutout có đoạn lặp, một số hàm hoặc thuộc tính chưa có nơi gọi. | Cân nhắc sau bản nộp. Rà `src/processing/pipeline.py`, `src/algorithms/otsu.py` và chỗ lưu ma trận ví dụ trong `src/ui/page_theory.py`. | P3 | Tăng phạm vi hồi quy gần ngày bảo vệ. | Nếu triển khai sau, bộ test và so sánh đầu ra giữ đúng hành vi hiện tại. |

### 6. Đóng gói

| Công việc | Vấn đề hiện tại | Cách giải quyết và file dự kiến sửa | Ưu tiên | Rủi ro | Điều kiện hoàn thành |
|---|---|---|---|---|---|
| Kiểm tra tệp bắt buộc khi tạo ZIP | Các module lõi, test, báo cáo và slide được lấy theo glob. Một tệp thiếu có thể không gây lỗi. | Mở rộng kiểm tra trong `scripts/package_submission.py` cho entry point, hai lõi, pipeline và I/O, bộ test, README, lockfile, báo cáo, slide, kịch bản và tài liệu phản biện. Giữ kiểm tra manifest. Thêm ca thiếu tệp bắt buộc ở `tests/test_package_submission.py` dùng `tmp_path`. | P1 | Danh sách quá cứng có thể buộc cập nhật khi đổi đường dẫn hợp lệ. | ZIP hợp lệ có đủ nhóm đầu vào và tài liệu bắt buộc. Ca thiếu source hoặc slide bị báo lỗi. Kiểm tra SHA-256 và CRC của gói chốt đạt. |
| Cài sạch từ gói phát hành | `.venv` phát triển lệch một số pin. Gói wheel đã qua kiểm tra danh sách và checksum. | Dùng `requirements.txt` và `requirements-dev.txt` theo README. Tạo môi trường mới bằng CPython 3.11 x64, cài offline với wheelhouse, chạy `pip check`, Ruff, pytest và Chrome. Cập nhật README nếu phát hiện bước cài chưa rõ. | P1 | Wheel sai hệ điều hành, kiến trúc hoặc ABI. Cache cũ có thể che lỗi cài. | Cài đủ đúng pin trên Windows x64, app mở được, bộ test đạt, kiểm tra OpenCV đạt và kịch bản trình duyệt hoàn tất. Khi tài liệu thay đổi ở bước 7, tạo lại ZIP nguồn từ phiên bản chốt và kiểm tra lại danh sách tệp cùng checksum. |

### 7. Báo cáo và thuyết trình

| Công việc | Vấn đề hiện tại | Cách giải quyết và file dự kiến sửa | Ưu tiên | Rủi ro | Điều kiện hoàn thành |
|---|---|---|---|---|---|
| Chốt báo cáo và bộ slide | Bìa thiếu thông tin nhóm, slide ghi mốc 560 test của một lần chạy trước. | Điền thông tin do nhóm cung cấp trong `docs/REPORT.md` và `docs/SLIDES.pptx`. Đồng bộ `docs/SLIDES_OUTLINE.md`, `docs/TEST_CASES.md` và kịch bản với kết quả từ bản sạch. Ghi cấu hình, nguồn ảnh và phạm vi đo cho số liệu minh họa. | P1 | Điền nhầm thông tin hoặc dùng kết quả chạy ở phiên bản khác. | Báo cáo và 15 slide mở được, bìa đủ thông tin, số liệu kiểm tra truy được về bản chốt và mọi số liệu hiệu năng được ghi theo lần đo thực tế. |
| Chuẩn bị phản biện | Cần giải thích rõ các trường hợp hai phương pháp cho mask khác nhau. | Tập với `docs/QUESTIONS_AND_ANSWERS.md` và `docs/DEMO_SCRIPT.md`. Dùng ma trận nhỏ để giải thích Lloyd, cách chọn ngưỡng Otsu, ý nghĩa SSE và IoU, ngưỡng đồng hạng, ảnh đồng nhất và giới hạn của gợi ý theo viền. | P1 | Dùng chỉ số trùng khớp giữa hai mask để suy ra độ chính xác khi thiếu ground truth. | Giải thích được K, seed, điều kiện dừng, hướng bất đẳng thức Otsu, khác biệt giữa ảnh nhị phân và mask foreground, cách chấm IoU và vai trò của từng thư viện. |

## Cách bàn giao giữa các bước

Triển khai theo thứ tự trên, mỗi thay đổi có ca xác nhận tương ứng. Ghi lệnh, môi trường và kết quả kiểm thử của bước vừa hoàn tất. Tạo commit riêng cho thay đổi đã kiểm tra để có thể đối chiếu khi hồi quy. Checklist trình duyệt được chuẩn bị ở bước 3, chạy chính thức sau cài sạch ở bước 6 và dùng làm bằng chứng ở bước 7. Điều kiện chốt là các mục P0 và P1 đạt, các hồi quy đã sửa có test, và ZIP cuối chứa đúng source cùng tài liệu đã diễn tập.
