# Checklist bàn giao · Đề tài 3

## Cây thư mục gói nguồn

```text
PTIT-DeTai3-ImageSegmentation/
├── app.py, README.md, .python-version, .gitignore
├── requirements.in, requirements.txt
├── requirements-dev.in, requirements-dev.txt
├── pytest.ini, ruff.toml, .streamlit/config.toml
├── .github/workflows/ci.yml
├── src/
│   ├── algorithms/               K-means và Otsu tự cài đặt
│   ├── processing/               Tiền xử lý, tách nền và xuất kết quả
│   ├── ui/                       Các trang Streamlit và tài nguyên giao diện
│   └── utils/                    Đọc/ghi, chỉ số đánh giá và biểu đồ
├── data/
│   ├── samples.json              Danh mục 11 mẫu
│   ├── sample_images/            9 ảnh tổng hợp
│   ├── sample_csv/               2 ma trận hợp lệ và 3 tệp kiểm tra lỗi
│   └── ground_truth/             8 mask tham chiếu
├── scripts/                      Tạo mẫu, thực nghiệm và đóng gói
├── tests/                        Pytest và kiểm thử trình duyệt
└── docs/
    ├── REPORT.md, SLIDES.pptx, SLIDES_OUTLINE.md
    ├── ACCEPTANCE.md, TEST_CASES.md, PERFORMANCE.md
    ├── DEMO_SCRIPT.md, ALGORITHMS.md, QUESTIONS_AND_ANSWERS.md
    └── figures/, screenshots/, results/ và các ghi chép kiểm tra
```

## Thành phần bàn giao

| Thành phần | Mục đích | Vị trí |
| --- | --- | --- |
| Mã nguồn và cấu hình | Cài đặt và chạy từ `app.py` | `app.py`, `src/`, `.streamlit/` |
| Khóa phiên bản | Tái lập môi trường chạy và kiểm thử trên Python 3.11 | `requirements*.in`, `requirements*.txt` |
| Hướng dẫn sử dụng | Chạy Windows/Linux, thao tác với ảnh và CSV, tùy chọn ngoại tuyến | [README](../README.md) |
| Dữ liệu và mask mẫu | Trình diễn offline và đối chiếu kết quả | `data/` |
| Test và dấu vết nghiệm thu | Kiểm tra thuật toán, I/O, ứng dụng và nội dung tệp tải xuống | `tests/`, [test case](TEST_CASES.md), [nghiệm thu](ACCEPTANCE.md) |
| Báo cáo và slide | Thuyết minh thuật toán, kết quả thực nghiệm và demo | [Báo cáo](REPORT.md), [slide](SLIDES.pptx), [kịch bản demo](DEMO_SCRIPT.md) |
| Wheel Windows x64 | Cài ngoại tuyến trên CPython 3.11 khi cần | `PTIT-DeTai3-Wheels-Windows-x64-Python311.zip`, đặt cạnh gói nguồn |

`scripts/package_submission.py` tạo ZIP theo danh sách tệp cho phép. Gói nguồn gồm tài liệu và các bằng chứng cần để xem lại. Môi trường ảo, cache, `.git/`, `outputs/`, `.streamlit/secrets.toml`, dữ liệu nghiên cứu tải xuống và gói wheel nằm ngoài ZIP nguồn. Script dừng nếu thiếu báo cáo, slide hoặc tệp được khai báo trong danh mục ảnh mẫu.

## Trước khi gửi

- [x] Có nguồn, cấu hình, khóa dependency, README, test, dữ liệu demo, báo cáo và slide.
- [x] Đã cài và kiểm thử lại từ ZIP giải nén trên Windows x64 với CPython 3.11.7. Kiểm tra nghiệm thu gồm K-means, Otsu, CSV, phục hồi sau tệp lỗi và tải kết quả.
- [x] Đã kiểm tra OpenCV và xác minh SHA-256 cho hai ZIP. `SHA256SUMS.txt` nằm cạnh hai ZIP bàn giao.
- [ ] Điền tên nhóm trên trang bìa slide và họ tên, mã sinh viên, công việc, tỷ lệ đóng góp trong bản báo cáo dùng để nộp. Thông tin nhóm hiện để trống.
- [ ] Nếu dùng máy trình chiếu khác, chạy thử theo kịch bản trên máy đó. Dùng gói wheel riêng nếu máy trình chiếu cần cài ngoại tuyến trên Windows x64.
