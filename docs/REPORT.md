# BÁO CÁO BÀI TẬP LỚN MÔN XỬ LÝ ẢNH

**Đề tài 3: Hệ thống phân đoạn ảnh bằng K-means và Thresholding (Otsu)**

Học phần: Xử lý ảnh (INT13146) — Học viện Công nghệ Bưu chính Viễn thông
Giảng viên: Nguyễn Xuân Đức

| Thành viên | Mã sinh viên | Công việc | % đóng góp |
|---|---|---|---|
| *(điền)* | *(điền)* | *(điền)* | *(điền)* |
| *(điền)* | *(điền)* | *(điền)* | *(điền)* |
| *(điền)* | *(điền)* | *(điền)* | *(điền)* |

> Đề bài yêu cầu nhóm ghi rõ % đóng góp của từng thành viên (Slide0, trang 5).

---

## Chương 1. Giới thiệu đề tài

### 1.1 Lý do lựa chọn

Phân đoạn ảnh là bước tách ảnh thành các vùng có ý nghĩa, đặc biệt là tách **đối tượng** khỏi **nền**. Giáo trình
nhận định đây là bước “quan trọng nhất nhưng cũng khó khăn nhất” trong hệ thống xử lý ảnh và là tiền đề cho trích chọn
đặc trưng, nhận dạng (Chương 5, trang 64). Hai cách tiếp cận cơ bản, dễ giải thích và áp dụng rộng là **phân ngưỡng**
(Otsu) và **phân cụm** (K-means). Đề tài 3 (Slide0, trang 8; tối đa 8 điểm) yêu cầu cài đặt và so sánh hai phương
pháp này.

### 1.2 Mục tiêu

1. Tự cài đặt K-means và Otsu bằng Python/NumPy, **không dùng OpenCV** (Slide0, trang 5) và không dùng hàm thuật
   toán có sẵn (`sklearn.cluster.KMeans`, `skimage.filters.threshold_otsu`).
2. Xây dựng ứng dụng cho phép **tải/lưu ảnh (file CSV/ảnh thật), xem kết quả trực quan** (yêu cầu nghiệm thu).
3. Tách nền và đối tượng bằng cả hai phương pháp; so sánh trên cùng ảnh.
4. Tự chuẩn bị test case (Slide0, trang 5), kiểm thử tự động, đánh giá định lượng khi có ground truth.

### 1.3 Phạm vi

- Ảnh PNG/JPG/BMP (xám, RGB, có alpha, có EXIF) và ma trận CSV mức xám.
- K-means với K = 2..8 trên đặc trưng mức xám hoặc RGB; Otsu toàn cục hai lớp trên ảnh xám.
- Không dùng thông tin không gian (lân cận), không dùng học sâu, chạy hoàn toàn cục bộ.

### 1.4 Kết quả cần đạt

Ứng dụng Streamlit 7 trang chạy trên Windows; 11 ảnh mẫu offline có ground truth khi phù hợp; 86 kiểm thử tự động;
báo cáo, slide, kịch bản demo và câu hỏi phản biện.

---

## Chương 2. Cơ sở lý thuyết

### 2.1 Ảnh số, ảnh xám và ảnh màu

Ảnh số là tập hợp các điểm ảnh, biểu diễn bởi mảng hai chiều I(n, p) (giáo trình mục 2.6). Có ba loại chính
(mục 2.6.1): ảnh nhị phân (2 mức), ảnh xám (thường 256 mức, 8 bit) và ảnh màu (3 thành phần R, G, B). Ảnh màu được
chuyển sang xám bằng trung bình cộng (R + G + B)/3 (mục 4.2.1) hoặc theo độ chói 0.299R + 0.587G + 0.114B
(chuẩn ITU-R BT.601, kiến thức bổ sung). Histogram h(g) là số điểm ảnh có mức xám g (mục 3.1.1).

### 2.2 Phân đoạn ảnh

Phân đoạn chia miền ảnh X thành các vùng R₁..R_N thoả: ⋃Rᵢ = X, Rᵢ ∩ Rⱼ = ∅, P(Rᵢ) = TRUE, P(Rᵢ ∪ Rⱼ) = FALSE
với P là vị từ đồng nhất (công thức 5.1–5.4). Giáo trình phân ba nhóm kỹ thuật: cục bộ, toàn thể (dựa trên histogram)
và tách/hợp vùng (trang 65); các hướng gồm phân vùng theo ngưỡng biên độ (5.2), theo miền đồng nhất (5.3), theo đường
biên (5.4) và theo kết cấu (5.5).

- **Phân đoạn dựa trên ngưỡng:** I_B(x, y) = Max nếu I(x, y) ≥ θ, ngược lại Min (kỹ thuật tách ngưỡng, mục 3.1.3);
  ngưỡng chọn từ histogram (mục 5.2).
- **Phân đoạn dựa trên phân cụm:** mỗi pixel là một vector đặc trưng; nhóm các vector gần nhau thành cụm (nhận dạng
  không giám sát, mục 6.1 và 6.2.4).

### 2.3 Thuật toán K-means

**Bài toán** (mục 6.2.4.2): cho N vector x₁..x_N và K, tìm tâm Z₁..Z_K làm cực tiểu

  J = Σ_k Σ_{x∈C_k} ‖x − Z_k‖²   (công thức 6.9)

Với phép gán cố định, ∂J/∂Z_k = 0 ⇒ Z_k = (1/N_k) Σ_{x∈C_k} x (công thức 6.10) — vì thế gọi là “K trung bình”.

**Giả mã:**

```
1. Khởi tạo K tâm (K-means++ hoặc K giá trị phân biệt ngẫu nhiên)
2. Lặp tới khi hội tụ hoặc đạt max_iter:
   a) Gán mỗi pixel vào tâm gần nhất theo khoảng cách Euclid
   b) Cập nhật tâm = trung bình các pixel của cụm (cụm rỗng: đặt tâm tại điểm xa nhất)
   c) Dừng nếu độ dịch tâm lớn nhất ≤ tol
3. Gán nhãn lần cuối, tính SSE, số pixel mỗi cụm
```

Mỗi bước không làm tăng J nên thuật toán dừng sau hữu hạn bước, nhưng chỉ tới **cực tiểu địa phương**. Khởi tạo
K-means++ (Arthur & Vassilvitskii, 2007 — bổ sung) chọn tâm mới với xác suất ∝ D(x)² để giảm rủi ro này.

### 2.4 Thuật toán Otsu

Kiến thức chuẩn (N. Otsu, 1979 — bổ sung). Với pᵢ = h(i)/N và ngưỡng t chia C₀ = {I ≤ t}, C₁ = {I > t}:

  ω₀(t) = Σ_{i≤t} pᵢ,  ω₁ = 1 − ω₀,  μ(t) = Σ_{i≤t} i·pᵢ,  μ_T = μ(255)
  μ₀ = μ(t)/ω₀,  μ₁ = (μ_T − μ(t))/ω₁
  σ²_B(t) = ω₀ω₁(μ₀ − μ₁)² = (μ_T·ω₀ − μ(t))² / (ω₀ω₁),  t* = argmax σ²_B(t)

Vì σ²_T = σ²_W + σ²_B không đổi, cực đại σ²_B ⇔ cực tiểu phương sai trong lớp.

**Liên hệ giáo trình.** Mục 3.1.6 *Kỹ thuật tìm tách ngưỡng tự động* (trang 31–32) dùng
f(g) = t(g)/(mn − t(g)) · [m(g) − m(G−1)]². Thay t(g)/(mn − t(g)) = ω₀/ω₁, m(g) = μ₀, m(G−1) = μ_T và
μ₀ − μ_T = ω₁(μ₀ − μ₁) được **f(g) = ω₀ω₁(μ₀ − μ₁)² = σ²_B(g)**. Vậy kỹ thuật ở mục 3.1.6 chính là tiêu chí Otsu,
dù giáo trình không gọi tên. Các kỹ thuật khác ở mục 5.2 — đẳng liệu (5.2.1), đối xứng nền (5.2.2), tam giác (5.2.3)
— là những phương pháp chọn ngưỡng **khác** Otsu.

**Giả mã:**

```
1. h = histogram 256 mức;  p = h/N
2. ω₀ = cumsum(p);  μ = cumsum(i·p);  μ_T = μ[255]
3. Với mọi t làm hai lớp khác rỗng: σ²_B(t) = (μ_T·ω₀ − μ)² / (ω₀(1 − ω₀))
4. t* = floor(trung bình các t đạt cực đại)
5. B = 255 nếu I > t*, ngược lại 0
```

### 2.5 Ưu, nhược điểm

| | K-means | Otsu |
|---|---|---|
| Ưu điểm | Dùng được màu; tách nhiều vùng; tổng quát | Không tham số; tối ưu toàn cục; rất nhanh O(N + 256) |
| Nhược điểm | Phải chọn K; phụ thuộc khởi tạo; chậm hơn; không biết cụm nào là đối tượng | Chỉ 2 lớp; chỉ dùng mức xám; kém khi histogram không hai đỉnh, đối tượng nhỏ, chiếu sáng không đều |
| Chung | Không dùng thông tin lân cận ⇒ nhạy nhiễu muối tiêu | |

Chi tiết công thức và quyết định cài đặt: [ALGORITHMS.md](ALGORITHMS.md).

---

## Chương 3. Phân tích và thiết kế hệ thống

### 3.1 Yêu cầu chức năng

| Mã | Yêu cầu |
|---|---|
| F1 | Tải ảnh PNG/JPG/JPEG/BMP, CSV; chọn ảnh mẫu; hiển thị ảnh gốc và thông tin (kích thước, số kênh, định dạng, dung lượng) |
| F2 | K-means tự cài đặt với K, khởi tạo, số vòng lặp, ngưỡng hội tụ, seed; hiển thị ảnh phân cụm, bản đồ nhãn, thống kê, biểu đồ |
| F3 | Chọn cụm làm đối tượng ⇒ mask, ảnh tách nền |
| F4 | Otsu tự cài đặt; histogram có đường ngưỡng; ngưỡng thủ công để đối chiếu; chọn lớp đối tượng |
| F5 | So sánh hai thuật toán trên cùng ảnh; IoU/Dice khi có ground truth |
| F6 | Tải xuống PNG/CSV/JSON đúng với kết quả đang hiển thị |

### 3.2 Yêu cầu phi chức năng

Chạy cục bộ trên Windows, không cần Internet; không dùng OpenCV; không treo với ảnh lớn (giới hạn tệp 30 MB, 40 triệu
pixel, thu nhỏ theo cạnh dài); không crash với dữ liệu sai (thông báo tiếng Việt); thuật toán tách khỏi giao diện để
kiểm thử; kết quả tái lập được (seed).

### 3.3 Sơ đồ xử lý tổng thể

```
 Ảnh/CSV ──► Kiểm tra & đọc ──► Giới hạn kích thước ──► Chuyển xám
                                                         │
                     ┌───────────────────────────────────┴──────────────┐
                     ▼                                                  ▼
          K-means (xám hoặc RGB)                              Otsu (histogram)
          nhãn K cụm ──► người dùng chọn cụm đối tượng        ngưỡng t* ──► chọn lớp sáng/tối
                     └──────────────► Mask nhị phân ◄───────────────────┘
                                          │
                         Ảnh tách nền, overlay, thống kê, IoU/Dice (nếu có GT)
                                          │
                                   Tải xuống PNG/CSV/JSON
```

### 3.4 Kiến trúc chương trình

Bốn tầng, phụ thuộc một chiều từ trên xuống:

| Tầng | Module | Vai trò |
|---|---|---|
| Giao diện | `app.py`, `src/ui/*` | Điều hướng, widget, hiển thị; không chứa thuật toán |
| Luồng xử lý | `src/processing/pipeline.py` | Ghép tiền xử lý + thuật toán + hậu xử lý + tạo tệp xuất; dùng chung cho UI, thực nghiệm, kiểm thử |
| Xử lý ảnh | `src/processing/preprocessing.py`, `postprocessing.py` | Chuyển xám, đổi kích thước; mask, tách nền, tô màu nhãn, gợi ý nền |
| Thuật toán | `src/algorithms/kmeans.py`, `otsu.py` | Chỉ phụ thuộc NumPy; trả dataclass kết quả |
| Tiện ích | `src/utils/image_io.py`, `metrics.py`, `visualization.py` | Đọc/ghi và kiểm tra dữ liệu; IoU/Dice; biểu đồ |

### 3.5 Luồng dữ liệu và trạng thái

Thanh bên đọc đầu vào và tạo `InputContext` (ảnh đã chuẩn bị + **chữ ký** = hash nội dung tệp + giới hạn kích thước +
cách chuyển xám). Khi bấm nút chạy, kết quả được lưu trong `st.session_state` cùng chữ ký. Mỗi lần vẽ trang, nếu chữ
ký không khớp đầu vào hiện tại thì kết quả bị ẩn và có cảnh báo ⇒ không thể tải nhầm kết quả của ảnh cũ. Tham số
nằm trong `st.form` nên thuật toán chỉ chạy khi bấm nút; việc chọn cụm đối tượng chỉ tính lại mask (rất nhanh).

---

## Chương 4. Xây dựng chương trình

### 4.1 Công nghệ

Python 3.11; NumPy (tính toán), Pillow (đọc/ghi/đổi kích thước ảnh), Streamlit 1.66 (giao diện), Matplotlib (biểu
đồ), Pandas (bảng), Pytest (kiểm thử). Không dùng OpenCV, scikit-learn, scikit-image, SciPy — có kiểm thử tự động quét
mã nguồn để bảo đảm (`test_no_opencv_or_library_algorithms_imported`).

### 4.2 Cấu trúc mã nguồn

Xem README mục 4. Tổng cộng 4 module thuật toán/xử lý, 3 module tiện ích, 10 module giao diện, 2 script, 4 tệp kiểm thử.

### 4.3 Cài đặt K-means (`src/algorithms/kmeans.py`)

- `assign_labels`: khoảng cách bình phương theo khai triển ‖x‖² − 2x·c + ‖c‖², tính **theo khối 262 144 điểm** để
  bộ nhớ tạm không phụ thuộc kích thước ảnh.
- `kmeans`: vòng lặp Lloyd có trọng số; cập nhật tâm bằng `np.bincount(labels, weights=w·x_d)`; xử lý cụm rỗng; giảm K
  khi thiếu giá trị phân biệt; gán nhãn lần cuối; sắp cụm theo độ sáng; lưu lịch sử SSE, độ dịch tâm, tâm từng vòng.
- `segment_image_kmeans`: gom pixel trùng giá trị (mã hoá R·65536 + G·256 + B rồi `np.unique`), chạy K-means có trọng
  số trên các giá trị phân biệt rồi trả nhãn về từng pixel. Đây là cùng bài toán với K-means trên mọi pixel (đã kiểm
  thử). Đo với cùng tâm khởi tạo (nên cùng số vòng lặp), K = 4: cảnh RGB 86 400 pixel / 20 451 màu 158 → 47 ms
  (3.4×); 4 hình màu 76 800 pixel / 5 màu 37 → 10 ms (3.6×); ảnh xám nhiễu 65 536 pixel / 256 mức 126 → 6 ms (19.6×).

### 4.4 Cài đặt Otsu (`src/algorithms/otsu.py`)

- `compute_histogram`: `np.bincount(minlength=256)`.
- `otsu_from_histogram`: tính ω₀, μ bằng `cumsum`; σ²_B chỉ tại ngưỡng hợp lệ (số pixel tích luỹ nguyên 0 < c₀ < N);
  ngưỡng đồng hạng ⇒ floor(trung bình); ảnh một mức xám ⇒ `valid = False`; trả về ω₀, ω₁, μ₀, μ₁, σ²_T, η.
- `apply_threshold`: pixel > t ⇒ 255.

### 4.5 Giao diện

Bảy trang: Tổng quan, K-means, Otsu Thresholding, So sánh thuật toán, Thư viện ảnh mẫu, Kiến thức thuật toán, Hướng
dẫn. Thanh bên chứa đầu vào dùng chung. Ảnh nhỏ được phóng to bằng láng giềng gần nhất **chỉ khi hiển thị**. Bản đồ
nhãn và biểu đồ cụm dùng chung một bảng 8 màu cố định theo thứ tự (K ≤ 8 nên không lặp màu).

Hình minh hoạ (ảnh chụp thật từ ứng dụng):

- `docs/screenshots/01_tong_quan.png` — trang Tổng quan
- `docs/screenshots/02_kmeans_ket_qua.png` — kết quả K-means
- `docs/screenshots/03_kmeans_tach_nen.png` — chọn cụm đối tượng, mask, ảnh tách nền, IoU với GT
- `docs/screenshots/04_kmeans_thong_ke.png` — bảng tâm cụm, biểu đồ tỷ lệ, SSE theo vòng lặp
- `docs/screenshots/05_otsu_histogram.png` — histogram, ngưỡng t*, σ²_B(t)
- `docs/screenshots/06_so_sanh.png` — so sánh hai thuật toán
- `docs/screenshots/07_kien_thuc_otsu.png` — đối chiếu ví dụ giáo trình
- `docs/screenshots/08_thu_vien.png`, `09_tai_xuong.png`

### 4.6 Quy trình sử dụng

Chọn ảnh mẫu hoặc tải ảnh/CSV ở thanh bên → trang K-means: chọn tham số, bấm *Chạy K-means*, chọn cụm đối tượng ở tab
*Tách nền* → trang Otsu: bấm *Chạy Otsu*, xem histogram → trang So sánh → tab *Tải xuống*. Chi tiết: trang Hướng dẫn
trong ứng dụng và [DEMO_SCRIPT.md](DEMO_SCRIPT.md).

---

## Chương 5. Thực nghiệm và đánh giá

### 5.1 Môi trường

Windows 10 Pro (10.0.19045), CPU Intel Family 6 Model 165, Python 3.11.7, NumPy 2.4.6. Số liệu sinh bởi
`python scripts\run_experiments.py` (tệp gốc: `docs/results/experiment_results.json` và `.md`).

### 5.2 Bộ dữ liệu

11 mẫu **tổng hợp** sinh bằng `scripts/generate_samples.py` (seed cố định, có ground truth khi phù hợp): hình tròn
đen/nền trắng; chữ nhật sáng/nền tối; 4 hình màu (RGB); 3 khối xám 90/150/215 trên nền 35; hình tròn nhiễu Gauss σ=45
+ muối tiêu; tương phản thấp 150/130; ảnh một màu; cảnh RGB nhiều vùng; ảnh 10×10; CSV 32×32; ma trận 5×6 của giáo
trình. Không có ảnh chụp thật trong bộ dữ liệu (xem 5.7).

### 5.3 Quy tắc thử nghiệm

K-means: K và đặc trưng theo gợi ý của mẫu, K-means++, seed 0, tol 1e-4, max 100 vòng, cụm đối tượng = gợi ý theo viền
ảnh. Otsu: lớp đối tượng = gợi ý theo viền ảnh. Thời gian: trung vị 5 lần, chỉ phần thuật toán.

### 5.4 Kết quả

| Mẫu | Kích thước | K | Vòng | K-means (ms) | t* | Otsu (ms) | IoU K-means | IoU Otsu | Dice K-means | Dice Otsu |
|---|---|---|---|---|---|---|---|---|---|---|
| Hình tròn đen | 256×256 | 2 xám | 1 | 2.8 | 127 | 0.38 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Chữ nhật sáng | 256×192 | 2 xám | 1 | 3.6 | 124 | 0.31 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Nhiều hình màu | 320×240 | 5 RGB | 1 | 17.0 | 148 | 0.47 | **1.0000** | **0.7964** | 1.0000 | 0.8867 |
| Nhiều mức xám | 300×200 | 4 xám | 1 | 4.5 | 119 | 0.38 | **1.0000** | **0.6667** | 1.0000 | 0.8000 |
| Tròn có nhiễu | 256×256 | 2 xám | 3 | 4.4 | 126 | 0.30 | 0.8397 | 0.8397 | 0.9129 | 0.9129 |
| Tương phản thấp | 256×256 | 2 xám | 6 | 4.8 | 139 | 0.38 | 0.9532 | 0.9532 | 0.9761 | 0.9761 |
| Một màu | 128×128 | 1 (giảm từ 2) | 1 | 0.6 | 128 (không hợp lệ) | 0.14 | — | — | — | — |
| Cảnh RGB | 360×240 | 6 RGB | 8 | 42.7 | 139 | 0.55 | — | — | — | — |
| Ảnh 10×10 | 10×10 | 2 xám | 2 | 0.4 | 125 | 0.11 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| CSV 32×32 | 32×32 | 2 xám | 2 | 0.6 | 119 | 0.13 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Ma trận giáo trình | 6×5 | 2 xám | 2 | 0.5 | **1** | 0.09 | — | — | — | — |

“—”: không có ground truth nên không tính. Hình kết quả từng mẫu: `docs/figures/<mã mẫu>.png`
(ảnh gốc | K-means | mask K-means | Otsu | mask Otsu).

### 5.5 Thời gian thực thi

Otsu luôn dưới 1 ms với ảnh ≤ 360×240 và 4.4 ms với ảnh 1024×1024. K-means tỷ lệ với số giá trị phân biệt × K × số
vòng: trường hợp xấu nhất (ảnh ngẫu nhiên, mọi pixel khác màu) K = 4, 20 vòng: 35 ms (128²), 178 ms (256²),
859 ms (512²), 3.7 s (1024²). Vì vậy giao diện mặc định giới hạn cạnh dài 800 px.

### 5.6 So sánh và phân tích

1. **Ảnh hai mức tách biệt** (tròn, chữ nhật, CSV, 10×10): cả hai đạt IoU = 1. Otsu không cần tham số và nhanh hơn
   khoảng 4–12 lần (ví dụ 0.38 ms so với 2.8 ms).
2. **Nhiều đối tượng khác màu/khác mức xám**: K-means với K phù hợp đạt IoU = 1, Otsu chỉ 0.7964 và 0.6667 vì chỉ có
   2 lớp: hình vàng (xám ≈ 187) bị gộp với nền (≈ 234); khối xám 90 bị gộp với nền 35. Đây là ưu điểm rõ nhất của
   K-means: nhiều cụm + dùng màu.
3. **Nhiễu**: cả hai cùng IoU 0.8397 — do không dùng thông tin lân cận, mọi pixel nhiễu vượt ngưỡng đều bị phân sai.
   Đây là **trường hợp thất bại chung**; khắc phục bằng lọc trung vị trước khi phân đoạn (giáo trình 3.2.3) hoặc phép
   toán hình thái (4.4) sau phân đoạn — ngoài phạm vi đề tài.
4. **Tương phản thấp**: η = 0.778, thấp nhất; IoU 0.9532 — hai đỉnh histogram chồng lấn do nhiễu.
5. **Ảnh xám, K = 2**: K-means và Otsu cho **cùng IoU** trên cả 6 ảnh xám chạy với K = 2. Giải thích: điểm dừng của K-means 1 chiều
   với K = 2 thoả t = (μ₀ + μ₁)/2 — chính là điều kiện của thuật toán đẳng liệu (giáo trình 5.2.1) — và trên các ảnh
   này nó trùng phân hoạch cực đại σ²_B. Nói chung hai tiêu chí khác nhau và có thể cho ngưỡng khác nhau khi hai lớp
   có kích thước hoặc phương sai chênh lệch lớn.
6. **Ảnh một màu**: cả hai báo cảnh báo thay vì bịa kết quả.
7. **Chọn K** (cảnh RGB): SSE giảm 235.8 → 133.7 → 103.0 → **11.0** → 6.5 → 3.8 → 2.9 triệu với K = 2..8; “khuỷu” rõ ở
   K = 5 — đúng số vùng màu chính.
8. **Phụ thuộc khởi tạo** (cảnh RGB, K = 6, khởi tạo ngẫu nhiên, seed 0..4): SSE 6.51 / **70.41** / 6.51 / 6.51 /
   11.00 triệu — seed 1 kẹt ở cực tiểu địa phương kém gấp ~10.8 lần.
9. **Gợi ý nền theo viền ảnh** chọn đúng cụm/lớp đối tượng trên cả 8 ảnh có ground truth (IoU < 1 ở ảnh nhiễu và
   tương phản thấp là do pixel bị phân sai, không phải do chọn nhầm lớp). Đây vẫn là heuristic: sẽ sai khi đối tượng
   chạm viền ảnh hoặc nền nhiều màu.
10. **Ví dụ giáo trình**: t* = 1 trùng θ = 1; các giá trị f(g) khớp tới sai số làm tròn của giáo trình.

### 5.7 Thử nghiệm với ảnh chụp thật

*Chưa thực hiện trong bản này — nhóm tải ảnh chụp thật của mình lên ứng dụng, ghi tham số và nhận xét vào bảng dưới.*
Không có ground truth thì chỉ nhận xét trực quan, không ghi IoU.

| Ảnh | Kích thước | Phương pháp & tham số | Ngưỡng / K | Thời gian | Nhận xét |
|---|---|---|---|---|---|
| *(chờ điền)* | | | | | |

---

## Chương 6. Kết luận

### 6.1 Kết quả đạt được

- Tự cài đặt K-means (Lloyd, K-means++, cụm rỗng, có trọng số, tính theo khối) và Otsu (từ histogram, xử lý lớp rỗng,
  ngưỡng đồng hạng, ảnh một mức xám) bằng NumPy, không dùng OpenCV.
- Chứng minh và kiểm chứng số liệu: hàm f(g) ở giáo trình mục 3.1.6 là tiêu chí Otsu.
- Ứng dụng 7 trang: tải/lưu ảnh thật và CSV, xem kết quả trực quan, tách nền, so sánh, tải 7 loại tệp mỗi thuật toán.
- 11 test case offline, 86 kiểm thử tự động đều đạt, số liệu thực nghiệm sinh tự động.

### 6.2 Hạn chế

- Cả hai thuật toán không dùng thông tin không gian ⇒ nhạy nhiễu muối tiêu, chiếu sáng không đều.
- Otsu chỉ 2 lớp toàn cục; K-means cần chọn K, phụ thuộc khởi tạo, và không biết cụm nào là đối tượng — gợi ý theo
  viền ảnh chỉ là heuristic.
- K-means chậm với ảnh lớn có rất nhiều màu khác nhau (~3.7 s cho 1 triệu màu, 20 vòng).
- Bộ dữ liệu là ảnh tổng hợp; chưa đánh giá định lượng trên ảnh thật có ground truth.

### 6.3 Hướng phát triển

Lọc trung vị/trung bình trước phân đoạn (giáo trình 3.2.3–3.2.4); phép mở/đóng hình thái sau phân đoạn (4.4); Otsu
nhiều ngưỡng; Otsu cục bộ theo khối cho chiếu sáng không đều (gợi ý ở trang 68); thêm toạ độ (x, y) vào vector đặc trưng
K-means; chọn K tự động theo phương pháp khuỷu; so sánh thêm thuật toán đẳng liệu (5.2.1).

---

## Tài liệu tham khảo

**Tài liệu gốc (được cung cấp):**

1. Đỗ Năng Toàn, *Bài giảng môn học Xử lý ảnh*, Học viện Công nghệ Bưu chính Viễn thông, Hà Nội, 06/2010
   (`Giaotrinh_XLA_CNTT.pdf`) — các mục 2.6, 3.1.1, 3.1.3, 3.1.6, 4.2.1, Chương 5 (5.1, 5.2, 5.3), 6.2.4.2.
2. Nguyễn Xuân Đức, Slide học phần Xử lý ảnh INT13146 — đề bài tập lớn, Đề tài 3 (`Slide0.pdf`, trang 5 và 8).

**Tài liệu bổ sung (cho nội dung ngoài giáo trình):**

3. N. Otsu, “A Threshold Selection Method from Gray-Level Histograms,” *IEEE Transactions on Systems, Man, and
   Cybernetics*, vol. 9, no. 1, pp. 62–66, 1979.
4. D. Arthur, S. Vassilvitskii, “k-means++: The Advantages of Careful Seeding,” *Proc. 18th ACM-SIAM Symposium on
   Discrete Algorithms (SODA)*, 2007.
5. S. Lloyd, “Least Squares Quantization in PCM,” *IEEE Transactions on Information Theory*, vol. 28, no. 2, 1982.
6. R. C. Gonzalez, R. E. Woods, *Digital Image Processing*, 4th ed., Pearson, 2018 — mục phân ngưỡng Otsu và phân cụm
   K-means (giáo trình chính thứ hai theo Slide0, trang 3).
7. ITU-R Recommendation BT.601 — trọng số độ chói 0.299, 0.587, 0.114.
