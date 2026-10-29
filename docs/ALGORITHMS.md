# Thuật toán và quyết định cài đặt

Tài liệu gốc: **Đỗ Năng Toàn, *Bài giảng môn học Xử lý ảnh*, Học viện Công nghệ BCVT, 2010** (gọi tắt “giáo trình”).
Nội dung không có trong giáo trình được đánh dấu **[bổ sung]** kèm nguồn.

---

## 1. Biểu diễn ảnh và chuyển mức xám

- Ảnh số là mảng hai chiều I(x, y); ảnh xám mỗi pixel 8 bit trong [0, 255]; ảnh màu RGB mỗi pixel 3 thành phần
  (giáo trình mục 2.6, 2.6.1). Ứng dụng dùng mảng NumPy `uint8` dạng (H, W) hoặc (H, W, 3).
- Chuyển ảnh màu sang xám, hai lựa chọn:
  - Trung bình cộng `Gray = (R + G + B) / 3` — giáo trình mục 4.2.1 (trang 48).
  - Độ chói `Gray = 0.299 R + 0.587 G + 0.114 B` — chuẩn ITU-R BT.601 **[bổ sung]**, mặc định vì gần cảm nhận độ
    sáng của mắt hơn.
  Kết quả làm tròn và giới hạn về [0, 255] (`src/processing/preprocessing.py::to_grayscale`).

## 2. Phân đoạn ảnh

Giáo trình Chương 5 (trang 64): chia miền ảnh X thành các vùng R₁..R_N sao cho ⋃Rᵢ = X, Rᵢ ∩ Rⱼ = ∅,
P(Rᵢ) = TRUE và P(Rᵢ ∪ Rⱼ) = FALSE với P là vị từ đồng nhất. Bài toán của đề tài là trường hợp N = 2:
**nền** và **đối tượng**.

- **Otsu** thuộc nhóm *phân vùng theo ngưỡng biên độ* (mục 5.2): vị từ P(R): f(k, l) < T (công thức 5.5).
- **K-means** là *phân cụm không giám sát* (mục 6.2.4.2) áp dụng lên vector đặc trưng của từng pixel; kết quả gồm K
  vùng đồng nhất về màu/mức xám, sau đó người dùng gộp một số cụm thành đối tượng.

Cả hai đều là kỹ thuật **toàn thể** theo phân loại ở trang 65: chỉ dùng giá trị pixel, không dùng quan hệ lân cận.

---

## 3. K-means

### 3.1 Bài toán

Cho N vector đặc trưng x₁..x_N (pixel: x = (g) với ảnh xám, x = (R, G, B) với ảnh màu) và số cụm K. Tìm tâm
Z₁..Z_K và phép gán c(i) ∈ {1..K} làm cực tiểu (giáo trình công thức 6.9):

```
J = Σ_k Σ_{x ∈ C_k} ‖x − Z_k‖²          (SSE, inertia)
```

Với phép gán cố định, ∂J/∂Z_k = 0 cho Z_k = (1/N_k) Σ_{x∈C_k} x (công thức 6.10) — tâm cụm là trung bình cộng.

### 3.2 Thuật toán Lloyd (giáo trình mục 6.2.4.2 b)

```
Vào: X (N×D), K, max_iter, tol, seed, init
1. Khởi tạo K tâm Z (K-means++ hoặc K giá trị PHÂN BIỆT chọn ngẫu nhiên)
2. for t = 1..max_iter:
     a) c(i) = argmin_k ‖xᵢ − Z_k‖²              (gán nhãn)
     b) Z_k  = trung bình các xᵢ có c(i) = k     (cập nhật)
        nếu cụm k rỗng: Z_k = điểm xa tâm của nó nhất (không trùng tâm khác)
     c) nếu max_k ‖Z_k(mới) − Z_k(cũ)‖ ≤ tol và không có cụm rỗng: dừng (hội tụ)
3. Gán nhãn lần cuối theo Z cuối; tính SSE và số pixel mỗi cụm
4. Sắp xếp cụm theo độ sáng tâm (cụm 0 tối nhất)
Ra: nhãn (H×W), tâm, SSE, số vòng, cờ hội tụ, lịch sử SSE/độ dịch tâm
```

**Hội tụ.** Bước (a) không làm tăng J (mỗi điểm chọn tâm gần nhất), bước (b) không làm tăng J (trung bình là cực tiểu
với nhãn cố định). J ≥ 0 và số phép gán là hữu hạn ⇒ dừng sau hữu hạn bước — nhưng chỉ đạt **cực tiểu địa phương**.
Kiểm thử `test_sse_is_non_increasing` xác nhận SSE không tăng.

### 3.3 Quyết định cài đặt (`src/algorithms/kmeans.py`)

| Vấn đề | Cách xử lý |
|---|---|
| Khoảng cách | ‖x − c‖² = ‖x‖² − 2x·c + ‖c‖², tính bằng phép nhân ma trận NumPy, **theo khối 262 144 điểm** ⇒ bộ nhớ tạm ≤ ~16 MB với K = 8 dù ảnh lớn |
| Khởi tạo K-means++ **[bổ sung]** | Arthur & Vassilvitskii (2007), *k-means++: The Advantages of Careful Seeding*, SODA. Tâm đầu ngẫu nhiên; tâm tiếp theo chọn với xác suất ∝ w(x)·D(x)², D(x) = khoảng cách tới tâm gần nhất đã chọn |
| Khởi tạo ngẫu nhiên | Chọn K giá trị **phân biệt** (tránh hai tâm trùng nhau ngay từ đầu) |
| K > số giá trị phân biệt | Giảm K về số giá trị phân biệt, ghi cảnh báo (ví dụ ảnh một màu ⇒ K = 1) |
| Cụm rỗng | Đặt lại tâm tại điểm xa tâm hiện tại của nó nhất, không trùng tâm khác; không được tính là hội tụ ở vòng đó |
| Tái lập | `numpy.random.default_rng(seed)` |
| Nhất quán | Sau vòng lặp gán nhãn lại theo tâm cuối ⇒ nhãn, số pixel, SSE khớp với tâm báo cáo (`test_labels_consistent_with_centers_and_sse`) |
| Thứ tự cụm | Sắp theo độ chói của tâm ⇒ cùng ảnh cho cùng thứ tự cụm, dễ giải thích |
| **Tăng tốc có trọng số** | Ảnh `uint8` có nhiều pixel trùng giá trị. `segment_image_kmeans` gom pixel thành U giá trị phân biệt với trọng số w = số pixel, chạy K-means **có trọng số** J = Σ wᵢ‖xᵢ − Z‖², Z_k = Σwᵢxᵢ / Σwᵢ, rồi trả nhãn về từng pixel. Đây là **cùng một bài toán** với K-means trên mọi pixel (pixel trùng giá trị luôn cùng nhãn). Ảnh xám còn ≤ 256 điểm. Kiểm thử `test_weighted_equals_expanded_data` và `test_image_result_is_exact_kmeans_on_all_pixels` chứng minh |

### 3.4 Chọn đối tượng

K-means **không biết** cụm nào là đối tượng. Người dùng chọn một hoặc nhiều cụm làm foreground. Lựa chọn mặc định
là **gợi ý heuristic**: cụm chiếm nhiều pixel viền ảnh nhất được coi là nền (giả định đối tượng nằm trong ảnh, không
phủ viền). Heuristic này sai khi đối tượng chạm viền hoặc nền nhiều màu — người dùng phải chỉnh tay.

### 3.5 Độ phức tạp

Thời gian O(U·K·D·T) với U ≤ N là số giá trị phân biệt, T là số vòng lặp; cộng O(N log N) cho bước gom pixel.
Bộ nhớ O(N) cho nhãn + O(B·K) cho khối khoảng cách.

---

## 4. Otsu

### 4.1 Công thức **[bổ sung — N. Otsu, “A Threshold Selection Method from Gray-Level Histograms”, IEEE Trans. Systems, Man, and Cybernetics, 9(1), 1979]**

Histogram h(i), i = 0..255 (giáo trình mục 3.1.1), N pixel, pᵢ = h(i)/N. Ngưỡng t chia
C₀ = {I ≤ t}, C₁ = {I > t}:

```
ω₀(t) = Σ_{i≤t} pᵢ        ω₁(t) = 1 − ω₀(t)
μ(t)  = Σ_{i≤t} i·pᵢ       μ_T = μ(255)
μ₀ = μ(t)/ω₀               μ₁ = (μ_T − μ(t))/ω₁
σ²_B(t) = ω₀ω₁(μ₀ − μ₁)² = (μ_T·ω₀(t) − μ(t))² / (ω₀(t)·ω₁(t))
t* = argmax_t σ²_B(t)
```

Vì σ²_T = σ²_W + σ²_B không phụ thuộc t, cực đại σ²_B tương đương cực tiểu phương sai trong lớp σ²_W. Độ tách
η = σ²_B(t*)/σ²_T ∈ [0, 1]. Toàn bộ tính bằng `cumsum` ⇒ O(256) sau khi dựng histogram O(N); t* là **tối ưu toàn cục**
vì duyệt mọi t.

### 4.2 Liên hệ với giáo trình mục 3.1.6 “Kỹ thuật tìm tách ngưỡng tự động” (trang 31–32)

Giáo trình định nghĩa t(g) = số pixel ≤ g, m(g) = (1/t(g)) Σ_{i≤g} i·h(i) và

```
f(g) = t(g) / (m·n − t(g)) · [m(g) − m(G−1)]²,   θ = argmax f(g)
```

mà **không gọi tên Otsu**. Biến đổi: t(g)/(mn − t(g)) = ω₀/ω₁, m(g) = μ₀, m(G−1) = μ_T; từ μ_T = ω₀μ₀ + ω₁μ₁ có
μ₀ − μ_T = ω₁(μ₀ − μ₁). Do đó

```
f(g) = (ω₀/ω₁)·ω₁²(μ₀ − μ₁)² = ω₀ω₁(μ₀ − μ₁)² = σ²_B(g)
```

⇒ **f(g) của giáo trình chính là tiêu chí Otsu**, viết dưới dạng khác. Kiểm chứng trên ví dụ 5×6 của giáo trình:
chương trình cho t* = 1, trùng θ = 1 (`test_textbook_example_section_3_1_6`). Giá trị chính xác f(1) = 1.6806 —
giáo trình ghi 1.66 do làm tròn m(g) trước khi tính.

**Phân biệt với các kỹ thuật khác trong giáo trình (không phải Otsu):**

- Mục 5.2.1 *Thuật toán đẳng liệu* (Ridler–Calvard): lặp θ_k = (m_f + m_b)/2 tới khi ổn định — tìm điểm cân bằng
  hai trung bình, không cực đại σ²_B.
- Mục 5.2.2 *Đối xứng nền*: dựa vào đỉnh nền và tính đối xứng của histogram.
- Mục 5.2.3 *Tam giác* (Zack): khoảng cách lớn nhất từ histogram tới đường nối đỉnh và mức thấp nhất.
- Mục 3.1.5 *Cân bằng histogram* và 3.1.4 *Bó cụm*: biến đổi mức xám, không phải chọn ngưỡng.

### 4.3 Quyết định cài đặt (`src/algorithms/otsu.py`)

| Vấn đề | Cách xử lý |
|---|---|
| Lớp rỗng (ω₀ = 0 hoặc ω₁ = 0) | t đó không hợp lệ, σ²_B = 0, không chia cho 0. Xác định bằng số pixel tích luỹ **nguyên** (không phụ thuộc sai số float) |
| Ảnh một mức xám | Không có t hợp lệ ⇒ `valid = False`, cảnh báo, t = mức xám đó, toàn ảnh là nền |
| Nhiều t cùng cực đại | Chọn **floor(trung bình các t đồng hạng)** (cùng quy tắc MATLAB `graythresh`). Ảnh tổng hợp có khoảng trống trong histogram nên thường có cả một đoạn t cho cùng phân hoạch; lấy giữa đoạn cho ngưỡng “đẹp” |
| Quy ước nhị phân | pixel > t ⇒ 255 (C₁), ngược lại 0 (giống công thức 3.1.3 với θ = t + 1) |
| Lớp nào là đối tượng | Lựa chọn: lớp sáng, lớp tối, hoặc **gợi ý theo viền ảnh** (lớp chiếm đa số viền là nền) — heuristic, cùng giả định như K-means |
| Ngưỡng thủ công | Dùng `apply_threshold` với T người dùng chọn; t* vẫn được tính và vẽ để đối chiếu |

Kiểm chứng: so với cài đặt duyệt trực tiếp từng ngưỡng theo định nghĩa ω₀ω₁(μ₀ − μ₁)² trên 5 ảnh ngẫu nhiên
(`test_random_images_match_brute_force`).

---

## 5. Đánh giá

Chỉ khi có **ground truth** (GT):

```
IoU  = |A ∩ B| / |A ∪ B|          Dice = 2|A ∩ B| / (|A| + |B|)
Độ chính xác pixel = (TP + TN) / N
```

Quy ước cả hai mask rỗng ⇒ IoU = Dice = 1. Khi không có GT, ứng dụng **không** tính các chỉ số này; trang So sánh chỉ
báo *mức trùng khớp giữa hai mask* và ghi rõ đó không phải độ chính xác.

## 6. So sánh

| | K-means | Otsu |
|---|---|---|
| Bản chất | Phân cụm, cực tiểu SSE | Ngưỡng toàn cục, cực đại σ²_B |
| Đầu vào | Mức xám hoặc RGB | Chỉ histogram mức xám |
| Đầu ra | K cụm, cần chọn cụm đối tượng | 2 lớp |
| Tham số | K, khởi tạo, seed, max_iter, tol | Không (chỉ chọn lớp sáng/tối) |
| Tối ưu | Cục bộ, phụ thuộc khởi tạo | Toàn cục |
| Thời gian | O(U·K·D·T) | O(N + 256) |
| Hạn chế chung | Không dùng thông tin không gian ⇒ nhạy nhiễu muối tiêu, chiếu sáng không đều | như bên trái |
