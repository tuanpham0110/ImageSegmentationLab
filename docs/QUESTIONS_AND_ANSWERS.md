# Câu hỏi phản biện và trả lời

**1. K-means hoạt động như thế nào?**
Mỗi pixel là một vector đặc trưng (mức xám hoặc R, G, B). Chọn K tâm ban đầu, rồi lặp hai bước: gán mỗi pixel vào tâm
gần nhất theo khoảng cách Euclid; cập nhật mỗi tâm bằng trung bình các pixel của cụm (giáo trình công thức 6.10). Dừng
khi tâm không dịch chuyển quá ngưỡng `tol` hoặc đạt số vòng tối đa. Ảnh phân đoạn thay mỗi pixel bằng màu tâm cụm.

**2. Vì sao cần chọn K?**
K-means cực tiểu J với K cố định; J luôn giảm khi K tăng (K = N cho J = 0), nên không thể chọn K bằng cách cực tiểu J.
K phản ánh số vùng màu ta muốn. Ứng dụng cho chọn K = 2..8; thực nghiệm cho thấy “khuỷu” của SSE (ảnh cảnh RGB: SSE
103 → 11 triệu khi K từ 4 lên 5, sau đó giảm chậm) là gợi ý tốt.

**3. Tâm cụm được cập nhật như thế nào?**
Z_k = (1/N_k) Σ_{x∈C_k} x — trung bình cộng theo từng kênh. Cài đặt dùng `np.bincount(labels, weights=x_d)` chia cho
số điểm. Khi chạy trên ảnh, các pixel trùng giá trị được gom lại với trọng số w = số pixel: Z_k = Σwx/Σw — cho đúng
cùng trung bình như tính trên mọi pixel. Cụm rỗng: đặt tâm tại điểm xa tâm của nó nhất.

**4. Khi nào K-means hội tụ? Có chắc chắn hội tụ không?**
Bước gán và bước cập nhật đều không làm tăng J ≥ 0, số cách gán hữu hạn ⇒ luôn dừng sau hữu hạn bước. Nhưng chỉ đạt
**cực tiểu địa phương**: với cùng ảnh cảnh RGB, K = 6, khởi tạo ngẫu nhiên, seed 1 cho SSE gấp ~10.8 lần seed 2.
Ứng dụng dừng khi độ dịch tâm lớn nhất ≤ tol (mặc định 1e-4) và có `max_iter` để chặn trên.

**5. K-means khác Otsu ở đâu?**
K-means là phân cụm, nhiều cụm, dùng được màu, có tham số và cho cực tiểu địa phương. Otsu là phân ngưỡng một ngưỡng,
chỉ 2 lớp, chỉ dùng histogram mức xám, không tham số và cho tối ưu toàn cục (duyệt đủ 256 ngưỡng). Đầu ra K-means là
nhãn cụm, cần người chọn cụm đối tượng; Otsu cho thẳng ảnh nhị phân.

**6. Otsu tối ưu theo tiêu chí nào?**
Cực đại phương sai giữa hai lớp σ²_B(t) = ω₀ω₁(μ₀ − μ₁)², tương đương cực tiểu phương sai trong lớp vì
σ²_T = σ²_W + σ²_B không đổi. Trong giáo trình mục 3.1.6, hàm f(g) = t(g)/(mn − t(g))·[m(g) − m(G−1)]² biến đổi được
đúng thành σ²_B(g); ví dụ trang 32 cho θ = 1, chương trình cũng cho t* = 1.

**7. Vì sao cần histogram?**
σ²_B chỉ phụ thuộc phân bố mức xám, không phụ thuộc vị trí pixel. Histogram nén N pixel thành 256 số, nên mọi ω₀, μ
tính bằng tổng tích luỹ O(256) và duyệt hết các ngưỡng gần như tức thời (4.4 ms cho ảnh 1 triệu pixel, gồm cả dựng
histogram).

**8. Otsu có hoạt động tốt với ảnh nhiều đối tượng không?**
Chỉ khi các đối tượng cùng phía so với nền về độ sáng. Ảnh mẫu 3 khối xám 90/150/215 trên nền 35: Otsu chọn t* = 119,
khối 90 bị gộp vào nền, IoU = 0.6667; K-means K = 4 tách đủ, IoU = 1. Có thể mở rộng Otsu nhiều ngưỡng nhưng ngoài phạm
vi đề tài.

**9. Vì sao ảnh phức tạp khó tách nền?**
Cả hai phương pháp chỉ dùng giá trị pixel: nếu đối tượng và nền có màu/độ sáng chồng lấn (bóng đổ, chiếu sáng không
đều, nền nhiều màu, nhiễu) thì không ngưỡng hay cụm nào tách được. Ngoài ra khái niệm “đối tượng” là ngữ nghĩa — thuật
toán chỉ thấy màu, nên K-means cần người chọn cụm và Otsu cần chọn lớp sáng/tối.

**10. Vì sao không sử dụng OpenCV?**
Đề bài cấm (Slide0, trang 5) và mục tiêu là tự cài đặt thuật toán của giáo trình. Dự án chỉ dùng NumPy cho tính toán,
Pillow để đọc/ghi/đổi kích thước ảnh, Matplotlib vẽ biểu đồ, Streamlit giao diện. Kiểm thử
`test_no_opencv_or_library_algorithms_imported` quét toàn bộ mã nguồn để chắc chắn không import `cv2`, `sklearn`,
`skimage`, `scipy`.

**11. Phần nào nhóm tự lập trình?**
Toàn bộ K-means (`src/algorithms/kmeans.py`: khoảng cách theo khối, K-means++, vòng lặp Lloyd có trọng số, cụm rỗng,
hội tụ, sắp xếp cụm), toàn bộ Otsu (`src/algorithms/otsu.py`: histogram, ω₀, μ, σ²_B, lớp rỗng, ngưỡng đồng hạng),
chuyển xám, mask, tách nền, IoU/Dice, đọc–kiểm tra CSV, giao diện. Pillow chỉ dùng để giải mã/mã hoá tệp ảnh và đổi kích
thước.

**12. Độ phức tạp thời gian?**
Otsu: O(N) dựng histogram + O(L) với L = 256. K-means: O(U·K·D·T) với U ≤ N số giá trị phân biệt, D số kênh, T số vòng,
cộng O(N log N) gom pixel. Bộ nhớ K-means: O(N) cho nhãn và O(B·K) cho một khối khoảng cách (B = 262 144) nên không
tạo ma trận N×K khổng lồ.

**13. Trường hợp nào nên chọn K-means?**
Ảnh màu; nhiều vùng/nhiều đối tượng khác màu; đối tượng và nền khác màu nhưng gần độ sáng (ví dụ hình vàng trên nền
trắng ngà); cần lượng tử hoá màu. Chấp nhận chọn K và chọn cụm đối tượng.

**14. Trường hợp nào nên chọn Otsu?**
Ảnh xám có histogram hai đỉnh rõ: văn bản quét, đối tượng tương phản với nền, ảnh công nghiệp chiếu sáng đều; khi cần
nhanh, không tham số, kết quả tất định.

**15. Những hạn chế thực tế của sản phẩm?**
Không dùng thông tin lân cận ⇒ nhạy nhiễu (ảnh nhiễu: IoU 0.84 cho cả hai); Otsu toàn cục kém khi chiếu sáng không
đều; K-means chậm với ảnh lớn rất nhiều màu (~3.7 s với 1 triệu màu, 20 vòng) nên mặc định thu nhỏ cạnh dài 800 px;
gợi ý nền theo viền ảnh sai khi đối tượng chạm viền; bộ dữ liệu đánh giá định lượng là ảnh tổng hợp.

**16. Với ảnh xám, K = 2, vì sao K-means và Otsu cho cùng kết quả trên các ảnh thử?**
Điểm dừng của K-means 1 chiều với K = 2 là ngưỡng t thoả t = (μ₀ + μ₁)/2 — đúng điều kiện của thuật toán đẳng liệu
(giáo trình 5.2.1). Otsu cực đại σ²_B, nói chung là tiêu chí khác; trên các ảnh thử (hai lớp tách khá rõ) hai tiêu chí
cho cùng phân hoạch. Khi hai lớp chênh lệch mạnh về kích thước hoặc phương sai, hai ngưỡng có thể khác nhau.

**17. Vì sao Otsu báo “nhiều ngưỡng cùng đạt cực đại”? Chọn ngưỡng nào?**
Nếu histogram có khoảng trống (ví dụ ảnh chỉ có mức 0 và 255), mọi t trong khoảng trống cho cùng phân hoạch nên cùng
σ²_B. Nhóm chọn floor(trung bình các t đồng hạng) — giữa khoảng trống, cùng quy tắc với MATLAB `graythresh`. Mọi lựa
chọn trong đoạn đều cho cùng ảnh nhị phân.

**18. Ảnh chỉ có một màu thì sao?**
Otsu: không có ngưỡng nào làm cả hai lớp khác rỗng ⇒ `valid = False`, cảnh báo, toàn ảnh là nền. K-means: số giá trị
phân biệt = 1 < K ⇒ K giảm về 1, cảnh báo. Không chia cho 0, không crash (kiểm thử `test_uniform_image_*`,
`test_all_black_image`, `test_all_white_image`).

**19. Gom pixel trùng giá trị có làm thay đổi kết quả K-means không?**
Không. Các pixel cùng giá trị luôn có cùng khoảng cách tới mọi tâm nên luôn cùng nhãn; thay chúng bằng một điểm trọng
số w cho cùng J, cùng tâm. Kiểm thử `test_weighted_equals_expanded_data` và `test_image_result_is_exact_kmeans_on_all_pixels`
xác nhận nhãn = tâm gần nhất của từng pixel, tâm = trung bình pixel của cụm, SSE tính trên mọi pixel.

**20. IoU/Dice được tính thế nào và khi nào?**
IoU = |A∩B|/|A∪B|, Dice = 2|A∩B|/(|A|+|B|) giữa mask dự đoán và ground truth. Chỉ tính khi có ground truth (ảnh mẫu
tổng hợp có sẵn, hoặc người dùng tải mask). Không có ground truth thì ứng dụng không hiển thị các chỉ số này; độ trùng
khớp giữa hai mask được ghi rõ là “không phải độ chính xác”.
