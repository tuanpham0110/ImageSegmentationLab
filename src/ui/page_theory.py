"""Trang Kiến thức thuật toán: công thức, giả mã, và minh hoạ từng bước tính trên dữ liệu nhỏ."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.algorithms.kmeans import segment_image_kmeans
from src.algorithms.otsu import otsu_threshold
from src.ui.components import page_header, show_image
from src.ui.state import DATA_DIR, sample_by_id
from src.utils.image_io import load_path

TEXTBOOK = np.array([[0, 1, 2, 3, 4, 5], [0, 0, 1, 2, 3, 4], [0, 0, 0, 1, 2, 3],
                     [0, 0, 0, 0, 1, 2], [0, 0, 0, 0, 0, 1]], dtype=np.uint8)


def render() -> None:
    page_header("Kiến thức thuật toán", "Cơ sở lý thuyết, công thức và giả mã — đối chiếu với giáo trình Xử lý ảnh (PTIT).")
    t1, t2, t3 = st.tabs(["K-means", "Otsu", "So sánh & độ phức tạp"])
    with t1:
        _kmeans_theory()
    with t2:
        _otsu_theory()
    with t3:
        _complexity()


def _kmeans_theory() -> None:
    st.markdown("""
**Nguồn:** Giáo trình mục 6.2.4.2 *Thuật toán K trung bình* (trang 84); phân vùng ảnh — Chương 5.

Cho tập N vector đặc trưng $x_1,\\dots,x_N$ (mỗi pixel là $x=(g)$ với ảnh xám hoặc $x=(R,G,B)$ với ảnh màu)
và số cụm K. K-means tìm các tâm $Z_1,\\dots,Z_K$ làm cực tiểu hàm mục tiêu (giáo trình công thức 6.9):
""")
    st.latex(r"J = \sum_{k=1}^{K} \sum_{x \in C_k} \lVert x - Z_k \rVert^2 \quad(\text{SSE / inertia})")
    st.markdown("Lấy đạo hàm theo $Z_k$ và cho bằng 0 được tâm cụm là trung bình cộng (công thức 6.10):")
    st.latex(r"\frac{\partial J}{\partial Z_k} = 0 \;\Rightarrow\; Z_k = \frac{1}{N_k}\sum_{x \in C_k} x")
    st.code("""Vào: dữ liệu X (N×D), số cụm K, max_iter, tol, seed
1. Khởi tạo K tâm Z (K-means++ hoặc chọn ngẫu nhiên K giá trị phân biệt)
2. Lặp t = 1..max_iter:
     a) Gán nhãn:  c(x) = argmin_k ||x − Z_k||²        (tính theo khối, tránh tràn RAM)
     b) Cập nhật:  Z_k = trung bình các x có c(x) = k
        – nếu cụm k rỗng: đặt Z_k = điểm xa tâm của nó nhất
     c) Nếu max_k ||Z_k(mới) − Z_k(cũ)|| ≤ tol: dừng (hội tụ)
3. Gán nhãn lần cuối theo Z, tính SSE, số pixel mỗi cụm
Ra: nhãn (H×W), tâm cụm, SSE, số vòng lặp""", language="text")
    st.markdown("""
- **Hội tụ**: mỗi bước (a) và (b) đều không làm tăng J, và số cách gán nhãn là hữu hạn ⇒ thuật toán dừng sau hữu hạn bước,
  nhưng chỉ đảm bảo **cực tiểu địa phương** (phụ thuộc khởi tạo) — vì vậy có random seed.
- **K-means++** (Arthur & Vassilvitskii, 2007 — *kiến thức bổ sung ngoài giáo trình*): tâm mới được chọn với xác suất tỉ lệ
  $D(x)^2$, giúp các tâm ban đầu trải đều.
- **K > số giá trị phân biệt**: K được giảm xuống bằng số giá trị phân biệt và có cảnh báo.
- **Cụm được sắp xếp theo độ sáng tâm** (cụm 0 tối nhất) để kết quả ổn định giữa các lần chạy.
""")
    st.markdown("##### Minh hoạ từng bước trên ảnh 10×10")
    entry = sample_by_id("tiny_10x10")
    if entry is None:
        st.info("Thiếu ảnh mẫu tiny_10x10 — chạy scripts/generate_samples.py.")
        return
    img = load_path(DATA_DIR / entry["file"]).array
    c1, c2 = st.columns([1, 2])
    with c1:
        show_image(img, "Ảnh 10×10")
        seed = st.number_input("Seed", 0, 999, 3, key="theory_seed")
    res = segment_image_kmeans(img, 2, feature="gray", seed=int(seed), init="random", tol=0.0)
    rows = []
    for i, c in enumerate(res.core.centers_history):
        rows.append({"Vòng": i, "Tâm Z0": round(float(c[0, 0]), 3), "Tâm Z1": round(float(c[1, 0]), 3),
                     "SSE (sau gán nhãn)": round(res.core.sse_history[i], 2) if i < len(res.core.sse_history) else None,
                     "Độ dịch tâm": round(res.core.shift_history[i - 1], 4) if i > 0 else None})
    with c2:
        st.caption("Khởi tạo ngẫu nhiên (để thấy tâm thay đổi qua các vòng), K = 2, tol = 0. Vòng 0 là tâm khởi tạo.")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.write(f"Hội tụ sau **{res.core.n_iter}** vòng. Số pixel mỗi cụm: {res.core.counts.tolist()}.")
    with st.expander("Ma trận mức xám và ma trận nhãn"):
        a, b = st.columns(2)
        a.dataframe(pd.DataFrame(img), width="stretch")
        b.dataframe(pd.DataFrame(res.labels), width="stretch")


def _otsu_theory() -> None:
    st.markdown("""
**Nguồn:** Kỹ thuật tách ngưỡng — giáo trình mục 3.1.3; tìm ngưỡng tự động — mục 3.1.6 (trang 31–32);
phân vùng theo ngưỡng biên độ — mục 5.2. Thuật toán Otsu chuẩn: N. Otsu (1979), *A Threshold Selection Method from
Gray-Level Histograms*, IEEE Trans. SMC — **kiến thức bổ sung**, giáo trình không gọi tên “Otsu”.

Với histogram $h(i)$, $i = 0..255$, tổng N pixel, đặt $p_i = h(i)/N$. Ngưỡng t chia ảnh thành
$C_0 = \\{I \\le t\\}$ và $C_1 = \\{I > t\\}$:
""")
    st.latex(r"\omega_0(t)=\sum_{i=0}^{t}p_i,\quad \omega_1=1-\omega_0,\quad \mu(t)=\sum_{i=0}^{t} i\,p_i,\quad \mu_T=\mu(255)")
    st.latex(r"\mu_0=\frac{\mu(t)}{\omega_0},\qquad \mu_1=\frac{\mu_T-\mu(t)}{\omega_1}")
    st.latex(r"\sigma_B^2(t)=\omega_0\,\omega_1\,(\mu_0-\mu_1)^2=\frac{\big(\mu_T\,\omega_0(t)-\mu(t)\big)^2}{\omega_0(t)\,\omega_1(t)},"
             r"\qquad t^*=\arg\max_t \sigma_B^2(t)")
    st.markdown("""
Vì $\\sigma_T^2 = \\sigma_W^2 + \\sigma_B^2$ không đổi, cực đại $\\sigma_B^2$ tương đương cực tiểu phương sai trong lớp $\\sigma_W^2$.

**Liên hệ với giáo trình mục 3.1.6.** Giáo trình dùng $f(g) = \\dfrac{t(g)}{mn - t(g)}\\,[m(g) - m(G-1)]^2$,
với $t(g)$ = số pixel ≤ g, $m(g)$ = trung bình các pixel ≤ g. Ta có $\\dfrac{t(g)}{mn-t(g)} = \\dfrac{\\omega_0}{\\omega_1}$,
$m(g)=\\mu_0$, $m(G-1)=\\mu_T$ và $\\mu_0-\\mu_T=\\omega_1(\\mu_0-\\mu_1)$, nên
""")
    st.latex(r"f(g)=\frac{\omega_0}{\omega_1}\,\omega_1^2(\mu_0-\mu_1)^2=\omega_0\,\omega_1(\mu_0-\mu_1)^2=\sigma_B^2(g)")
    st.markdown("""
⇒ hai tiêu chí **cho cùng ngưỡng**. Các kỹ thuật khác trong giáo trình như *thuật toán đẳng liệu* (5.2.1, lặp trung bình hai
lớp), *đối xứng nền* (5.2.2) và *tam giác* (5.2.3) **không** phải Otsu.

**Quy ước cài đặt:** chỉ xét t làm cả hai lớp khác rỗng (tránh chia 0); nhiều t cùng cực đại → chọn floor(trung bình);
ảnh một mức xám → không hợp lệ, toàn ảnh là nền; nhị phân: pixel > t → 255.
""")
    st.code("""Vào: ảnh xám I (H×W, uint8)
1. h = histogram 256 mức;  p = h / N
2. ω0 = cumsum(p);  μ = cumsum(i·p);  μT = μ[255]
3. Với t có 0 < số pixel ≤ t < N:  σB²(t) = (μT·ω0(t) − μ(t))² / (ω0(t)·(1 − ω0(t)))
4. t* = floor(trung bình các t đạt max σB²)
5. B(x,y) = 255 nếu I(x,y) > t*, ngược lại 0
Ra: t*, σB², ảnh nhị phân""", language="text")

    st.markdown("##### Đối chiếu ví dụ giáo trình (mục 3.1.6, trang 32)")
    res = otsu_threshold(TEXTBOOK)
    c1, c2 = st.columns([1, 2])
    with c1:
        st.caption("Ảnh I (5×6) trong giáo trình")
        st.table(pd.DataFrame(TEXTBOOK))
    h = res.histogram
    t_cum = np.cumsum(h)
    gh = np.cumsum(np.arange(256) * h)
    rows = []
    for g in range(6):
        rows.append({"g": g, "h(g)": int(h[g]), "t(g)": int(t_cum[g]), "Σ i·h(i)": int(gh[g]),
                     "m(g)": round(gh[g] / t_cum[g], 4),
                     "f(g) = σ²_B": round(float(res.between_class_variance[g]), 4) if res.valid_thresholds[g] else "∞ / không hợp lệ"})
    with c2:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.write(f"Ngưỡng tìm được **t* = {res.threshold}**, trùng với θ = 1 trong giáo trình. "
                 "Giáo trình làm tròn trung gian nên ghi f(1) ≈ 1.66 (giá trị chính xác 1.6806).")


def _complexity() -> None:
    st.markdown("""
| | K-means | Otsu |
|---|---|---|
| Độ phức tạp thời gian | O(N · K · D · số vòng lặp) | O(N) dựng histogram + O(L) với L = 256 |
| Bộ nhớ thêm | O(N) nhãn + O(B · K) cho mỗi khối khoảng cách (B = 262 144 điểm) | O(L) |
| Kết quả | Cực tiểu địa phương, phụ thuộc khởi tạo | Tối ưu toàn cục theo σ²_B (duyệt mọi t) |
| Thông tin dùng | Giá trị pixel (màu/xám) | Chỉ histogram mức xám |
| Không dùng | Vị trí không gian của pixel | Vị trí không gian của pixel |

Cả hai đều **không dùng quan hệ lân cận** giữa các pixel, nên nhạy với nhiễu muối tiêu và chiếu sáng không đều —
đây là hạn chế chung, thấy rõ trên ảnh mẫu “Hình tròn có nhiễu”.
""")
