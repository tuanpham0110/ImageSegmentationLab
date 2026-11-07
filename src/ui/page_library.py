"""Thư viện ảnh mẫu: các tình huống kiểm thử chạy offline."""

from __future__ import annotations

import streamlit as st

from src.ui import components as ui
from src.ui.state import DATA_DIR, load_manifest, thumbnail
from src.utils.image_io import InputError


def _use_sample(sid: str) -> None:
    # Callback chạy trước khi sidebar được vẽ lại nên được phép đặt giá trị widget.
    st.session_state["input_source"] = "Ảnh mẫu"
    st.session_state["sample_id"] = sid
    st.session_state["_library_notice"] = sid


def render() -> None:
    ui.page_header("Thư viện ảnh mẫu", "Bộ test case tổng hợp (sinh bằng NumPy/Pillow), chạy offline, có ground truth khi phù hợp.")
    notice = st.session_state.pop("_library_notice", None)
    if notice:
        st.success(f"Đã chọn mẫu **{notice}**. Mở trang K-means, Otsu hoặc So sánh để xử lý.")
    try:
        samples = load_manifest()
    except InputError as exc:
        st.error(str(exc))
        return
    if not samples:
        st.error("Không tìm thấy data/samples.json. Chạy: `python scripts/generate_samples.py`.")
        return
    st.caption("Tất cả là ảnh tổng hợp, không phải ảnh chụp thật. Có thể tải ảnh chụp của bạn ở thanh bên.")
    with st.container(key="library-grid"):
        for row in range(0, len(samples), 2):
            cols = st.columns(2, gap="medium")
            for col, s in zip(cols, samples[row : row + 2]):
                with col, st.container(border=True, height="stretch", key=f"sample-card-{s['id']}"):
                    try:
                        st.image(thumbnail(str(DATA_DIR / s["file"])), width="stretch", alt=s["title"])
                    except (InputError, OSError, MemoryError) as exc:
                        st.error(f"Không đọc được ảnh mẫu: {exc}")
                    st.markdown(f"**{s['title']}**")
                    chips = [s["kind"].upper(), "có ground truth" if s["ground_truth"] else "không có ground truth",
                             f"gợi ý K = {s['suggested']['k']}", f"đặc trưng {s['suggested']['feature']}"]
                    st.markdown("".join(f"<span class='isl-chip'>{c}</span>" for c in chips), unsafe_allow_html=True)
                    st.caption(f"**Mục đích:** {s['purpose']}")
                    st.caption(f"**Kỳ vọng:** {s['expected']}")
                    st.space("stretch")
                    st.button("Dùng ảnh này", key=f"use_{s['id']}", on_click=_use_sample, args=(s["id"],),
                              width="stretch")
