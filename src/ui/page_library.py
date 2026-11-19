"""Thư viện ảnh mẫu: các tình huống kiểm thử chạy offline."""

from __future__ import annotations

from html import escape

import streamlit as st

from src.ui import components as ui
from src.ui.state import DATA_DIR, get_context, load_manifest, thumbnail
from src.utils.image_io import InputError

FEATURE_LABELS = {"rgb": "RGB", "lab": "Lab", "gray": "mức xám"}


def _use_sample(sid: str) -> None:
    # Callback chạy trước khi sidebar được vẽ lại nên được phép đặt giá trị widget.
    st.session_state["input_source"] = "Ảnh mẫu"
    st.session_state["sample_id"] = sid
    st.session_state["_library_notice"] = sid


def render() -> None:
    ui.page_header("Thư viện ảnh mẫu", "Chọn mẫu tổng hợp để thử các vùng màu, mức xám và ảnh có nhiễu.")
    try:
        samples = load_manifest()
    except InputError as exc:
        st.error(str(exc))
        return
    if not samples:
        st.error("Không tìm thấy data/samples.json. Chạy: `python scripts/generate_samples.py`.")
        return

    ctx = get_context()
    selected = ctx.sample if ctx is not None else None
    notice = st.session_state.pop("_library_notice", None)
    if selected is not None and selected["id"] == notice:
        st.success(f"Đã chọn mẫu {selected['title']}.")
        pages = st.session_state["_nav_pages"]
        with st.container(horizontal=True, gap="small", key="library-next"):
            st.page_link(pages["kmeans"], label="Phân cụm K-means", icon=":material/palette:")
            st.page_link(pages["otsu"], label="Phân ngưỡng Otsu", icon=":material/contrast:")
            st.page_link(pages["compare"], label="So sánh kết quả", icon=":material/compare:")

    with st.container(key="library-grid"):
        for sample in samples:
            _sample_card(sample, selected is not None and sample["id"] == selected["id"])


def _sample_card(sample: dict, selected: bool) -> None:
    sid, title = sample["id"], sample["title"]
    path = DATA_DIR / sample["file"]
    with st.container(border=True, height="stretch", key=f"sample-card-{sid}"):
        try:
            image = thumbnail(str(path), size=320)
        except (InputError, OSError, MemoryError) as exc:
            st.error(f"Không đọc được ảnh mẫu: {exc}")
            image = None
        else:
            st.image(image, width="stretch", alt=title)

        kind = path.suffix.removeprefix(".").upper()
        reference = "Có mask tham chiếu" if sample["ground_truth"] else "Đánh giá trực quan"
        status = "<span class='isl-sample-current'>Ảnh đang dùng</span>" if selected else ""
        st.html(
            f"<p class='isl-sample-meta'>{escape(kind)} · {reference}</p>{status}"
            f"<h2 class='isl-sample-heading'>{escape(title)}</h2>"
        )
        st.caption(sample["purpose"])
        with st.expander("Tham số và kết quả dự kiến"):
            suggested = sample["suggested"]
            st.caption(f"K-means · K = {suggested['k']} · {FEATURE_LABELS[suggested['feature']]}")
            st.markdown(sample["expected"])
        st.space("stretch")
        st.button(
            "Dùng ảnh này", key=f"use_{sid}", on_click=_use_sample, args=(sid,),
            icon=":material/arrow_forward:", width="stretch", disabled=image is None,
        )
