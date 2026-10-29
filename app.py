"""IMAGE SEGMENTATION LAB — điểm khởi chạy Streamlit.

Chạy:  python -m streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Image Segmentation Lab", page_icon="🧩", layout="wide",
                   initial_sidebar_state="expanded")

from src.ui import (  # noqa: E402
    page_compare,
    page_guide,
    page_kmeans,
    page_library,
    page_otsu,
    page_overview,
    page_theory,
)
from src.ui.components import inject_css  # noqa: E402
from src.ui.sidebar import render_input_sidebar  # noqa: E402

PAGES = {
    "overview": (page_overview.render, "Tổng quan", "🏠"),
    "kmeans": (page_kmeans.render, "K-means", "🎨"),
    "otsu": (page_otsu.render, "Otsu Thresholding", "🌗"),
    "compare": (page_compare.render, "So sánh thuật toán", "⚖️"),
    "library": (page_library.render, "Thư viện ảnh mẫu", "🖼️"),
    "theory": (page_theory.render, "Kiến thức thuật toán", "📐"),
    "guide": (page_guide.render, "Hướng dẫn", "📘"),
}


def main() -> None:
    inject_css()
    # Trang mặc định (Tổng quan) nằm ở URL gốc "/"; các trang khác ở "/kmeans", "/otsu", ...
    pages = [st.Page(fn, title=title, icon=icon, default=True) if key == "overview"
             else st.Page(fn, title=title, icon=icon, url_path=key)
             for key, (fn, title, icon) in PAGES.items()]
    nav = st.navigation(pages)
    render_input_sidebar()
    test_page = st.session_state.get("_test_page")  # chỉ dùng khi kiểm thử tự động (AppTest)
    if test_page in PAGES:
        PAGES[test_page][0]()
    else:
        nav.run()


main()
