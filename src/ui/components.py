"""Thành phần giao diện dùng chung: CSS, tiêu đề trang, hiển thị ảnh, thông tin đầu vào, nút tải xuống."""

from __future__ import annotations

import numpy as np
import streamlit as st
from PIL import Image

from src.utils import visualization as viz

MIN_DISPLAY_SIDE = 256  # ảnh nhỏ hơn được phóng to (nearest) khi HIỂN THỊ để nhìn rõ từng pixel

CSS = """
<style>
:root {
  --isl-ink: #1f2328; --isl-ink-2: #57606a; --isl-line: #d8dee4;
  --isl-surface: #ffffff; --isl-soft: #f4f7fb; --isl-accent: #2a78d6;
}
.block-container { padding-top: 2.2rem; max-width: 1280px; }
.isl-hero {
  border-radius: 16px; padding: 1.5rem 1.75rem; margin-bottom: 1.25rem;
  background: linear-gradient(120deg, #0f3d75 0%, #2a78d6 60%, #1baf7a 140%); color: #fff;
}
.isl-hero h1 { color: #fff; margin: 0; font-size: 1.9rem; letter-spacing: .02em; }
.isl-hero p { margin: .35rem 0 0 0; opacity: .92; font-size: 1.02rem; }
.isl-page-title { margin: 0 0 .2rem 0; }
.isl-subtitle { color: var(--isl-ink-2); margin-bottom: 1rem; }
.isl-card {
  border: 1px solid var(--isl-line); border-radius: 12px; padding: .9rem 1rem;
  background: var(--isl-surface); height: 100%;
}
.isl-card h4 { margin: 0 0 .35rem 0; font-size: 1rem; }
.isl-card p, .isl-card li { color: var(--isl-ink-2); font-size: .92rem; margin: 0; }
.isl-step {
  border: 1px solid var(--isl-line); border-radius: 12px; padding: .75rem .9rem; background: var(--isl-soft);
  min-height: 110px;
}
.isl-step b { display: block; color: var(--isl-accent); font-size: .8rem; text-transform: uppercase; letter-spacing: .05em; }
.isl-chip {
  display: inline-block; padding: .1rem .55rem; border-radius: 999px; font-size: .78rem;
  border: 1px solid var(--isl-line); background: var(--isl-soft); color: var(--isl-ink); margin: 0 .25rem .25rem 0;
}
[data-testid="stImageCaption"], .stImage figcaption { font-size: .85rem; }
div[data-testid="stMetric"] { background: var(--isl-soft); border-radius: 10px; padding: .55rem .8rem; }
div[data-testid="stMetricValue"] p { font-size: 1.3rem !important; font-weight: 600; line-height: 1.3; }
div[data-testid="stMetricValue"] .stMarkdownEllipsis, div[data-testid="stMetricValue"] p {
  white-space: normal !important; overflow: visible !important; text-overflow: clip !important; overflow-wrap: anywhere;
}
div[data-testid="stMetricLabel"] p { font-size: .82rem; color: var(--isl-ink-2); }
.isl-swatch { display: inline-block; width: .8rem; height: .8rem; border-radius: 3px; vertical-align: -1px; margin-right: .3rem; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str) -> None:
    st.markdown(f"<h2 class='isl-page-title'>{title}</h2><div class='isl-subtitle'>{subtitle}</div>",
                unsafe_allow_html=True)


def _for_display(arr: np.ndarray) -> tuple[np.ndarray, bool]:
    """Phóng to ảnh nhỏ bằng nội suy láng giềng gần nhất (chỉ để hiển thị, không đổi dữ liệu)."""
    a = np.asarray(arr)
    h, w = a.shape[:2]
    if max(h, w) >= MIN_DISPLAY_SIDE:
        return a, False
    f = int(np.ceil(MIN_DISPLAY_SIDE / max(h, w)))
    return np.asarray(Image.fromarray(a).resize((w * f, h * f), Image.Resampling.NEAREST)), True


def show_image(arr: np.ndarray, caption: str) -> None:
    disp, scaled = _for_display(arr)
    h, w = np.asarray(arr).shape[:2]
    cap = f"{caption} · {w}×{h}" + (" (phóng to để hiển thị)" if scaled else "")
    st.image(disp, caption=cap, width="stretch")


def show_images(items: list[tuple[np.ndarray, str]]) -> None:
    cols = st.columns(len(items))
    for col, (arr, cap) in zip(cols, items):
        with col:
            show_image(arr, cap)


def show_figure(fig) -> None:
    st.pyplot(fig, width="stretch")
    viz.close(fig)


def compact_number(x: float) -> str:
    """Số lớn dạng gọn để vừa thẻ metric: 208974262 -> '209.0 triệu'."""
    if abs(x) >= 1e9:
        return f"{x / 1e9:.2f} tỷ"
    if abs(x) >= 1e6:
        return f"{x / 1e6:.1f} triệu"
    return f"{x:,.0f}"


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB"):
        if n < 1024 or unit == "MB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} MB"


def input_summary(ctx) -> None:
    """Thẻ thông tin ảnh đầu vào: tên, định dạng, kích thước, số kênh, dung lượng, kích thước xử lý."""
    loaded, prep = ctx.loaded, ctx.prepared
    ow, oh = prep.resize.original_size
    pw, ph = prep.resize.processed_size
    chans = "3 (RGB)" if loaded.is_color else "1 (mức xám)"
    cols = st.columns([1.4, 1, 1, 1, 1])
    cols[0].metric("Tệp", loaded.source_name if len(loaded.source_name) < 26 else loaded.source_name[:23] + "…")
    cols[1].metric("Định dạng", loaded.file_format)
    cols[2].metric("Kích thước gốc", f"{ow}×{oh}")
    cols[3].metric("Số kênh", chans)
    cols[4].metric("Dung lượng", human_bytes(loaded.file_bytes))
    if prep.resize.resized:
        st.info(f"Ảnh được thu nhỏ để xử lý: **{ow}×{oh} → {pw}×{ph}** (giới hạn cạnh dài trong thanh bên). "
                f"Mọi kết quả và tệp tải xuống có kích thước {pw}×{ph}.")
    for note in loaded.notes:
        st.caption("ℹ️ " + note)


def require_input():
    """Lấy InputContext; nếu chưa có hoặc lỗi thì hiển thị thông báo và trả None."""
    from src.ui.state import get_context, get_input_error

    err = get_input_error()
    if err:
        st.error(f"Không dùng được dữ liệu đầu vào: {err}")
        return None
    ctx = get_context()
    if ctx is None:
        st.info("Hãy chọn ảnh mẫu hoặc tải ảnh/CSV lên ở thanh bên trái.")
    return ctx


def download_buttons(files: dict[str, tuple[bytes, str]], labels: dict[str, str], key_prefix: str) -> None:
    """Hiển thị các nút tải xuống theo lưới 3 cột."""
    names = list(files.keys())
    for row_start in range(0, len(names), 3):
        cols = st.columns(3)
        for col, name in zip(cols, names[row_start : row_start + 3]):
            data, mime = files[name]
            with col:
                st.download_button(
                    labels.get(name, name), data=data, file_name=name, mime=mime,
                    key=f"{key_prefix}_{name}", width="stretch",
                )
                st.caption(f"{name} · {human_bytes(len(data))}")
