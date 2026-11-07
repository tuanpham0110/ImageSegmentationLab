"""Thành phần giao diện dùng chung: CSS, tiêu đề trang, hiển thị ảnh, thông tin đầu vào, nút tải xuống."""

from __future__ import annotations

from collections.abc import Callable
from html import escape
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from src.processing.filters import MORPH_LABELS
from src.processing.pipeline import MaskPostprocess, PreparedInput
from src.utils import visualization as viz

MIN_DISPLAY_SIDE = 256  # ảnh nhỏ hơn được phóng to (nearest) khi HIỂN THỊ để nhìn rõ từng pixel

CSS = "<style>" + Path(__file__).with_name("styles.css").read_text(encoding="utf-8") + "</style>"


def inject_css() -> None:
    st.html(CSS)


def page_header(title: str, subtitle: str) -> None:
    st.html(f"<header class='isl-page-header'><div class='isl-eyebrow'>PHÒNG THỰC HÀNH XỬ LÝ ẢNH</div>"
            f"<h1 class='isl-page-title'>{escape(title)}</h1><p class='isl-subtitle'>{escape(subtitle)}</p></header>")


def metric_columns(count: int, key: str):
    """Các thẻ tự xuống hàng theo chiều rộng còn lại, không ép năm cột hẹp."""
    with st.container(horizontal=True, gap="xsmall", key=f"metrics-{key}"):
        return [st.container(width=160) for _ in range(count)]


def workspace(key: str):
    """Giữ đủ chỗ cho tham số; xếp dọc khi vùng nội dung không còn đủ rộng."""
    with st.container(key=f"workspace-{key}"):
        return st.columns([1.05, 2], gap="medium")


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
    with st.container(border=True, key=None):
        st.image(disp, caption=cap, width="stretch", alt=caption)


def source_panels(prep: PreparedInput, with_gray: bool = False) -> list[tuple[np.ndarray, str]]:
    """Ô "Ảnh gốc" (chưa lọc) và, nếu có lọc trung vị, thêm ô ảnh đã lọc mà thuật toán thực sự nhận."""
    k = prep.median_size
    items = [(prep.original, "Ảnh gốc")]
    if prep.is_filtered:
        items.append((prep.image, f"Sau lọc trung vị {k}×{k} (đầu vào thuật toán)"))
    if with_gray:
        items.append((prep.gray, "Ảnh xám" + (" (sau lọc)" if prep.is_filtered else "")))
    return items


def show_images(items: list[tuple[np.ndarray, str]]) -> None:
    # Hai ảnh mỗi hàng vẫn đọc rõ khi nằm trong cột kết quả.
    for start in range(0, len(items), 2):
        row = items[start:start + 2]
        cols = st.columns(min(2, len(items)), gap="small")
        for col, (arr, cap) in zip(cols, row):
            with col:
                show_image(arr, cap)


def show_figure(fig) -> None:
    try:
        st.pyplot(fig, width="stretch")
    finally:
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
    st.html(f"<p class='isl-file-summary'>Ảnh đầu vào <strong>{escape(loaded.source_name)}</strong></p>")
    cols = metric_columns(4, "input")
    cols[0].metric("Định dạng", loaded.file_format)
    cols[1].metric("Kích thước gốc", f"{ow}×{oh}")
    cols[2].metric("Số kênh", chans)
    cols[3].metric("Dung lượng", human_bytes(loaded.file_bytes))
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


def cached_export_files(run: dict, key: tuple, build: Callable[[], dict[str, tuple[bytes, str]]]):
    """Giữ duy nhất một bộ tệp cho kết quả hiện tại; thay foreground thì tạo lại.

    Cache nằm trong payload của lần chạy, được bỏ khi chạy lại hoặc đổi đầu vào.
    Lỗi tạo tệp không được biến thành nút tải dữ liệu cũ.
    """
    cached = run.get("_export_cache")
    if cached is not None and cached["key"] == key:
        return cached["files"]
    run.pop("_export_cache", None)
    try:
        files = build()
    except (ValueError, OSError, MemoryError) as exc:
        st.error("Không tạo được tệp tải xuống. Hãy kiểm tra kết quả hoặc giảm giới hạn cạnh dài. " + str(exc))
        return None
    run["_export_cache"] = {"key": key, "files": files}
    return files


MORPH_SIZES = (3, 5, 7, 9)


def morphology_controls(key_prefix: str) -> MaskPostprocess:
    """Chọn phép hình thái học hậu xử lý mask (giáo trình Chương 7). Giá trị giữ qua điều hướng trang."""
    from src.ui.state import remember_widget, restore_widget

    op_key, size_key = f"{key_prefix}_morph", f"{key_prefix}_morph_size"
    restore_widget(op_key, "none")
    restore_widget(size_key, 3)
    op = st.selectbox("Hậu xử lý mask (hình thái học)", list(MORPH_LABELS), key=op_key,
                      format_func=lambda o: MORPH_LABELS[o],
                      help="Mở = co rồi giãn: xoá đốm nhiễu nhỏ hơn phần tử cấu trúc. "
                           "Đóng = giãn rồi co: lấp lỗ thủng/khe hở nhỏ. Phần tử cấu trúc vuông k×k.")
    size = st.selectbox("Kích thước k", MORPH_SIZES, key=size_key, format_func=lambda k: f"{k}×{k}")
    remember_widget(op_key)
    remember_widget(size_key)
    return MaskPostprocess(op, int(size))


def reset_morphology(key_prefix: str) -> None:
    """Đặt lại hậu xử lý mask về "không" (cả giá trị widget lẫn bản sao bền giữ qua điều hướng)."""
    from src.ui.state import remember_widget

    for key, value in ((f"{key_prefix}_morph", "none"), (f"{key_prefix}_morph_size", 3)):
        st.session_state[key] = value
        remember_widget(key)


def download_buttons(files: dict[str, tuple[bytes, str]], labels: dict[str, str], key_prefix: str) -> None:
    """Hiển thị các nút tải xuống theo lưới 2 cột, đủ chỗ cho tên tệp."""
    st.caption("Trình duyệt xử lý vị trí lưu và xác nhận ghi đè tệp.")
    names = list(files.keys())
    for row_start in range(0, len(names), 2):
        cols = st.columns(2)
        for col, name in zip(cols, names[row_start : row_start + 2]):
            data, mime = files[name]
            with col:
                requested = st.download_button(
                    labels.get(name, name), data=data, file_name=name, mime=mime,
                    key=f"{key_prefix}_{name}", width="stretch",
                )
                if requested:
                    st.info(f"Đã yêu cầu trình duyệt tải {name}. Kiểm tra mục tải xuống của trình duyệt.")
                st.caption(f"{name} · {human_bytes(len(data))}")
