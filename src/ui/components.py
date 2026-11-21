"""Thành phần giao diện dùng chung: CSS, tiêu đề trang, hiển thị ảnh, thông tin đầu vào, nút tải xuống."""

from __future__ import annotations

from collections.abc import Callable
from html import escape
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from src.processing.filters import MORPH_LABELS
from src.processing.pipeline import MaskPostprocess, PreparedInput, original_size_png_files
from src.utils import visualization as viz

MIN_DISPLAY_SIDE = 256  # ảnh nhỏ hơn được phóng to (nearest) khi HIỂN THỊ để nhìn rõ từng pixel
METHOD_PAGES = {
    "kmeans": ("K-means", ":material/palette:"),
    "otsu": ("Otsu", ":material/contrast:"),
    "compare": ("So sánh", ":material/compare:"),
}


def inject_css() -> None:
    # Read on rerun so a running app also picks up stylesheet changes.
    st.html("<style>" + Path(__file__).with_name("styles.css").read_text(encoding="utf-8") + "</style>")


def page_header(title: str, subtitle: str) -> None:
    current = st.session_state.get("_studio_page")
    if current in METHOD_PAGES:
        pages = st.session_state["_nav_pages"]
        with st.container(horizontal=True, gap="xsmall", key="method-navigation"):
            for key, (label, icon) in METHOD_PAGES.items():
                st.page_link(pages[key], label=label, icon=icon)
    st.html(f"<header class='isl-page-header'>"
            f"<h1 class='isl-page-title'>{escape(title)}</h1><p class='isl-subtitle'>{escape(subtitle)}</p></header>")


def metric_columns(count: int, key: str):
    """Các thẻ tự xuống hàng theo chiều rộng còn lại, không ép năm cột hẹp."""
    with st.container(horizontal=True, gap="xsmall", key=f"metrics-{key}"):
        return [st.container(width=130) for _ in range(count)]


def workspace(key: str):
    """Hai vùng bố cục, dùng chung cho các trang thực hành."""
    with st.container(horizontal=True, gap="small", key=f"workspace-{key}"):
        return st.container(key=f"controls-{key}"), st.container(key=f"results-{key}")


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
    h, w = arr.shape[:2]
    label = escape(caption)
    note = " title='Ảnh được phóng to để hiển thị'" if scaled else ""
    with st.container(border=True):
        st.html(f"<div class='isl-frame-heading'><strong>{label}</strong>"
                f"<span{note}>{w}×{h} px</span></div>")
        st.image(disp, width="stretch", alt=caption)


def source_panels(prep: PreparedInput, with_gray: bool = False) -> list[tuple[np.ndarray, str]]:
    """Ô "Ảnh gốc" (chưa lọc) và, nếu có lọc trung vị, thêm ô ảnh đã lọc mà thuật toán thực sự nhận."""
    k = prep.median_size
    items = [(prep.original, "Ảnh gốc (đã thu nhỏ)" if prep.resize.resized else "Ảnh gốc")]
    if prep.is_filtered:
        items.append((prep.image, f"Sau lọc trung vị {k}×{k} (đầu vào thuật toán)"))
    if with_gray:
        items.append((prep.gray, "Ảnh xám" + (" (sau lọc)" if prep.is_filtered else "")))
    return items


def show_images(items: list[tuple[np.ndarray, str]], *, columns: int = 2) -> None:
    """Đặt các ảnh cần đối chiếu trên cùng hàng khi vùng kết quả đủ rộng."""
    for start in range(0, len(items), columns):
        row = items[start:start + columns]
        cols = st.columns(len(row), gap="small")
        for col, (arr, cap) in zip(cols, row):
            with col:
                show_image(arr, cap)


def run_state(label: str, *, tone: str = "ready", detail: str = "") -> None:
    description = f"<span>{escape(detail)}</span>" if detail else ""
    st.html(f"<div class='isl-run-state' data-tone='{escape(tone, quote=True)}' role='status'>"
            f"<i aria-hidden='true'></i><strong>{escape(label)}</strong>{description}</div>")


def result_empty(prep: PreparedInput, action: str, *, with_gray: bool = False, stale: bool = False) -> None:
    if stale:
        st.warning(f"Ảnh đầu vào đã thay đổi, kết quả cũ đã được ẩn. Bấm **{action}** để cập nhật.")
    else:
        with st.container(border=True, key="workbench-empty"):
            st.subheader("Chưa có kết quả")
            st.caption(f"Chọn thiết lập và bấm **{action}**.")
    show_images(source_panels(prep, with_gray=with_gray))


def show_pending_settings(action: str) -> None:
    st.warning(f"Thiết lập đã đổi. Đang xem kết quả lần chạy trước. Bấm **{action}** để cập nhật và tải kết quả.")


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
    chans = "RGB · 3 kênh" if loaded.is_color else "Mức xám · 1 kênh"
    filename = escape(loaded.source_name, quote=True)
    cells = [("Kích thước gốc", f"{ow} × {oh} px"),
             ("Định dạng", f"{loaded.file_format} · {chans}"), ("Dung lượng", human_bytes(loaded.file_bytes))]
    st.html(
        "<dl class='isl-input-summary' aria-label='Thông tin ảnh đầu vào'>"
        f"<div class='isl-file-cell'><dt>Ảnh đang sử dụng</dt><dd title='{filename}'>{filename}</dd></div>"
        + "".join(f"<div><dt>{label}</dt><dd>{escape(str(value))}</dd></div>" for label, value in cells)
        + "</dl>"
    )
    if prep.resize.resized:
        st.info(f"Ảnh được thu nhỏ để xử lý: **{ow}×{oh} → {pw}×{ph}** (giới hạn cạnh dài trong thanh bên). "
                f"Kết quả trên màn hình và tệp xuất mặc định có kích thước {pw}×{ph}.")
    for note in loaded.notes:
        st.caption("ℹ️ " + note)
    if ctx.ground_truth_warning:
        st.warning(ctx.ground_truth_warning)


def require_input():
    """Lấy InputContext; nếu chưa có hoặc lỗi thì hiển thị thông báo và trả None."""
    from src.ui.state import get_context, get_input_error

    err = get_input_error()
    if err:
        st.error(f"Không dùng được dữ liệu đầu vào: {err}")
        return None
    ctx = get_context()
    if ctx is None:
        with st.container(border=True, key="input-empty"):
            st.subheader("Thêm ảnh để bắt đầu")
            st.info("Hãy chọn ảnh mẫu hoặc tải ảnh/CSV lên ở thanh bên trái.")
    return ctx


def cached_export_files(run: dict, key: tuple, build: Callable[[], dict[str, tuple[bytes, str]]],
                        *, cache_key: str = "_export_cache"):
    """Giữ duy nhất một bộ tệp cho kết quả hiện tại; thay foreground thì tạo lại.

    Cache nằm trong payload của lần chạy, được bỏ khi chạy lại hoặc đổi đầu vào.
    Lỗi tạo tệp không được biến thành nút tải dữ liệu cũ.
    """
    cached = run.get(cache_key)
    if cached is not None and cached["key"] == key:
        return cached["files"]
    run.pop(cache_key, None)
    try:
        files = build()
    except (ValueError, OSError, MemoryError) as exc:
        st.error("Không tạo được tệp tải xuống. Hãy kiểm tra kết quả hoặc giảm giới hạn cạnh dài. " + str(exc))
        return None
    run[cache_key] = {"key": key, "files": files}
    return files


def offer_original_size_pngs(ctx, run: dict, key: tuple, prefix: str, mask: np.ndarray,
                             images: dict[str, np.ndarray]) -> None:
    if not ctx.prepared.resize.resized:
        return
    if not st.checkbox("Thêm PNG kích thước gốc", key=f"{prefix}_export_original",
                       help="Phóng nhãn/mask bằng láng giềng gần nhất. Ảnh tách nền dùng pixel ảnh nguồn."):
        run.pop("_original_export_cache", None)
        return
    w, h = ctx.prepared.resize.original_size
    st.caption(f"{w} × {h} px · ranh giới được ánh xạ từ kết quả đã xử lý.")
    files = cached_export_files(run, key, lambda: original_size_png_files(ctx.loaded.array, mask, prefix, images),
                                cache_key="_original_export_cache")
    if files is not None:
        download_buttons(files, {}, f"{prefix}_original_dl")


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
    with st.container(key=f"exports-{key_prefix}"):
        st.caption("Vị trí lưu và lỗi ghi tệp được hiển thị trong mục Tải xuống của trình duyệt.")
        names = list(files)
        if names and names[0].endswith(".zip"):
            name = names.pop(0)
            _download_file(name, files[name], labels.get(name, name), key_prefix, primary=True)
        for row_start in range(0, len(names), 2):
            row = names[row_start:row_start + 2]
            cols = st.columns(len(row), gap="small")
            for col, name in zip(cols, row):
                with col:
                    _download_file(name, files[name], labels.get(name, name), key_prefix)


def _download_file(name: str, file: tuple[bytes, str], label: str, key_prefix: str, *, primary: bool = False) -> None:
    data, mime = file
    st.download_button(
        label, data=data, file_name=name, mime=mime, key=f"{key_prefix}_{name}", width="stretch",
        on_click="ignore", type="primary" if primary else "secondary",
    )
    st.caption(f"{name} · {human_bytes(len(data))}")
