"""Render a report plate from the measured segmentation outputs for one image."""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np

from src.processing.pipeline import OtsuOutputs
from src.utils.visualization import ACCENT, GRID, NEUTRAL, SERIES_MAIN, TEXT_PRIMARY, TEXT_SECONDARY

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def _score_text(row: dict) -> str:
    if row.get("iou") is None or row.get("dice") is None:
        return "Chưa có điểm với GT"
    return f"IoU {row['iou']:.4f} · Dice {row['dice']:.4f}"


def _show_pixels(ax, pixels: np.ndarray | None, title: str, caption: str) -> None:
    ax.set_title(title, fontsize=12, color=TEXT_PRIMARY, fontweight="bold", pad=8)
    ax.axis("off")
    if pixels is None:
        ax.text(0.5, 0.54, "Không có mask tham chiếu", ha="center", va="center", fontsize=10,
                color=TEXT_SECONDARY, bbox={"facecolor": "#eff5f2", "edgecolor": GRID, "pad": 8})
        return
    ax.imshow(pixels, cmap="gray" if pixels.ndim == 2 else None, vmin=0, vmax=255, interpolation="nearest")
    ax.text(0.5, -0.09, caption, transform=ax.transAxes, ha="center", va="top", fontsize=10,
            color=TEXT_SECONDARY, linespacing=1.35)


def _style_plot(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=12, color=TEXT_PRIMARY, fontweight="bold", pad=8)
    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=(-3, 258), xticks=(0, 64, 128, 192, 255))
    ax.tick_params(axis="both", labelsize=8, colors=TEXT_SECONDARY)
    ax.xaxis.label.set_color(TEXT_SECONDARY)
    ax.yaxis.label.set_color(TEXT_SECONDARY)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.6)


def _show_otsu_plots(hist_ax, variance_ax, otsu: OtsuOutputs) -> None:
    result = otsu.result
    levels = np.arange(result.histogram.size)
    threshold = result.threshold
    colors = np.where((levels <= threshold) & result.valid, NEUTRAL, SERIES_MAIN)
    hist_ax.bar(levels, result.histogram, width=1, color=colors)
    hist_ax.set_ylim(bottom=0)
    _style_plot(hist_ax, "Histogram ảnh xám", "Mức xám", "Số pixel")

    _style_plot(variance_ax, "Phương sai giữa hai lớp", "Ngưỡng t", r"$\sigma_B^2(t)$")
    if result.valid:
        hist_ax.axvline(threshold + 0.5, color=ACCENT, linewidth=1.5, label=f"t* = {threshold}")
        hist_ax.legend(fontsize=9, frameon=False, loc="upper right")
        variance_ax.plot(levels, np.where(result.valid_thresholds, result.between_class_variance, np.nan),
                         color=SERIES_MAIN, linewidth=1.8)
        variance_ax.axvline(threshold, color=ACCENT, linewidth=1.5)
        variance_ax.scatter([threshold], [result.max_variance], color=ACCENT, s=24, zorder=3)
        variance_ax.set_ylim(bottom=0)
    else:
        variance_ax.set_yticks([])
        variance_ax.text(0.5, 0.54, "Không có ngưỡng chia\nhai lớp khác rỗng", transform=variance_ax.transAxes,
                         ha="center", va="center", color=TEXT_SECONDARY, fontsize=10)


def render_case_figure(
    path: Path,
    case: dict,
    original: np.ndarray,
    reference: np.ndarray | None,
    km_panels: list[tuple[dict, np.ndarray, np.ndarray]],
    otsu: OtsuOutputs,
    otsu_row: dict,
) -> None:
    """Use the measured outputs directly and keep 128 visible only on trimap references."""
    columns = max(4, 1 + len(km_panels))
    figure, axes = plt.subplots(3, columns, figsize=(columns * 3.25, 9.0), dpi=165, facecolor="#fcfdfb")
    figure.subplots_adjust(left=0.066, right=0.982, bottom=0.105, top=0.811, hspace=0.73, wspace=0.36)
    width, height = case["processed_size"]
    original_width, original_height = case["original_size"]
    filter_size = case.get("median_size", 1)
    filter_text = f"Trung vị {filter_size}×{filter_size}" if filter_size > 1 else "Tắt lọc trung vị"
    title = textwrap.fill(case["title"], width=76)
    figure.text(0.066, 0.952, title, color=TEXT_PRIMARY, fontsize=23, weight="bold", va="top")
    subtitle_y = 0.895 - 0.034 * title.count("\n")
    figure.text(0.066, subtitle_y,
                f"Ảnh gốc {original_width}×{original_height} px   ·   Xử lý {width}×{height} px   ·   {filter_text}",
                color=TEXT_SECONDARY, fontsize=11, va="top")

    original_caption = f"{original.shape[1]}×{original.shape[0]} px"
    reference_caption = "Xám 128 = vùng bỏ qua" if case.get("gt_kind") == "trimap" else "Trắng = đối tượng"
    result = otsu.result
    threshold_text = f"t* = {result.threshold}" if result.valid else f"Ngưỡng dự phòng {result.threshold}"
    binary_caption = f"{threshold_text} · {otsu_row['time_ms']:.2f} ms"
    polarity = "Lớp tối" if otsu.invert else "Lớp sáng"
    mask_caption = f"{polarity if result.valid else 'Mask tự động rỗng'}\n{_score_text(otsu_row)}"
    for ax, (pixels, panel_title, caption) in zip(axes[0], (
        (original, "Ảnh đầu vào", original_caption),
        (reference, "Mask tham chiếu (GT)", reference_caption),
        (otsu.binary, "Ảnh nhị phân Otsu", binary_caption),
        (otsu.mask, "Mask đối tượng Otsu", mask_caption),
    )):
        _show_pixels(ax, pixels, panel_title, caption)
    for ax in axes[0, 4:]:
        ax.axis("off")

    for column, (row, segmented, mask) in enumerate(km_panels, start=1):
        requested, effective = row["k_requested"], row["k_effective"]
        k_text = f"K={requested}" if requested == effective else f"K={requested}→{effective}"
        config_text = f"{row['feature'].upper()} · {row['iterations']} vòng · {row['time_ms']:.2f} ms"
        _show_pixels(axes[1, column], segmented, f"K-means · {k_text}", config_text)
        _show_pixels(axes[2, column], mask, f"Mask K-means · {k_text}", _score_text(row))
    for ax in axes[1:, 1 + len(km_panels):].flat:
        ax.axis("off")
    _show_otsu_plots(axes[1, 0], axes[2, 0], otsu)
    figure.text(0.066, 0.027,
                "Mask kết quả: trắng là đối tượng. Thời gian tính trên lõi thuật toán.",
                fontsize=10, color=TEXT_SECONDARY)

    try:
        figure.savefig(path, facecolor=figure.get_facecolor())
    finally:
        plt.close(figure)
