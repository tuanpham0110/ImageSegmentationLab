"""Biểu đồ Matplotlib cho giao diện và báo cáo (backend Agg, không cần màn hình)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.processing.postprocessing import CLUSTER_PALETTE_HEX  # noqa: E402

TEXT_PRIMARY = "#1f2328"
TEXT_SECONDARY = "#57606a"
GRID = "#d8dee4"
SERIES_MAIN = "#2a78d6"
ACCENT = "#e34948"
NEUTRAL = "#8c959f"


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=11, color=TEXT_PRIMARY, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=9, color=TEXT_SECONDARY)
    ax.set_ylabel(ylabel, fontsize=9, color=TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def histogram_figure(
    hist: np.ndarray,
    threshold: int | None = None,
    manual_threshold: int | None = None,
    title: str = "Histogram 256 mức xám",
):
    """Histogram với đường ngưỡng Otsu (đỏ, liền) và ngưỡng thủ công (xám, nét đứt)."""
    fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=120)
    levels = np.arange(256)
    h = np.asarray(hist)
    if threshold is not None:
        ax.bar(levels[: threshold + 1], h[: threshold + 1], width=1.0, color=NEUTRAL, label=f"Lớp C0 (≤ {threshold})")
        ax.bar(levels[threshold + 1 :], h[threshold + 1 :], width=1.0, color=SERIES_MAIN, label=f"Lớp C1 (> {threshold})")
        ax.axvline(threshold + 0.5, color=ACCENT, linewidth=2, label=f"Ngưỡng Otsu t* = {threshold}")
    else:
        ax.bar(levels, h, width=1.0, color=SERIES_MAIN)
    if manual_threshold is not None:
        ax.axvline(manual_threshold + 0.5, color=TEXT_PRIMARY, linewidth=1.5, linestyle="--",
                   label=f"Ngưỡng thủ công T = {manual_threshold}")
    ax.set_xlim(-1, 256)
    _style(ax, title, "Mức xám g", "Số pixel h(g)")
    if threshold is not None or manual_threshold is not None:
        ax.legend(fontsize=8, frameon=True, facecolor="white", edgecolor=GRID, framealpha=0.95, loc="best")
    fig.tight_layout()
    return fig


def variance_figure(sigma_b: np.ndarray, valid: np.ndarray, threshold: int | None):
    """Đồ thị phương sai giữa hai lớp sigma_B^2(t) theo ngưỡng t."""
    fig, ax = plt.subplots(figsize=(7.2, 2.8), dpi=120)
    t = np.arange(256)
    y = np.where(valid, sigma_b, np.nan)
    ax.plot(t, y, color=SERIES_MAIN, linewidth=2)
    if threshold is not None and valid.any():
        ax.axvline(threshold, color=ACCENT, linewidth=1.5)
        ax.plot([threshold], [sigma_b[threshold]], "o", color=ACCENT, markersize=7)
        ax.annotate(f"t* = {threshold}", (threshold, sigma_b[threshold]), textcoords="offset points",
                    xytext=(8, -16), fontsize=9, color=TEXT_PRIMARY)
    ax.set_xlim(-1, 256)
    _style(ax, "Phương sai giữa hai lớp σ²_B(t)", "Ngưỡng t", "σ²_B(t)")
    fig.tight_layout()
    return fig


def cluster_share_figure(counts: np.ndarray, centers: np.ndarray):
    """Biểu đồ cột ngang: tỷ lệ pixel của từng cụm (màu = màu cụm trên bản đồ nhãn)."""
    counts = np.asarray(counts)
    k = counts.size
    share = counts / counts.sum() * 100
    fig, ax = plt.subplots(figsize=(4.6, 0.45 * k + 1.1), dpi=130)
    ypos = np.arange(k)
    ax.barh(ypos, share, color=[CLUSTER_PALETTE_HEX[i] for i in range(k)], height=0.6)
    labels = []
    for i in range(k):
        c = centers[i]
        desc = f"{c[0]:.0f}" if c.size == 1 else f"({c[0]:.0f}, {c[1]:.0f}, {c[2]:.0f})"
        labels.append(f"Cụm {i}  {desc}")
    ax.set_yticks(ypos, labels)
    ax.invert_yaxis()
    for i, v in enumerate(share):
        ax.text(v + 0.8, i, f"{v:.1f}%", va="center", fontsize=8, color=TEXT_PRIMARY)
    ax.set_xlim(0, max(share.max() * 1.18, 10))
    _style(ax, "Tỷ lệ pixel theo cụm", "% số pixel", "")
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def sse_figure(sse_history: list[float]):
    """SSE (inertia) sau bước gán nhãn ở mỗi vòng lặp — phải không tăng."""
    fig, ax = plt.subplots(figsize=(4.6, 2.6), dpi=130)
    it = np.arange(1, len(sse_history) + 1)
    ax.plot(it, sse_history, color=SERIES_MAIN, linewidth=2, marker="o", markersize=5)
    _style(ax, "SSE theo vòng lặp", "Vòng lặp", "SSE")
    ax.xaxis.get_major_locator().set_params(integer=True)
    fig.tight_layout()
    return fig


def close(fig) -> None:
    plt.close(fig)
