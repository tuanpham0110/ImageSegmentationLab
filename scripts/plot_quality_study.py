"""Vẽ biểu đồ tổng hợp từ JSON thực nghiệm K-means và Otsu đã đo."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


K_VALUES = (2, 3, 5)
METHODS = (*K_VALUES, "otsu")
METHOD_LABELS = ("K=2", "K=3", "K=5", "Otsu")
COHORTS = {"synthetic": "Ảnh mẫu", "photo": "Ảnh chụp thật"}
METHOD_COLORS = ("#287eba", "#087b6a", "#6952a3", "#bc5834")
INIT_STYLES = {
    "kmeans++": ("K-means++", "#087b6a", "o"),
    "random": ("Ngẫu nhiên", "#bc5834", "s"),
}
INK = "#172e2c"
MUTED = "#46615d"
GRID = "#dce8e4"


def _finite(value) -> bool:
    return value is not None and math.isfinite(value)


def _point_distribution(ax, x: float, values: list[float], color: str, marker: str = "o", spread: float = 0.16,
                        *, aggregate=statistics.mean) -> None:
    """Mỗi chấm là một ảnh, hình thoi là một giá trị tổng hợp theo ảnh."""
    if not values:
        return
    step = (2 * spread) / max(1, len(values) - 1)
    offsets = [0.0] if len(values) == 1 else [(i * step - spread) for i in range(len(values))]
    ax.scatter([x + offset for offset in offsets], values, color=color, marker=marker, s=26, alpha=0.48,
               linewidths=0, zorder=2)
    ax.scatter([x], [aggregate(values)], marker="D", s=59, color=color, edgecolor="white", linewidth=0.9, zorder=3)


def _style_axes(ax, xlabels, title, ylabel, *, bounded: bool = False) -> None:
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold", pad=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.set_xticks(range(len(xlabels)), xlabels)
    ax.set_xlim(-0.5, len(xlabels) - 0.5)
    if bounded:
        ax.set_ylim(-0.035, 1.035)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.tick_params(colors=MUTED, labelsize=9, length=0, pad=5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.8)


def _save(fig, figures: Path, name: str, title: str, note: str) -> Path:
    path = figures / name
    height_points = fig.get_figheight() * 72
    fig.text(0.065, 0.966, title, ha="left", va="top", fontsize=17, fontweight="bold", color=INK)
    fig.text(0.065, 0.966 - 25 / height_points, note, ha="left", va="top", fontsize=9, color=MUTED)
    fig.subplots_adjust(left=0.065, right=0.98, bottom=0.08, top=0.966 - 61 / height_points,
                        hspace=0.44, wspace=0.24)
    try:
        fig.savefig(path, dpi=180, facecolor="white", metadata={"Title": title, "Description": note})
    finally:
        plt.close(fig)
    return path


def _main_results(cases: list[dict], runs: list[dict]) -> dict:
    allowed = {case["id"] for case in cases if case["group"] in COHORTS}
    by_case = defaultdict(dict)
    for run in runs:
        if run["phase"] != "main" or run["case_id"] not in allowed:
            continue
        if run["algorithm"] == "otsu":
            by_case[run["case_id"]]["otsu"] = run
        elif (run["algorithm"] == "kmeans" and run["k_requested"] in K_VALUES
              and run["seed"] == 0 and run["init"] == "kmeans++"):
            by_case[run["case_id"]][run["k_requested"]] = run
    return by_case


def _paired_cases(cases: list[dict], main: dict, keys: tuple[str, ...]) -> dict[str, list]:
    cohorts = {}
    for group, label in COHORTS.items():
        complete = []
        for case in cases:
            if case["group"] != group:
                continue
            methods = main.get(case["id"], {})
            if all(method in methods and all(_finite(methods[method][key]) for key in keys) for method in METHODS):
                complete.append(methods)
        if complete:
            cohorts[label] = complete
    return cohorts


def _quality_figure(cases: list[dict], main: dict, figures: Path) -> Path | None:
    cohorts = _paired_cases(cases, main, ("iou", "dice"))
    if not cohorts:
        return None
    fig, axes = plt.subplots(len(cohorts), 2, figsize=(10.5, 3.5 + 2.25 * (len(cohorts) - 1)), squeeze=False)
    for (label, cohort), row in zip(cohorts.items(), axes):
        for metric, ylabel, ax in zip(("iou", "dice"), ("IoU", "Dice"), row):
            _style_axes(ax, METHOD_LABELS, f"{label} · {len(cohort)} ảnh", ylabel, bounded=True)
            for index, (method, color) in enumerate(zip(METHODS, METHOD_COLORS)):
                _point_distribution(ax, index, [record[method][metric] for record in cohort], color)
    return _save(fig, figures, "quality_by_k.png", "Chất lượng mask theo phương pháp và K",
                 "Các cấu hình dùng cùng ảnh có tham chiếu · K-means++ seed 0 · Chấm = một ảnh · Hình thoi = trung bình")


def _runtime_figure(cases: list[dict], main: dict, figures: Path) -> Path | None:
    cohorts = _paired_cases(cases, main, ("time_ms",))
    if not cohorts:
        return None
    fig, axes = plt.subplots(1, len(cohorts), figsize=(10.5, 4.35), squeeze=False, sharey=True)
    for (label, cohort), ax in zip(cohorts.items(), axes[0]):
        _style_axes(ax, METHOD_LABELS, f"{label} · {len(cohort)} ảnh", "Thời gian lõi (ms)")
        ax.set_yscale("log")
        for index, (method, color) in enumerate(zip(METHODS, METHOD_COLORS)):
            times = [record[method]["time_ms"] for record in cohort]
            if any(value <= 0 for value in times):
                plt.close(fig)
                raise ValueError("Thời gian lõi phải dương để vẽ theo thang log.")
            _point_distribution(ax, index, times, color, aggregate=statistics.median)
    return _save(fig, figures, "runtime_by_method.png", "Thời gian thực thi trên cùng bộ ảnh",
                 "Trục dọc log · Chấm = trung vị lượt đo của một ảnh · Hình thoi = trung vị theo ảnh · K-means++ seed 0")


def _seed_measurements(cases: list[dict], runs: list[dict]) -> dict:
    allowed = {case["id"] for case in cases if case["group"] in COHORTS}
    seeds_by_config = defaultdict(dict)
    for run in runs:
        if (run["algorithm"] == "kmeans" and run["case_id"] in allowed and run["k_requested"] in K_VALUES
                and run["init"] in INIT_STYLES):
            key = (run["case_id"], run["k_requested"], run["init"])
            seeds_by_config[key][run["seed"]] = run

    values = defaultdict(lambda: {"agreement": [], "range": []})
    for (_, k, init), by_seed in seeds_by_config.items():
        if 0 not in by_seed or len(by_seed) < 2:
            continue
        agreement = [run["mask_agreement_seed0"] for seed, run in by_seed.items()
                     if seed != 0 and _finite(run.get("mask_agreement_seed0"))]
        if len(agreement) == len(by_seed) - 1:
            values[k, init]["agreement"].append(min(agreement))
        scores = [run["iou"] for run in by_seed.values()]
        if all(_finite(score) for score in scores):
            values[k, init]["range"].append(max(scores) - min(scores))
    return values


def _stability_figure(cases: list[dict], runs: list[dict], figures: Path) -> Path | None:
    values = _seed_measurements(cases, runs)
    metrics = [("agreement", "Mức trùng thấp nhất với seed 0", "IoU giữa hai mask dự đoán"),
               ("range", "Độ dao động điểm theo seed", "IoU lớn nhất − nhỏ nhất với GT")]
    present = [entry for entry in metrics if any(config[entry[0]] for config in values.values())]
    if not present:
        return None
    inits = [init for init in INIT_STYLES if any(key[1] == init for key in values)]
    fig, axes = plt.subplots(1, len(present), figsize=(10.5, 4.55), squeeze=False)
    for (metric, title, ylabel), ax in zip(present, axes[0]):
        _style_axes(ax, METHOD_LABELS[: len(K_VALUES)], title, ylabel, bounded=True)
        for init_index, init in enumerate(inits):
            label, color, marker = INIT_STYLES[init]
            offset = 0.0 if len(inits) == 1 else 0.17 * (2 * init_index - 1)
            n_by_k = []
            for index, k in enumerate(K_VALUES):
                row = values.get((k, init))
                scores = row[metric] if row else []
                n_by_k.append(len(scores))
                _point_distribution(ax, index + offset, scores, color, marker, spread=0.09,
                                    aggregate=statistics.median)
            if any(n_by_k):
                counts = "/".join(str(n) for n in n_by_k)
                ax.scatter([], [], color=color, marker=marker, s=26, label=f"{label} · n={counts}")
        ax.legend(loc="best", fontsize=8, frameon=True, facecolor="white", edgecolor=GRID, framealpha=0.95)
    return _save(fig, figures, "seed_stability.png", "Ảnh hưởng của seed đến mask K-means",
                 "Số ảnh n theo K=2/3/5 · Chấm = một ảnh có quét seed · Hình thoi = trung vị · GT = mask tham chiếu")


def render_study_figures(cases: list[dict], runs: list[dict], output_dir: Path) -> list[Path]:
    """Xuất đồ thị theo các cấu hình đã đo; chỉ dùng cặp có đủ dữ liệu để so sánh."""
    output_dir.mkdir(parents=True, exist_ok=True)
    main = _main_results(cases, runs)
    with plt.rc_context({"font.family": "DejaVu Sans", "figure.facecolor": "white", "axes.facecolor": "white"}):
        paths = (_quality_figure(cases, main, output_dir), _runtime_figure(cases, main, output_dir),
                 _stability_figure(cases, runs, output_dir))
    return [path for path in paths if path is not None]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="JSON thực nghiệm chứa cases và runs.")
    parser.add_argument("--output-dir", type=Path, help="Thư mục PNG, mặc định figures cạnh JSON.")
    args = parser.parse_args()
    results = json.loads(args.results.read_text(encoding="utf-8"))
    output_dir = args.output_dir or args.results.parent / "figures"
    paths = render_study_figures(results["cases"], results["runs"], output_dir)
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
