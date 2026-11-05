"""Chạy thực nghiệm trên toàn bộ ảnh mẫu, ghi kết quả thật vào docs/results/ và hình vào docs/figures/.

Chạy:  python scripts/run_experiments.py
Quy tắc (cố định, không chỉnh tay theo kết quả):
  * K-means: K và đặc trưng theo gợi ý trong data/samples.json, K-means++, seed 0, tol 1e-4, tối đa 100 vòng;
    cụm đối tượng = gợi ý tự động theo viền ảnh (cụm chiếm nhiều pixel viền nhất là nền).
  * Otsu: ngưỡng tự động; lớp đối tượng (sáng/tối) = gợi ý theo viền ảnh (lớp chiếm đa số viền ảnh là nền).
  * Thời gian: trung vị của 5 lần chạy.
  * IoU/Dice chỉ tính khi mẫu có ground truth.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.algorithms.kmeans import segment_image_kmeans  # noqa: E402
from src.algorithms.otsu import segment_image_otsu  # noqa: E402
from src.processing.preprocessing import to_grayscale  # noqa: E402
from src.processing.pipeline import (  # noqa: E402
    TIMING_SCOPE,
    KMeansParams,
    default_foreground,
    fit_ground_truth,
    kmeans_outputs,
    prepare_input,
    run_kmeans,
    run_otsu,
)
from src.utils.image_io import load_path  # noqa: E402
from src.utils.metrics import compare_masks  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "docs" / "results"
FIG = ROOT / "docs" / "figures"
REPEATS = 5


def timed(fn, repeats=REPEATS, elapsed=None):
    """Lấy trung vị; elapsed chọn đồng hồ phân đoạn bên trong kết quả nếu có."""
    times, result = [], None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        wall_time = time.perf_counter() - t0
        times.append(wall_time if elapsed is None else elapsed(result))
    return result, statistics.median(times)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-figures", action="store_true", help="Chỉ cập nhật số liệu, giữ nguyên ảnh minh hoạ hiện có.")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.skip_figures:
        FIG.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((DATA / "samples.json").read_text(encoding="utf-8"))
    rows = []
    for s in manifest:
        prepared = prepare_input(load_path(DATA / s["file"]), 800)
        gt = None
        if s["ground_truth"]:
            gt = fit_ground_truth(np.asarray(Image.open(DATA / s["ground_truth"]).convert("L")), prepared)
        params = KMeansParams(k=s["suggested"]["k"], feature=s["suggested"]["feature"], seed=0)
        km, km_t = timed(lambda: run_kmeans(prepared, params), elapsed=lambda result: result.core.elapsed_seconds)
        fg = default_foreground(km)
        kout = kmeans_outputs(prepared, km, fg)
        otsu, otsu_t = timed(lambda: run_otsu(prepared, polarity="auto"),
                            elapsed=lambda result: result.result.elapsed_seconds)
        invert = otsu.invert
        row = {
            "id": s["id"], "title": s["title"], "size": f"{prepared.image.shape[1]}x{prepared.image.shape[0]}",
            "channels": 3 if prepared.is_color else 1,
            "k_requested": km.core.k_requested, "k_effective": km.core.k_effective, "feature": km.feature,
            "kmeans_iter": km.core.n_iter, "kmeans_converged": km.core.converged, "kmeans_sse": km.core.sse,
            "kmeans_fg_clusters": fg, "kmeans_time_ms": km_t * 1000,
            "otsu_threshold": otsu.result.threshold, "otsu_valid": otsu.result.valid, "otsu_invert": invert,
            "otsu_eta": otsu.result.separability, "otsu_time_ms": otsu_t * 1000,
            "warnings": km.core.warnings + otsu.warnings,
        }
        if gt is not None:
            ks, os_ = compare_masks(kout.mask, gt), compare_masks(otsu.mask, gt)
            row.update(kmeans_iou=ks.iou, kmeans_dice=ks.dice, otsu_iou=os_.iou, otsu_dice=os_.dice)
        rows.append(row)
        if not args.skip_figures:
            _figure(s, prepared, kout, otsu)
        print(f"{s['id']:<24} K={row['k_effective']} it={row['kmeans_iter']:<3} t*={row['otsu_threshold']:<3} "
              + (f"IoU km={row['kmeans_iou']:.4f} otsu={row['otsu_iou']:.4f}" if gt is not None else "(không có GT)"))

    # Thí nghiệm ảnh hưởng của K trên ảnh RGB nhiều vùng
    entry = next(s for s in manifest if s["id"] == "rgb_regions")
    prepared = prepare_input(load_path(DATA / entry["file"]), 800)
    k_sweep = []
    for k in range(2, 9):
        km, t = timed(lambda: run_kmeans(prepared, KMeansParams(k=k, feature="rgb", seed=0)), 3,
                      elapsed=lambda result: result.core.elapsed_seconds)
        k_sweep.append({"k": k, "sse": km.core.sse, "iter": km.core.n_iter, "time_ms": t * 1000})
    # Ảnh hưởng của seed (khởi tạo ngẫu nhiên) trên cùng ảnh, K = 6
    seed_sweep = []
    for seed in range(5):
        km = run_kmeans(prepared, KMeansParams(k=6, feature="rgb", seed=seed, init="random"))
        seed_sweep.append({"seed": seed, "sse": km.core.sse, "iter": km.core.n_iter})
    # Thời gian theo kích thước ảnh (ảnh ngẫu nhiên RGB, K = 4)
    rng = np.random.default_rng(0)
    scale = []
    for side in (128, 256, 512, 1024):
        img = rng.integers(0, 256, (side, side, 3), dtype=np.uint8)
        km, t_km = timed(lambda: segment_image_kmeans(img, 4, "rgb", max_iter=20, tol=0.0, seed=0), 3,
                         elapsed=lambda result: result.core.elapsed_seconds)
        gray = to_grayscale(img)
        _, t_otsu = timed(lambda: segment_image_otsu(gray), 3, elapsed=lambda result: result[0].elapsed_seconds)
        scale.append({"side": side, "pixels": side * side, "kmeans_20iter_ms": t_km * 1000,
                      "kmeans_iter": km.core.n_iter, "kmeans_converged": km.core.converged,
                      "otsu_ms": t_otsu * 1000})

    env = {"python": platform.python_version(), "os": platform.platform(), "processor": platform.processor(),
            "numpy": np.__version__, "measured_at": datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat(timespec="seconds"),
            "timing_scope": TIMING_SCOPE, "sample_repeats": REPEATS, "sweep_repeats": 3}
    (OUT / "experiment_results.json").write_text(
        json.dumps({"environment": env, "samples": rows, "k_sweep_rgb_regions": k_sweep,
                    "seed_sweep_rgb_regions_k6_random": seed_sweep, "scaling": scale},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(env, rows, k_sweep, seed_sweep, scale)
    print("Đã ghi docs/results/experiment_results.json và .md. "
          + ("Giữ nguyên ảnh minh hoạ." if args.skip_figures else "Đã ghi hình trong docs/figures/."))
    return 0


def _figure(s, prepared, kout, otsu) -> None:
    panels = [(prepared.image, "Ảnh gốc"), (kout.segmented, "K-means"), (kout.mask, "Mask K-means"),
              (otsu.binary, f"Otsu (t* = {otsu.result.threshold})"), (otsu.mask, "Mask Otsu")]
    fig, axes = plt.subplots(1, 5, figsize=(13, 2.9), dpi=110)
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None, vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle(s["title"], fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / f"{s['id']}.png")
    plt.close(fig)


def _write_markdown(env, rows, k_sweep, seed_sweep, scale) -> None:
    lines = ["# Kết quả thực nghiệm (sinh tự động bởi scripts/run_experiments.py)", "",
             f"Môi trường: Python {env['python']}, NumPy {env['numpy']}, {env['os']}, CPU: {env['processor']}.",
             f"Thời điểm đo: {env['measured_at']} (Asia/Ho_Chi_Minh).",
             "Bảng mẫu: trung vị 5 lần chạy; quét K và kích thước: trung vị 3 lần. Ảnh mẫu đều ≤ 800 px.",
             "Phạm vi đo: " + TIMING_SCOPE, "",
             "| Mẫu | Kích thước | K (thực tế) | Vòng lặp | K-means (ms) | t* Otsu | Otsu (ms) | IoU K-means | IoU Otsu | Dice K-means | Dice Otsu |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        f = lambda key: f"{r[key]:.4f}" if key in r else "—"  # noqa: E731
        lines.append(f"| {r['id']} | {r['size']} | {r['k_effective']} ({r['feature']}) | {r['kmeans_iter']} | "
                     f"{r['kmeans_time_ms']:.1f} | {r['otsu_threshold']}{'' if r['otsu_valid'] else ' (không hợp lệ)'}"
                     f"{' đảo' if r['otsu_invert'] else ''} | {r['otsu_time_ms']:.2f} | {f('kmeans_iou')} | {f('otsu_iou')} | "
                     f"{f('kmeans_dice')} | {f('otsu_dice')} |")
    lines += ["", "Cảnh báo phát sinh:", ""]
    for r in rows:
        for w in r["warnings"]:
            lines.append(f"- `{r['id']}`: {w}")
    lines += ["", "## Ảnh hưởng của K (rgb_regions, RGB, K-means++, seed 0)", "", "| K | SSE | Vòng lặp | Thời gian (ms) |", "|---|---|---|---|"]
    lines += [f"| {r['k']} | {r['sse']:,.0f} | {r['iter']} | {r['time_ms']:.1f} |" for r in k_sweep]
    lines += ["", "## Ảnh hưởng của seed (rgb_regions, K = 6, khởi tạo ngẫu nhiên)", "", "| Seed | SSE | Vòng lặp |", "|---|---|---|"]
    lines += [f"| {r['seed']} | {r['sse']:,.0f} | {r['iter']} |" for r in seed_sweep]
    lines += ["", "## Thời gian theo kích thước ảnh (ảnh RGB ngẫu nhiên; K-means K = 4, tối đa 20 vòng)", "",
              "| Cạnh | Số pixel | K-means (ms) | Vòng lặp | Hội tụ | Otsu (ms) |", "|---|---|---|---|---|---|"]
    lines += [f"| {r['side']} | {r['pixels']:,} | {r['kmeans_20iter_ms']:.1f} | {r['kmeans_iter']} | "
              f"{'Có' if r['kmeans_converged'] else 'Chưa'} | {r['otsu_ms']:.2f} |" for r in scale]
    (OUT / "experiment_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
