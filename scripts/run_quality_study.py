"""Measure K-means and Otsu on the bundled samples and the fixed Oxford test set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from scripts.evaluate_real_images import DATA as OXFORD  # noqa: E402
from src.processing.pipeline import (  # noqa: E402
    TIMING_SCOPE,
    KMeansParams,
    default_foreground,
    prepare_input,
    run_kmeans,
    run_otsu,
)
from src.processing.postprocessing import mask_from_labels  # noqa: E402
from src.utils.image_io import load_any, load_image_bytes, to_png_bytes  # noqa: E402
from src.utils.metrics import compare_masks, compare_trimap  # noqa: E402

DATA = ROOT / "data"
KS = (2, 3, 5)
SEEDS = range(5)
SEED_CASES = {"multi_shapes_color", "rgb_regions", "noisy_circle", "low_contrast", "banana1", "boat_2007_000837"}
MAX_SIDE = 800
MAX_ITER = 100
TOL = 1e-4


@dataclass(frozen=True)
class Case:
    id: str
    title: str
    group: str
    path: Path | None
    reference: Path | None
    feature: str
    extra_k: int | None = None
    median_size: int = 1


def study_cases(synthetic_only: bool) -> list[Case]:
    manifest = json.loads((DATA / "samples.json").read_text(encoding="utf-8"))
    cases = [Case(s["id"], s["title"], "synthetic", DATA / s["file"],
                  DATA / s["ground_truth"] if s["ground_truth"] else None,
                  s["suggested"]["feature"], s["suggested"]["k"]) for s in manifest]
    cases.append(Case("equal_luma_color", "Hai màu RGB có cùng mức xám", "synthetic", None, None, "rgb"))
    noisy = next(case for case in cases if case.id == "noisy_circle")
    cases.append(Case("noisy_circle_median3", "Hình tròn nhiễu sau lọc trung vị 3×3", "filtered",
                      noisy.path, noisy.reference, "gray", median_size=3))
    if synthetic_only:
        return cases
    photos = sorted((p for p in (OXFORD / "images").glob("*") if p.suffix.lower() in (".jpg", ".png", ".bmp")),
                    key=lambda p: hashlib.sha256(p.name.encode()).hexdigest())
    if len(photos) < 30:
        raise ValueError("Thiếu ảnh Oxford. Chuẩn bị cache bằng scripts/evaluate_real_images.py hoặc dùng --synthetic-only.")
    for photo in photos[6:30]:
        reference = OXFORD / "images-gt" / f"{photo.stem}.png"
        if not reference.is_file():
            raise ValueError(f"Thiếu trimap Oxford của {photo.name}.")
        cases.append(Case(photo.stem, f"Oxford VGG iSeg · {photo.stem}", "photo", photo, reference, "rgb"))
    return cases


def equal_luma_case():
    """A fixed geometric reference, independent of the predictions."""
    target = np.zeros((144, 192), dtype=bool)
    target[36:108, 48:144] = True
    rgb = np.empty((*target.shape, 3), dtype=np.uint8)
    rgb[:] = (255, 0, 0)
    rgb[target] = (0, 130, 0)
    data = to_png_bytes(rgb)
    return load_image_bytes(data, "equal_luma_color.png"), target.astype(np.uint8) * 255, data


def load_reference(path: Path | None, generated, size, *, trimap: bool):
    if path is None:
        return generated
    with Image.open(path) as source:
        reference = np.asarray(source.convert("L").resize(size, Image.Resampling.NEAREST))
    allowed = (0, 128, 255) if trimap else (0, 255)
    if not np.all(np.isin(reference, allowed)):
        raise ValueError(f"Nhãn tham chiếu không hợp lệ trong {path}.")
    return reference


def cluster(prepared, case: Case, k: int, seed: int = 0, init: str = "kmeans++"):
    return run_kmeans(prepared, KMeansParams(k, case.feature, MAX_ITER, TOL, seed, init))


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes()).hexdigest()


def partition_digest(labels: np.ndarray) -> str:
    """Canonicalize labels by the order in which each cluster first appears."""
    _, first, inverse = np.unique(labels.ravel(), return_index=True, return_inverse=True)
    canonical = np.argsort(np.argsort(first)).astype(np.uint8)
    return digest(canonical[inverse])


def measure(prepared, case: Case, ks: list[int], repeats: int):
    """One warm-up and repeated rotated execution order on one prepared image."""
    order = [*ks, None]
    times = {key: [] for key in order}
    signatures = {}
    results = {}
    for trial in range(repeats + 1):
        offset = trial % len(order)
        for k in order[offset:] + order[:offset]:
            if k is None:
                result = run_otsu(prepared, polarity="auto")
                elapsed = result.result.elapsed_seconds
                signature = (result.threshold_used, digest(result.binary), digest(result.mask))
            else:
                result = cluster(prepared, case, k)
                elapsed = result.core.elapsed_seconds
                signature = (digest(result.labels), digest(result.centers), digest(result.segmented))
            if trial == 0:
                signatures[k] = signature
            else:
                if signatures[k] != signature:
                    raise RuntimeError(f"Kết quả không tái lập trên {case.id}, K={k}.")
                times[k].append(elapsed * 1000)
                results[k] = result
    return results, times


def verify_otsu(gray, output) -> None:
    """Independent exhaustive integer-ratio reference, including tied thresholds."""
    bins, counts = np.unique(gray, return_counts=True)
    hist = dict(zip(map(int, bins), map(int, counts)))
    n = gray.size
    total = sum(g * count for g, count in hist.items())
    if not np.array_equal(output.result.histogram, [hist.get(g, 0) for g in range(256)]):
        raise RuntimeError("Histogram khác phép đếm độc lập.")
    n0 = moment0 = 0
    scores = {}
    for t in range(255):
        n0 += hist.get(t, 0)
        moment0 += t * hist.get(t, 0)
        if 0 < n0 < n:
            scores[t] = Fraction((moment0 * n - total * n0) ** 2, n0 * (n - n0))
    if scores:
        best = max(scores.values())
        tied = [t for t, value in scores.items() if value == best]
        center = sum(tied) // len(tied)
        chosen = min(tied, key=lambda t: (abs(t - center), t))
        if (not output.result.valid or output.result.tied_thresholds != tied or output.threshold_used != chosen
                or not np.isclose(output.result.max_variance, float(best / (n * n)), rtol=1e-12, atol=0)):
            raise RuntimeError("Ngưỡng Otsu khác đối chứng hữu tỉ.")
    elif output.result.valid or output.mask.any():
        raise RuntimeError("Otsu tự động tạo đối tượng trên ảnh đồng nhất.")
    expected = np.where(gray > output.threshold_used, 255, 0).astype(np.uint8)
    if not np.array_equal(expected, output.binary):
        raise RuntimeError("Ảnh nhị phân khác điều kiện I > t.")


def score_mask(mask, reference, group: str) -> dict:
    if reference is None:
        return {key: None for key in ("iou", "dice", "tp", "fp", "fn", "tn", "pixel_accuracy")}
    metric = compare_trimap if group == "photo" else compare_masks
    return asdict(metric(mask, reference))


def timing_summary(times: list[float]) -> dict:
    p25, p75 = np.percentile(times, [25, 75])
    return {"time_ms": statistics.median(times), "time_p25_ms": float(p25), "time_p75_ms": float(p75),
            "times_ms": times}


def describe_kmeans(case, result, reference, k, seed, init, phase, times):
    core = result.core
    foreground = default_foreground(result)
    mask = mask_from_labels(result.labels, foreground, n_clusters=core.k_effective)
    row = {"case_id": case.id, "algorithm": "kmeans", "phase": phase, "k_requested": k, "seed": seed, "init": init,
           "feature": result.feature, "k_effective": core.k_effective, "iterations": core.n_iter,
           "converged": core.converged, "sse": core.sse, "sse_per_pixel": core.sse / result.labels.size,
           "foreground_clusters": foreground, "foreground_ratio": float(np.count_nonzero(mask) / mask.size),
           "centers": result.centers.tolist(), "counts": core.counts.tolist(), "warnings": core.warnings,
           "partition_sha256": partition_digest(result.labels), "mask_sha256": digest(mask)}
    return {**row, **timing_summary(times), **score_mask(mask, reference, case.group)}, mask


def describe_otsu(case, output, reference, times):
    r = output.result
    row = {"case_id": case.id, "algorithm": "otsu", "phase": "main", "k_requested": None, "seed": None, "init": None,
           "feature": "gray", "threshold": output.threshold_used, "threshold_valid": r.valid,
           "threshold_ties": len(r.tied_thresholds), "polarity": ("dark" if output.invert else "bright") if r.valid else None,
           "eta": r.separability, "between_class_variance": r.max_variance, "foreground_ratio": output.foreground_ratio,
           "weight0": r.weight0, "weight1": r.weight1, "mean0": r.mean0 if r.valid else None,
           "mean1": r.mean1 if r.valid else None, "mask_sha256": digest(output.mask), "warnings": output.warnings}
    return {**row, **timing_summary(times), **score_mask(output.mask, reference, case.group)}


def seed_runs(case, prepared, reference, baseline):
    rows = []
    for k in KS:
        for init in ("kmeans++", "random"):
            base_row, base_mask = baseline[k] if init == "kmeans++" else (None, None)
            for seed in SEEDS:
                if init == "kmeans++" and seed == 0:
                    continue
                result = cluster(prepared, case, k, seed, init)
                row, mask = describe_kmeans(case, result, reference, k, seed, init, "seed",
                                            [result.core.elapsed_seconds * 1000])
                if seed == 0:
                    base_row, base_mask = row, mask
                row["partition_same_seed0"] = row["partition_sha256"] == base_row["partition_sha256"]
                row["mask_agreement_seed0"] = compare_masks(mask, base_mask).iou
                rows.append(row)
    return rows


def run_case(case: Case, repeats: int, output_dir: Path):
    from scripts.plot_quality_case import render_case_figure

    if case.path is None:
        loaded, generated_reference, image_bytes = equal_luma_case()
    else:
        image_bytes = case.path.read_bytes()
        loaded, generated_reference = load_any(image_bytes, case.path.name), None
    prepared = prepare_input(loaded, MAX_SIDE, median_size=case.median_size)
    ks = sorted(set(KS) | ({case.extra_k} if case.extra_k else set()))
    results, times = measure(prepared, case, ks, repeats)
    otsu = results[None]
    verify_otsu(prepared.gray, otsu)
    trimap = case.group == "photo"
    reference = load_reference(case.reference, generated_reference, prepared.resize.processed_size, trimap=trimap)
    gt_kind = ("trimap" if trimap else "binary") if reference is not None else None
    figure_path = Path("images") / f"{case.id}.png"
    entry = {"id": case.id, "title": case.title, "group": case.group, "gt_kind": gt_kind,
             "image_path": case.path.relative_to(ROOT).as_posix() if case.path else None,
             "reference_path": case.reference.relative_to(ROOT).as_posix() if case.reference else None,
             "original_size": list(prepared.resize.original_size), "processed_size": list(prepared.resize.processed_size),
             "median_size": case.median_size, "figure": figure_path.as_posix(), "source_sha256": hashlib.sha256(image_bytes).hexdigest(),
             "reference_sha256": (hashlib.sha256(case.reference.read_bytes()).hexdigest() if case.reference
                                  else digest(reference) if reference is not None else None),
             "ignored_reference_pixels": int(np.count_nonzero(reference == 128)) if trimap else None}
    otsu_row = describe_otsu(case, otsu, reference, times[None])
    panels, baseline = [], {}
    for k in ks:
        result = results[k]
        row, mask = describe_kmeans(case, result, reference, k, 0, "kmeans++", "main", times[k])
        panels.append((row, result.segmented, mask))
        baseline[k] = (row, mask)
    rows = [otsu_row, *(row for row, _, _ in panels)]
    if case.id in SEED_CASES:
        rows.extend(seed_runs(case, prepared, reference, baseline))
    r = otsu.result
    ties = set(r.tied_thresholds)
    hist = [{"case_id": case.id, "level": g, "count": int(r.histogram[g]),
             "probability": float(r.histogram[g] / prepared.gray.size),
             "between_class_variance": float(r.between_class_variance[g]) if r.valid_thresholds[g] else None,
             "optimal_threshold": g in ties} for g in range(256)]
    render_case_figure(output_dir / figure_path, entry, loaded.array, reference, panels, otsu, otsu_row)
    return entry, rows, hist


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def mask_observation(row: dict) -> str:
    fp, fn = row["fp"], row["fn"]
    if fp == fn == 0:
        return "khớp toàn bộ pixel được chấm"
    if fp > fn:
        return f"lấy nhầm nền nhiều hơn bỏ sót đối tượng ({fp:,} FP, {fn:,} FN)"
    if fn > fp:
        return f"bỏ sót đối tượng nhiều hơn lấy nhầm nền ({fn:,} FN, {fp:,} FP)"
    return f"lấy nhầm nền và bỏ sót đối tượng cùng {fp:,} pixel"


def write_case_report(path: Path, cases: list[dict], runs: list[dict], environment: dict) -> None:
    lines = ["# Đối chiếu từng ảnh", "",
             f"Một lượt làm nóng, {environment['repeats']} lần đo mỗi cấu hình chính. {environment['timing_scope']}",
             "Bảng sử dụng trung vị. Các cấu hình K-means chính dùng K-means++, seed 0, tol 1e-4, tối đa 100 vòng.",
             "Cùng chuyển xám BT.601, cạnh dài tối đa 800 px, không áp dụng hình thái. GT Oxford bỏ pixel có nhãn 128.",
             "Cụm/lớp đối tượng chọn theo viền và trung tâm. Trắng là đối tượng trong mask kết quả.",
             "Các dòng phase=seed trong runs.csv dùng để khảo sát khởi tạo, mỗi seed đo một lần.", ""]
    by_case = {case["id"]: [] for case in cases}
    for row in runs:
        if row["phase"] == "main":
            by_case[row["case_id"]].append(row)
    for case in cases:
        w, h = case["processed_size"]
        filtered = f"Lọc trung vị {case['median_size']}×{case['median_size']}." if case["median_size"] > 1 else "Tắt lọc trung vị."
        lines += [f"## {case['title']}", "", f"[{case['id']} — ảnh gốc, kết quả, mask và histogram]({case['figure']})", "",
                  f"Kích thước xử lý {w}×{h}. {filtered}", "",
                  "| Thuật toán và tham số | Vòng | Trung vị (ms) | IoU | Dice | FP | FN |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for row in by_case[case["id"]]:
            if row["algorithm"] == "otsu":
                if row["threshold_valid"]:
                    polarity = "tối" if row["polarity"] == "dark" else "sáng"
                    description = f"Otsu t={row['threshold']}, đối tượng {polarity}, {row['threshold_ties']} cực đại"
                else:
                    description = f"Otsu, ngưỡng dự phòng {row['threshold']}, mask rỗng"
                iterations = "—"
            else:
                k, actual = row["k_requested"], row["k_effective"]
                label = str(k) if k == actual else f"{k}→{actual}"
                description = f"K-means K={label}, {row['feature'].upper()}, FG {row['foreground_clusters']}, SSE {row['sse']:.3g}"
                iterations = str(row["iterations"]) + ("" if row["converged"] else " (chưa hội tụ)")
            scores = "— | — | — | —" if row["iou"] is None else f"{row['iou']:.4f} | {row['dice']:.4f} | {row['fp']} | {row['fn']}"
            lines.append(f"| {description} | {iterations} | {row['time_ms']:.3f} | {scores} |")
        otsu = by_case[case["id"]][0]
        if not otsu["threshold_valid"]:
            observation = "Histogram xám đồng nhất. Otsu không tìm được cách chia hai lớp khác rỗng."
        else:
            observation = (f"Otsu chia khối lượng histogram {otsu['weight0']:.1%} / {otsu['weight1']:.1%}, "
                           f"với mức xám trung bình {otsu['mean0']:.2f} / {otsu['mean1']:.2f} và η={otsu['eta']:.3f}.")
        if case["gt_kind"] is None:
            observation += " Ca này dùng để quan sát định tính vì thiếu mask tham chiếu."
        else:
            k5 = next(row for row in by_case[case["id"]] if row["k_requested"] == 5)
            observation += f" Mặt nạ Otsu {mask_observation(otsu)}. K-means yêu cầu K=5 {mask_observation(k5)}."
            if case["gt_kind"] == "trimap":
                observation += f" Đã bỏ {case['ignored_reference_pixels']:,} pixel có nhãn 128 khi tính điểm."
        lines += ["", observation, ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "quality-study")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--synthetic-only", action="store_true", help="Bỏ tập Oxford khi chưa có cache ảnh.")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats phải >= 1.")
    if (args.output_dir / "study.json").exists() or (args.output_dir / "images").exists():
        parser.error("Hãy chọn thư mục mới để giữ dữ liệu thực nghiệm cũ.")
    try:
        cases = study_cases(args.synthetic_only)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    (args.output_dir / "images").mkdir(parents=True)
    revision = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=False).stdout.strip() if shutil.which("git") else ""
    environment = {"source_revision": revision or None, "measured_at": datetime.now().astimezone().isoformat(),
                   "platform": platform.platform(), "processor": platform.processor(), "python": platform.python_version(),
                   "numpy": np.__version__, "timing_scope": TIMING_SCOPE, "warmups": 1, "repeats": args.repeats,
                   "max_side": MAX_SIDE, "tol": TOL, "max_iter": MAX_ITER, "gray_method": "luminance", "morphology": "none",
                   "threads": {name: os.environ[name] for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
                   "photo_split": "First 30 by SHA-256(filename), held-out items 7–30, gain 1; ignore trimap 128",
                   "foreground_rule": "Viền/trung tâm mặc định, được áp dụng trước khi tính điểm với GT"}
    entries, runs, histograms = [], [], []
    for number, case in enumerate(cases, 1):
        print(f"{number}/{len(cases)} {case.id}", flush=True)
        entry, rows, hist = run_case(case, args.repeats, args.output_dir)
        entries.append(entry)
        runs.extend(rows)
        histograms.extend(hist)
    payload = {"environment": environment, "cases": entries, "runs": runs}
    (args.output_dir / "study.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    write_csv(args.output_dir / "runs.csv", runs)
    write_csv(args.output_dir / "histograms.csv", histograms)
    write_case_report(args.output_dir / "cases.md", entries, runs, environment)
    from scripts.plot_quality_study import render_study_figures

    render_study_figures(entries, runs, args.output_dir / "charts")
    print(f"Đã ghi {len(entries)} ảnh, {len(runs)} cấu hình và {len(histograms)} hàng histogram vào {args.output_dir}.")


if __name__ == "__main__":
    main()
