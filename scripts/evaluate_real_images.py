"""Reproducible fixed-parameter evaluation on Oxford VGG interactive-segmentation photographs.

Images stay under ignored data/external; reports contain IDs, SHA-256, scores and timings.
No ground truth is passed to clustering, foreground selection, or parameter selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.processing.pipeline import (  # noqa: E402
    TIMING_SCOPE, KMeansParams, default_foreground, kmeans_outputs,
    prepare_input, run_kmeans, run_otsu,
)
from src.utils.image_io import load_image_bytes, load_path  # noqa: E402
from src.utils.metrics import compare_trimap  # noqa: E402

SOURCE_PAGE = "https://www.robots.ox.ac.uk/~vgg/data/iseg/"
ARCHIVES = {
    "images.tgz": "88b905dbde17ee980ce2efc78e6e07d54e14c604574a87542c3f2384ce90bd49",
    "images-gt.tgz": "9e95ab84f39c7118baaf01fe35f7a4a6208340f36d6cd20a24a39592ecdacec8",
}
DATA = ROOT / "data/external/oxford-iseg"


def fetch_dataset() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    for name, expected in ARCHIVES.items():
        archive = DATA / name
        existing = ROOT / "outputs" / name
        if not archive.exists():
            if existing.exists():
                content = existing.read_bytes()
            else:
                with urllib.request.urlopen(SOURCE_PAGE + "data/" + name, timeout=90) as response:
                    content = response.read(40_000_001)
                if len(content) > 40_000_000:
                    raise ValueError("Unexpected archive size")
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError(f"Archive checksum differs: {name}")
            archive.write_bytes(content)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Archive checksum differs: {name}")
        with tarfile.open(archive) as source:
            for member in source.getmembers():
                if member.isdir():
                    continue
                path = PurePosixPath(member.name)
                if not member.isfile() or path.is_absolute() or ".." in path.parts or len(path.parts) != 2:
                    raise ValueError(f"Unsafe archive entry: {member.name}")
                if path.parts[0] not in ("images", "images-gt") or path.suffix.lower() not in (".jpg", ".png", ".bmp"):
                    raise ValueError(f"Unexpected archive entry: {member.name}")
                destination = DATA.joinpath(*path.parts).resolve()
                if not destination.is_relative_to(DATA.resolve()):
                    raise ValueError("Archive path escapes destination")
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(source.extractfile(member).read())


def evaluate(image_path: Path, split: str, gain: float, repeats: int) -> dict:
    loaded = load_path(image_path)
    if gain != 1:
        from src.utils.image_io import to_png_bytes
        pixels = np.rint(loaded.array.astype(np.float64) * gain).clip(0, 255).astype(np.uint8)
        loaded = load_image_bytes(to_png_bytes(pixels), image_path.name)
    prepared = prepare_input(loaded, 800)
    times_km, times_otsu = [], []
    signatures = []
    for _ in range(repeats):
        km = run_kmeans(prepared, KMeansParams(k=5, feature="rgb", seed=0, max_iter=100))
        fg = default_foreground(km)
        output = kmeans_outputs(prepared, km, fg)
        otsu = run_otsu(prepared, polarity="auto")
        times_km.append(km.core.elapsed_seconds)
        times_otsu.append(otsu.result.elapsed_seconds)
        signatures.append(hashlib.sha256(output.mask.tobytes() + otsu.mask.tobytes()).hexdigest())
    assert len(set(signatures)) == 1, f"Non-repeatable segmentation: {image_path.name}"
    # Ground truth is read only after all predictions have been fixed.
    gt_path = DATA / "images-gt" / (image_path.stem + ".png")
    from PIL import Image
    with Image.open(gt_path) as mask:
        gt = np.asarray(mask.convert("L").resize(prepared.resize.processed_size, Image.Resampling.NEAREST))
    if not set(np.unique(gt)).issubset({0, 128, 255}):
        raise ValueError(f"Unexpected annotation labels: {gt_path.name}")
    valid = gt != 128  # Oxford trimap: 128 is uncertain boundary, not foreground.
    km_scores = compare_trimap(output.mask, gt)
    otsu_scores = compare_trimap(otsu.mask, gt)
    return {
        "id": image_path.stem, "split": split, "brightness_gain": gain,
        "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        "ground_truth_sha256": hashlib.sha256(gt_path.read_bytes()).hexdigest(),
        "original_size": list(prepared.resize.original_size), "processed_size": list(prepared.resize.processed_size),
        "k": km.core.k_effective, "iterations": km.core.n_iter, "converged": km.core.converged,
        "foreground_clusters_auto": fg, "otsu_threshold": otsu.threshold_used,
        "ground_truth_touches_border": bool((gt[0] == 255).any() or (gt[-1] == 255).any() or (gt[:, 0] == 255).any() or (gt[:, -1] == 255).any()),
        "foreground_ratio": float(np.count_nonzero(gt == 255) / gt.size),
        "evaluated_pixels": int(np.count_nonzero(valid)), "ignored_uncertain_boundary_pixels": int(np.count_nonzero(~valid)),
        "kmeans_iou": km_scores.iou, "kmeans_dice": km_scores.dice,
        "otsu_iou": otsu_scores.iou, "otsu_dice": otsu_scores.dice,
        "kmeans_median_ms": statistics.median(times_km) * 1000,
        "otsu_median_ms": statistics.median(times_otsu) * 1000,
        "kmeans_runs_ms": [t * 1000 for t in times_km], "otsu_runs_ms": [t * 1000 for t in times_otsu],
        "mask_sha256": signatures[0],
    }


def write_report(path: Path, environment: dict, rows: list[dict]) -> None:
    path.write_text(json.dumps({"environment": environment, "samples": rows}, indent=2), encoding="utf-8")
    lines = ["# Ảnh chụp thật — Oxford VGG iSeg", "", f"Nguồn: {SOURCE_PAGE}", "",
             "Chọn mẫu và chia dev/test bằng SHA-256 của tên tệp trước khi đọc ảnh/mask; không dùng GT để chỉnh K hoặc foreground.",
             "K=5, RGB, K-means++, seed=0, tối đa 100 vòng; foreground tự động. Otsu tự động trên cùng ảnh tiền xử lý, cạnh dài ≤800.",
             f"Thời gian: trung vị {environment['repeats']} lần, cùng phạm vi: {TIMING_SCOPE}",
             "Gain 0.5 là biến thể giảm sáng có kiểm soát của ảnh thật, không phải ảnh chụp thiếu sáng độc lập.",
             "Mask tham chiếu: 0 = nền, 255 = đối tượng; bỏ qua vùng biên chưa chắc chắn (128) khi tính IoU/Dice. Resize trimap bằng nearest.",
             "Đây là đánh giá hai thuật toán không giám sát với heuristic foreground, không phải kết quả của thuật toán GrabCut.", "",
             "| Ảnh | Tập | Gain | Sát viền | IoU KM | Dice KM | IoU Otsu | Dice Otsu | KM ms | Otsu ms |", "|---|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['id']} | {row['split']} | {row['brightness_gain']} | {row['ground_truth_touches_border']} | "
                     f"{row['kmeans_iou']:.4f} | {row['kmeans_dice']:.4f} | {row['otsu_iou']:.4f} | {row['otsu_dice']:.4f} | "
                     f"{row['kmeans_median_ms']:.2f} | {row['otsu_median_ms']:.2f} |")
    test = [row for row in rows if row["split"] == "test" and row["brightness_gain"] == 1]
    if test:
        lines += ["", "## Trung bình macro trên tập test gốc", ""]
        for key in ("kmeans_iou", "kmeans_dice", "otsu_iou", "otsu_dice"):
            lines.append(f"- {key}: {statistics.mean(row[key] for row in test):.4f}")
        failures = sorted(test, key=lambda row: min(row["kmeans_iou"], row["otsu_iou"]))[:5]
        lines += ["", "## Ca thất bại cần đọc cùng bảng kết quả", ""]
        lines += [f"- `{row['id']}`: KM IoU={row['kmeans_iou']:.4f}, Otsu IoU={row['otsu_iou']:.4f}; "
                  f"sát viền={row['ground_truth_touches_border']}, đối tượng chiếm {row['foreground_ratio']:.1%} ảnh."
                  for row in failures]
        lines += ["", "Màu của nền/đối tượng chồng lấn làm phân cụm hoặc ngưỡng không thể mô tả một mask ngữ nghĩa. "
                  "Heuristic viền/trung tâm có thể chọn nhầm lớp, nhất là vật thể sát mép. "
                  "IoU thấp là giới hạn thực tế, không được che bằng chọn cụm theo GT."]
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/results/cr_real_images.json")
    args = parser.parse_args()
    if args.samples < 5 or args.repeats < 1:
        parser.error("samples >= 5 and repeats >= 1 are required")
    if args.fetch:
        fetch_dataset()
    images = sorted((path for path in (DATA / "images").glob("*") if path.suffix.lower() in (".jpg", ".png", ".bmp")),
                    key=lambda path: hashlib.sha256(path.name.encode()).hexdigest())
    if len(images) < args.samples:
        parser.error("Dataset missing or too small; run with --fetch")
    environment = {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(),
                   "source": SOURCE_PAGE, "archive_sha256": ARCHIVES, "repeats": args.repeats,
                   "threads": {name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
                   "split_rule": "SHA256(filename), first 20% dev, remaining test", "parameters_frozen": {"k": 5, "feature": "rgb", "seed": 0, "max_iter": 100}}
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(images[:args.samples]):
        split = "dev" if index < args.samples // 5 else "test"
        for gain in (1.0, 0.5):
            row = evaluate(image, split, gain, args.repeats)
            rows.append(row)
            write_report(args.output, environment, rows)
            print(f"{index+1}/{args.samples} {image.stem} gain={gain}: IoU KM={row['kmeans_iou']:.3f}, Otsu={row['otsu_iou']:.3f}", flush=True)


if __name__ == "__main__":
    main()
