"""Measure comparable algorithm timings and process peaks in fresh workers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CASES = {
    "multi_shapes_color": ("data/sample_images/multi_shapes_color.png", 100),
    "noisy_circle": ("data/sample_images/noisy_circle.png", 100),
    "oxford_cat_2008_001885": ("data/external/oxford-iseg/images/cat_2008_001885.jpg", 100),
    "stress_rgb_800": (None, 20),
}
METHODS = ("kmeans-2", "kmeans-3", "kmeans-5", "otsu")
MAX_SIDE = 800
STRESS_SEED = 42


def measure(case: str, method: str) -> dict:
    import numpy as np

    from scripts.benchmark_kmeans import peak_memory
    from src.algorithms.otsu import segment_image_otsu
    from src.processing.pipeline import KMeansParams, prepare_input, run_kmeans
    from src.utils.image_io import LoadedImage, load_path

    path, max_iter = CASES[case]
    if path is None:
        pixels = np.random.default_rng(STRESS_SEED).integers(0, 256, (800, 800, 3), dtype=np.uint8)
        loaded = LoadedImage(pixels, case, "RGB", pixels.nbytes, "RGB")
    else:
        loaded = load_path(ROOT / path)
    prepared = prepare_input(loaded, MAX_SIDE)
    input_sha256 = hashlib.sha256(memoryview(prepared.image).cast("B")).hexdigest()
    before = peak_memory()

    if method == "otsu":
        result, binary = segment_image_otsu(prepared.gray)
        after = peak_memory()
        outputs = (binary, result.histogram)
        details = {"threshold": result.threshold, "valid": result.valid, "iterations": None, "converged": None}
        seconds = result.elapsed_seconds
    else:
        k = int(method.removeprefix("kmeans-"))
        result = run_kmeans(prepared, KMeansParams(k=k, max_iter=max_iter))
        after = peak_memory()
        outputs = (result.labels, result.segmented, result.centers)
        details = {"threshold": None, "valid": None, "iterations": result.core.n_iter,
                   "converged": result.core.converged}
        seconds = result.core.elapsed_seconds

    digest = hashlib.sha256()
    for output in outputs:
        digest.update(memoryview(np.ascontiguousarray(output)).cast("B"))
    if method == "otsu":
        digest.update(str(result.threshold).encode("ascii"))

    return {"seconds": seconds, "before": before, "after": after, "input_sha256": input_sha256,
            "output_sha256": digest.hexdigest(), "size": prepared.resize.processed_size, **details}


def run_benchmark(output_dir: Path, repeats: int, cases: list[str]) -> None:
    from src.processing.pipeline import TIMING_SCOPE

    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    environment = {"python": platform.python_version(), "numpy": importlib.metadata.version("numpy"),
                   "platform": platform.platform(), "processor": platform.processor(), "logical_cpus": os.cpu_count(),
                   "timing_scope": TIMING_SCOPE, "peak_rss_scope": "Whole fresh worker, including imports and input",
                   "peak_rise_scope": "Increase in process high-water mark since input preparation, not allocation",
                   "repeats": repeats, "threads": 1, "max_side": MAX_SIDE, "median_size": 1,
                   "gray_conversion": "luminance", "init": "kmeans++", "seed": 0, "tol": 1e-4,
                   "stress_rgb_data_seed": STRESS_SEED}
    rows = []
    raw = []
    skipped = []
    script = str(Path(__file__).resolve())
    for case in cases:
        path, max_iter = CASES[case]
        if path is not None and not (ROOT / path).is_file():
            skipped.append({"case": case, "reason": f"Input unavailable: {path}"})
            continue
        for method in METHODS:
            runs = []
            for _ in range(repeats):
                cmd = [sys.executable, "-B", script, "--worker", case, method]
                process = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, check=True,
                                         text=True, encoding="utf-8", timeout=300)
                runs.append(json.loads(process.stdout))
            first = runs[0]
            if any(r["output_sha256"] != first["output_sha256"] for r in runs):
                raise RuntimeError(f"Result differed between equivalent worker runs: {case}, {method}")
            raw.append({"case": case, "method": method, "input_path": path, "runs": runs})
            mib = 1024**2
            before = [r["before"]["peak_rss_bytes"] for r in runs]
            after = [r["after"]["peak_rss_bytes"] for r in runs]
            row = {"case": case, "method": method, "width": first["size"][0], "height": first["size"][1],
                   "max_iter": None if method == "otsu" else max_iter,
                   "median_ms": statistics.median(r["seconds"] * 1000 for r in runs),
                   "peak_rss_before_median_mib": statistics.median(before) / mib,
                   "peak_rss_after_median_mib": statistics.median(after) / mib,
                   "peak_rss_rise_median_mib": statistics.median(a - b for a, b in zip(after, before)) / mib,
                   "peak_rss_after_max_mib": max(after) / mib,
                   "iterations": first["iterations"], "converged": first["converged"],
                   "threshold": first["threshold"], "valid": first["valid"]}
            rows.append(row)
            print(f"{case:24s} {method:8s} {row['median_ms']:8.2f} ms  "
                  f"peak={row['peak_rss_after_median_mib']:6.2f} MiB  "
                  f"rise={row['peak_rss_rise_median_mib']:6.2f} MiB", flush=True)

    if not rows:
        raise RuntimeError(f"No inputs available: {skipped}")
    output_dir.mkdir(parents=True, exist_ok=True)
    document = {"environment": environment, "measurements": rows, "raw_runs": raw, "skipped": skipped}
    (output_dir / "memory_benchmark.json").write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n",
                                                     encoding="utf-8")
    with (output_dir / "memory_benchmark.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "quality-memory")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--worker", nargs=2, metavar=("CASE", "METHOD"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(measure(*args.worker)))
        return
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    run_benchmark(args.output_dir, args.repeats, args.cases)


if __name__ == "__main__":
    main()
