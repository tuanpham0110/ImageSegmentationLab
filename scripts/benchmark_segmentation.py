"""Benchmark exact segmentation outputs in fresh processes at native image dimensions."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import platform
import statistics
import subprocess
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SIZES = [(256, 256), (512, 512), (1024, 1024), (1920, 1080), (3840, 2160)]
METHODS = ("kmeans-rgb", "kmeans-gray", "otsu")
THREADS = {name: "1" for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}
SEED, K, MAX_ITER, TOL = 42, 4, 8, 1e-4


def load_algorithm(name: str, source_dir: Path | None = None):
    if source_dir is None:
        return importlib.import_module(f"src.algorithms.{name}")
    spec = importlib.util.spec_from_file_location(f"benchmark_baseline_{name}", source_dir / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fingerprint(result) -> str:
    import numpy as np

    digest = hashlib.sha256()

    def add(value):
        if isinstance(value, np.ndarray):
            digest.update(repr((value.dtype.str, value.shape)).encode())
            digest.update(memoryview(np.ascontiguousarray(value)).cast("B"))
        elif is_dataclass(value):
            for field in fields(value):
                if field.name != "elapsed_seconds":
                    digest.update(field.name.encode())
                    add(getattr(value, field.name))
        elif isinstance(value, (tuple, list)):
            digest.update(str(len(value)).encode())
            for item in value:
                add(item)
        else:
            digest.update(repr(value).encode())

    add(result)
    return digest.hexdigest()


def worker(method: str, size: tuple[int, int], source_dir: Path | None) -> dict:
    import numpy as np

    from scripts.benchmark_kmeans import peak_memory

    name, _, feature = method.partition("-")
    module = load_algorithm(name, source_dir)
    width, height = size
    dimensions = (height, width, 3) if feature == "rgb" else (height, width)
    rng = np.random.default_rng(SEED)
    warm_dimensions = (32, 32, 3) if feature == "rgb" else (32, 32)

    def segment(image):
        if name == "otsu":
            result, binary = module.segment_image_otsu(image)
            return (result, binary), result.elapsed_seconds, {"threshold": result.threshold}
        result = module.segment_image_kmeans(image, K, feature=feature, max_iter=MAX_ITER, tol=TOL, seed=0)
        return result, result.core.elapsed_seconds, {"iterations": result.core.n_iter, "converged": result.core.converged}

    segment(np.random.default_rng(SEED + 1).integers(0, 256, warm_dimensions, dtype=np.uint8))
    image = rng.integers(0, 256, dimensions, dtype=np.uint8)
    before = peak_memory()
    result, elapsed, details = segment(image)
    peak = peak_memory()
    return {"seconds": elapsed, **peak, "baseline_peak_rss_bytes": before["peak_rss_bytes"],
            "input_sha256": fingerprint(image), "output_sha256": fingerprint(result), **details}


def resolution(value: str) -> tuple[int, int]:
    try:
        width, height = map(int, value.lower().split("x"))
        if min(width, height) >= 1:
            return width, height
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("Expected a positive WIDTHxHEIGHT, e.g. 1920x1080.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/performance/measurements.json")
    parser.add_argument("--source-dir", type=Path, help="Optional directory containing the baseline algorithm files.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--sizes", nargs="+", type=resolution, default=SIZES)
    parser.add_argument("--include-4k-rgb", action="store_true", help="Add the high-color 4K stress case.")
    parser.add_argument("--worker", nargs=3, metavar=("METHOD", "WIDTH", "HEIGHT"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        method, width, height = args.worker
        print(json.dumps(worker(method, (int(width), int(height)), args.source_dir)))
        return
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.output.exists():
        parser.error("Choose a new output file to keep previous measurements.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source_dir = args.source_dir or ROOT / "src/algorithms"
    from importlib.metadata import version

    environment = {"python": platform.python_version(), "numpy": version("numpy"), "platform": platform.platform(),
                   "processor": platform.processor(), "threads": THREADS, "repeats": args.repeats, "input_seed": SEED,
                   "kmeans": {"k": K, "max_iter": MAX_ITER, "tol": TOL, "seed": 0, "init": "kmeans++"},
                   "input": "Native-size uniform RNG uint8; grayscale input shared between both methods",
                   "time_scope": "Algorithm elapsed after a 32x32 warmup, one measured image per worker",
                   "memory_scope": "Whole-process peak working set on Windows / RSS otherwise, sampled before hashing",
                   "source_sha256": {name: hashlib.sha256((source_dir / f"{name}.py").read_bytes()).hexdigest()
                                     for name in ("kmeans", "otsu")}}
    rows = []
    env = {**os.environ, **THREADS, "PYTHONUTF8": "1"}
    for width, height in args.sizes:
        for method in args.methods:
            if method == "kmeans-rgb" and (width, height) == SIZES[-1] and not args.include_4k_rgb:
                continue
            command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", method, str(width), str(height)]
            if args.source_dir:
                command += ["--source-dir", str(args.source_dir.resolve())]
            runs = []
            for _ in range(args.repeats):
                run = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
                if run.returncode:
                    raise RuntimeError(f"Benchmark worker failed for {method} {width}x{height}:\n{run.stderr}")
                runs.append(json.loads(run.stdout))
            if len({(run["input_sha256"], run["output_sha256"]) for run in runs}) != 1:
                raise RuntimeError(f"Non-repeatable result for {method} {width}x{height}.")
            seconds = statistics.median(run["seconds"] for run in runs)
            peak = statistics.median(run["peak_rss_bytes"] for run in runs)
            rows.append({"method": method, "width": width, "height": height, "median_seconds": seconds,
                         "median_peak_rss_bytes": peak, "runs": runs})
            args.output.write_text(json.dumps({"environment": environment, "measurements": rows}, indent=2), encoding="utf-8")
            print(f"{method:12} {width}x{height}: {seconds * 1000:.2f} ms, {peak / 2**20:.2f} MiB", flush=True)


if __name__ == "__main__":
    main()
