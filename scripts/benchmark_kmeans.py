"""Benchmark isolated processes: time, peak RSS, exact output hashes; no fixed time assertions."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def peak_memory() -> dict:
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        query = ctypes.WinDLL("psapi").GetProcessMemoryInfo
        query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not query(wintypes.HANDLE(-1), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError()
        return {"peak_rss_bytes": counters.PeakWorkingSetSize, "peak_private_bytes": counters.PeakPagefileUsage}
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"peak_rss_bytes": int(peak if sys.platform == "darwin" else peak * 1024)}


def worker(args):
    import numpy as np
    from PIL import Image

    if args.module:
        spec = importlib.util.spec_from_file_location("baseline_kmeans", args.module)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    else:
        from src.algorithms import kmeans as module
    if args.image:
        with Image.open(args.image) as photo:
            image = np.asarray(photo.convert("RGB").resize((args.side, args.side), Image.Resampling.BILINEAR))
        kind = "photograph_resized_to_square_for_stress_test"
    else:
        image = np.random.default_rng(args.seed).integers(0, 256, (args.side, args.side, 3), dtype=np.uint8)
        kind = "synthetic_high_color_stress_not_a_photograph"
    started = time.perf_counter()
    result = module.segment_image_kmeans(image, args.k, max_iter=args.max_iter, tol=0.0, seed=args.seed)
    elapsed = time.perf_counter() - started
    memory = peak_memory()  # Before hashing/counting adds allocations.
    digest = hashlib.sha256()
    for array in (result.labels, result.centers, result.core.counts, result.segmented):
        digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    digest.update(float(result.core.sse).hex().encode())
    digest.update(str((result.core.n_iter, result.core.converged)).encode())
    unique = np.unique(image.reshape(-1, 3), axis=0).shape[0]
    print(json.dumps({"seconds": elapsed, **memory, "output_sha256": digest.hexdigest(), "unique_colors": unique,
                      "iterations": result.core.n_iter, "converged": result.core.converged, "sse": result.core.sse,
                      "input_kind": kind}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--sizes", type=int, nargs="+", default=[800, 1600, 2048])
    parser.add_argument("--ks", type=int, nargs="+", default=[4, 16])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/results/cr_benchmark.json")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--module", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--side", type=int, default=800, help=argparse.SUPPRESS)
    parser.add_argument("--k", type=int, default=4, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 1 or args.max_iter < 1 or any(n < 1 for n in args.sizes + args.ks):
        parser.error("sizes, ks, repeats and max-iter must be positive")
    if args.worker:
        worker(args)
        return
    variants = {"current": None}
    if args.baseline:
        variants = {"baseline": args.baseline.resolve(), **variants}
    environment = {"python": platform.python_version(), "platform": platform.platform(), "processor": platform.processor(),
                   "cpu_count": os.cpu_count(), "seed": args.seed, "max_iter": args.max_iter,
                   "repeats": args.repeats, "threads": 1, "memory_scope": "Peak whole worker process, includes imports and input image",
                   "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest() if args.baseline else None}
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONUTF8": "1"}
    for side in args.sizes:
        for k in args.ks:
            hashes = {}
            for name, module in variants.items():
                runs = []
                for repeat in range(args.repeats):
                    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--side", str(side), "--k", str(k),
                               "--max-iter", str(args.max_iter), "--seed", str(args.seed)]
                    if module:
                        command += ["--module", str(module)]
                    if args.image:
                        command += ["--image", str(args.image.resolve())]
                    completed = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=900, check=True)
                    runs.append(json.loads(completed.stdout))
                    print(f"{name} {side}x{side} K={k} repeat={repeat+1}: {runs[-1]['seconds']:.3f}s", flush=True)
                hashes[name] = runs[0]["output_sha256"]
                assert len({run["output_sha256"] for run in runs}) == 1, "Non-repeatable output"
                rows.append({"variant": name, "side": side, "k": k, "median_seconds": statistics.median(r["seconds"] for r in runs),
                             "max_peak_rss_bytes": max(r["peak_rss_bytes"] for r in runs), "runs": runs})
                args.output.write_text(json.dumps({"environment": environment, "measurements": rows}, indent=2), encoding="utf-8")
            if "baseline" in hashes:
                assert hashes["baseline"] == hashes["current"], f"Changed exact output: side={side}, K={k}"


if __name__ == "__main__":
    main()
