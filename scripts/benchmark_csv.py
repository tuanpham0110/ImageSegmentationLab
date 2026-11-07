"""CSV at the 4,000,000 element limit; record baseline without a machine-specific time threshold."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.worker:
        import numpy as np
        from scripts.benchmark_kmeans import peak_memory
        from src.utils.image_io import parse_csv_matrix
        expected = np.random.default_rng(0).integers(0, 256, (2000, 2000), dtype=np.uint8)
        text = "\n".join(",".join(map(str, row)) for row in expected)
        started = time.perf_counter()
        actual = parse_csv_matrix(text)
        elapsed = time.perf_counter() - started
        memory = peak_memory()
        np.testing.assert_array_equal(actual, expected)
        print(json.dumps({"seconds": elapsed, "elements": actual.size, "input_bytes": len(text), **memory}))
        return
    if args.repeats < 1:
        parser.error("repeats must be positive")
    rows = []
    for _ in range(args.repeats):
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker"], cwd=ROOT,
                                capture_output=True, text=True, check=True, timeout=120,
                                env={**os.environ, "PYTHONUTF8": "1"})
        rows.append(json.loads(result.stdout))
    output = ROOT / "docs/results/cr_csv_benchmark.json"
    output.write_text(json.dumps({"median_seconds": statistics.median(r["seconds"] for r in rows),
                                  "memory_scope": "Whole worker, including CSV generation", "runs": rows}, indent=2), encoding="utf-8")
    print(output.read_text(), flush=True)


if __name__ == "__main__":
    main()
