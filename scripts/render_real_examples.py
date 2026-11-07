"""Render best/worst test examples after evaluation; images stay in ignored local outputs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from src.processing.pipeline import KMeansParams, default_foreground, kmeans_outputs, prepare_input, run_kmeans, run_otsu  # noqa: E402
from src.utils.image_io import load_path  # noqa: E402
from scripts.evaluate_real_images import DATA  # noqa: E402


def main():
    report = json.loads((ROOT / "docs/results/cr_real_images.json").read_text())
    rows = sorted((row for row in report["samples"] if row["split"] == "test" and row["brightness_gain"] == 1),
                  key=lambda row: min(row["kmeans_iou"], row["otsu_iou"]))
    selected = rows[:3] + rows[-3:]
    figure, axes = plt.subplots(len(selected), 4, figsize=(13, 3 * len(selected)), dpi=110)
    for axis, row in zip(axes, selected):
        image_path = next(path for path in (DATA / "images").iterdir() if path.stem == row["id"])
        prepared = prepare_input(load_path(image_path), 800)
        km = run_kmeans(prepared, KMeansParams(k=5))
        output = kmeans_outputs(prepared, km, default_foreground(km))
        otsu = run_otsu(prepared, polarity="auto")
        with Image.open(DATA / "images-gt" / (row["id"] + ".png")) as image:
            gt = np.asarray(image.convert("L").resize(prepared.resize.processed_size, Image.Resampling.NEAREST))
        panels = [(prepared.original, row["id"]), (gt, "Reference (gray = ignored)"),
                  (output.mask, f"K-means IoU={row['kmeans_iou']:.3f}"), (otsu.mask, f"Otsu IoU={row['otsu_iou']:.3f}")]
        for ax, (pixels, title) in zip(axis, panels):
            ax.imshow(pixels, cmap="gray", vmin=0, vmax=255)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
    figure.suptitle("Oxford VGG iSeg: three worst and three best held-out cases; fixed parameters", fontsize=12)
    figure.tight_layout()
    destination = ROOT / "outputs/cr-examples/qualitative_test.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination)
    plt.close(figure)
    print(destination)


if __name__ == "__main__":
    main()
