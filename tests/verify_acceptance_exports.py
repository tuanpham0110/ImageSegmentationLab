"""Read browser downloads back and check their contents against the acceptance inputs."""

import csv
import hashlib
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from matplotlib import get_data_path
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.image_io import load_path  # noqa: E402

DATA = ROOT / "data"
OUTPUT = ROOT / "outputs/stage8"


def png(path, mode=None):
    with Image.open(path) as image:
        assert image.format == "PNG", path
        assert mode is None or image.mode == mode, (path, image.mode)
        return np.asarray(image)


def matrix(path):
    expected = np.loadtxt(path, delimiter=",", dtype=np.uint8, ndmin=2)
    np.testing.assert_array_equal(load_path(path).array, expected, err_msg=str(path))
    return expected


def metadata(path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_kmeans(folder, source, k, feature):
    stats = metadata(folder / "kmeans_stats.json")
    assert stats["kich_thuoc_xu_ly_WxH"] == list(source.shape[:2][::-1])
    assert stats["k_yeu_cau"] == stats["k_thuc_te"] == k
    assert stats["tham_so"]["feature_thuc_te"] == feature
    labels = matrix(folder / "kmeans_labels.csv")
    points = source[..., None] if source.ndim == 2 else source
    centers = np.asarray(stats["tam_cum"])
    nearest = np.argmin(np.sum((points[:, :, None, :] - centers) ** 2, axis=-1), axis=2)
    np.testing.assert_array_equal(labels, nearest)
    np.testing.assert_array_equal(np.bincount(labels.ravel()), stats["so_pixel_moi_cum"])
    mapped = np.rint(centers[labels]).astype(np.uint8).reshape(source.shape)
    np.testing.assert_array_equal(png(folder / "kmeans_segmented.png", "L" if source.ndim == 2 else "RGB"), mapped)
    return stats, labels


def validate_otsu(folder, source, threshold):
    stats = metadata(folder / "otsu_stats.json")
    assert stats["otsu_hop_le"] and stats["nguong_otsu"] == stats["nguong_su_dung"] == threshold
    assert stats["kich_thuoc_xu_ly_WxH"] == list(source.shape[::-1])
    expected = (source > threshold).astype(np.uint8) * 255
    np.testing.assert_array_equal(png(folder / "otsu_binary.png", "L"), expected)
    return expected


def main():
    report = metadata(OUTPUT / "browser.json")
    assert set(report["cases"]) == set("ABCDEFGH")
    for key, case in report["cases"].items():
        assert case["status"] == "PASS", (key, case.get("error"))
        for item in case["files"]:
            file = OUTPUT / item["file"]
            payload = file.read_bytes()
            assert len(payload) == item["bytes"] > 0
            item["sha256"] = hashlib.sha256(payload).hexdigest()
            if file.suffix == ".png":
                image = png(file)
                item["shape"] = list(image.shape)

    a, b, c_otsu, c_kmeans, d, e, f = (OUTPUT / name for name in ("a", "b", "c_otsu", "c_kmeans", "d", "e", "f"))
    rgb = png(DATA / "sample_images/multi_shapes_color.png", "RGB")
    stats, labels = validate_kmeans(a, rgb, 5, "rgb")
    assert stats["tham_so"] == dict(k=5, feature="rgb", feature_thuc_te="rgb", max_iter=8, seed=9, tol=0, init="random")
    assert stats["sse"] == 0
    np.testing.assert_array_equal(rgb, png(a / "kmeans_segmented.png", "RGB"))
    for name in ("kmeans_labels.csv", "kmeans_segmented.png"):
        assert (a / name).read_bytes() == (OUTPUT / "a_repeat" / name).read_bytes()
    mask = np.isin(labels, stats["cum_doi_tuong"]).astype(np.uint8) * 255
    np.testing.assert_array_equal(png(a / "kmeans_mask.png", "L"), mask)
    np.testing.assert_array_equal(png(a / "kmeans_cutout.png", "RGBA")[..., 3], mask)
    assert png(a / "kmeans_label_map.png", "RGB").shape == rgb.shape
    with (a / "kmeans_clusters.csv").open(encoding="utf-8", newline="") as file:
        assert len(list(csv.DictReader(file))) == 5

    circle = png(DATA / "sample_images/circle_black_on_white.png", "L")
    truth = png(DATA / "ground_truth/circle_black_on_white_gt.png", "L")
    validate_otsu(b, circle, 127)
    np.testing.assert_array_equal(png(b / "otsu_mask.png", "L"), truth)
    np.testing.assert_array_equal(png(b / "otsu_cutout.png", "RGBA")[..., 3], truth)
    np.testing.assert_array_equal(png(b / "otsu_gray.png", "L"), circle)
    np.testing.assert_array_equal(matrix(b / "otsu_binary_matrix.csv"), truth // 255)
    with (b / "otsu_histogram.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 256
    np.testing.assert_array_equal([int(row["so_pixel"]) for row in rows], np.bincount(circle.ravel(), minlength=256))

    small = matrix(DATA / "sample_csv/textbook_otsu_5x6.csv")
    assert small.shape == (5, 6)
    binary = validate_otsu(c_otsu, small, 1)
    np.testing.assert_array_equal(matrix(c_otsu / "otsu_binary_matrix.csv"), binary // 255)
    validate_kmeans(c_kmeans, small, 2, "gray")

    rectangle = png(DATA / "sample_images/bright_rect_on_dark.png", "L")
    validate_otsu(d, rectangle, 124)
    assert metadata(d / "otsu_stats.json")["anh"] == "bright_rect_on_dark.png"

    gray = png(DATA / "sample_images/gray_levels_objects.png", "L")
    multi = metadata(e / "multiotsu_stats.json")
    assert multi["hop_le"] and multi["hau_xu_ly_mask"] == {"phep_hinh_thai": "open", "kich_thuoc": 5}
    t1, t2 = multi["nguong"]
    assert 35 <= t1 < 90 and 150 <= t2 < 215
    multi_labels = matrix(e / "multiotsu_labels.csv")
    np.testing.assert_array_equal(multi_labels, np.digitize(gray, (t1, t2), right=True))
    assert png(e / "multiotsu_label_map.png", "RGB").shape == (*gray.shape, 3)
    np.testing.assert_array_equal(png(e / "multiotsu_cutout.png", "RGBA")[..., 3], png(e / "multiotsu_mask.png", "L"))

    with ZipFile(f / "comparison_results.zip") as archive:
        expected_names = {file.name for folder in (a, b) for file in folder.iterdir()}
        assert archive.testzip() is None and set(archive.namelist()) == expected_names
        for name in ("kmeans_mask.png", "otsu_mask.png", "kmeans_cutout.png", "otsu_cutout.png"):
            assert archive.read(name) == (f / name).read_bytes()
        for name in ("kmeans_stats.json", "otsu_stats.json"):
            assert json.loads(archive.read(name))["anh"] == "circle_black_on_white.png"
    for name in ("kmeans_mask.png", "otsu_mask.png"):
        np.testing.assert_array_equal(png(f / name, "L"), truth)

    with Image.open(Path(get_data_path()) / "sample_data/grace_hopper.jpg") as photo:
        full = np.asarray(ImageOps.exif_transpose(photo).convert("RGB"))
    lab_stats = metadata(OUTPUT / "h_kmeans/kmeans_stats.json")
    assert lab_stats["anh"] == "grace_hopper.jpg" and lab_stats["tham_so"]["feature_thuc_te"] == "lab"
    assert lab_stats["chuyen_xam"] == "average" and lab_stats["tien_xu_ly"]["loc_trung_vi"] == "3x3"
    assert lab_stats["hau_xu_ly_mask"]["phep_hinh_thai"] == "erode"
    assert max(lab_stats["kich_thuoc_xu_ly_WxH"]) == 128
    for prefix, original in (("kmeans", "segmented"), ("otsu", "binary"), ("multiotsu", "label_map")):
        folder = OUTPUT / ("h_multi" if prefix == "multiotsu" else f"h_{prefix}")
        mask = png(folder / f"{prefix}_mask_original.png", "L")
        assert mask.shape == full.shape[:2] and set(np.unique(mask)) <= {0, 255}
        assert png(folder / f"{prefix}_{original}_original.png").shape[:2] == full.shape[:2]
        cutout = png(folder / f"{prefix}_cutout_original.png", "RGBA")
        np.testing.assert_array_equal(cutout[..., 3], mask)
        np.testing.assert_array_equal(cutout[mask > 0, :3], full[mask > 0])

    report["export_checks"] = {"status": "PASS", "files": sum(len(case["files"]) for case in report["cases"].values()),
                               "circle_iou_kmeans": 1, "circle_iou_otsu": 1, "repeated_kmeans_files_identical": True}
    (OUTPUT / "verified.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"PASS reopened {report['export_checks']['files']} files and verified PNG, CSV, JSON and ZIP contents")


if __name__ == "__main__":
    main()
