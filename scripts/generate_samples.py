"""Sinh bộ dữ liệu demo offline (ảnh tổng hợp + ground truth + CSV) bằng NumPy/Pillow.

Chạy:  python scripts/generate_samples.py
Kết quả ghi vào data/. Hạt giống cố định nên chạy lại cho kết quả giống hệt.
Tất cả ảnh đều là ẢNH TỔNG HỢP (không phải ảnh chụp thật).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

if hasattr(sys.stdout, "reconfigure"):  # console Windows mặc định cp1252 không in được tiếng Việt
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "data" / "sample_images"
GT_DIR = ROOT / "data" / "ground_truth"
CSV_DIR = ROOT / "data" / "sample_csv"
MANIFEST = ROOT / "data" / "samples.json"

RNG = np.random.default_rng(2026)


def _canvas(w: int, h: int, color, mode: str = "L") -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new(mode, (w, h), color)
    return img, ImageDraw.Draw(img)


def _mask_canvas(w: int, h: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    return _canvas(w, h, 0, "L")


def _save(img: Image.Image | np.ndarray, path: Path) -> None:
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def _save_csv(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, arr.astype(np.int64), fmt="%d", delimiter=",")


def circle_black_on_white():
    img, d = _canvas(256, 256, 255)
    m, dm = _mask_canvas(256, 256)
    box = (68, 58, 188, 178)
    d.ellipse(box, fill=0)
    dm.ellipse(box, fill=255)
    return img, m


def bright_rect_on_dark():
    img, d = _canvas(256, 192, 30)
    m, dm = _mask_canvas(256, 192)
    box = (60, 50, 200, 140)
    d.rectangle(box, fill=220)
    dm.rectangle(box, fill=255)
    return img, m


def multi_shapes_color():
    img, d = _canvas(320, 240, (235, 235, 230), "RGB")
    m, dm = _mask_canvas(320, 240)
    shapes = [
        ("ellipse", (30, 30, 120, 120), (220, 40, 40)),
        ("rectangle", (160, 40, 290, 110), (40, 90, 210)),
        ("polygon", [(70, 210), (130, 140), (190, 210)], (30, 160, 70)),
        ("ellipse", (215, 140, 295, 220), (245, 190, 20)),
    ]
    for kind, geo, color in shapes:
        getattr(d, kind)(geo, fill=color)
        getattr(dm, kind)(geo, fill=255)
    return img, m


def gray_levels_objects():
    img, d = _canvas(300, 200, 35)
    m, dm = _mask_canvas(300, 200)
    for box, g in [((25, 50, 95, 150), 90), ((115, 50, 185, 150), 150), ((205, 50, 275, 150), 215)]:
        d.rectangle(box, fill=g)
        dm.rectangle(box, fill=255)
    return img, m


def noisy_circle():
    base, gt = circle_black_on_white()
    arr = np.asarray(base, dtype=np.float64)
    arr += RNG.normal(0, 45, arr.shape)  # nhiễu Gauss sigma = 45
    sp = RNG.random(arr.shape)
    arr[sp < 0.03] = 0  # nhiễu muối tiêu 3% + 3%
    arr[sp > 0.97] = 255
    return np.clip(np.round(arr), 0, 255).astype(np.uint8), gt


def low_contrast():
    img, d = _canvas(256, 256, 130)
    m, dm = _mask_canvas(256, 256)
    box = (70, 70, 186, 186)
    d.ellipse(box, fill=150)
    dm.ellipse(box, fill=255)
    arr = np.asarray(img, dtype=np.float64) + RNG.normal(0, 4, (256, 256))
    return np.clip(np.round(arr), 0, 255).astype(np.uint8), m


def uniform_image():
    return np.full((128, 128), 128, dtype=np.uint8), None


def rgb_regions():
    """Cảnh tổng hợp: bầu trời gradient, mặt trời, núi, cỏ, hồ nước."""
    w, h = 360, 240
    y = np.linspace(0, 1, h)[:, None]
    sky = np.zeros((h, w, 3))
    sky[..., 0] = 120 + 60 * y
    sky[..., 1] = 170 + 40 * y
    sky[..., 2] = 235 - 10 * y
    img = Image.fromarray(np.round(sky).astype(np.uint8))
    d = ImageDraw.Draw(img)
    d.ellipse((270, 25, 330, 85), fill=(250, 210, 60))
    d.polygon([(0, 170), (90, 70), (180, 170)], fill=(110, 100, 120))
    d.polygon([(120, 170), (230, 90), (360, 170)], fill=(95, 85, 105))
    d.rectangle((0, 165, 360, 240), fill=(70, 150, 60))
    d.ellipse((150, 190, 320, 232), fill=(40, 90, 170))
    arr = np.asarray(img, dtype=np.float64) + RNG.normal(0, 3, (h, w, 3))
    return np.clip(np.round(arr), 0, 255).astype(np.uint8), None


def tiny_step_by_step():
    """Ảnh 10x10 hai vùng có nhiễu nhẹ — đủ nhỏ để theo dõi từng vòng lặp."""
    arr = np.full((10, 10), 60, dtype=np.int64)
    gt = np.zeros((10, 10), dtype=np.uint8)
    arr[2:8, 3:9] = 190
    gt[2:8, 3:9] = 255
    arr += RNG.integers(-12, 13, (10, 10))
    return np.clip(arr, 0, 255).astype(np.uint8), gt


def csv_rectangle_32():
    arr = np.full((32, 32), 40, dtype=np.int64)
    gt = np.zeros((32, 32), dtype=np.uint8)
    arr[8:24, 6:26] = 200
    gt[8:24, 6:26] = 255
    arr += RNG.integers(-15, 16, (32, 32))
    return np.clip(arr, 0, 255).astype(np.uint8), gt


# Ma trận ví dụ trong giáo trình, mục 3.1.6 (trang 32): ngưỡng tự động θ = 1.
TEXTBOOK_5X6 = np.array(
    [
        [0, 1, 2, 3, 4, 5],
        [0, 0, 1, 2, 3, 4],
        [0, 0, 0, 1, 2, 3],
        [0, 0, 0, 0, 1, 2],
        [0, 0, 0, 0, 0, 1],
    ],
    dtype=np.uint8,
)


def main() -> int:
    samples = []

    def add(sid, title, kind, fn_or_arr, purpose, expected, k=2, feature="gray", gt_available=True):
        result = fn_or_arr() if callable(fn_or_arr) else (fn_or_arr, None)
        img, gt = result
        if kind == "image":
            rel = f"sample_images/{sid}.png"
            _save(img, ROOT / "data" / rel)
        else:
            rel = f"sample_csv/{sid}.csv"
            _save_csv(np.asarray(img), ROOT / "data" / rel)
        gt_rel = None
        if gt is not None and gt_available:
            gt_rel = f"ground_truth/{sid}_gt.png"
            _save(gt, ROOT / "data" / gt_rel)
        samples.append({
            "id": sid, "title": title, "kind": kind, "file": rel, "ground_truth": gt_rel,
            "purpose": purpose, "expected": expected,
            "suggested": {"k": k, "feature": feature},
        })

    add("circle_black_on_white", "Hình tròn đen trên nền trắng", "image", circle_black_on_white,
        "Trường hợp lý tưởng: 2 mức xám tách biệt.",
        "Otsu và K-means (K=2) tách đúng hoàn toàn hình tròn (IoU = 1 khi đối tượng là lớp tối).")
    add("bright_rect_on_dark", "Hình chữ nhật sáng trên nền tối", "image", bright_rect_on_dark,
        "Đối tượng sáng trên nền tối — kiểm tra quy ước foreground của Otsu.",
        "Ngưỡng nằm giữa 30 và 220; đối tượng là lớp sáng (> t).")
    add("multi_shapes_color", "Nhiều hình màu khác nhau (RGB)", "image", multi_shapes_color,
        "Nhiều đối tượng màu khác nhau trên nền sáng.",
        "K-means RGB với K=5 tách mỗi màu thành một cụm; Otsu trên ảnh xám chỉ có 2 lớp nên có thể gộp sai hình vàng vào nền.",
        k=5, feature="rgb")
    add("gray_levels_objects", "Ảnh xám nhiều mức sáng", "image", gray_levels_objects,
        "Ba đối tượng có mức xám 90/150/215 trên nền 35.",
        "K-means K=4 tách 4 mức; Otsu chỉ chia 2 lớp nên có thể xếp đối tượng tối nhất vào nền.", k=4)
    add("noisy_circle", "Hình tròn có nhiễu Gauss + muối tiêu", "image", noisy_circle,
        "Đánh giá độ bền với nhiễu (thuật toán chỉ dùng giá trị pixel, không dùng lân cận).",
        "Hình dạng tổng thể đúng nhưng mask có nhiều điểm lốm đốm; IoU < 1.")
    add("low_contrast", "Ảnh độ tương phản thấp", "image", low_contrast,
        "Đối tượng 150 trên nền 130, nhiễu sigma = 4.",
        "Hai đỉnh histogram gần nhau; kết quả vẫn tách được nhưng nhạy với nhiễu.")
    add("uniform_gray", "Ảnh chỉ có một màu (ngoại lệ)", "image", uniform_image,
        "Kiểm tra xử lý ngoại lệ: không có gì để tách.",
        "Otsu báo không hợp lệ (một mức xám); K-means giảm K về 1 và cảnh báo.", gt_available=False)
    add("rgb_regions", "Cảnh RGB nhiều vùng màu (tổng hợp)", "image", rgb_regions,
        "Ảnh màu nhiều vùng: bầu trời gradient, mặt trời, núi, cỏ, hồ.",
        "K-means RGB với K=5–6 tách được các vùng màu; không có ground truth nên không tính IoU.",
        k=6, feature="rgb", gt_available=False)
    add("tiny_10x10", "Ảnh nhỏ 10x10 (minh hoạ từng bước)", "image", tiny_step_by_step,
        "Đủ nhỏ để xem tâm cụm ở từng vòng lặp và histogram từng mức.",
        "K-means K=2 hội tụ sau vài vòng; Otsu tách đúng vùng sáng.")
    add("csv_rectangle_32x32", "Ma trận CSV 32x32", "csv", csv_rectangle_32,
        "Kiểm tra đầu vào CSV: hình chữ nhật 200 trên nền 40, nhiễu đều ±15.",
        "Cả hai thuật toán tách đúng hình chữ nhật (IoU = 1).")
    add("textbook_otsu_5x6", "Ma trận ví dụ giáo trình mục 3.1.6", "csv", TEXTBOOK_5X6,
        "Đối chiếu với ví dụ tính tay trong giáo trình (trang 32).",
        "Ngưỡng Otsu = 1, trùng với θ = 1 của giáo trình.", gt_available=False)

    # CSV lỗi để demo thông báo lỗi
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    (CSV_DIR / "invalid_text.csv").write_text("10,20,30\n40,abc,60\n70,80,90\n", encoding="utf-8")
    (CSV_DIR / "invalid_range.csv").write_text("10,20,30\n40,300,60\n70,80,-5\n", encoding="utf-8")
    (CSV_DIR / "invalid_ragged.csv").write_text("10,20,30\n40,50\n70,80,90\n", encoding="utf-8")

    MANIFEST.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã sinh {len(samples)} mẫu vào {ROOT / 'data'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
