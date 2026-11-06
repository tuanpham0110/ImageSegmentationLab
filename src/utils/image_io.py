"""Đọc / ghi ảnh và ma trận CSV, kiểm tra dữ liệu đầu vào với thông báo lỗi tiếng Việt."""

from __future__ import annotations

import csv
import io
import json
import re
import struct
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
MAX_FILE_BYTES = 30 * 1024 * 1024  # 30 MB
MAX_DECODE_PIXELS = 40_000_000  # chặn ảnh quá lớn trước khi giải mã (chống treo máy)
MAX_CSV_CELLS = 4_000_000  # tối đa 2000 x 2000 phần tử


class InputError(ValueError):
    """Lỗi dữ liệu đầu vào — thông điệp hiển thị trực tiếp cho người dùng."""


@dataclass
class LoadedImage:
    array: np.ndarray  # (H, W) uint8 hoặc (H, W, 3) uint8
    source_name: str
    file_format: str  # PNG / JPEG / BMP / CSV
    file_bytes: int
    original_mode: str  # chế độ màu của Pillow, hoặc 'CSV'
    notes: list[str] = field(default_factory=list)

    @property
    def is_color(self) -> bool:
        return self.array.ndim == 3

    @property
    def channels(self) -> int:
        return 1 if self.array.ndim == 2 else self.array.shape[2]

    @property
    def height(self) -> int:
        return self.array.shape[0]

    @property
    def width(self) -> int:
        return self.array.shape[1]


def _composite_alpha(img: Image.Image, notes: list[str]) -> Image.Image:
    """Ghép ảnh có kênh alpha lên nền trắng (pixel trong suốt trở thành trắng)."""
    rgba = img.convert("RGBA")
    alpha = np.asarray(rgba)[..., 3]
    if alpha.min() < 255:
        notes.append("Ảnh có kênh alpha (trong suốt): các vùng trong suốt được ghép lên nền trắng.")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, rgba).convert("RGB")


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_COLOR_TYPES = {0: "xám", 2: "RGB", 3: "bảng màu", 4: "xám + alpha", 6: "RGBA"}


def _png_bit_depth(data: bytes) -> tuple[int, int] | None:
    """Đọc (bit depth, color type) từ chunk IHDR của PNG; None nếu không phải PNG hợp lệ.

    Pillow giải mã PNG màu 16 bit (RGB/RGBA/LA) thành chế độ 8 bit nên img.mode
    không cho biết độ sâu gốc — phải đọc trực tiếp header.
    """
    if len(data) < 33 or not data.startswith(_PNG_SIGNATURE) or data[12:16] != b"IHDR":
        return None
    bit_depth, color_type = struct.unpack(">BB", data[24:26])
    return bit_depth, color_type


def load_image_bytes(data: bytes, name: str = "image") -> LoadedImage:
    """Đọc ảnh PNG/JPG/BMP từ bytes thành mảng uint8.

    - Áp dụng hướng xoay EXIF.
    - Ảnh có alpha được ghép lên nền trắng.
    - Ảnh màu có R = G = B ở mọi pixel được coi là ảnh xám (H, W).
    """
    return _load_image_bytes(data, name, normalize_high_depth=True)


def _load_image_bytes(data: bytes, name: str, *, normalize_high_depth: bool) -> LoadedImage:
    """Dùng chung kiểm tra/giải mã; mask giữ thang sáng tuyệt đối thay vì giãn tương phản."""
    ext = Path(name).suffix.lower()
    if ext and ext not in SUPPORTED_IMAGE_EXTS:
        raise InputError(f"Định dạng '{ext}' không được hỗ trợ. Chỉ nhận PNG, JPG, JPEG, BMP.")
    if len(data) == 0:
        raise InputError("Tệp rỗng.")
    if len(data) > MAX_FILE_BYTES:
        raise InputError(f"Tệp quá lớn ({len(data) / 1e6:.1f} MB). Giới hạn {MAX_FILE_BYTES // (1024 * 1024)} MB.")
    notes: list[str] = []
    try:
        with Image.open(io.BytesIO(data)) as img:
            fmt = img.format or ext.lstrip(".").upper()
            if fmt not in ("PNG", "JPEG", "BMP", "MPO"):
                raise InputError(f"Nội dung tệp là định dạng {fmt}, không phải PNG/JPG/BMP.")
            w, h = img.size
            if w * h > MAX_DECODE_PIXELS:
                raise InputError(
                    f"Ảnh quá lớn ({w} x {h} = {w * h / 1e6:.1f} triệu pixel). Giới hạn {MAX_DECODE_PIXELS / 1e6:.0f} triệu pixel."
                )
            png_depth = _png_bit_depth(data) if fmt == "PNG" else None
            # PNG màu 16 bit: Pillow chỉ giữ byte cao của mỗi kênh, còn khóa tRNS vẫn ở thang
            # 0–65535 nên không thể so khớp đúng pixel trong suốt -> từ chối thay vì đọc sai âm thầm.
            if png_depth is not None and png_depth[0] == 16 and png_depth[1] in (2, 4, 6):
                if "transparency" in img.info:
                    raise InputError(
                        f"PNG {_PNG_COLOR_TYPES[png_depth[1]]} 16 bit có khóa trong suốt (tRNS) chưa được hỗ trợ. "
                        "Hãy lưu lại ảnh ở 8 bit/kênh hoặc dùng kênh alpha thay cho tRNS."
                    )
                notes.append(
                    f"PNG {_PNG_COLOR_TYPES[png_depth[1]]} 16 bit/kênh được giảm về 8 bit/kênh (giữ byte cao, "
                    "tức ⌊v / 256⌋). Không giãn tương phản như ảnh xám 16 bit; "
                    "các mức khác nhau dưới 1/256 thang sáng sẽ bị gộp."
                )
            img.load()
            arr, mode = _decoded_array(img, notes, normalize_high_depth)
    except InputError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        raise InputError(f"Không đọc được ảnh '{name}': tệp hỏng hoặc không phải ảnh hợp lệ ({exc}).") from exc

    return LoadedImage(
        array=np.ascontiguousarray(arr, dtype=np.uint8),
        source_name=name,
        file_format="JPEG" if fmt == "MPO" else fmt,
        file_bytes=len(data),
        original_mode=mode,
        notes=notes,
    )


def _decoded_array(img: Image.Image, notes: list[str], normalize_high_depth: bool) -> tuple[np.ndarray, str]:
    orientation = img.getexif().get(0x0112, 1)  # thẻ EXIF Orientation
    if orientation not in (1, None):
        img = ImageOps.exif_transpose(img)
        notes.append(f"Ảnh đã được xoay/lật theo thông tin hướng chụp EXIF (Orientation = {orientation}).")
    mode = img.mode

    # So khóa tRNS trong thang gốc trước khi chuyển 16 bit về 8 bit.
    # convert("RGBA") trực tiếp từ I;16 sẽ clip các pixel >255 thành trắng.
    if mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        raw = np.asarray(img, dtype=np.float64)
        if not np.all(np.isfinite(raw)):
            raise InputError("Ảnh chứa pixel NaN hoặc vô cực.")
        if normalize_high_depth:
            lo, hi = float(raw.min()), float(raw.max())
            arr = np.zeros(raw.shape, np.uint8) if hi == lo else np.round((raw - lo) / (hi - lo) * 255).astype(np.uint8)
            notes.append(f"Ảnh {mode} (độ sâu > 8 bit) được chuẩn hoá tuyến tính về [0, 255].")
        else:
            arr = np.round(np.clip(raw / np.iinfo(np.uint16).max, 0, 1) * 255).astype(np.uint8)
            notes.append("Mask 16 bit được đổi từ thang [0, 65535] về [0, 255], giữ độ sáng tuyệt đối.")
        transparent = img.info.get("transparency")
        if transparent is not None:
            transparent_pixels = raw == transparent
            if np.any(transparent_pixels):
                arr[transparent_pixels] = 255
                notes.append("Ảnh có kênh alpha (trong suốt): các vùng trong suốt được ghép lên nền trắng.")
    elif mode in ("LA", "RGBA", "PA") or "transparency" in img.info:
        arr = np.asarray(_composite_alpha(img, notes))
    elif mode in ("1", "L"):
        arr = np.asarray(img.convert("L"))
    else:
        arr = np.asarray(img.convert("RGB"))
        if mode not in ("RGB",):
            notes.append(f"Ảnh chế độ {mode} được chuyển sang RGB.")

    if arr.ndim == 3 and np.array_equal(arr[..., 0], arr[..., 1]) and np.array_equal(arr[..., 1], arr[..., 2]):
        arr = arr[..., 0].copy()
        notes.append("Ảnh có R = G = B tại mọi pixel nên được xử lý như ảnh xám 1 kênh.")

    return arr, mode


def _csv_rows(lines: list[str], line_numbers: list[int], delimiter: str | None):
    """Sinh (số dòng vật lý trong tệp, hàng ô) để báo lỗi đúng vị trí dù tệp có dòng trống.

    Chuyển lỗi cú pháp/ô quá dài của csv.reader thành lỗi đầu vào có kiểm soát.
    """
    if delimiter is None:
        yield from zip(line_numbers, (line.split() for line in lines))
        return
    reader = csv.reader(lines, delimiter=delimiter, strict=True)
    consumed = 0  # số dòng (đã lọc) csv.reader đã đọc trước hàng hiện tại
    try:
        for row in reader:
            # Một hàng có thể trải nhiều dòng nếu ô chứa xuống dòng trong dấu nháy: lấy dòng bắt đầu.
            yield line_numbers[consumed], row
            consumed = reader.line_num
    except csv.Error as exc:
        where = line_numbers[min(consumed, len(line_numbers) - 1)]
        raise InputError(f"CSV không hợp lệ tại dòng {where} (cú pháp dấu nháy hoặc ô dữ liệu quá dài): {exc}.") from exc


def parse_csv_matrix(text: str) -> np.ndarray:
    """Đọc ma trận mức xám từ văn bản CSV.

    Định dạng: mỗi dòng là một hàng ảnh, các giá trị nguyên 0..255 phân cách bởi
    dấu phẩy (hoặc chấm phẩy / tab / khoảng trắng), không có dòng tiêu đề,
    mọi dòng cùng số cột. Dòng trống bị bỏ qua; số dòng trong thông báo lỗi là
    số dòng thực trong tệp (tính cả dòng trống).
    """
    text = text.lstrip("﻿")
    # Giữ ký tự xuống dòng để csv.reader không ghép "1\n2" trong dấu nháy thành 12.
    numbered = [(i, ln) for i, ln in enumerate(text.splitlines(keepends=True), start=1) if ln.strip()]
    if not numbered:
        raise InputError("CSV rỗng: không có dòng dữ liệu nào.")

    line_numbers = [i for i, _ in numbered]
    lines = [ln for _, ln in numbered]
    sample = lines[0]
    if "," in sample:
        delim = ","
    elif ";" in sample:
        delim = ";"
    elif "\t" in sample:
        delim = "\t"
    else:
        delim = None  # khoảng trắng

    rows: list[list[int]] = []
    n_cols = None
    first_line = line_numbers[0]
    number = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)$")
    for r_idx, raw_row in _csv_rows(lines, line_numbers, delim):
        cells = [c.strip() for c in raw_row]
        if delim and cells and cells[-1] == "":  # cho phép dấu phân cách thừa cuối dòng
            cells = cells[:-1]
        if n_cols is None:
            n_cols = len(cells)
        elif len(cells) != n_cols:
            raise InputError(f"Dòng {r_idx} có {len(cells)} cột, khác với {n_cols} cột của dòng {first_line}.")
        if (len(rows) + 1) * n_cols > MAX_CSV_CELLS:
            raise InputError(f"CSV quá lớn (vượt {MAX_CSV_CELLS:,} phần tử).")
        row: list[int] = []
        for c_idx, cell in enumerate(cells, start=1):
            if cell == "":
                raise InputError(f"Ô trống tại dòng {r_idx}, cột {c_idx}.")
            if not number.match(cell):
                raise InputError(f"Giá trị không phải số tại dòng {r_idx}, cột {c_idx}: '{cell[:20]}'.")
            shown = cell if len(cell) <= 40 else cell[:37] + "..."
            try:
                value = Decimal(cell)
            except InvalidOperation as exc:
                raise InputError(f"Giá trị không phải số tại dòng {r_idx}, cột {c_idx}: '{shown}'.") from exc
            if not 0 <= value <= 255:
                raise InputError(f"Giá trị {shown} tại dòng {r_idx}, cột {c_idx} nằm ngoài khoảng [0, 255].")
            if value != value.to_integral_value():
                raise InputError(f"Giá trị không phải số nguyên tại dòng {r_idx}, cột {c_idx}: {shown}.")
            row.append(int(value))
        rows.append(row)
    if not n_cols:
        raise InputError("CSV không có cột dữ liệu.")
    return np.array(rows, dtype=np.uint8)


def load_csv_bytes(data: bytes, name: str = "matrix.csv") -> LoadedImage:
    if len(data) == 0:
        raise InputError("Tệp CSV rỗng.")
    if len(data) > MAX_FILE_BYTES:
        raise InputError("Tệp CSV quá lớn.")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise InputError("Tệp CSV không phải văn bản UTF-8 hợp lệ.") from exc
    arr = parse_csv_matrix(text)
    return LoadedImage(array=arr, source_name=name, file_format="CSV", file_bytes=len(data), original_mode="CSV")


def load_any(data: bytes, name: str) -> LoadedImage:
    """Đọc ảnh hoặc CSV dựa theo phần mở rộng tên tệp."""
    if Path(name).suffix.lower() == ".csv":
        return load_csv_bytes(data, name)
    return load_image_bytes(data, name)


def load_path(path: str | Path) -> LoadedImage:
    p = Path(path)
    return load_any(p.read_bytes(), p.name)


def load_mask_bytes(data: bytes, name: str = "mask.png") -> np.ndarray:
    """Đọc mask: pixel 8 bit >127 (16 bit >=32768) là đối tượng; trả uint8 0/255.

    Không giãn tương phản mask: ảnh trắng 16 bit phải giữ nguyên toàn đối tượng.
    """
    loaded = (load_csv_bytes(data, name) if Path(name).suffix.lower() == ".csv"
              else _load_image_bytes(data, name, normalize_high_depth=False))
    arr = loaded.array if loaded.array.ndim == 2 else loaded.array.mean(axis=2)
    return np.where(arr > 127, 255, 0).astype(np.uint8)


# ----------------------------- Ghi kết quả -----------------------------

def to_png_bytes(array: np.ndarray) -> bytes:
    """Mã hoá mảng uint8 (H,W), (H,W,3) hoặc (H,W,4) thành PNG."""
    arr = np.asarray(array)
    if (arr.ndim not in (2, 3) or arr.size == 0
            or (arr.ndim == 3 and arr.shape[2] not in (3, 4))):
        raise ValueError("PNG yêu cầu ảnh không rỗng dạng (H,W), (H,W,3) hoặc (H,W,4).")
    if arr.dtype != np.uint8:
        if (arr.dtype.kind not in "buif" or not np.all(np.isfinite(arr))
                or np.any(arr < 0) or np.any(arr > 255) or np.any(arr != np.round(arr))):
            raise ValueError("Pixel PNG phải là số nguyên trong [0, 255].")
        arr = arr.astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def matrix_to_csv_bytes(matrix: np.ndarray) -> bytes:
    """Ghi ma trận 2 chiều số nguyên thành CSV (không tiêu đề), đọc lại được bằng parse_csv_matrix."""
    m = np.asarray(matrix)
    if m.ndim != 2 or m.size == 0:
        raise ValueError("Chỉ ghi được ma trận 2 chiều không rỗng ra CSV.")
    if m.dtype.kind not in "buif" or not np.all(np.isfinite(m)) or np.any(m != np.floor(m)):
        raise ValueError("Ma trận CSV phải chứa số nguyên hữu hạn.")
    buf = io.StringIO()
    np.savetxt(buf, m, fmt="%d", delimiter=",")
    return buf.getvalue().encode("utf-8")


def table_to_csv_bytes(rows: list[dict]) -> bytes:
    if not rows:
        return b""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8-sig")  # BOM để Excel hiển thị đúng tiếng Việt


def to_json_bytes(obj: dict) -> bytes:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=_json_default, allow_nan=False).encode("utf-8")


def _json_default(o):
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Không chuyển được {type(o)} sang JSON")
