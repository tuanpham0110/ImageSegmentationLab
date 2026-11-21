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
MAX_CSV_CELLS = 2048**2  # nhận lại ma trận ở kích thước xử lý tối đa của giao diện


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


MASK_TRANSPARENCY_ERROR = (
    "Ground-truth mask có vùng trong suốt (kênh alpha hoặc khóa tRNS). Mask phải là ảnh đục: "
    "trắng = đối tượng, đen = nền. Ứng dụng không tự đoán vùng trong suốt là nền hay đối tượng — "
    "hãy lưu lại mask không có độ trong suốt (ví dụ ghép lên nền đen)."
)


def _rgb_is_gray(pixels: np.ndarray) -> bool:
    return np.array_equal(pixels[..., 0], pixels[..., 1]) and np.array_equal(pixels[..., 1], pixels[..., 2])


def _binary_reference_mask(pixels: np.ndarray, white: int = 255) -> np.ndarray:
    """Xác nhận một cách mã hóa nhị phân và đưa nhãn về uint8 0/255."""
    message = f"Mask tham chiếu phải là mask nhị phân 0/1 hoặc 0/{white}; 0 là nền."
    if pixels.ndim == 3:
        if not _rgb_is_gray(pixels):
            raise InputError(message)
        pixels = pixels[..., 0]
    foreground = pixels.max()
    if foreground not in (0, 1, white) or np.any((pixels != 0) & (pixels != foreground)):
        raise InputError(message)
    return (pixels != 0).astype(np.uint8) * np.uint8(255)


def _composite_alpha(img: Image.Image, notes: list[str], reject_transparency: bool = False) -> Image.Image:
    """Ghép ảnh có kênh alpha lên nền trắng (pixel trong suốt trở thành trắng).

    Với mask (reject_transparency=True), ghép lên nền trắng sẽ biến nền trong suốt
    thành đối tượng, nên pixel không đục hoàn toàn bị từ chối thay vì đổi nghĩa âm thầm.
    """
    rgba = img.convert("RGBA")
    alpha = np.asarray(rgba)[..., 3]
    if alpha.min() < 255 and reject_transparency:
        raise InputError(MASK_TRANSPARENCY_ERROR)
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


# Rawmode lấy byte thấp của mỗi mẫu PNG 16 bit big-endian:
#   RGB/RGBA: rawmode hậu tố '16L' lấy byte thứ hai của từng kênh.
#   LA 16 bit (4 byte/pixel = L_hi L_lo A_hi A_lo): 'RGBA' 8 bit đọc nguyên 4 byte thô.
# Các rawmode giữ nguyên số byte/pixel nên bộ khử lọc PNG vẫn đúng.
_PNG16_LOW_RAWMODE = {"RGB;16B": "RGB;16L", "RGBA;16B": "RGBA;16L", "LA;16B": "RGBA"}


def _png16_low_bytes(data: bytes) -> np.ndarray | None:
    """Đọc byte thấp. Trả None nếu cấu trúc decoder Pillow khác dự kiến."""
    try:
        with Image.open(io.BytesIO(data)) as img:
            tiles = list(img.tile)
            rawmodes = {t[3] for t in tiles} if tiles else set()
            if (img.mode not in ("RGB", "RGBA") or len(rawmodes) != 1 or any(t[0] != "zip" for t in tiles)
                    or next(iter(rawmodes)) not in _PNG16_LOW_RAWMODE):
                return None
            low = _PNG16_LOW_RAWMODE[next(iter(rawmodes))]
            img.tile = [t._replace(args=low) if hasattr(t, "_replace") else (*t[:3], low) for t in tiles]
            ImageOps.exif_transpose(img, in_place=True)
            return np.asarray(img)
    except (OSError, ValueError, SyntaxError, IndexError, TypeError, AttributeError):
        return None


def _png16_logical_mask(img: Image.Image, data: bytes, color_type: int, *, strict: bool = False) -> np.ndarray | None:
    """Giữ nhãn 0/1 ở độ sâu gốc và kiểm tra mask đục trước khi đọc độ sáng."""
    high = np.asarray(img)
    has_alpha = color_type in (4, 6)
    if has_alpha and np.any(high[..., 3] != 255):
        raise InputError(MASK_TRANSPARENCY_ERROR)
    has_high_color = np.any(high[..., :3])
    if has_high_color and not has_alpha and not strict:
        return None  # Ảnh không thể là mask 0/1 và không cần kiểm tra alpha.
    low = _png16_low_bytes(data)
    if low is None:
        raise InputError(
            f"Mask PNG {_PNG_COLOR_TYPES[color_type]} 16 bit: chưa kiểm tra chính xác được nhãn "
            "hoặc độ trong suốt. Hãy lưu lại mask ở 8 bit/kênh."
        )
    if has_alpha and np.any(low[..., 3] != 255):
        raise InputError(MASK_TRANSPARENCY_ERROR)
    color = low[..., 1:2] if color_type == 4 else low[..., :3]
    if strict:
        if color_type != 4 and (not _rgb_is_gray(high) or not _rgb_is_gray(color)):
            raise InputError("Mask tham chiếu phải là mask nhị phân với ba kênh màu bằng nhau.")
        gray = (high[..., 0].astype(np.uint16) << 8) | color[..., 0]
        return _binary_reference_mask(gray, white=65535)
    if not has_high_color and np.all(color <= 1) and np.all(color == color[..., :1]):
        return color[..., 0] * np.uint8(255)
    return None


def _scale_low_depth_gray_key(img: Image.Image, png_depth: tuple[int, int] | None) -> None:
    """Đưa khóa tRNS của PNG xám 2/4 bit về cùng thang 8 bit với pixel.

    Pillow giãn pixel xám 2/4 bit lên 0..255 (×85 / ×17) nhưng giữ khóa trong suốt ở thang
    gốc, nên pixel trong suốt không được nhận ra (ví dụ 4 bit, khóa 1: pixel thành 17, khóa vẫn 1).
    PNG 1 bit đã được Pillow quy đổi sẵn. Chỉ giãn khi khóa còn nằm trong thang gốc, để không
    giãn hai lần nếu phiên bản Pillow sau này tự sửa.
    """
    if png_depth is None or png_depth[1] != 0 or png_depth[0] not in (2, 4) or img.mode != "L":
        return
    key = img.info.get("transparency")
    max_raw = (1 << png_depth[0]) - 1
    if isinstance(key, int) and not isinstance(key, bool) and 0 <= key <= max_raw:
        img.info["transparency"] = key * (255 // max_raw)


def load_image_bytes(data: bytes, name: str = "image") -> LoadedImage:
    """Đọc ảnh PNG/JPG/BMP từ bytes thành mảng uint8.

    - Áp dụng hướng xoay EXIF.
    - Ảnh có alpha được ghép lên nền trắng.
    - Ảnh màu có R = G = B ở mọi pixel được coi là ảnh xám (H, W).
    """
    return _load_image_bytes(data, name, normalize_high_depth=True)


def _load_image_bytes(data: bytes, name: str, *, normalize_high_depth: bool, strict_mask: bool = False) -> LoadedImage:
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
            if strict_mask and fmt not in ("PNG", "BMP"):
                raise InputError(f"Nội dung tệp mask là định dạng {fmt}; chỉ nhận ảnh mask PNG/BMP.")
            if fmt not in ("PNG", "JPEG", "BMP", "MPO"):
                raise InputError(f"Nội dung tệp là định dạng {fmt}, không phải PNG/JPG/BMP.")
            w, h = img.size
            if w * h > MAX_DECODE_PIXELS:
                raise InputError(
                    f"Ảnh quá lớn ({w} x {h} = {w * h / 1e6:.1f} triệu pixel). Giới hạn {MAX_DECODE_PIXELS / 1e6:.0f} triệu pixel."
                )
            png_depth = _png_bit_depth(data) if fmt == "PNG" else None
            if fmt == "PNG":
                # load() có thể giải mã được pixel ngay cả khi CRC IDAT sai hoặc thiếu IEND.
                with Image.open(io.BytesIO(data), formats=("PNG",)) as checker:
                    checker.verify()
            _scale_low_depth_gray_key(img, png_depth)
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
            img.load()  # tệp hỏng báo lỗi trước bước kiểm tra byte thấp
            mask16 = (png_depth[1] if not normalize_high_depth and png_depth is not None
                      and png_depth[0] == 16 and png_depth[1] in (2, 4, 6) else None)
            arr, mode = _decoded_array(img, notes, normalize_high_depth,
                                       png16_mask=(data, mask16) if mask16 is not None else None,
                                       strict_mask=strict_mask)
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


def _decoded_array(img: Image.Image, notes: list[str], normalize_high_depth: bool,
                   png16_mask: tuple[bytes, int] | None = None, *, strict_mask: bool = False) -> tuple[np.ndarray, str]:
    orientation = img.getexif().get(0x0112, 1)  # thẻ EXIF Orientation
    if orientation not in (1, None):
        img = ImageOps.exif_transpose(img)
        notes.append(f"Ảnh đã được xoay/lật theo thông tin hướng chụp EXIF (Orientation = {orientation}).")
    mode = img.mode

    if png16_mask is not None:
        logical = _png16_logical_mask(img, *png16_mask, strict=strict_mask)
        if logical is not None:
            return logical, mode

    # So khóa tRNS trong thang gốc trước khi chuyển 16 bit về 8 bit.
    # convert("RGBA") trực tiếp từ I;16 sẽ clip các pixel >255 thành trắng.
    if mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        raw = np.asarray(img, dtype=np.float64)
        if not np.all(np.isfinite(raw)):
            raise InputError("Ảnh chứa pixel NaN hoặc vô cực.")
        transparent = img.info.get("transparency")
        hidden = raw == transparent if transparent is not None else None
        has_transparency = hidden is not None and hidden.any()
        if has_transparency and not normalize_high_depth:
            raise InputError(MASK_TRANSPARENCY_ERROR)
        if strict_mask:
            return _binary_reference_mask(raw, white=65535), mode
        if normalize_high_depth:
            if has_transparency:
                visible = ~hidden
                lo = raw.min(where=visible, initial=np.inf)
                hi = raw.max(where=visible, initial=-np.inf)
            else:
                lo, hi = raw.min(), raw.max()
            if hi > lo:
                scaled = (raw - lo) / (hi - lo) * 255
                if has_transparency:
                    scaled[hidden] = 255
                arr = np.rint(scaled).astype(np.uint8)
            else:
                arr = np.zeros(raw.shape, np.uint8)
            notes.append(f"Ảnh {mode} (độ sâu > 8 bit) được chuẩn hoá tuyến tính về [0, 255].")
        elif np.all((raw == 0) | (raw == 1)):
            # Mask nhị phân có thể lưu trực tiếp nhãn 0/1 trong PNG 16 bit.
            # Chia cho 65535 sẽ làm mất mọi pixel đối tượng trước bước đọc mask.
            arr = (raw * 255).astype(np.uint8)
            notes.append("Mask nhãn 0/1 được đổi thành 0/255, giữ nguyên vùng đối tượng.")
        else:
            arr = np.round(np.clip(raw / np.iinfo(np.uint16).max, 0, 1) * 255).astype(np.uint8)
            notes.append("Mask 16 bit được đổi từ thang [0, 65535] về [0, 255], giữ độ sáng tuyệt đối.")
        if has_transparency:
            arr[hidden] = 255
            notes.append("Ảnh có kênh alpha (trong suốt): các vùng trong suốt được ghép lên nền trắng.")
    elif mode in ("LA", "RGBA", "PA") or "transparency" in img.info:
        arr = np.asarray(_composite_alpha(img, notes, reject_transparency=not normalize_high_depth))
    elif mode in ("1", "L"):
        arr = np.asarray(img.convert("L"))
    else:
        arr = np.asarray(img.convert("RGB"))
        if mode not in ("RGB",):
            notes.append(f"Ảnh chế độ {mode} được chuyển sang RGB.")

    if strict_mask:
        return _binary_reference_mask(arr), mode

    if arr.ndim == 3 and _rgb_is_gray(arr):
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
            if row and row[-1] == "" and lines[reader.line_num - 1].rstrip("\r\n").endswith(delimiter):
                # Tương thích với một dấu phân cách thừa, nhưng không bỏ ô viết "" hoặc chứa khoảng trắng.
                row.pop()
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
    elif all(len(ln.split()) <= 1 for ln in lines):
        # Một cột: vẫn đọc theo cú pháp CSV để bỏ dấu nháy ("0") và hỗ trợ ô nhiều dòng.
        delim = ","
    else:
        delim = None  # khoảng trắng

    # Mỗi hàng hợp lệ được giữ dưới dạng chuỗi "v1,v2,..." rồi NumPy phân tích một lần ở cuối.
    rows: list[str] = []
    n_cols = None
    first_line = line_numbers[0]
    for r_idx, raw_row in _csv_rows(lines, line_numbers, delim):
        cells = [c.strip() for c in raw_row]
        if n_cols is None:
            n_cols = len(cells)
        elif len(cells) != n_cols:
            raise InputError(f"Dòng {r_idx} có {len(cells)} cột, khác với {n_cols} cột của dòng {first_line}.")
        if (len(rows) + 1) * n_cols > MAX_CSV_CELLS:
            raise InputError(f"CSV quá lớn (vượt {MAX_CSV_CELLS:,} phần tử).")
        joined = ",".join(cells)
        # Đường nhanh: cả hàng chỉ gồm số nguyên 0..255 viết thường (một phép so regex ở tầng C cho cả hàng).
        # Số dấu phẩy phải đúng n_cols - 1: ô trong dấu nháy chứa dấu phẩy ("1,2") làm chuỗi nối có thêm
        # dấu phẩy, khi đó hàng phải đi đường chậm để báo lỗi đúng ô thay vì bị tách thành nhiều giá trị.
        # Hàng khác (dấu +/-, phần thập phân, giá trị lỗi) đi đường kiểm tra từng ô để báo lỗi đúng vị trí.
        fast = joined.count(",") == len(cells) - 1 and _FAST_CSV_ROW.fullmatch(joined)
        rows.append(joined if fast else _parse_csv_row_slow(cells, r_idx))
    if not n_cols:
        raise InputError("CSV không có cột dữ liệu.")
    # Các hàng đã được xác nhận/canonical hoá. Đọc thẳng uint8 để tránh tạo
    # hàng triệu chuỗi Python và một bản sao int64 chỉ để đổi lại về 8 bit.
    values = np.fromstring(",".join(rows), sep=",", dtype=np.uint8)
    return values.reshape(len(rows), n_cols)


_CSV_BYTE = r"(?:[01]?[0-9]?[0-9]|2[0-4][0-9]|25[0-5])"  # số ASCII; Unicode đi qua Decimal để chuẩn hoá
_FAST_CSV_ROW = re.compile(rf"{_CSV_BYTE}(?:,{_CSV_BYTE})*")
_CSV_NUMBER = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)$")


def _parse_csv_row_slow(cells: list[str], r_idx: int) -> str:
    """Kiểm tra từng ô bằng Decimal (chính xác, không làm tròn số thực) và báo lỗi tại ô đầu tiên sai."""
    row: list[str] = []
    for c_idx, cell in enumerate(cells, start=1):
        if cell == "":
            raise InputError(f"Ô trống tại dòng {r_idx}, cột {c_idx}.")
        if not _CSV_NUMBER.match(cell):
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
        row.append(str(int(value)))
    return ",".join(row)


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


def read_input_bytes(path: str | Path) -> bytes:
    """Đọc tệp đầu vào cục bộ với cùng giới hạn dung lượng như tệp tải lên."""
    p = Path(path)
    try:
        with p.open("rb") as stream:
            data = stream.read(MAX_FILE_BYTES + 1)
    except FileNotFoundError as exc:
        raise InputError(f"Không tìm thấy tệp '{p.name}'.") from exc
    except PermissionError as exc:
        raise InputError(f"Không có quyền đọc tệp '{p.name}'.") from exc
    except (OSError, ValueError) as exc:
        raise InputError(f"Không đọc được tệp '{p.name}': {exc}.") from exc
    if len(data) > MAX_FILE_BYTES:
        raise InputError(f"Tệp '{p.name}' quá lớn. Giới hạn {MAX_FILE_BYTES:,} byte.")
    return data


def load_path(path: str | Path) -> LoadedImage:
    p = Path(path)
    return load_any(read_input_bytes(p), p.name)


def load_mask_bytes(data: bytes, name: str = "mask.png", *, strict: bool = False) -> np.ndarray:
    """Đọc mask PNG/BMP/CSV, trả uint8 0/255.

    Mask chỉ chứa 0/1 được hiểu là nhãn nhị phân (1 = đối tượng), kể cả toàn 1.
    CSV yêu cầu 0/1 hoặc 0/255, tránh nhận nhầm ma trận nhãn nhiều cụm.
    Với ảnh mask mức xám khác, pixel 8 bit >127 (16 bit >=32768) là đối tượng.
    Không giãn tương phản mask: ảnh trắng 16 bit giữ nguyên toàn đối tượng.
    Mask có pixel trong suốt (alpha < 255 hoặc khớp khóa tRNS) bị từ chối; mask RGBA/LA
    đục hoàn toàn vẫn được nhận.
    strict=True dùng cho đánh giá: PNG/BMP phải có nhãn 0/1 hoặc đen/trắng ở độ sâu gốc.
    """
    if Path(name).suffix.lower() == ".csv":
        return _binary_reference_mask(load_csv_bytes(data, name).array)
    loaded = _load_image_bytes(data, name, normalize_high_depth=False, strict_mask=strict)
    if strict:
        return loaded.array
    arr = loaded.array if loaded.array.ndim == 2 else loaded.array.mean(axis=2)
    png_depth = _png_bit_depth(data)
    high_depth = (loaded.original_mode in ("I;16", "I;16B", "I;16L", "I", "F")
                  or (png_depth is not None and png_depth[0] == 16))
    # Không nhận nhầm ảnh 16 bit gần đen (ví dụ 256 -> 1 sau giảm bit) là
    # nhãn 0/1. Nhãn xám 16 bit thật đã được nhận diện ở thang gốc phía trên.
    logical = not high_depth and np.all((arr == 0) | (arr == 1))
    return np.where(arr > (0 if logical else 127), 255, 0).astype(np.uint8)


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
    """Ghi ma trận số nguyên thành CSV không tiêu đề.

    Ma trận thuộc miền 0..255 và trong giới hạn số ô đọc lại được bằng parse_csv_matrix.
    """
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
