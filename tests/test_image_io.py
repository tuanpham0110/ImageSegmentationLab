"""Kiểm thử đọc/ghi ảnh, CSV và kiểm tra dữ liệu đầu vào."""

import io

import numpy as np
import pytest
from PIL import Image

from src.processing.preprocessing import limit_size, to_grayscale
from src.utils import image_io
from src.utils.image_io import (
    InputError,
    load_any,
    load_csv_bytes,
    load_image_bytes,
    load_mask_bytes,
    matrix_to_csv_bytes,
    parse_csv_matrix,
    to_png_bytes,
    to_json_bytes,
)


def encode(img: Image.Image, fmt: str, **kw) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt, **kw)
    return buf.getvalue()


def test_png_rgb_roundtrip():
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (20, 30, 3), dtype=np.uint8)
    loaded = load_image_bytes(to_png_bytes(arr), "a.png")
    assert loaded.file_format == "PNG"
    np.testing.assert_array_equal(loaded.array, arr)
    assert loaded.channels == 3 and loaded.width == 30 and loaded.height == 20


def test_png_grayscale():
    arr = np.arange(256, dtype=np.uint8).reshape(16, 16)
    loaded = load_image_bytes(to_png_bytes(arr), "g.png")
    assert loaded.array.ndim == 2
    np.testing.assert_array_equal(loaded.array, arr)


def test_jpeg_and_bmp_load():
    arr = np.zeros((32, 32, 3), np.uint8)
    arr[:, 16:] = (200, 30, 30)
    jpg = load_image_bytes(encode(Image.fromarray(arr), "JPEG", quality=95), "x.jpg")
    assert jpg.file_format == "JPEG" and jpg.array.shape == (32, 32, 3)
    assert abs(int(jpg.array[5, 25, 0]) - 200) < 10  # JPEG có mất mát nhẹ
    bmp = load_image_bytes(encode(Image.fromarray(arr), "BMP"), "x.bmp")
    np.testing.assert_array_equal(bmp.array, arr)


def test_rgb_with_equal_channels_becomes_gray():
    gray = np.arange(64, dtype=np.uint8).reshape(8, 8)
    rgb = np.stack([gray] * 3, axis=-1)
    loaded = load_image_bytes(to_png_bytes(rgb), "eq.png")
    assert loaded.array.ndim == 2
    assert any("xám" in n for n in loaded.notes)


def test_alpha_is_composited_on_white():
    rgba = np.zeros((4, 4, 4), np.uint8)
    rgba[..., 0] = 255  # đỏ
    rgba[:2, :, 3] = 255  # nửa trên đục, nửa dưới trong suốt
    loaded = load_image_bytes(to_png_bytes(rgba), "alpha.png")
    assert loaded.array.shape == (4, 4, 3)
    assert tuple(loaded.array[0, 0]) == (255, 0, 0)
    assert tuple(loaded.array[3, 3]) == (255, 255, 255)
    assert any("alpha" in n for n in loaded.notes)


def test_exif_orientation_applied():
    arr = np.zeros((10, 20, 3), np.uint8)  # rộng 20, cao 10
    arr[:, :5] = 255
    img = Image.fromarray(arr)
    exif = img.getexif()
    exif[0x0112] = 6  # xoay 90 độ theo chiều kim đồng hồ khi hiển thị
    loaded = load_image_bytes(encode(img, "JPEG", exif=exif.tobytes(), quality=95), "rot.jpg")
    assert loaded.array.shape[:2] == (20, 10)
    assert any("EXIF" in n for n in loaded.notes)


def test_invalid_file_rejected():
    with pytest.raises(InputError):
        load_image_bytes(b"day khong phai anh", "fake.png")
    with pytest.raises(InputError):
        load_image_bytes(b"", "empty.png")
    with pytest.raises(InputError):
        load_any(b"GIF89a", "anim.gif")


def test_too_large_image_rejected(monkeypatch):
    monkeypatch.setattr(image_io, "MAX_DECODE_PIXELS", 1000)
    data = to_png_bytes(np.zeros((50, 50), np.uint8))
    with pytest.raises(InputError, match="quá lớn"):
        load_image_bytes(data, "big.png")


def test_csv_parsing_and_roundtrip():
    arr = np.array([[0, 128, 255], [1, 2, 3]], dtype=np.uint8)
    data = matrix_to_csv_bytes(arr)
    loaded = load_csv_bytes(data, "m.csv")
    np.testing.assert_array_equal(loaded.array, arr)
    assert loaded.file_format == "CSV"


@pytest.mark.parametrize("text", ["1;2;3\n4;5;6\n", "1\t2\t3\n4\t5\t6", "1 2 3\n4 5 6\n", "﻿1,2,3\n4,5,6,\n\n"])
def test_csv_delimiters(text):
    np.testing.assert_array_equal(parse_csv_matrix(text), [[1, 2, 3], [4, 5, 6]])


@pytest.mark.parametrize(
    "text,fragment",
    [
        ("1,2,3\n4,abc,6\n", "không phải số"),
        ("1,2,3\n4,300,6\n", "ngoài khoảng"),
        ("1,2,3\n4,-1,6\n", "ngoài khoảng"),
        ("1,2,3\n4,5\n", "cột"),
        ("1,2.5,3\n", "số nguyên"),
        ("1,,3\n", "Ô trống"),
        ("", "rỗng"),
    ],
)
def test_csv_errors(text, fragment):
    with pytest.raises(InputError, match=fragment):
        parse_csv_matrix(text)


def test_csv_non_utf8_rejected():
    with pytest.raises(InputError):
        load_csv_bytes(b"\xff\xfe\x00\x81", "bad.csv")


def test_grayscale_conversion():
    rgb = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [90, 90, 90]]], dtype=np.uint8)
    np.testing.assert_array_equal(to_grayscale(rgb, "luminance"), [[76, 150, 29, 90]])
    np.testing.assert_array_equal(to_grayscale(rgb, "average"), [[85, 85, 85, 90]])
    gray = np.zeros((2, 2), np.uint8)
    assert to_grayscale(gray) is gray or np.array_equal(to_grayscale(gray), gray)


def test_limit_size_keeps_aspect_ratio():
    img = np.zeros((300, 1200, 3), np.uint8)
    out, info = limit_size(img, 600)
    assert out.shape == (150, 600, 3)
    assert info.resized and info.original_size == (1200, 300) and info.processed_size == (600, 150)
    small, info2 = limit_size(np.zeros((10, 10), np.uint8), 600)
    assert small.shape == (10, 10) and not info2.resized


def test_save_and_reload_rgba_png():
    rgba = np.zeros((5, 6, 4), np.uint8)
    rgba[..., 3] = 128
    img = Image.open(io.BytesIO(to_png_bytes(rgba)))
    assert img.mode == "RGBA" and img.size == (6, 5)
    np.testing.assert_array_equal(np.asarray(img), rgba)


@pytest.mark.parametrize("text,fragment", [("9" * 400, "ngoài khoảng"),
                                         ("1.00000000000000001", "số nguyên"),
                                         ("255.00000000000000001", "ngoài khoảng"),
                                         ('"1,2', "CSV không hợp lệ"),
                                         ('0,"1\n2"', "không phải số"),
                                         ("9" * 150_000 + ",0", "CSV không hợp lệ")])
def test_csv_extreme_values_and_syntax_are_input_errors(text, fragment):
    with pytest.raises(InputError, match=fragment):
        parse_csv_matrix(text)


def test_csv_exact_integer_decimal_notation_is_preserved():
    np.testing.assert_array_equal(parse_csv_matrix("1.0000000000000000000,+002,-0,255.0"), [[1, 2, 0, 255]])


@pytest.mark.parametrize("mode,pixels,transparent", [
    ("L", [[0, 80]], 0),
    ("RGB", [[[255, 0, 0], [0, 60, 120]]], (255, 0, 0)),
])
def test_png_transparency_key_is_composited_on_white(mode, pixels, transparent):
    image = Image.fromarray(np.array(pixels, dtype=np.uint8))
    assert image.mode == mode
    loaded = load_image_bytes(encode(image, "PNG", transparency=transparent), "key.png")
    np.testing.assert_array_equal(loaded.array[0, 0], 255 if mode == "L" else [255, 255, 255])
    np.testing.assert_array_equal(loaded.array[0, 1], pixels[0][1])
    assert any("alpha" in note for note in loaded.notes)


def test_csv_cell_limit_is_checked_before_converting_an_oversized_row(monkeypatch):
    monkeypatch.setattr(image_io, "MAX_CSV_CELLS", 3)
    with pytest.raises(InputError, match="quá lớn"):
        parse_csv_matrix("1,2\ninvalid,4")


def test_16bit_transparency_preserves_opaque_pixel_contrast():
    pixels = np.array([[1000, 2000, 3000]], dtype=np.uint16)
    loaded = load_image_bytes(encode(Image.fromarray(pixels), "PNG", transparency=1000), "depth.png")
    np.testing.assert_array_equal(loaded.array, [[255, 128, 255]])
    assert any("alpha" in note for note in loaded.notes)
    assert any("16" in note for note in loaded.notes)


@pytest.mark.parametrize("level,expected", [(0, 0), (32767, 0), (32768, 255), (65535, 255)])
def test_16bit_ground_truth_preserves_absolute_black_and_white(level, expected):
    data = encode(Image.fromarray(np.full((3, 4), level, np.uint16)), "PNG")
    np.testing.assert_array_equal(load_mask_bytes(data, "gt.png"), np.full((3, 4), expected, np.uint8))


@pytest.mark.parametrize("array", [np.array([[1.5]]), np.array([[np.nan]]), np.array([[np.inf]]),
                                   np.array([[-1]]), np.array([[256]]), np.zeros((2, 2, 5), np.uint8),
                                   np.empty((0, 2), np.uint8)])
def test_png_export_rejects_data_that_would_be_silently_changed(array):
    with pytest.raises(ValueError):
        to_png_bytes(array)


@pytest.mark.parametrize("matrix", [np.array([[1.5]]), np.array([[np.nan]]), np.array([[np.inf]]),
                                    np.array([[1j]]), np.array([["1"]]), np.empty((0, 2))])
def test_matrix_export_rejects_non_integer_or_empty_data(matrix):
    with pytest.raises(ValueError):
        matrix_to_csv_bytes(matrix)


def test_matrix_export_preserves_unsigned_integer_range():
    maximum = np.iinfo(np.uint64).max
    assert matrix_to_csv_bytes(np.array([[maximum]], np.uint64)) == f"{maximum}\n".encode()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), np.array([np.nan])])
def test_json_export_does_not_emit_nonstandard_numeric_literals(value):
    with pytest.raises(ValueError):
        to_json_bytes({"nested": {"value": value}})


def test_json_export_supports_numpy_boolean_values():
    import json

    assert json.loads(to_json_bytes({"valid": np.bool_(True)})) == {"valid": True}


# ---------------- Hồi quy: PNG màu 16 bit, CSV có dòng trống, hậu xử lý ----------------

def _png16(arr: np.ndarray, color_type: int, trns: tuple[int, ...] | None = None) -> bytes:
    """Tự mã hoá PNG 16 bit/kênh (Pillow không ghi được RGB/RGBA 16 bit)."""
    import struct
    import zlib

    def chunk(tag: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)

    h, w = arr.shape[:2]
    raw = b"".join(b"\x00" + arr[y].astype(">u2").tobytes() for y in range(h))
    out = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 16, color_type, 0, 0, 0))
    if trns is not None:
        out += chunk(b"tRNS", struct.pack(">" + "H" * len(trns), *trns))
    return out + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def test_16bit_rgb_png_with_transparency_key_is_rejected_not_misread():
    pixels = np.array([[[256, 256, 256], [1000, 2000, 3000]]])
    with pytest.raises(InputError, match="16 bit có khóa trong suốt"):
        load_image_bytes(_png16(pixels, 2, trns=(1000, 2000, 3000)), "key16.png")


@pytest.mark.parametrize("color_type,pixels,expected", [
    (2, [[[0, 0, 0], [65535, 32768, 512]]], [[[0, 0, 0], [255, 128, 2]]]),
    (6, [[[0, 0, 0, 65535], [65535, 0, 0, 65535]]], [[[0, 0, 0], [255, 0, 0]]]),
])
def test_16bit_color_png_is_reduced_to_8bit_with_explicit_note(color_type, pixels, expected):
    loaded = load_image_bytes(_png16(np.array(pixels), color_type), "deep.png")
    np.testing.assert_array_equal(loaded.array, expected)
    assert any("16 bit/kênh được giảm về 8 bit" in n for n in loaded.notes)


def test_16bit_grayscale_png_still_uses_contrast_stretch_without_color_note():
    loaded = load_image_bytes(_png16(np.array([[256, 257]]), 0), "gray16.png")
    np.testing.assert_array_equal(loaded.array, [[0, 255]])
    assert not any("giảm về 8 bit" in n for n in loaded.notes)


def test_8bit_png_has_no_16bit_note():
    loaded = load_image_bytes(to_png_bytes(np.zeros((2, 2, 3), np.uint8)), "a.png")
    assert not any("16 bit" in n for n in loaded.notes)


@pytest.mark.parametrize("text,fragment", [
    ("1,2\n\n3,x\n", "dòng 3, cột 2"),
    ("\n\n1,2\n3,4,5\n", "Dòng 4 có 3 cột, khác với 2 cột của dòng 3"),
    ("1 2\n   \n3 x\n", "dòng 3, cột 2"),
    ('"1",2\n\n"3\n",4\n5,x\n', "dòng 5, cột 2"),  # hàng 2 trải dòng 3–4 do ô trong dấu nháy
])
def test_csv_error_reports_physical_line_number_with_blank_lines(text, fragment):
    with pytest.raises(InputError, match=fragment):
        parse_csv_matrix(text)


@pytest.mark.parametrize("image", [
    np.array([[-1, 256]]),
    np.array([[1.5, 2.0]]),
    np.zeros((2, 2, 4), np.uint8),
    np.zeros((2, 2, 1), np.uint8),
    np.array([[np.nan, 0.0]]),
])
def test_postprocessing_rejects_invalid_images_instead_of_wrapping(image):
    from src.processing.postprocessing import cutout_rgba

    with pytest.raises(ValueError):
        cutout_rgba(image, np.ones(image.shape[:2], bool))


def test_postprocessing_accepts_integer_valued_non_uint8_images():
    from src.processing.postprocessing import cutout_rgba

    out = cutout_rgba(np.array([[0, 255]], dtype=np.int64), np.array([[True, False]]))
    np.testing.assert_array_equal(out, [[[0, 0, 0, 255], [255, 255, 255, 0]]])


# ---------------- Hồi quy: ground truth có vùng trong suốt (RISK-004) ----------------

def _transparent_masks():
    rgba = np.zeros((4, 4, 4), np.uint8)
    rgba[1:3, 1:3] = (255, 255, 255, 255)
    la = Image.fromarray(np.dstack([np.full((2, 2), 255, np.uint8), np.array([[255, 0], [255, 255]], np.uint8)]), "LA")
    palette = Image.fromarray(np.array([[0, 1]], np.uint8), "P")
    palette.putpalette([0, 0, 0, 255, 255, 255])
    gray_key = Image.fromarray(np.array([[0, 255]], np.uint8), "L")
    gray16_key = Image.fromarray(np.array([[0, 65535]], np.uint16))
    return [
        to_png_bytes(rgba),
        encode(la, "PNG"),
        encode(palette, "PNG", transparency=0),
        encode(gray_key, "PNG", transparency=0),
        encode(gray16_key, "PNG", transparency=0),
    ]


@pytest.mark.parametrize("data", _transparent_masks(), ids=["rgba", "la", "palette_trns", "gray_trns", "gray16_trns"])
def test_ground_truth_with_transparency_is_rejected(data):
    with pytest.raises(InputError, match="vùng trong suốt"):
        load_mask_bytes(data, "gt.png")


def test_fully_opaque_rgba_ground_truth_is_accepted():
    rgba = np.zeros((2, 2, 4), np.uint8)
    rgba[..., 3] = 255
    rgba[0, 0, :3] = 255
    np.testing.assert_array_equal(load_mask_bytes(to_png_bytes(rgba), "gt.png"), [[255, 0], [0, 0]])


def test_input_image_transparency_is_still_composited_on_white():
    rgba = np.zeros((1, 2, 4), np.uint8)
    rgba[0, 1] = (10, 20, 30, 255)
    np.testing.assert_array_equal(load_image_bytes(to_png_bytes(rgba), "in.png").array, [[[255, 255, 255], [10, 20, 30]]])
