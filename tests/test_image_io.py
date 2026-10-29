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
    matrix_to_csv_bytes,
    parse_csv_matrix,
    to_png_bytes,
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
