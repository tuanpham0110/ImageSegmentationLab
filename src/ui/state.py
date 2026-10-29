"""Quản lý ảnh đầu vào dùng chung giữa các trang và bộ nhớ đệm kết quả.

Mỗi lần chạy thuật toán, kết quả được lưu trong st.session_state kèm "chữ ký" của đầu vào
(hash nội dung tệp + giới hạn kích thước + cách chuyển xám). Khi đầu vào đổi, chữ ký đổi
và kết quả cũ không được hiển thị/tải xuống nữa.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from src.processing.pipeline import PreparedInput, fit_ground_truth, prepare_input
from src.utils.image_io import InputError, LoadedImage, load_any, load_mask_bytes

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MANIFEST_PATH = DATA_DIR / "samples.json"

MAX_SIDE_OPTIONS = [128, 256, 400, 600, 800, 1024, 1600, 2048]
DEFAULT_MAX_SIDE = 800


@dataclass
class InputContext:
    loaded: LoadedImage
    prepared: PreparedInput
    signature: str
    sample: dict | None  # mục trong samples.json nếu là ảnh mẫu
    ground_truth: np.ndarray | None  # mask GT ở kích thước xử lý (nếu có)


@st.cache_data(show_spinner=False)
def load_manifest() -> list[dict]:
    if not MANIFEST_PATH.exists():
        return []
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def sample_by_id(sid: str) -> dict | None:
    return next((s for s in load_manifest() if s["id"] == sid), None)


@st.cache_data(show_spinner=False, max_entries=16)
def _load(data: bytes, name: str) -> LoadedImage:
    return load_any(data, name)


@st.cache_data(show_spinner=False, max_entries=16)
def _prepare(data: bytes, name: str, max_side: int, gray_method: str) -> PreparedInput:
    return prepare_input(_load(data, name), max_side, gray_method)


@st.cache_data(show_spinner=False, max_entries=32)
def thumbnail(path: str, size: int = 220) -> np.ndarray:
    loaded = load_any(Path(path).read_bytes(), Path(path).name)
    img = Image.fromarray(loaded.array)
    resample = Image.Resampling.NEAREST if max(img.size) < size else Image.Resampling.BOX
    scale = size / max(img.size)
    return np.asarray(img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), resample))


def build_context(data: bytes, name: str, max_side: int, gray_method: str, sample: dict | None,
                  gt_bytes: bytes | None = None) -> InputContext:
    """Đọc + chuẩn bị đầu vào. Ném InputError nếu dữ liệu không hợp lệ."""
    loaded = _load(data, name)
    prepared = _prepare(data, name, max_side, gray_method)
    digest = hashlib.sha1(data).hexdigest()[:16]
    signature = f"{digest}|{max_side}|{gray_method}"
    gt = None
    if gt_bytes is not None:
        gt = fit_ground_truth(load_mask_bytes(gt_bytes, "gt.png"), prepared)
    return InputContext(loaded=loaded, prepared=prepared, signature=signature, sample=sample, ground_truth=gt)


def set_context(ctx: InputContext | None, error: str | None = None) -> None:
    st.session_state["_input_ctx"] = ctx
    st.session_state["_input_error"] = error


def get_context() -> InputContext | None:
    return st.session_state.get("_input_ctx")


def get_input_error() -> str | None:
    return st.session_state.get("_input_error")


def store_result(key: str, signature: str, payload: dict) -> None:
    st.session_state[key] = {"signature": signature, **payload}


def get_result(key: str, signature: str) -> dict | None:
    """Trả về kết quả đã lưu nếu khớp với đầu vào hiện tại, ngược lại None."""
    res = st.session_state.get(key)
    if res is None or res.get("signature") != signature:
        return None
    return res


def has_stale_result(key: str, signature: str) -> bool:
    res = st.session_state.get(key)
    return res is not None and res.get("signature") != signature


__all__ = [
    "InputContext", "InputError", "MAX_SIDE_OPTIONS", "DEFAULT_MAX_SIDE", "DATA_DIR",
    "load_manifest", "sample_by_id", "thumbnail", "build_context", "set_context", "get_context",
    "get_input_error", "store_result", "get_result", "has_stale_result",
]
