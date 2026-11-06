"""Quản lý ảnh đầu vào dùng chung giữa các trang và bộ nhớ đệm kết quả.

Mỗi lần chạy thuật toán, kết quả được lưu trong st.session_state kèm "chữ ký" của đầu vào
(hash nội dung tệp + tên tệp + giới hạn kích thước + cách chuyển xám). Khi đầu vào đổi, chữ ký đổi
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
RESULT_KEYS = ("km_result", "otsu_result", "cmp_result")


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
    try:
        samples = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InputError(f"Không đọc được danh mục ảnh mẫu data/samples.json: {exc}.") from exc
    text_fields = ("id", "title", "kind", "file", "purpose", "expected")
    if not isinstance(samples, list) or any(
        not isinstance(s, dict)
        or any(not isinstance(s.get(key), str) or not s[key] for key in text_fields)
        or "ground_truth" not in s
        or (s["ground_truth"] is not None and not isinstance(s["ground_truth"], str))
        or not isinstance(s.get("suggested"), dict)
        or type(s["suggested"].get("k")) is not int
        or not 1 <= s["suggested"]["k"] <= 8
        or s["suggested"].get("feature") not in ("rgb", "gray")
        for s in samples
    ):
        raise InputError("Danh mục ảnh mẫu data/samples.json sai cấu trúc hoặc thiếu trường bắt buộc.")
    if len({s["id"] for s in samples}) != len(samples):
        raise InputError("Danh mục ảnh mẫu data/samples.json có id trùng lặp.")
    return samples


def sample_by_id(sid: str) -> dict | None:
    return next((s for s in load_manifest() if s["id"] == sid), None)


def _load(data: bytes, name: str) -> LoadedImage:
    # Ảnh gốc chỉ được giữ trong InputContext hiện tại của phiên, không đưa
    # nhiều ảnh giải mã (tới 120 MB/ảnh RGB) vào cache dùng chung.
    return load_any(data, name)


@st.cache_data(show_spinner=False, max_entries=4, ttl=300)
def _prepare(data: bytes, name: str, max_side: int, gray_method: str,
             _loaded: LoadedImage | None = None) -> PreparedInput:
    # _loaded không tham gia hash: bytes + tên tệp đã định danh đầy đủ đầu vào.
    # Truyền ảnh vừa đọc để cache miss không phải giải mã thêm một lần.
    return prepare_input(_load(data, name) if _loaded is None else _loaded, max_side, gray_method)


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
    digest = hashlib.sha1(data).hexdigest()[:16]
    current = get_context()
    if current is not None and current.signature.split("|", 1)[0] == digest and current.loaded.source_name == name:
        loaded = current.loaded
    else:
        loaded = _load(data, name)
    prepared = _prepare(data, name, max_side, gray_method, _loaded=loaded)
    signature = f"{digest}|{max_side}|{gray_method}|{name}"
    gt = None
    if gt_bytes is not None:
        gt = fit_ground_truth(load_mask_bytes(gt_bytes, "gt.png"), prepared)
    return InputContext(loaded=loaded, prepared=prepared, signature=signature, sample=sample, ground_truth=gt)


def set_context(ctx: InputContext | None, error: str | None = None) -> None:
    for key in RESULT_KEYS:
        result = st.session_state.get(key)
        if result is not None and (ctx is None or result.get("signature") != ctx.signature):
            # Giữ dấu báo kết quả cũ, giải phóng mảng ảnh và bytes xuất.
            # Dấu stale ngăn khôi phục payload đã bỏ nếu người dùng chọn lại ảnh cũ.
            st.session_state[key] = {"signature": result.get("signature"), "_stale": True}
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
    if res is None or res.get("_stale") or res.get("signature") != signature:
        return None
    return res


def has_stale_result(key: str, signature: str) -> bool:
    res = st.session_state.get(key)
    return res is not None and (res.get("_stale", False) or res.get("signature") != signature)


# ---------------- Giữ giá trị widget qua điều hướng trang ----------------
# Streamlit xoá state của widget có key nếu widget không được vẽ trong một lượt chạy
# (ví dụ khi chuyển sang trang khác). Khi quay lại, widget dựng lại từ giá trị mặc định,
# làm lựa chọn của người dùng (cụm đối tượng, cực tính, tham số form) tự đổi.
# Hai hàm dưới lưu một bản sao ở key không phải widget (không bị xoá) và khôi phục lại.
_PERSIST_PREFIX = "_persist__"


def restore_widget(key: str, default) -> None:
    """Gọi TRƯỚC khi vẽ widget: khôi phục giá trị đã lưu, hoặc dùng mặc định nếu chưa có."""
    if key not in st.session_state:
        st.session_state[key] = st.session_state.get(_PERSIST_PREFIX + key, default)


def remember_widget(key: str) -> None:
    """Gọi SAU khi vẽ widget: lưu giá trị hiện tại vào bản sao bền."""
    if key in st.session_state:
        value = st.session_state[key]
        st.session_state[_PERSIST_PREFIX + key] = list(value) if isinstance(value, list) else value


def persistent_foreground(run: dict, key: str, options: list[int], default: list[int]) -> None:
    """Khôi phục lựa chọn cụm đối tượng từ payload kết quả (gắn với đúng ảnh và lần chạy)."""
    if key not in st.session_state:
        saved = run.get("foreground")
        st.session_state[key] = ([c for c in saved if c in options] if saved is not None
                                 else list(default))


def save_foreground(run: dict, fg) -> None:
    """Lưu lựa chọn hiện tại vào payload để giữ qua điều hướng và dùng cho dữ liệu xuất."""
    run["foreground"] = sorted(int(c) for c in fg)


__all__ = [
    "InputContext", "InputError", "MAX_SIDE_OPTIONS", "DEFAULT_MAX_SIDE", "DATA_DIR",
    "load_manifest", "sample_by_id", "thumbnail", "build_context", "set_context", "get_context",
    "get_input_error", "store_result", "get_result", "has_stale_result",
    "restore_widget", "remember_widget", "persistent_foreground", "save_foreground",
]
