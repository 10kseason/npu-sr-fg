import csv
import ctypes
from collections import deque
import queue
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
from tkinter import messagebox, ttk
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageGrab, ImageTk

THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from lunasr_universal import (
    apply_luna_profile_overrides,
    apply_performance_fallback_step,
    auto_tune,
    lunasr_upscale_bgr_internal,
    mode_defaults,
    resolve_luna_profile,
)

DEFAULT_OV_MODEL = "model/ir/fixed_sr_algo_x2_temporal.xml"
DEFAULT_OV_MODEL_T192 = "model/ir/fixed_sr_algo_x2_temporal_t192.xml"
DEFAULT_OV_MODEL_T256 = "model/ir/fixed_sr_algo_x2_temporal_t256.xml"
DEFAULT_FG_MODEL = "model/ir/fixed_fg_algo_mid.xml"
DEFAULT_OV_PRESET = "speed"
DEFAULT_OV_INTERNAL_SCALE = 0.50
DEFAULT_OV_CACHE_DIR = ".ov_cache"
DEFAULT_CAPTURE_BACKEND = "gdi"
DEFAULT_OUTPUT_PRESET = "AUTO"
DEFAULT_GPU_VIDEO_PATH = True
DEFAULT_TEMPORAL_RESTORE = True
DEFAULT_TEMPORAL_STRENGTH = 0.35
DEFAULT_BENCHMARK_ENABLED = False
DEFAULT_BENCHMARK_CSV = "bench_realtime_overlay.csv"
DEFAULT_STRICT_NPU_ONLY = True
DEFAULT_STRICT_GPU_ONLY = False
DEFAULT_OV_PARALLEL_REQS_NPU = 0
DEFAULT_OV_PARALLEL_REQS_NPU_HARD_CAP = 256
DEFAULT_OV_PARALLEL_REQS_GPU = 0
DEFAULT_OV_PARALLEL_REQS_GPU_HARD_CAP = 32
DEFAULT_OV_PARALLEL_REQS_CPU = 2
GPU_TILE_BASE_MAX_W = 1280
GPU_TILE_BASE_MAX_H = 720
CAPTURE_FRAME_BUFFER_SIZE = 4
OVERLAY_POLL_INTERVAL_MS = 8
STATUS_POLL_INTERVAL_MS = 100
LUNASR_ONLY_MODE = False


user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
GWL_EXSTYLE = -20
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_NOACTIVATE = 0x08000000
LWA_ALPHA = 0x00000002
SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOZORDER = 0x0004
SWP_NOACTIVATE = 0x0010
SWP_FRAMECHANGED = 0x0020
HWND_TOPMOST = -1
WDA_NONE = 0x00000000
WDA_EXCLUDEFROMCAPTURE = 0x00000011
WINDOW_TITLE_MAX = 512
MONITOR_DEFAULTTONEAREST = 0x00000002
SRCCOPY = 0x00CC0020
CAPTUREBLT = 0x40000000
DIB_RGB_COLORS = 0
BI_RGB = 0
PW_RENDERFULLCONTENT = 0x00000002

user32.MonitorFromWindow.argtypes = [ctypes.c_void_p, ctypes.c_uint]
user32.MonitorFromWindow.restype = ctypes.c_void_p
user32.GetMonitorInfoW.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
user32.GetMonitorInfoW.restype = ctypes.c_bool
user32.GetWindowDC.argtypes = [ctypes.c_void_p]
user32.GetWindowDC.restype = ctypes.c_void_p
user32.ReleaseDC.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
user32.ReleaseDC.restype = ctypes.c_int
user32.EnableWindow.argtypes = [ctypes.c_void_p, ctypes.c_bool]
user32.EnableWindow.restype = ctypes.c_bool
user32.GetForegroundWindow.argtypes = []
user32.GetForegroundWindow.restype = ctypes.c_void_p
user32.SetForegroundWindow.argtypes = [ctypes.c_void_p]
user32.SetForegroundWindow.restype = ctypes.c_bool
user32.PrintWindow.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
user32.PrintWindow.restype = ctypes.c_bool
gdi32.CreateCompatibleDC.argtypes = [ctypes.c_void_p]
gdi32.CreateCompatibleDC.restype = ctypes.c_void_p
gdi32.DeleteDC.argtypes = [ctypes.c_void_p]
gdi32.DeleteDC.restype = ctypes.c_bool
gdi32.CreateCompatibleBitmap.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
gdi32.CreateCompatibleBitmap.restype = ctypes.c_void_p
gdi32.DeleteObject.argtypes = [ctypes.c_void_p]
gdi32.DeleteObject.restype = ctypes.c_bool
gdi32.SelectObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
gdi32.SelectObject.restype = ctypes.c_void_p
gdi32.BitBlt.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_uint,
]
gdi32.BitBlt.restype = ctypes.c_bool


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", ctypes.c_uint32),
        ("biWidth", ctypes.c_long),
        ("biHeight", ctypes.c_long),
        ("biPlanes", ctypes.c_ushort),
        ("biBitCount", ctypes.c_ushort),
        ("biCompression", ctypes.c_uint32),
        ("biSizeImage", ctypes.c_uint32),
        ("biXPelsPerMeter", ctypes.c_long),
        ("biYPelsPerMeter", ctypes.c_long),
        ("biClrUsed", ctypes.c_uint32),
        ("biClrImportant", ctypes.c_uint32),
    ]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [
        ("bmiHeader", BITMAPINFOHEADER),
        ("bmiColors", ctypes.c_uint32 * 3),
    ]


gdi32.GetDIBits.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_void_p,
    ctypes.POINTER(BITMAPINFO),
    ctypes.c_uint,
]
gdi32.GetDIBits.restype = ctypes.c_int


def enable_dpi_awareness() -> str:
    # Make window/capture coordinates use physical pixels to avoid DPI virtualized zoom.
    try:
        set_ctx = getattr(user32, "SetProcessDpiAwarenessContext")
        set_ctx.argtypes = [ctypes.c_void_p]
        set_ctx.restype = ctypes.c_bool
        # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4
        if set_ctx(ctypes.c_void_p(-4 & 0xFFFFFFFFFFFFFFFF)):
            return "per_monitor_v2"
    except Exception:
        pass

    try:
        shcore = ctypes.WinDLL("shcore", use_last_error=True)
        set_proc = getattr(shcore, "SetProcessDpiAwareness")
        set_proc.argtypes = [ctypes.c_int]
        set_proc.restype = ctypes.c_int
        # PROCESS_PER_MONITOR_DPI_AWARE = 2, S_OK=0, E_ACCESSDENIED=-2147024891(already set)
        hr = int(set_proc(2))
        if hr == 0 or hr == -2147024891:
            return "per_monitor_v1"
    except Exception:
        pass

    try:
        set_aware = getattr(user32, "SetProcessDPIAware")
        set_aware.argtypes = []
        set_aware.restype = ctypes.c_bool
        if set_aware():
            return "system_aware"
    except Exception:
        pass

    return "unknown"


DPI_AWARENESS_MODE = enable_dpi_awareness()


def detect_cuda_available() -> bool:
    try:
        if not hasattr(cv2, "cuda"):
            return False
        cnt = int(cv2.cuda.getCudaEnabledDeviceCount())
        return cnt > 0
    except Exception:
        return False


CUDA_AVAILABLE = detect_cuda_available()


def resize_bgr(
    image: np.ndarray,
    size_wh: Tuple[int, int],
    interpolation: int,
    use_cuda: bool,
) -> np.ndarray:
    if not use_cuda:
        return cv2.resize(image, size_wh, interpolation=interpolation)
    try:
        gpu = cv2.cuda_GpuMat()
        gpu.upload(image)
        out_gpu = cv2.cuda.resize(gpu, size_wh, interpolation=interpolation)
        return out_gpu.download()
    except Exception:
        return cv2.resize(image, size_wh, interpolation=interpolation)


def temporal_restore_bgr(
    prev_bgr: np.ndarray,
    cur_bgr: np.ndarray,
    strength: float,
) -> Tuple[np.ndarray, float]:
    s = max(0.0, min(1.0, float(strength)))
    if s <= 0.0:
        return cur_bgr, 0.0
    if prev_bgr is None or cur_bgr is None:
        return cur_bgr, 0.0
    if prev_bgr.ndim != 3 or cur_bgr.ndim != 3:
        return cur_bgr, 0.0
    h, w = int(cur_bgr.shape[0]), int(cur_bgr.shape[1])
    if h <= 1 or w <= 1 or prev_bgr.shape[0] != h or prev_bgr.shape[1] != w:
        return cur_bgr, 0.0

    t0 = time.perf_counter()
    max_side = max(h, w)
    motion_max_side = 320
    motion_scale = min(1.0, float(motion_max_side) / float(max(1, max_side)))
    sw = max(16, int(round(w * motion_scale)))
    sh = max(16, int(round(h * motion_scale)))
    sw = max(16, sw - (sw % 2))
    sh = max(16, sh - (sh % 2))

    if sw != w or sh != h:
        prev_small = cv2.resize(prev_bgr, (sw, sh), interpolation=cv2.INTER_AREA)
        cur_small = cv2.resize(cur_bgr, (sw, sh), interpolation=cv2.INTER_AREA)
    else:
        prev_small = prev_bgr
        cur_small = cur_bgr

    prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    cur_gray = cv2.cvtColor(cur_small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    shift, response = cv2.phaseCorrelate(prev_gray, cur_gray)
    dx = float(shift[0]) * (float(w) / float(max(1, sw)))
    dy = float(shift[1]) * (float(h) / float(max(1, sh)))
    max_shift = 0.25 * float(max_side)
    if abs(dx) > max_shift or abs(dy) > max_shift:
        dx, dy, response = 0.0, 0.0, 0.0

    affine = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    prev_warp = cv2.warpAffine(
        prev_bgr,
        affine,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )

    if sw != w or sh != h:
        warp_small = cv2.resize(prev_warp, (sw, sh), interpolation=cv2.INTER_AREA)
    else:
        warp_small = prev_warp
    warp_gray = cv2.cvtColor(warp_small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    diff_mean = float(np.mean(np.abs(cur_gray - warp_gray)))
    stability = float(np.clip((0.16 - diff_mean) / 0.16, 0.0, 1.0))
    resp_conf = float(np.clip(response, 0.0, 1.0))
    alpha = float(np.clip(s * stability * (0.25 + (0.75 * resp_conf)), 0.0, 0.95))

    if alpha <= 1e-4:
        ms = (time.perf_counter() - t0) * 1000.0
        return cur_bgr, ms
    out_u8 = cv2.addWeighted(prev_warp, alpha, cur_bgr, 1.0 - alpha, 0.0)
    ms = (time.perf_counter() - t0) * 1000.0
    return out_u8, ms


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


class MONITORINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", ctypes.c_uint),
        ("rcMonitor", RECT),
        ("rcWork", RECT),
        ("dwFlags", ctypes.c_uint),
    ]


@dataclass
class WindowInfo:
    hwnd: int
    title: str
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)

    @property
    def label(self) -> str:
        return f"[0x{self.hwnd:08X}] {self.title} ({self.width}x{self.height})"


@dataclass
class RuntimeConfig:
    hwnd: int
    backend: str
    capture_backend: str
    preset: str
    output_preset: str
    gpu_video_path: bool
    upscale_enabled: bool
    frame_budget_ms: float
    device: str
    ov_model: str
    ov_preset: str
    ov_internal_scale: float
    ov_reactive_scale: bool
    ov_reactive_target_fps: float
    temporal_restore: bool
    temporal_strength: float
    benchmark_enabled: bool
    benchmark_csv: str
    strict_npu_only: bool
    strict_gpu_only: bool
    ov_cache_dir: str
    allow_cpu_fallback: bool
    overlay_alpha: float
    overlay_click_through: bool
    overlay_exclude_from_capture: bool
    overlay_fullscreen_upscale: bool
    fg_enabled: bool
    fg_interp_only: bool
    fg_model: str
    fg_precision: str
    fg_size: int
    fg_timestep: float


def get_window_text(hwnd: int) -> str:
    length = user32.GetWindowTextLengthW(hwnd)
    if length <= 0:
        return ""
    buff = ctypes.create_unicode_buffer(min(length + 1, WINDOW_TITLE_MAX))
    user32.GetWindowTextW(hwnd, buff, WINDOW_TITLE_MAX)
    text = buff.value
    for token in ("\u200b", "\u200c", "\u200d", "\ufeff"):
        text = text.replace(token, "")
    text = "".join(ch for ch in text if ch.isprintable())
    return text.strip()


def get_window_rect(hwnd: int) -> Optional[Tuple[int, int, int, int]]:
    rect = RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return None
    left, top, right, bottom = int(rect.left), int(rect.top), int(rect.right), int(rect.bottom)
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def get_monitor_rect_for_window(hwnd: int) -> Optional[Tuple[int, int, int, int]]:
    monitor = user32.MonitorFromWindow(int(hwnd), MONITOR_DEFAULTTONEAREST)
    if not monitor:
        return None
    info = MONITORINFO()
    info.cbSize = ctypes.sizeof(MONITORINFO)
    if not user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
        return None
    left = int(info.rcMonitor.left)
    top = int(info.rcMonitor.top)
    right = int(info.rcMonitor.right)
    bottom = int(info.rcMonitor.bottom)
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


TARGET_HEIGHT_PRESETS: Dict[str, int] = {
    "HD": 720,
    "FHD": 1080,
    "QHD": 1440,
    "4K": 2160,
}


def auto_output_preset_for_source_height(src_h: int) -> str:
    h = max(1, int(src_h))
    # Default policy (source -> output): <720->HD, <1080->FHD, <1440->QHD, >=1440->4K.
    if h < 720:
        return "HD"
    if h < 1080:
        return "FHD"
    if h < 1440:
        return "QHD"
    return "4K"


def normalize_output_preset(text: str) -> str:
    token = (text or "").strip().upper()
    if token in ("AUTO", "DEFAULT", "SOURCE", "NATIVE"):
        token = "AUTO"
    if token in ("720P",):
        token = "HD"
    if token in ("1080P",):
        token = "FHD"
    if token in ("1440P",):
        token = "QHD"
    if token in ("UHD", "2160P"):
        token = "4K"
    if token == "AUTO":
        return token
    if token not in TARGET_HEIGHT_PRESETS:
        return DEFAULT_OUTPUT_PRESET
    return token


def resolve_target_size_from_preset(src_w: int, src_h: int, preset: str) -> Tuple[int, int]:
    p = normalize_output_preset(preset)
    if p == "AUTO":
        p = auto_output_preset_for_source_height(src_h)
    preset_h = int(TARGET_HEIGHT_PRESETS[p])
    target_h = max(src_h, preset_h)
    scale = float(target_h) / float(max(1, src_h))
    target_w = max(src_w, int(round(src_w * scale)))
    # Keep even size for display compatibility.
    target_w = max(2, target_w - (target_w % 2))
    target_h = max(2, target_h - (target_h % 2))
    return target_w, target_h


def clamp_size_preserve_aspect(src_w: int, src_h: int, max_w: int, max_h: int) -> Tuple[int, int]:
    w = max(1, int(src_w))
    h = max(1, int(src_h))
    mw = max(1, int(max_w))
    mh = max(1, int(max_h))
    if w <= mw and h <= mh:
        return w, h
    scale = min(float(mw) / float(w), float(mh) / float(h))
    cw = max(1, int(round(w * scale)))
    ch = max(1, int(round(h * scale)))
    cw = max(2, cw - (cw % 2))
    ch = max(2, ch - (ch % 2))
    return cw, ch


def build_centered_overlay_rect(
    src_rect: Tuple[int, int, int, int],
    target_w: int,
    target_h: int,
    bounds_rect: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[int, int, int, int]:
    left, top, right, bottom = src_rect
    cx = int(round((left + right) * 0.5))
    cy = int(round((top + bottom) * 0.5))
    w = max(1, int(target_w))
    h = max(1, int(target_h))
    nl = cx - (w // 2)
    nt = cy - (h // 2)
    nr = nl + w
    nb = nt + h

    if bounds_rect is not None:
        bl, bt, br, bb = bounds_rect
        bw = max(1, br - bl)
        bh = max(1, bb - bt)
        if w > bw:
            nl, nr = bl, br
        else:
            if nl < bl:
                nr += (bl - nl)
                nl = bl
            if nr > br:
                nl -= (nr - br)
                nr = br
        if h > bh:
            nt, nb = bt, bb
        else:
            if nt < bt:
                nb += (bt - nt)
                nt = bt
            if nb > bb:
                nt -= (nb - bb)
                nb = bb
    return int(nl), int(nt), int(nr), int(nb)


def enumerate_windows() -> List[WindowInfo]:
    result: List[WindowInfo] = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def enum_cb(hwnd, _lparam):
        hwnd_i = int(hwnd)
        if not user32.IsWindowVisible(hwnd_i):
            return True
        title = get_window_text(hwnd_i)
        if not title:
            return True
        if title.lower().startswith("lunasr realtime gui"):
            return True
        ex_style = user32.GetWindowLongW(hwnd_i, GWL_EXSTYLE)
        if ex_style & WS_EX_TOOLWINDOW:
            return True
        rect = get_window_rect(hwnd_i)
        if rect is None:
            return True
        left, top, right, bottom = rect
        if (right - left) < 160 or (bottom - top) < 120:
            return True
        result.append(
            WindowInfo(
                hwnd=hwnd_i,
                title=title,
                left=left,
                top=top,
                right=right,
                bottom=bottom,
            )
        )
        return True

    user32.EnumWindows(enum_cb, 0)
    result.sort(key=lambda x: x.title.lower())
    return result


def capture_window_bgr_gdi(hwnd: int) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    rect = get_window_rect(hwnd)
    if rect is None:
        return None
    left, top, right, bottom = rect
    w = max(1, right - left)
    h = max(1, bottom - top)

    hwnd_dc = ctypes.c_void_p(0)
    mem_dc = ctypes.c_void_p(0)
    bmp = ctypes.c_void_p(0)
    old_obj = ctypes.c_void_p(0)
    try:
        hwnd_dc = ctypes.c_void_p(user32.GetWindowDC(int(hwnd)))
        if not hwnd_dc.value:
            return None

        mem_dc = ctypes.c_void_p(gdi32.CreateCompatibleDC(hwnd_dc))
        if not mem_dc.value:
            return None

        bmp = ctypes.c_void_p(gdi32.CreateCompatibleBitmap(hwnd_dc, w, h))
        if not bmp.value:
            return None

        old_obj = ctypes.c_void_p(gdi32.SelectObject(mem_dc, bmp))
        if not old_obj.value:
            return None

        # Capture target hwnd directly to avoid recursive overlay feedback.
        ok = bool(user32.PrintWindow(int(hwnd), mem_dc, PW_RENDERFULLCONTENT))
        if not ok:
            ok = bool(user32.PrintWindow(int(hwnd), mem_dc, 0))
        if not ok:
            return None

        bmi = BITMAPINFO()
        bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.bmiHeader.biWidth = w
        bmi.bmiHeader.biHeight = -h  # top-down
        bmi.bmiHeader.biPlanes = 1
        bmi.bmiHeader.biBitCount = 32
        bmi.bmiHeader.biCompression = BI_RGB
        buf = (ctypes.c_ubyte * (w * h * 4))()
        rows = gdi32.GetDIBits(mem_dc, bmp, 0, h, ctypes.byref(buf), ctypes.byref(bmi), DIB_RGB_COLORS)
        if rows != h:
            return None

        frame_bgra = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
        frame_bgr = frame_bgra[:, :, :3].copy()
        return frame_bgr, (left, top, right, bottom)
    except Exception:
        return None
    finally:
        try:
            if mem_dc.value and old_obj.value:
                gdi32.SelectObject(mem_dc, old_obj)
        except Exception:
            pass
        try:
            if bmp.value:
                gdi32.DeleteObject(bmp)
        except Exception:
            pass
        try:
            if mem_dc.value:
                gdi32.DeleteDC(mem_dc)
        except Exception:
            pass
        try:
            if hwnd_dc.value:
                user32.ReleaseDC(int(hwnd), hwnd_dc)
        except Exception:
            pass


def capture_window_bgr_pil(hwnd: int) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    rect = get_window_rect(hwnd)
    if rect is None:
        return None
    left, top, right, bottom = rect

    img = None
    try:
        # Prefer direct window capture to avoid recursive self-capture from overlay.
        img = ImageGrab.grab(window=int(hwnd), include_layered_windows=False)
    except TypeError:
        img = None
    except Exception:
        img = None

    if img is None:
        try:
            img = ImageGrab.grab(
                bbox=(left, top, right, bottom),
                include_layered_windows=False,
                all_screens=True,
            )
        except Exception:
            return None

    frame_rgb = np.array(img)
    if frame_rgb.ndim != 3 or frame_rgb.shape[2] < 3:
        return None
    if frame_rgb.shape[2] == 4:
        frame_rgb = frame_rgb[:, :, :3]

    w = max(1, right - left)
    h = max(1, bottom - top)
    if frame_rgb.shape[1] != w or frame_rgb.shape[0] != h:
        interp = cv2.INTER_AREA if (frame_rgb.shape[1] > w or frame_rgb.shape[0] > h) else cv2.INTER_CUBIC
        frame_rgb = cv2.resize(frame_rgb, (w, h), interpolation=interp)

    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), (left, top, right, bottom)


def capture_window_bgr(
    hwnd: int,
    backend: str = DEFAULT_CAPTURE_BACKEND,
) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    token = (backend or DEFAULT_CAPTURE_BACKEND).strip().lower()
    if token == "pil":
        frame = capture_window_bgr_pil(hwnd)
        if frame is not None:
            return frame
        return capture_window_bgr_gdi(hwnd)

    frame = capture_window_bgr_gdi(hwnd)
    if frame is not None:
        return frame
    return capture_window_bgr_pil(hwnd)


def parse_float(text: str, fallback: float) -> float:
    try:
        return float(text)
    except Exception:
        return fallback


def parse_int(text: str, fallback: int) -> int:
    try:
        return int(text)
    except Exception:
        return fallback


def resolve_model_path(model_path: str) -> Path:
    raw = (model_path or "").strip().strip('"').strip("'")
    if not raw:
        raise RuntimeError("OpenVINO model path is empty.")

    p = Path(raw).expanduser()
    candidates: List[Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append((Path.cwd() / p).resolve())
        candidates.append((PROJECT_DIR / p).resolve())
        candidates.append((THIS_DIR / p).resolve())

    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def resolve_auto_ov_model_for_output(
    selected_model: str,
    output_preset: str,
) -> str:
    selected = (selected_model or "").strip() or DEFAULT_OV_MODEL
    auto_models = (
        DEFAULT_OV_MODEL,
        DEFAULT_OV_MODEL_T192,
        DEFAULT_OV_MODEL_T256,
    )
    selected_norm = selected.replace("\\", "/").lower()
    auto_norm = {m.replace("\\", "/").lower() for m in auto_models}
    # Respect explicit custom model paths.
    if selected_norm not in auto_norm:
        return selected

    p = normalize_output_preset(output_preset)
    if p in ("HD", "4K"):
        preferred = DEFAULT_OV_MODEL_T256
    else:
        preferred = DEFAULT_OV_MODEL_T192

    preferred_path = resolve_model_path(preferred)
    if preferred_path.exists():
        return preferred

    for fallback in (DEFAULT_OV_MODEL_T192, DEFAULT_OV_MODEL_T256, DEFAULT_OV_MODEL):
        if resolve_model_path(fallback).exists():
            return fallback
    return selected


def build_luna_runtime(preset: str, frame_budget_ms: float) -> Dict[str, Any]:
    profile_name = resolve_luna_profile(preset)
    tier = "quality"
    tune_strategy = "default"
    internal_scale = 1.0
    auto_tune_on = True
    (
        tier,
        tune_strategy,
        internal_scale,
        auto_tune_on,
        frame_budget_ms,
        profile_note,
    ) = apply_luna_profile_overrides(
        profile_name=profile_name,
        tier=tier,
        tune_strategy=tune_strategy,
        internal_scale=internal_scale,
        auto_tune_on=auto_tune_on,
        frame_budget_ms=frame_budget_ms,
    )
    return {
        "params": mode_defaults(tier),
        "tune_strategy": tune_strategy,
        "internal_scale": float(internal_scale),
        "auto_tune_on": bool(auto_tune_on),
        "frame_budget_ms": float(frame_budget_ms),
        "note": profile_note,
    }


def np_dtype_from_element_type(element_type: Any):
    name = str(element_type).lower()
    if "f16" in name or "bf16" in name:
        return np.float16
    if "f32" in name or "float" in name:
        return np.float32
    if "i64" in name:
        return np.int64
    if "i32" in name:
        return np.int32
    if "i16" in name:
        return np.int16
    if "i8" in name:
        return np.int8
    if "u8" in name:
        return np.uint8
    if "bool" in name:
        return np.bool_
    return np.float32


def to_nchw_tensor(image_bgr: np.ndarray, c: int, h: int, w: int, dtype) -> np.ndarray:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[0] != h or rgb.shape[1] != w:
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)

    if c == 1:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)[:, :, None]
    elif c == 4:
        alpha = np.full((h, w, 1), 255, dtype=np.uint8)
        rgb = np.concatenate([rgb, alpha], axis=2)
    elif c != 3:
        raise RuntimeError(f"Unsupported channel count for image input: C={c}")

    x = np.transpose(rgb, (2, 0, 1))[None, ...]
    if np.issubdtype(dtype, np.floating):
        x = (x.astype(np.float32) / 255.0).astype(dtype)
    elif np.issubdtype(dtype, np.integer):
        x = x.astype(dtype)
    else:
        x = x.astype(np.float32)
    return x


def postprocess_tensor(y: np.ndarray) -> np.ndarray:
    if y.ndim == 4:
        y = y[0]
    if y.ndim == 3:
        y = np.transpose(y, (1, 2, 0))
    if y.ndim != 3 or y.shape[2] not in (1, 3, 4):
        raise RuntimeError(f"Output tensor is not image-like: shape={y.shape}")

    if np.issubdtype(y.dtype, np.floating):
        if float(np.max(y)) > 1.5:
            y = np.clip(y, 0.0, 255.0).astype(np.uint8)
        else:
            y = np.clip(y, 0.0, 1.0)
            y = (y * 255.0 + 0.5).astype(np.uint8)
    else:
        y = y.astype(np.uint8)

    if y.shape[2] == 1:
        y = np.repeat(y, 3, axis=2)
    elif y.shape[2] == 4:
        y = y[:, :, :3]
    return cv2.cvtColor(y, cv2.COLOR_RGB2BGR)


def build_ov_compile_cfg(device: str, preset: str, cache_dir: str) -> Dict[str, object]:
    cfg: Dict[str, object] = {}
    if cache_dir:
        cfg["CACHE_DIR"] = cache_dir

    dev = (device or "AUTO").upper()
    if "NPU" in dev:
        # NPU path: force int8-oriented compile hints.
        cfg["INFERENCE_PRECISION_HINT"] = "i8"
        cfg["NPU_COMPILER_DYNAMIC_QUANTIZATION"] = True
        cfg["NPU_QDQ_OPTIMIZATION"] = True
        cfg["NPU_QDQ_OPTIMIZATION_AGGRESSIVE"] = True
    elif preset == "quality":
        cfg["INFERENCE_PRECISION_HINT"] = "f16"
    elif "GPU" in dev:
        cfg["INFERENCE_PRECISION_HINT"] = "f16"
    else:
        cfg["INFERENCE_PRECISION_HINT"] = "f32"
    if "GPU" in dev:
        # Intel GPU path: prioritize throughput so async request pool can run wider in parallel.
        cfg["PERFORMANCE_HINT"] = "THROUGHPUT"
    return cfg


def resolve_ov_device_name(available_devices: List[str], preferred: str) -> Optional[str]:
    token = (preferred or "").strip().upper()
    if not token:
        return None
    for dev in available_devices:
        name = str(dev).strip()
        up = name.upper()
        if up == token or up.startswith(token):
            return name
    return None


def build_relaxed_ov_compile_cfg(device: str, preset: str, cache_dir: str) -> Dict[str, object]:
    cfg: Dict[str, object] = {}
    if cache_dir:
        cfg["CACHE_DIR"] = cache_dir
    dev = (device or "AUTO").upper()
    cfg["INFERENCE_PRECISION_HINT"] = "f16" if (preset == "quality" or "NPU" in dev or "GPU" in dev) else "f32"
    return cfg


def summarize_exc(exc: Exception, limit: int = 220) -> str:
    msg = " ".join(str(exc).strip().splitlines())
    if len(msg) > limit:
        return msg[: limit - 3] + "..."
    return msg


def resize_to_x2(
    tile: np.ndarray,
    valid_h: int,
    valid_w: int,
    model_scale_h: float,
    model_scale_w: float,
    downsample_interp: int,
) -> np.ndarray:
    src_h = int(round(valid_h * model_scale_h))
    src_w = int(round(valid_w * model_scale_w))
    src = tile[:src_h, :src_w]
    dst_h = valid_h * 2
    dst_w = valid_w * 2
    if src_h == dst_h and src_w == dst_w:
        return src
    interp = downsample_interp if (src_h >= dst_h and src_w >= dst_w) else cv2.INTER_CUBIC
    return cv2.resize(src, (dst_w, dst_h), interpolation=interp)


def compile_ov_runtime(
    model_path: str,
    device: str,
    preset: str,
    cache_dir: str,
    allow_cpu_fallback: bool = True,
    strict_npu_only: bool = False,
    strict_gpu_only: bool = False,
) -> Dict[str, Any]:
    import openvino as ov

    model_resolved = resolve_model_path(model_path)
    if not model_resolved.exists():
        raise FileNotFoundError(f"OpenVINO model not found: {model_resolved}")

    core = ov.Core()
    model = core.read_model(str(model_resolved))
    if len(model.outputs) != 1:
        raise RuntimeError("OpenVINO realtime mode expects single-output image model.")

    image_inputs: List[Dict[str, Any]] = []
    for idx, inp in enumerate(model.inputs):
        shape = [int(x) for x in inp.shape]
        if len(shape) != 4 or int(shape[0]) != 1 or int(shape[1]) not in (1, 3, 4):
            continue
        image_inputs.append(
            {
                "name": str(inp.any_name),
                "shape": shape,
                "dtype": np_dtype_from_element_type(inp.element_type),
                "index": int(idx),
            }
        )
    if not image_inputs:
        all_shapes = [[int(x) for x in p.shape] for p in model.inputs]
        raise RuntimeError(f"No image-like 4D input found. inputs={all_shapes}")
    if len(image_inputs) > 4:
        raise RuntimeError(f"At most four image inputs are supported. got={len(image_inputs)}")

    # Ensure deterministic input order.
    image_inputs = sorted(image_inputs, key=lambda m: int(m.get("index", 0)))
    primary_in = image_inputs[0]
    in_shape = [int(x) for x in primary_in["shape"]]
    out_shape = [int(x) for x in model.outputs[0].shape]
    if len(out_shape) != 4:
        raise RuntimeError(f"Expected 4D NCHW output. got output={out_shape}")

    in_c, in_h, in_w = int(in_shape[1]), int(in_shape[2]), int(in_shape[3])
    out_h, out_w = int(out_shape[2]), int(out_shape[3])
    scale_h = float(out_h) / float(max(1, in_h))
    scale_w = float(out_w) / float(max(1, in_w))

    ov_inputs: List[Dict[str, Any]] = []
    temporal_model = False
    temporal_prev_depth = 0

    def _guess_named_prev_idx(name: str) -> Optional[int]:
        n = str(name).lower()
        if any(tok in n for tok in ("curr", "current", "img0", "input0", "frame0")):
            return 0
        if any(tok in n for tok in ("prev3", "history3", "past3", "state3")):
            return 3
        if any(tok in n for tok in ("prev2", "history2", "past2", "state2")):
            return 2
        if any(tok in n for tok in ("prev", "history", "past", "state", "img1", "input1", "frame1")):
            return 1
        return None

    curr_idx = None
    for idx, meta in enumerate(image_inputs):
        if _guess_named_prev_idx(str(meta["name"])) == 0:
            curr_idx = idx
            break
    if curr_idx is None:
        curr_idx = 0

    for idx, meta in enumerate(image_inputs):
        mh = int(meta["shape"][2])
        mw = int(meta["shape"][3])
        if mh != in_h or mw != in_w:
            raise RuntimeError(
                f"All image inputs must match shape of first input. got first={in_shape}, this={meta['shape']}"
            )

        if idx == curr_idx:
            role = "curr"
        else:
            guessed = _guess_named_prev_idx(str(meta["name"]))
            if guessed is None or guessed <= 0:
                guessed = temporal_prev_depth + 1
            if guessed == 1:
                role = "prev"
            else:
                role = f"prev{int(guessed)}"
            temporal_prev_depth = max(temporal_prev_depth, int(guessed))

        ov_inputs.append(
            {
                "name": str(meta["name"]),
                "dtype": meta["dtype"],
                "c": int(meta["shape"][1]),
                "h": int(meta["shape"][2]),
                "w": int(meta["shape"][3]),
                "role": str(role),
            }
        )
    temporal_model = any(str(m.get("role", "")).startswith("prev") for m in ov_inputs)

    available_devices = [str(x) for x in core.available_devices]
    if strict_npu_only and strict_gpu_only:
        raise RuntimeError("strict_npu_only and strict_gpu_only cannot both be enabled.")

    requested_device = (device or "AUTO").strip() or "AUTO"
    if strict_npu_only:
        requested_device = "NPU"
    elif strict_gpu_only:
        requested_device = "GPU"
    requested_up = requested_device.upper()
    primary_device = (
        resolve_ov_device_name(available_devices, requested_up) or requested_device
        if requested_up != "AUTO"
        else "AUTO"
    )

    compile_cfg = build_ov_compile_cfg(device=primary_device, preset=preset, cache_dir=cache_dir)
    compile_note = ""
    compile_device = primary_device
    attempts: List[Tuple[str, str, Dict[str, object]]] = []
    seen_attempts = set()

    def add_attempt(tag: str, target_device: str, cfg: Dict[str, object]) -> None:
        key = (str(target_device).upper(), tuple(sorted((str(k), str(v)) for k, v in cfg.items())))
        if key in seen_attempts:
            return
        seen_attempts.add(key)
        attempts.append((tag, target_device, cfg))

    add_attempt("primary", primary_device, compile_cfg)
    if strict_npu_only:
        relaxed_cfg = build_relaxed_ov_compile_cfg(device=primary_device, preset=preset, cache_dir=cache_dir)
        if tuple(sorted((str(k), str(v)) for k, v in relaxed_cfg.items())) != tuple(
            sorted((str(k), str(v)) for k, v in compile_cfg.items())
        ):
            add_attempt("npu_relaxed_hint", primary_device, relaxed_cfg)
    elif strict_gpu_only:
        relaxed_cfg = build_relaxed_ov_compile_cfg(device=primary_device, preset=preset, cache_dir=cache_dir)
        if tuple(sorted((str(k), str(v)) for k, v in relaxed_cfg.items())) != tuple(
            sorted((str(k), str(v)) for k, v in compile_cfg.items())
        ):
            add_attempt("gpu_relaxed_hint", primary_device, relaxed_cfg)
    elif "NPU" not in str(primary_device).upper():
        add_attempt(
            "relaxed_hint",
            primary_device,
            build_relaxed_ov_compile_cfg(device=primary_device, preset=preset, cache_dir=cache_dir),
        )

    if (not strict_npu_only) and (not strict_gpu_only) and (("NPU" in requested_up) or (requested_up == "AUTO")):
        gpu_dev = resolve_ov_device_name(available_devices, "GPU")
        if gpu_dev:
            add_attempt(
                "gpu_fallback",
                gpu_dev,
                build_ov_compile_cfg(device=gpu_dev, preset=preset, cache_dir=cache_dir),
            )

    cpu_dev = resolve_ov_device_name(available_devices, "CPU")
    if (not strict_npu_only) and (not strict_gpu_only) and allow_cpu_fallback and cpu_dev:
        cpu_cfg = build_ov_compile_cfg(device=cpu_dev, preset=preset, cache_dir=cache_dir)
        cpu_cfg["INFERENCE_PRECISION_HINT"] = "f32"
        cpu_cfg.pop("NPU_COMPILER_DYNAMIC_QUANTIZATION", None)
        cpu_cfg.pop("NPU_QDQ_OPTIMIZATION", None)
        cpu_cfg.pop("NPU_QDQ_OPTIMIZATION_AGGRESSIVE", None)
        add_attempt("cpu_fallback", cpu_dev, cpu_cfg)

    t0 = time.perf_counter()
    compiled = None
    errors: List[str] = []
    for tag, target_device, cfg in attempts:
        try:
            compiled = core.compile_model(model, target_device, cfg)
            compile_cfg = cfg
            compile_device = target_device
            if tag != "primary":
                compile_note = f"compile fallback applied: {tag}@{target_device}"
                if errors:
                    compile_note += f" (last_error={errors[-1]})"
            break
        except Exception as exc:
            errors.append(f"{tag}@{target_device}: {summarize_exc(exc)}")
    if compiled is None:
        detail = " | ".join(errors[-3:]) if errors else "unknown compile error"
        raise RuntimeError(
            f"OpenVINO compile failed. requested={requested_device}, available={available_devices}, detail={detail}"
        )
    t1 = time.perf_counter()

    downsample_interp = cv2.INTER_LANCZOS4 if preset == "quality" else cv2.INTER_AREA
    final_interp = cv2.INTER_LANCZOS4 if preset == "quality" else cv2.INTER_CUBIC

    exec_devices = "unknown"
    try:
        v = compiled.get_property("EXECUTION_DEVICES")
        if isinstance(v, (list, tuple)):
            exec_devices = ",".join(str(x) for x in v)
        else:
            exec_devices = str(v)
    except Exception:
        pass

    if strict_npu_only:
        compile_up = str(compile_device).upper()
        exec_up = str(exec_devices).upper()
        if ("NPU" not in compile_up) and ("NPU" not in exec_up):
            raise RuntimeError(
                f"strict_npu_only enabled, but compiled execution is not NPU "
                f"(compile_device={compile_device}, exec_devices={exec_devices})"
            )
    if strict_gpu_only:
        compile_up = str(compile_device).upper()
        exec_up = str(exec_devices).upper()
        if ("GPU" not in compile_up) and ("GPU" not in exec_up):
            raise RuntimeError(
                f"strict_gpu_only enabled, but compiled execution is not GPU "
                f"(compile_device={compile_device}, exec_devices={exec_devices})"
            )

    compile_device_up = str(compile_device).upper()
    optimal_reqs = 0
    try:
        optimal_reqs = int(compiled.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS"))
    except Exception:
        optimal_reqs = 0
    if "NPU" in compile_device_up:
        ov_parallel_reqs_target = int(DEFAULT_OV_PARALLEL_REQS_NPU)
        req_pool_hard_cap = int(DEFAULT_OV_PARALLEL_REQS_NPU_HARD_CAP)
    elif "GPU" in compile_device_up:
        ov_parallel_reqs_target = int(DEFAULT_OV_PARALLEL_REQS_GPU)
        req_pool_hard_cap = max(1, int(DEFAULT_OV_PARALLEL_REQS_GPU_HARD_CAP))
        if ov_parallel_reqs_target <= 0 and optimal_reqs > 0:
            ov_parallel_reqs_target = max(2, min(req_pool_hard_cap, int(optimal_reqs)))
    else:
        ov_parallel_reqs_target = int(DEFAULT_OV_PARALLEL_REQS_CPU)
        req_pool_hard_cap = max(1, int(DEFAULT_OV_PARALLEL_REQS_CPU))

    req_pool: List[Any] = []
    req_pool_error = ""
    auto_mode = bool(ov_parallel_reqs_target <= 0)
    if auto_mode:
        req_pool_hard_cap = max(1, req_pool_hard_cap)
        while len(req_pool) < req_pool_hard_cap:
            try:
                req_pool.append(compiled.create_infer_request())
            except Exception as exc:
                req_pool_error = summarize_exc(exc)
                break
    else:
        ov_parallel_reqs_target = max(1, ov_parallel_reqs_target)
        for _ in range(ov_parallel_reqs_target):
            try:
                req_pool.append(compiled.create_infer_request())
            except Exception as exc:
                req_pool_error = summarize_exc(exc)
                break

    if not req_pool:
        mode = f"auto<= {req_pool_hard_cap}" if auto_mode else f"target={ov_parallel_reqs_target}"
        raise RuntimeError(
            f"Failed to create infer request pool on {compile_device} "
            f"({mode}): {req_pool_error or 'unknown'}"
        )
    if auto_mode:
        auto_msg = f"req_pool_auto={len(req_pool)}"
        if req_pool_error:
            auto_msg += f" ({req_pool_error})"
        compile_note = f"{compile_note} | {auto_msg}" if compile_note else auto_msg
    elif len(req_pool) < ov_parallel_reqs_target:
        cap_msg = f"req_pool_capped={len(req_pool)}/{ov_parallel_reqs_target}"
        if req_pool_error:
            cap_msg += f" ({req_pool_error})"
        compile_note = f"{compile_note} | {cap_msg}" if compile_note else cap_msg
    if ("GPU" in compile_device_up) and (optimal_reqs > 0):
        opt_msg = f"ov_optimal_reqs={optimal_reqs}"
        compile_note = f"{compile_note} | {opt_msg}" if compile_note else opt_msg

    runtime = {
        "compiled": compiled,
        "req": req_pool[0],
        "req_pool": req_pool,
        "ov_parallel_reqs": int(len(req_pool)),
        "in_name": str(primary_in["name"]),
        "out_name": compiled.outputs[0].any_name,
        "in_dtype": primary_in["dtype"],
        "in_c": in_c,
        "in_h": in_h,
        "in_w": in_w,
        "ov_inputs": ov_inputs,
        "temporal_model": bool(temporal_model),
        "temporal_prev_depth": int(temporal_prev_depth),
        "tile_h": in_h,
        "tile_w": in_w,
        "tile_overlap": max(0, min(16, (in_h - 1) // 2, (in_w - 1) // 2)),
        "model_scale_h": scale_h,
        "model_scale_w": scale_w,
        "downsample_interp": downsample_interp,
        "final_interp": final_interp,
        "compile_ms": (t1 - t0) * 1000.0,
        "model_path": str(model_resolved),
        "device": requested_device,
        "compile_device": compile_device,
        "exec_devices": exec_devices,
        "compile_cfg": compile_cfg,
        "compile_note": compile_note,
        "strict_npu_only": bool(strict_npu_only),
        "strict_gpu_only": bool(strict_gpu_only),
    }
    return runtime


def build_fg_compile_cfg(device: str, precision: str, cache_dir: str) -> Dict[str, object]:
    cfg: Dict[str, object] = {}
    if cache_dir:
        cfg["CACHE_DIR"] = cache_dir

    prec = (precision or "f16").lower()
    if prec not in ("f16", "f32", "i8"):
        prec = "f16"
    cfg["INFERENCE_PRECISION_HINT"] = prec

    if prec == "i8" and "NPU" in (device or "").upper():
        cfg["NPU_COMPILER_DYNAMIC_QUANTIZATION"] = True
        cfg["NPU_QDQ_OPTIMIZATION"] = True
        cfg["NPU_QDQ_OPTIMIZATION_AGGRESSIVE"] = True
    return cfg


def compile_fg_runtime(
    model_path: str,
    device: str,
    cache_dir: str,
    precision: str,
    fg_size: int,
    allow_cpu_fallback: bool = True,
    strict_npu_only: bool = False,
    strict_gpu_only: bool = False,
) -> Dict[str, Any]:
    import openvino as ov

    if fg_size < 16:
        raise RuntimeError(f"FG size must be >=16. got {fg_size}")

    model_resolved = resolve_model_path(model_path)
    if not model_resolved.exists():
        raise FileNotFoundError(f"FG model not found: {model_resolved}")

    core = ov.Core()
    model = core.read_model(str(model_resolved))

    reshape_map: Dict[str, List[int]] = {}
    for inp in model.inputs:
        rank = len(inp.partial_shape)
        if rank == 4:
            reshape_map[inp.any_name] = [1, 3, fg_size, fg_size]
        elif rank == 1:
            reshape_map[inp.any_name] = [1]
    if reshape_map:
        model.reshape(reshape_map)

    available_devices = [str(x) for x in core.available_devices]
    if strict_npu_only and strict_gpu_only:
        raise RuntimeError("strict_npu_only and strict_gpu_only cannot both be enabled for FG.")

    requested_device = (device or "AUTO").strip() or "AUTO"
    if strict_npu_only:
        requested_device = "NPU"
    elif strict_gpu_only:
        requested_device = "GPU"
    requested_up = requested_device.upper()
    primary_device = (
        resolve_ov_device_name(available_devices, requested_up) or requested_device
        if requested_up != "AUTO"
        else "AUTO"
    )

    compile_cfg = build_fg_compile_cfg(device=primary_device, precision=precision, cache_dir=cache_dir)
    compile_device = primary_device
    compile_note = ""
    attempts: List[Tuple[str, str, Dict[str, object]]] = []
    seen_attempts = set()

    def add_attempt(tag: str, target_device: str, cfg: Dict[str, object]) -> None:
        key = (str(target_device).upper(), tuple(sorted((str(k), str(v)) for k, v in cfg.items())))
        if key in seen_attempts:
            return
        seen_attempts.add(key)
        attempts.append((tag, target_device, cfg))

    add_attempt("primary", primary_device, compile_cfg)
    if str(compile_cfg.get("INFERENCE_PRECISION_HINT", "")).lower() != "f16":
        add_attempt(
            "relaxed_hint",
            primary_device,
            build_fg_compile_cfg(device=primary_device, precision="f16", cache_dir=cache_dir),
        )

    if (not strict_npu_only) and (not strict_gpu_only) and (("NPU" in requested_up) or (requested_up == "AUTO")):
        gpu_dev = resolve_ov_device_name(available_devices, "GPU")
        if gpu_dev:
            add_attempt(
                "gpu_fallback",
                gpu_dev,
                build_fg_compile_cfg(device=gpu_dev, precision="f16", cache_dir=cache_dir),
            )

    cpu_dev = resolve_ov_device_name(available_devices, "CPU")
    if (not strict_npu_only) and (not strict_gpu_only) and allow_cpu_fallback and cpu_dev:
        add_attempt(
            "cpu_fallback",
            cpu_dev,
            build_fg_compile_cfg(device=cpu_dev, precision="f32", cache_dir=cache_dir),
        )

    t0 = time.perf_counter()
    compiled = None
    errors: List[str] = []
    for tag, target_device, cfg in attempts:
        try:
            compiled = core.compile_model(model, target_device, cfg)
            compile_cfg = cfg
            compile_device = target_device
            if tag != "primary":
                compile_note = f"compile fallback applied: {tag}@{target_device}"
                if errors:
                    compile_note += f" (last_error={errors[-1]})"
            break
        except Exception as exc:
            errors.append(f"{tag}@{target_device}: {summarize_exc(exc)}")
    if compiled is None:
        detail = " | ".join(errors[-3:]) if errors else "unknown compile error"
        raise RuntimeError(
            f"FG compile failed. requested={requested_device}, available={available_devices}, detail={detail}"
        )
    t1 = time.perf_counter()

    exec_devices = "unknown"
    try:
        v = compiled.get_property("EXECUTION_DEVICES")
        if isinstance(v, (list, tuple)):
            exec_devices = ",".join(str(x) for x in v)
        else:
            exec_devices = str(v)
    except Exception:
        pass

    if strict_npu_only:
        compile_up = str(compile_device).upper()
        exec_up = str(exec_devices).upper()
        if ("NPU" not in compile_up) and ("NPU" not in exec_up):
            raise RuntimeError(
                f"strict_npu_only enabled for FG, but execution is not NPU "
                f"(compile_device={compile_device}, exec_devices={exec_devices})"
            )
    if strict_gpu_only:
        compile_up = str(compile_device).upper()
        exec_up = str(exec_devices).upper()
        if ("GPU" not in compile_up) and ("GPU" not in exec_up):
            raise RuntimeError(
                f"strict_gpu_only enabled for FG, but execution is not GPU "
                f"(compile_device={compile_device}, exec_devices={exec_devices})"
            )

    input_meta: List[Dict[str, Any]] = []
    image_inputs = 0
    for inp in compiled.inputs:
        name = inp.any_name
        lname = name.lower()
        shape = [int(x) for x in inp.shape]
        dtype = np_dtype_from_element_type(inp.element_type)

        kind = "other"
        if len(shape) == 4 and shape[0] == 1 and shape[1] in (1, 3, 4):
            kind = "image"
            image_inputs += 1
        elif "timestep" in lname and int(np.prod(shape)) == 1:
            kind = "timestep"

        input_meta.append(
            {
                "name": name,
                "shape": shape,
                "dtype": dtype,
                "kind": kind,
            }
        )

    if image_inputs < 2:
        raise RuntimeError("FG model must expose at least two image inputs.")

    runtime = {
        "req": compiled.create_infer_request(),
        "out_name": compiled.outputs[0].any_name,
        "inputs": input_meta,
        "model_path": str(model_resolved),
        "compile_ms": (t1 - t0) * 1000.0,
        "device": requested_device,
        "compile_device": compile_device,
        "compile_cfg": compile_cfg,
        "compile_note": compile_note,
        "exec_devices": exec_devices,
        "fg_size": int(fg_size),
        "precision": str(compile_cfg.get("INFERENCE_PRECISION_HINT", precision or "f16")).lower(),
        "strict_npu_only": bool(strict_npu_only),
        "strict_gpu_only": bool(strict_gpu_only),
    }
    return runtime


def reduce_fg_ghosting(
    mid: np.ndarray,
    frame0: np.ndarray,
    frame1: np.ndarray,
    strength: float = 0.18,
    motion_low: float = 10.0,
    motion_high: float = 48.0,
) -> np.ndarray:
    g = max(0.0, min(1.0, float(strength)))
    if g <= 1e-6:
        return mid
    if frame0.shape[:2] != mid.shape[:2]:
        frame0 = cv2.resize(frame0, (mid.shape[1], mid.shape[0]), interpolation=cv2.INTER_CUBIC)
    if frame1.shape[:2] != mid.shape[:2]:
        frame1 = cv2.resize(frame1, (mid.shape[1], mid.shape[0]), interpolation=cv2.INTER_CUBIC)

    base = cv2.addWeighted(frame0, 0.5, frame1, 0.5, 0.0)
    diff = cv2.absdiff(frame0, frame1)
    motion = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    motion = cv2.GaussianBlur(motion, (0, 0), 1.0)
    low = float(motion_low)
    high = max(low + 1.0, float(motion_high))
    mask = np.clip((motion.astype(np.float32) - low) / (high - low), 0.0, 1.0)
    alpha = (mask[:, :, None] * g).astype(np.float32)

    out = mid.astype(np.float32) * (1.0 - alpha) + base.astype(np.float32) * alpha
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def run_fg_mid(
    frame0: np.ndarray,
    frame1: np.ndarray,
    fg_ctx: Dict[str, Any],
    timestep: float,
) -> Tuple[np.ndarray, float]:
    req = fg_ctx["req"]
    out_name = str(fg_ctx["out_name"])
    inputs: List[Dict[str, Any]] = fg_ctx["inputs"]

    feed: Dict[str, np.ndarray] = {}
    image_inputs: List[Dict[str, Any]] = []

    for meta in inputs:
        name = str(meta["name"])
        shape = [int(x) for x in meta["shape"]]
        dtype = meta["dtype"]
        kind = str(meta["kind"])

        if kind == "image":
            image_inputs.append(meta)
            continue
        if kind == "timestep":
            feed[name] = np.full(shape, float(timestep), dtype=dtype)
            continue
        feed[name] = np.zeros(shape, dtype=dtype)

    for idx, meta in enumerate(image_inputs):
        name = str(meta["name"])
        lname = name.lower()
        shape = [int(x) for x in meta["shape"]]
        dtype = meta["dtype"]
        _, c, h, w = shape

        if "img0" in lname:
            src = frame0
        elif "img1" in lname:
            src = frame1
        else:
            src = frame0 if idx == 0 else frame1
        feed[name] = to_nchw_tensor(src, c, h, w, dtype)

    t0 = time.perf_counter()
    outputs = req.infer(feed)
    t1 = time.perf_counter()

    y = outputs[out_name] if out_name in outputs else next(iter(outputs.values()))
    mid = postprocess_tensor(np.asarray(y))
    mid = reduce_fg_ghosting(mid=mid, frame0=frame0, frame1=frame1, strength=0.18)
    return mid, (t1 - t0) * 1000.0


def run_ov_infer_tiled(
    image: np.ndarray,
    ov_ctx: Dict[str, Any],
    temporal_images: Optional[List[np.ndarray]] = None,
) -> Tuple[np.ndarray, float]:
    src_h, src_w = image.shape[:2]
    dst = np.zeros((src_h * 2, src_w * 2, 3), dtype=np.uint8)

    tile_h = int(ov_ctx["tile_h"])
    tile_w = int(ov_ctx["tile_w"])
    tile_overlap = max(0, int(ov_ctx.get("tile_overlap", 0)))
    core_h = max(1, tile_h - tile_overlap * 2)
    core_w = max(1, tile_w - tile_overlap * 2)
    tiles_x = (src_w + core_w - 1) // core_w
    tiles_y = (src_h + core_h - 1) // core_h

    req_pool_raw = ov_ctx.get("req_pool")
    req_pool: List[Any]
    if isinstance(req_pool_raw, list) and req_pool_raw:
        req_pool = list(req_pool_raw)
    else:
        req_pool = [ov_ctx["req"]]

    in_name = str(ov_ctx["in_name"])
    out_name = str(ov_ctx["out_name"])
    in_c = int(ov_ctx["in_c"])
    in_h = int(ov_ctx["in_h"])
    in_w = int(ov_ctx["in_w"])
    in_dtype = ov_ctx["in_dtype"]
    ov_inputs_raw = ov_ctx.get("ov_inputs")
    ov_inputs: List[Dict[str, Any]]
    if isinstance(ov_inputs_raw, list) and ov_inputs_raw:
        ov_inputs = list(ov_inputs_raw)
    else:
        ov_inputs = [
            {
                "name": in_name,
                "dtype": in_dtype,
                "c": in_c,
                "h": in_h,
                "w": in_w,
                "role": "curr",
            }
        ]
    model_scale_h = float(ov_ctx["model_scale_h"])
    model_scale_w = float(ov_ctx["model_scale_w"])
    downsample_interp = int(ov_ctx["downsample_interp"])

    free_reqs: List[Any] = list(req_pool)
    inflight: List[Tuple[Any, Dict[str, Any], float]] = []
    async_enabled = len(req_pool) > 1
    infer_lat_ms: List[float] = []

    def finalize_tile(job: Dict[str, Any], outputs: Any) -> None:
        y = outputs[out_name] if out_name in outputs else next(iter(outputs.values()))
        out_tile = postprocess_tensor(np.asarray(y))
        out_x2 = resize_to_x2(
            tile=out_tile,
            valid_h=int(job["valid_h"]),
            valid_w=int(job["valid_w"]),
            model_scale_h=model_scale_h,
            model_scale_w=model_scale_w,
            downsample_interp=downsample_interp,
        )
        cx0 = int(job["core_x0"]) * 2
        cy0 = int(job["core_y0"]) * 2
        cx1 = cx0 + int(job["core_out_w"])
        cy1 = cy0 + int(job["core_out_h"])
        out_core = out_x2[cy0:cy1, cx0:cx1]
        x0 = int(job["x0"])
        y0 = int(job["y0"])
        x1 = int(job["x1"])
        y1 = int(job["y1"])
        dst[y0 * 2 : y1 * 2, x0 * 2 : x1 * 2] = out_core

    def request_outputs(req_obj: Any) -> Any:
        try:
            return req_obj.results
        except Exception:
            tensor = req_obj.get_output_tensor(0)
            return {out_name: np.asarray(tensor.data)}

    for ty in range(tiles_y):
        for tx in range(tiles_x):
            x0 = tx * core_w
            y0 = ty * core_h
            x1 = min(x0 + core_w, src_w)
            y1 = min(y0 + core_h, src_h)

            wx0 = x0 - tile_overlap
            wy0 = y0 - tile_overlap
            wx1 = x1 + tile_overlap
            wy1 = y1 + tile_overlap

            rx0 = max(0, wx0)
            ry0 = max(0, wy0)
            rx1 = min(src_w, wx1)
            ry1 = min(src_h, wy1)

            patch = image[ry0:ry1, rx0:rx1]
            valid_h = ry1 - ry0
            valid_w = rx1 - rx0
            core_x0 = x0 - rx0
            core_y0 = y0 - ry0
            core_out_h = (y1 - y0) * 2
            core_out_w = (x1 - x0) * 2

            if valid_h != in_h or valid_w != in_w:
                patch = cv2.copyMakeBorder(
                    patch,
                    0,
                    max(0, in_h - valid_h),
                    0,
                    max(0, in_w - valid_w),
                    borderType=cv2.BORDER_REFLECT_101,
                )

            def _prev_index_from_role(role: str) -> int:
                r = str(role).lower()
                if r == "prev":
                    return 1
                if r.startswith("prev"):
                    suf = r[4:]
                    if suf.isdigit():
                        return max(1, int(suf))
                return 1

            def _slice_temporal_patch(prev_idx: int) -> np.ndarray:
                if temporal_images is None or prev_idx <= 0:
                    return patch
                k = prev_idx - 1
                if k >= len(temporal_images):
                    return patch
                src = temporal_images[k]
                if src is None:
                    return patch
                sh = int(src.shape[0]) if hasattr(src, "shape") and len(src.shape) >= 2 else 0
                sw = int(src.shape[1]) if hasattr(src, "shape") and len(src.shape) >= 2 else 0
                if sh != src_h or sw != src_w:
                    return patch
                p = src[ry0:ry1, rx0:rx1]
                if p.shape[0] != valid_h or p.shape[1] != valid_w:
                    return patch
                return p

            job = {
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "valid_h": valid_h,
                "valid_w": valid_w,
                "core_x0": core_x0,
                "core_y0": core_y0,
                "core_out_h": core_out_h,
                "core_out_w": core_out_w,
            }

            feed: Dict[str, np.ndarray] = {}
            for meta in ov_inputs:
                name = str(meta["name"])
                role = str(meta.get("role", "curr")).lower()
                c = int(meta.get("c", in_c))
                h = int(meta.get("h", in_h))
                w = int(meta.get("w", in_w))
                dtype = meta.get("dtype", in_dtype)
                if role.startswith("prev"):
                    src_patch = _slice_temporal_patch(_prev_index_from_role(role))
                else:
                    src_patch = patch
                feed[name] = to_nchw_tensor(
                    image_bgr=src_patch,
                    c=c,
                    h=h,
                    w=w,
                    dtype=dtype,
                )
            if not feed:
                feed[in_name] = to_nchw_tensor(
                    image_bgr=patch,
                    c=in_c,
                    h=in_h,
                    w=in_w,
                    dtype=in_dtype,
                )
            job["feed"] = feed

            if async_enabled:
                while not free_reqs:
                    req_done, job_done, t_started = inflight.pop(0)
                    req_done.wait()
                    infer_lat_ms.append((time.perf_counter() - t_started) * 1000.0)
                    finalize_tile(job_done, request_outputs(req_done))
                    free_reqs.append(req_done)

                req = free_reqs.pop(0)
                try:
                    t_started = time.perf_counter()
                    req.start_async(job["feed"])
                    inflight.append((req, job, t_started))
                    continue
                except Exception:
                    # If async path is unavailable for this backend/runtime, fall back to sync.
                    async_enabled = False
                    for req_done, job_done, t_start in inflight:
                        req_done.wait()
                        infer_lat_ms.append((time.perf_counter() - t_start) * 1000.0)
                        finalize_tile(job_done, request_outputs(req_done))
                        free_reqs.append(req_done)
                    inflight.clear()
                    free_reqs.insert(0, req)

            req_sync = free_reqs[0] if free_reqs else ov_ctx["req"]
            t0 = time.perf_counter()
            outputs = req_sync.infer(job["feed"])
            infer_lat_ms.append((time.perf_counter() - t0) * 1000.0)
            finalize_tile(job, outputs)

    for req_done, job_done, t_started in inflight:
        req_done.wait()
        infer_lat_ms.append((time.perf_counter() - t_started) * 1000.0)
        finalize_tile(job_done, request_outputs(req_done))

    avg_tile_ms = float(np.mean(infer_lat_ms)) if infer_lat_ms else 0.0
    return dst, avg_tile_ms


class OverlayWindow:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.win: Optional[tk.Toplevel] = None
        self.label: Optional[tk.Label] = None
        self.photo: Optional[ImageTk.PhotoImage] = None
        self.hwnd: int = 0
        self.visible = False

    def _ensure(self):
        if self.win is not None:
            return
        self.win = tk.Toplevel(self.root)
        self.win.withdraw()
        self.win.overrideredirect(True)
        self.win.attributes("-topmost", True)
        self.label = tk.Label(self.win, bd=0, highlightthickness=0)
        self.label.pack(fill=tk.BOTH, expand=True)
        self.win.update_idletasks()
        self.hwnd = int(self.win.winfo_id())

    def hide(self):
        if self.win is not None:
            self.win.withdraw()
        self.visible = False

    def show_frame(
        self,
        frame_bgr: np.ndarray,
        rect: Tuple[int, int, int, int],
        alpha: float,
        click_through: bool,
        exclude_from_capture: bool,
        target_hwnd: int = 0,
    ) -> None:
        self._ensure()
        assert self.win is not None
        assert self.label is not None

        left, top, right, bottom = rect
        w = max(1, int(right - left))
        h = max(1, int(bottom - top))
        if frame_bgr.shape[1] != w or frame_bgr.shape[0] != h:
            interp = cv2.INTER_AREA if (frame_bgr.shape[1] > w or frame_bgr.shape[0] > h) else cv2.INTER_CUBIC
            frame_bgr = cv2.resize(frame_bgr, (w, h), interpolation=interp)

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.label.configure(image=self.photo)

        self.win.geometry(f"{w}x{h}+{left}+{top}")
        self.win.attributes("-topmost", True)

        a = max(0.2, min(1.0, float(alpha)))
        if self.hwnd:
            ex_style = int(user32.GetWindowLongW(self.hwnd, GWL_EXSTYLE))
            ex_style |= WS_EX_LAYERED | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE
            if click_through:
                ex_style |= WS_EX_TRANSPARENT
            else:
                ex_style &= ~WS_EX_TRANSPARENT
            user32.SetWindowLongW(self.hwnd, GWL_EXSTYLE, ex_style)
            # Disable window input path entirely when click-through is enabled.
            try:
                user32.EnableWindow(self.hwnd, ctypes.c_bool(not click_through))
            except Exception:
                pass
            # Force style re-apply and keep topmost without activating focus.
            user32.SetWindowPos(
                self.hwnd,
                HWND_TOPMOST,
                left,
                top,
                w,
                h,
                SWP_NOACTIVATE | SWP_FRAMECHANGED,
            )
            user32.SetLayeredWindowAttributes(
                self.hwnd,
                0,
                int(round(a * 255.0)),
                LWA_ALPHA,
            )
            try:
                user32.SetWindowDisplayAffinity(
                    self.hwnd,
                    WDA_EXCLUDEFROMCAPTURE if exclude_from_capture else WDA_NONE,
                )
            except Exception:
                pass

            if click_through and target_hwnd > 0:
                try:
                    fg_hwnd = int(user32.GetForegroundWindow())
                    root_hwnd = int(self.root.winfo_id())
                    if fg_hwnd == self.hwnd or fg_hwnd == root_hwnd:
                        user32.SetForegroundWindow(int(target_hwnd))
                except Exception:
                    pass
        else:
            self.win.attributes("-alpha", a)

        if not self.visible:
            self.win.deiconify()
        self.visible = True


class LatestCapturePump:
    def __init__(
        self,
        get_cfg_fn,
        stop_event: threading.Event,
        buffer_size: int = CAPTURE_FRAME_BUFFER_SIZE,
    ):
        self.get_cfg_fn = get_cfg_fn
        self.stop_event = stop_event
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.buffer = deque(maxlen=max(1, int(buffer_size)))
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_rect: Optional[Tuple[int, int, int, int]] = None
        self.latest_id: int = 0
        self.latest_ts: float = 0.0
        self.last_error: str = ""
        self.last_error_ts: float = 0.0
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        if not self.thread.is_alive():
            self.thread.start()

    def join(self, timeout: float = 1.0) -> None:
        try:
            self.thread.join(timeout=max(0.0, float(timeout)))
        except Exception:
            pass

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                cfg = self.get_cfg_fn()
                hwnd = int(getattr(cfg, "hwnd", 0))
                backend = str(getattr(cfg, "capture_backend", DEFAULT_CAPTURE_BACKEND))
            except Exception:
                time.sleep(0.01)
                continue

            if hwnd <= 0:
                time.sleep(0.01)
                continue

            captured = capture_window_bgr(hwnd, backend=backend)
            now = time.perf_counter()
            if captured is None:
                self.last_error = "capture_failed"
                self.last_error_ts = now
                time.sleep(0.003)
                continue

            frame, rect = captured
            with self.cond:
                self.latest_id += 1
                frame_id = int(self.latest_id)
                self.buffer.append((frame_id, float(now), frame, rect))
                self.latest_frame = frame
                self.latest_rect = rect
                self.latest_ts = now
                self.cond.notify_all()
            self.last_error = ""

    def get_latest_after(
        self,
        min_id: int,
        wait_timeout_s: float = 0.008,
    ) -> Optional[Tuple[int, float, np.ndarray, Tuple[int, int, int, int]]]:
        timeout_s = max(0.0, float(wait_timeout_s))
        deadline = time.perf_counter() + timeout_s
        with self.cond:
            while True:
                if self.buffer:
                    latest = self.buffer[-1]
                    if int(latest[0]) > int(min_id):
                        return latest

                remain = deadline - time.perf_counter()
                if remain <= 0.0:
                    return None
                self.cond.wait(timeout=remain)

    def latest_error(self) -> str:
        return self.last_error


class LunaRealtimeWorker(threading.Thread):
    def __init__(
        self,
        config_lock: threading.Lock,
        config_holder: Dict[str, RuntimeConfig],
        status_q: queue.Queue,
        overlay_q: queue.Queue,
    ):
        super().__init__(daemon=True)
        self.config_lock = config_lock
        self.config_holder = config_holder
        self.status_q = status_q
        self.overlay_q = overlay_q
        self.stop_event = threading.Event()
        self.bench_fp: Optional[Any] = None
        self.bench_writer: Optional[csv.writer] = None
        self.bench_path: str = ""
        self.bench_rows_since_flush: int = 0

    def stop(self):
        self.stop_event.set()

    def _current_config(self) -> RuntimeConfig:
        with self.config_lock:
            return self.config_holder["cfg"]

    def _push_status(self, payload: Dict[str, object]) -> None:
        try:
            self.status_q.put_nowait(payload)
        except queue.Full:
            pass

    def _push_overlay(self, payload: Dict[str, object]) -> None:
        try:
            self.overlay_q.put_nowait(payload)
        except queue.Full:
            try:
                self.overlay_q.get_nowait()
            except Exception:
                pass
            try:
                self.overlay_q.put_nowait(payload)
            except Exception:
                pass

    def _close_benchmark_writer(self) -> None:
        fp = self.bench_fp
        self.bench_fp = None
        self.bench_writer = None
        self.bench_path = ""
        self.bench_rows_since_flush = 0
        if fp is not None:
            try:
                fp.flush()
            except Exception:
                pass
            try:
                fp.close()
            except Exception:
                pass

    def _ensure_benchmark_writer(self, cfg: RuntimeConfig) -> None:
        if not bool(cfg.benchmark_enabled):
            self._close_benchmark_writer()
            return

        raw = (cfg.benchmark_csv or DEFAULT_BENCHMARK_CSV).strip()
        if not raw:
            raw = DEFAULT_BENCHMARK_CSV
        bench_path = Path(raw).expanduser()
        if not bench_path.is_absolute():
            bench_path = (Path.cwd() / bench_path).resolve()

        if self.bench_fp is not None and self.bench_path == str(bench_path):
            return

        self._close_benchmark_writer()
        bench_path.parent.mkdir(parents=True, exist_ok=True)
        has_data = bool(bench_path.exists() and bench_path.stat().st_size > 0)
        fp = bench_path.open("a", newline="", encoding="utf-8")
        writer = csv.writer(fp)
        self.bench_fp = fp
        self.bench_writer = writer
        self.bench_path = str(bench_path)
        self.bench_rows_since_flush = 0
        if not has_data:
            writer.writerow(
                [
                    "ts_iso",
                    "epoch_ms",
                    "frame_idx",
                    "hwnd",
                    "backend",
                    "device",
                    "capture",
                    "output_preset",
                    "use_sr",
                    "fg_enabled",
                    "fg_interp_only",
                    "ov_scale",
                    "ov_parallel_reqs",
                    "ov_reactive",
                    "ov_target_fps",
                    "temporal_on",
                    "temporal_strength",
                    "input_w",
                    "input_h",
                    "output_w",
                    "output_h",
                    "fps",
                    "loop_ms",
                    "capture_age_ms",
                    "frame_ms",
                    "infer_ms",
                    "fg_ms",
                    "temporal_ms",
                    "tile_ms",
                ]
            )
            fp.flush()
        self._push_status({"benchmark_note": f"Benchmark logging: {self.bench_path}"})

    def _write_benchmark_row(
        self,
        cfg: RuntimeConfig,
        frame_idx: int,
        use_sr: bool,
        frame: np.ndarray,
        out: np.ndarray,
        fps: float,
        loop_ms: float,
        capture_age_ms: float,
        elapsed_ms: float,
        infer_ms: float,
        fg_ms: float,
        temporal_ms: float,
        tile_ms: float,
        ov_scale: float,
        ov_parallel_reqs: int,
    ) -> None:
        writer = self.bench_writer
        fp = self.bench_fp
        if writer is None or fp is None:
            return
        now = time.time()
        writer.writerow(
            [
                datetime.now().isoformat(timespec="milliseconds"),
                int(round(now * 1000.0)),
                int(frame_idx),
                int(cfg.hwnd),
                str(cfg.backend),
                str(cfg.device),
                str(cfg.capture_backend),
                str(cfg.output_preset),
                int(bool(use_sr)),
                int(bool(cfg.fg_enabled)),
                int(bool(cfg.fg_interp_only)),
                f"{float(ov_scale):.4f}",
                int(ov_parallel_reqs),
                int(bool(cfg.ov_reactive_scale)),
                f"{float(cfg.ov_reactive_target_fps):.2f}",
                int(bool(cfg.temporal_restore)),
                f"{float(cfg.temporal_strength):.3f}",
                int(frame.shape[1]),
                int(frame.shape[0]),
                int(out.shape[1]),
                int(out.shape[0]),
                f"{float(fps):.4f}",
                f"{float(loop_ms):.4f}",
                f"{float(capture_age_ms):.4f}",
                f"{float(elapsed_ms):.4f}",
                f"{float(infer_ms):.4f}",
                f"{float(fg_ms):.4f}",
                f"{float(temporal_ms):.4f}",
                f"{float(tile_ms):.4f}",
            ]
        )
        self.bench_rows_since_flush += 1
        if self.bench_rows_since_flush >= 30:
            fp.flush()
            self.bench_rows_since_flush = 0

    def run(self):
        frame_idx = 0
        last_tick = time.perf_counter()
        seed_base = int(time.time() * 1000.0) & 0xFFFFFFFF
        capture_pump = LatestCapturePump(self._current_config, self.stop_event)
        capture_pump.start()
        last_capture_id = 0
        last_capture_warn_ts = 0.0

        runtime_key = None
        luna_runtime: Optional[Dict[str, Any]] = None
        ov_runtime: Optional[Dict[str, Any]] = None
        fg_runtime: Optional[Dict[str, Any]] = None
        fg_prev_frame: Optional[np.ndarray] = None
        ov_prev_work_frames: List[np.ndarray] = []
        fg_failure_note_shown = False
        temporal_prev_out: Optional[np.ndarray] = None
        temporal_failure_note_shown = False
        bench_failure_note_shown = False
        tuned = False
        fallback_stage = 0
        reactive_scale_cur = float(DEFAULT_OV_INTERNAL_SCALE)
        reactive_ema_ms = 0.0
        reactive_last_adjust_frame = -1000

        while not self.stop_event.is_set():
            cfg = self._current_config()
            if cfg.hwnd <= 0:
                time.sleep(0.05)
                continue

            key = (
                cfg.backend,
                cfg.capture_backend,
                cfg.preset,
                cfg.output_preset,
                bool(cfg.gpu_video_path),
                round(cfg.frame_budget_ms, 3),
                cfg.device,
                cfg.ov_model,
                cfg.ov_preset,
                round(cfg.ov_internal_scale, 3),
                bool(cfg.ov_reactive_scale),
                round(cfg.ov_reactive_target_fps, 3),
                bool(cfg.temporal_restore),
                round(cfg.temporal_strength, 3),
                bool(cfg.benchmark_enabled),
                str(cfg.benchmark_csv),
                bool(cfg.strict_npu_only),
                bool(cfg.strict_gpu_only),
                cfg.ov_cache_dir,
                bool(cfg.allow_cpu_fallback),
                round(cfg.overlay_alpha, 2),
                bool(cfg.overlay_click_through),
                bool(cfg.overlay_exclude_from_capture),
                bool(cfg.overlay_fullscreen_upscale),
                bool(cfg.fg_enabled),
                bool(cfg.fg_interp_only),
                cfg.fg_model,
                cfg.fg_precision,
                int(cfg.fg_size),
                round(cfg.fg_timestep, 3),
            )
            if key != runtime_key:
                runtime_key = key
                tuned = False
                fallback_stage = 0
                reactive_scale_cur = max(0.50, min(2.0, float(cfg.ov_internal_scale)))
                reactive_ema_ms = 0.0
                reactive_last_adjust_frame = -1000
                fg_prev_frame = None
                ov_prev_work_frames = []
                fg_failure_note_shown = False
                temporal_prev_out = None
                temporal_failure_note_shown = False
                bench_failure_note_shown = False
                last_capture_warn_ts = 0.0
                try:
                    self._ensure_benchmark_writer(cfg)
                except Exception as e:
                    self._close_benchmark_writer()
                    self._push_status({"warning": f"Benchmark logger setup failed: {e}"})
                backend_note = ""
                if cfg.backend == "lunasr":
                    try:
                        luna_runtime = build_luna_runtime(cfg.preset, cfg.frame_budget_ms)
                        ov_runtime = None
                        backend_note = str(luna_runtime["note"])
                    except Exception as e:
                        self._push_status({"error": f"LunaSR setup failed: {e}"})
                        time.sleep(0.2)
                        continue
                elif cfg.backend == "openvino_sr":
                    try:
                        ov_runtime = compile_ov_runtime(
                            model_path=cfg.ov_model,
                            device=cfg.device,
                            preset=cfg.ov_preset,
                            cache_dir=cfg.ov_cache_dir,
                            allow_cpu_fallback=bool(cfg.allow_cpu_fallback),
                            strict_npu_only=bool(cfg.strict_npu_only),
                            strict_gpu_only=bool(cfg.strict_gpu_only),
                        )
                        luna_runtime = None
                        backend_note = (
                            f"backend=openvino_sr, device={cfg.device}, preset={cfg.ov_preset}, "
                            f"capture={cfg.capture_backend}, output={cfg.output_preset}, "
                            f"gpu_video={'cuda' if (cfg.gpu_video_path and CUDA_AVAILABLE) else 'cpu'}, "
                            f"reactive={'on' if cfg.ov_reactive_scale else 'off'}@{cfg.ov_reactive_target_fps:.0f}fps, "
                            f"temporal={'on' if cfg.temporal_restore else 'off'}@{cfg.temporal_strength:.2f}, "
                            f"dpi={DPI_AWARENESS_MODE}, "
                            f"npu_i8={'on' if 'NPU' in str(ov_runtime['compile_device']).upper() else 'off'}, "
                            f"gpu_tile_base_720p={'on' if 'GPU' in str(ov_runtime['compile_device']).upper() else 'off'}, "
                            f"cpu_fallback={'on' if cfg.allow_cpu_fallback else 'off'}, "
                            f"strict_npu={'on' if cfg.strict_npu_only else 'off'}, "
                            f"strict_gpu={'on' if cfg.strict_gpu_only else 'off'}, "
                            f"temporal_model={'on' if ov_runtime.get('temporal_model', False) else 'off'}, "
                            f"model={ov_runtime['model_path']}, compile_device={ov_runtime['compile_device']}, "
                            f"exec={ov_runtime['exec_devices']}, "
                            f"req_pool={ov_runtime.get('ov_parallel_reqs', 1)}, "
                            f"compile_ms={ov_runtime['compile_ms']:.1f}"
                        )
                        if ov_runtime["compile_note"]:
                            backend_note = backend_note + f" | {ov_runtime['compile_note']}"
                    except Exception as e:
                        self._push_status({"error": f"OpenVINO setup failed: {e}"})
                        time.sleep(0.3)
                        continue
                else:
                    self._push_status({"error": f"Unknown backend: {cfg.backend}"})
                    time.sleep(0.2)
                    continue

                fg_note = "fg=off"
                if cfg.fg_enabled:
                    try:
                        fg_runtime = compile_fg_runtime(
                            model_path=cfg.fg_model,
                            device=cfg.device,
                            cache_dir=cfg.ov_cache_dir,
                            precision=cfg.fg_precision,
                            fg_size=cfg.fg_size,
                            allow_cpu_fallback=bool(cfg.allow_cpu_fallback),
                            strict_npu_only=bool(cfg.strict_npu_only),
                            strict_gpu_only=bool(cfg.strict_gpu_only),
                        )
                        fg_note = (
                            f"fg=on model={fg_runtime['model_path']} "
                            f"prec={fg_runtime['precision']} size={fg_runtime['fg_size']} "
                            f"compile_device={fg_runtime['compile_device']} "
                            f"exec={fg_runtime['exec_devices']} "
                            f"compile_ms={fg_runtime['compile_ms']:.1f}"
                        )
                        if fg_runtime["compile_note"]:
                            fg_note = fg_note + f" | {fg_runtime['compile_note']}"
                        if cfg.fg_interp_only:
                            fg_note = fg_note + " mode=interp_only"
                    except Exception as e:
                        fg_runtime = None
                        fg_note = f"fg=on setup_failed: {e}"
                        self._push_status({"warning": f"FG setup failed, fallback=blend: {e}"})
                else:
                    fg_runtime = None

                self._push_status({"profile_note": f"{backend_note} | {fg_note}"})

            latest = capture_pump.get_latest_after(last_capture_id, wait_timeout_s=0.010)
            if latest is None:
                now_wait = time.perf_counter()
                if (now_wait - last_capture_warn_ts) >= 1.0:
                    cap_err = capture_pump.latest_error()
                    if cap_err:
                        self._push_status({"warning": f"Capture waiting: {cap_err}"})
                    else:
                        self._push_status({"warning": "Capture waiting: no new frame yet."})
                    last_capture_warn_ts = now_wait
                continue
            capture_id, capture_ts, frame, src_rect = latest
            last_capture_id = capture_id
            capture_age_ms = max(0.0, (time.perf_counter() - float(capture_ts)) * 1000.0)

            start = time.perf_counter()
            out = frame
            infer_ms = 0.0
            tile_ms = 0.0
            fg_ms = 0.0
            temporal_ms = 0.0
            work_frame = frame
            use_gpu_video = bool(cfg.gpu_video_path and CUDA_AVAILABLE and cfg.backend == "openvino_sr")
            target_w, target_h = resolve_target_size_from_preset(
                src_w=int(frame.shape[1]),
                src_h=int(frame.shape[0]),
                preset=cfg.output_preset,
            )

            if cfg.fg_enabled:
                if fg_prev_frame is not None:
                    if fg_runtime is not None:
                        try:
                            fg_mid, fg_ms = run_fg_mid(
                                frame0=fg_prev_frame,
                                frame1=frame,
                                fg_ctx=fg_runtime,
                                timestep=float(cfg.fg_timestep),
                            )
                            if fg_mid.shape[1] != frame.shape[1] or fg_mid.shape[0] != frame.shape[0]:
                                fg_mid = resize_bgr(
                                    fg_mid,
                                    (frame.shape[1], frame.shape[0]),
                                    cv2.INTER_CUBIC,
                                    use_cuda=use_gpu_video,
                                )
                            work_frame = fg_mid
                        except Exception as e:
                            work_frame = cv2.addWeighted(fg_prev_frame, 0.5, frame, 0.5, 0.0)
                            if not fg_failure_note_shown:
                                self._push_status({"warning": f"FG infer failed, fallback=blend: {e}"})
                                fg_failure_note_shown = True
                    else:
                        work_frame = cv2.addWeighted(fg_prev_frame, 0.5, frame, 0.5, 0.0)
                fg_prev_frame = frame.copy()
            else:
                fg_prev_frame = None

            use_sr = bool(cfg.upscale_enabled and not cfg.fg_interp_only)

            if use_sr:
                if cfg.backend == "lunasr":
                    if luna_runtime is None:
                        time.sleep(0.05)
                        continue

                    if luna_runtime["auto_tune_on"] and not tuned:
                        try:
                            y = cv2.cvtColor(work_frame, cv2.COLOR_BGR2YCrCb).astype(np.float32)[:, :, 0] / 255.0
                            tuned_params, _ = auto_tune(
                                y,
                                luna_runtime["params"],
                                strategy=luna_runtime["tune_strategy"],
                                is_video=True,
                            )
                            luna_runtime["params"] = tuned_params
                        except Exception:
                            pass
                        tuned = True

                    if luna_runtime["tune_strategy"] == "c":
                        seed = seed_base + (frame_idx // 120)
                    else:
                        seed = seed_base + frame_idx

                    target_w, target_h = resolve_target_size_from_preset(
                        src_w=int(work_frame.shape[1]),
                        src_h=int(work_frame.shape[0]),
                        preset=cfg.output_preset,
                    )
                    target_scale = float(target_h) / float(max(1, int(work_frame.shape[0])))

                    t0 = time.perf_counter()
                    out = lunasr_upscale_bgr_internal(
                        frame_bgr=work_frame,
                        scale=target_scale,
                        p=luna_runtime["params"],
                        seed=seed,
                        internal_scale=float(luna_runtime["internal_scale"]),
                    )
                    t1 = time.perf_counter()
                    infer_ms = (t1 - t0) * 1000.0

                    frame_budget_ms = float(luna_runtime["frame_budget_ms"])
                    if frame_budget_ms > 0.0 and infer_ms > frame_budget_ms:
                        new_p, fallback_stage, msg = apply_performance_fallback_step(
                            luna_runtime["params"], fallback_stage
                        )
                        luna_runtime["params"] = new_p
                        if msg != "fallback_step_done":
                            self._push_status({"fallback": msg})

                else:
                    if ov_runtime is None:
                        time.sleep(0.05)
                        continue

                    target_w, target_h = resolve_target_size_from_preset(
                        src_w=int(work_frame.shape[1]),
                        src_h=int(work_frame.shape[0]),
                        preset=cfg.output_preset,
                    )
                    if bool(cfg.ov_reactive_scale):
                        ov_internal_scale = max(0.50, min(2.0, float(reactive_scale_cur)))
                    else:
                        ov_internal_scale = max(0.50, min(2.0, float(cfg.ov_internal_scale)))
                    base_in_h = max(1, int(round(target_h * 0.5)))
                    base_in_w = max(1, int(round(target_w * 0.5)))
                    infer_h = max(1, int(round(base_in_h * ov_internal_scale)))
                    infer_w = max(1, int(round(base_in_w * ov_internal_scale)))
                    if "GPU" in str(ov_runtime.get("compile_device", "")).upper():
                        infer_w, infer_h = clamp_size_preserve_aspect(
                            src_w=infer_w,
                            src_h=infer_h,
                            max_w=GPU_TILE_BASE_MAX_W,
                            max_h=GPU_TILE_BASE_MAX_H,
                        )
                    if infer_h != work_frame.shape[0] or infer_w != work_frame.shape[1]:
                        interp_in = cv2.INTER_AREA if (
                            infer_h < work_frame.shape[0] or infer_w < work_frame.shape[1]
                        ) else cv2.INTER_CUBIC
                        ov_in = resize_bgr(
                            work_frame,
                            (infer_w, infer_h),
                            interp_in,
                            use_cuda=use_gpu_video,
                        )
                    else:
                        ov_in = work_frame

                    ov_temporal_inputs: Optional[List[np.ndarray]] = None
                    if bool(ov_runtime.get("temporal_model", False)):
                        prev_depth = max(1, int(ov_runtime.get("temporal_prev_depth", 1)))
                        ov_temporal_inputs = []
                        strength_base = max(0.0, min(1.0, float(cfg.temporal_strength)))
                        for hist_idx in range(prev_depth):
                            if not bool(cfg.temporal_restore):
                                prev_base = work_frame
                            elif hist_idx < len(ov_prev_work_frames):
                                prev_base = ov_prev_work_frames[hist_idx]
                            elif ov_prev_work_frames:
                                prev_base = ov_prev_work_frames[-1]
                            else:
                                prev_base = work_frame

                            if prev_base.shape[1] != work_frame.shape[1] or prev_base.shape[0] != work_frame.shape[0]:
                                prev_base = resize_bgr(
                                    prev_base,
                                    (work_frame.shape[1], work_frame.shape[0]),
                                    cv2.INTER_CUBIC,
                                    use_cuda=use_gpu_video,
                                )
                            if bool(cfg.temporal_restore):
                                # Decay strength for older history to reduce trailing artifacts.
                                t_strength = strength_base * (0.70 ** hist_idx)
                                if t_strength < 0.999:
                                    prev_base = cv2.addWeighted(
                                        work_frame,
                                        1.0 - t_strength,
                                        prev_base,
                                        t_strength,
                                        0.0,
                                    )
                            if prev_base.shape[1] != infer_w or prev_base.shape[0] != infer_h:
                                prev_interp = cv2.INTER_AREA if (
                                    infer_h < prev_base.shape[0] or infer_w < prev_base.shape[1]
                                ) else cv2.INTER_CUBIC
                                prev_base = resize_bgr(
                                    prev_base,
                                    (infer_w, infer_h),
                                    prev_interp,
                                    use_cuda=use_gpu_video,
                                )
                            ov_temporal_inputs.append(prev_base)

                    t0 = time.perf_counter()
                    out_internal, tile_ms = run_ov_infer_tiled(ov_in, ov_runtime, temporal_images=ov_temporal_inputs)
                    out = out_internal
                    if out_internal.shape[1] != target_w or out_internal.shape[0] != target_h:
                        interp_target = (
                            cv2.INTER_AREA
                            if (out_internal.shape[1] > target_w or out_internal.shape[0] > target_h)
                            else cv2.INTER_CUBIC
                        )
                        out = resize_bgr(
                            out_internal,
                            (target_w, target_h),
                            interp_target,
                            use_cuda=use_gpu_video,
                        )
                    t1 = time.perf_counter()
                    infer_ms = (t1 - t0) * 1000.0
                    if bool(ov_runtime.get("temporal_model", False)):
                        prev_depth = max(1, int(ov_runtime.get("temporal_prev_depth", 1)))
                        ov_prev_work_frames.insert(0, work_frame.copy())
                        if len(ov_prev_work_frames) > prev_depth:
                            del ov_prev_work_frames[prev_depth:]
                    else:
                        ov_prev_work_frames = []
            elif cfg.fg_enabled:
                # FG interpolation only path (no SR upscale).
                out = work_frame
                ov_prev_work_frames = []
            else:
                ov_prev_work_frames = []

            temporal_in_model = bool(
                use_sr
                and cfg.backend == "openvino_sr"
                and ov_runtime is not None
                and bool(ov_runtime.get("temporal_model", False))
            )
            if use_sr and bool(cfg.temporal_restore) and not temporal_in_model:
                raw_out = out
                if temporal_prev_out is not None:
                    try:
                        out, temporal_ms = temporal_restore_bgr(
                            prev_bgr=temporal_prev_out,
                            cur_bgr=raw_out,
                            strength=float(cfg.temporal_strength),
                        )
                    except Exception as e:
                        out = raw_out
                        temporal_ms = 0.0
                        if not temporal_failure_note_shown:
                            self._push_status({"warning": f"Temporal restore failed, fallback=off: {e}"})
                            temporal_failure_note_shown = True
                temporal_prev_out = raw_out.copy()
            else:
                temporal_prev_out = None

            elapsed_ms = (time.perf_counter() - start) * 1000.0
            now = time.perf_counter()
            loop_ms = (now - last_tick) * 1000.0
            last_tick = now
            fps = (1000.0 / loop_ms) if loop_ms > 0 else 0.0

            if use_sr and cfg.backend == "openvino_sr" and bool(cfg.ov_reactive_scale):
                target_fps = max(20.0, min(240.0, float(cfg.ov_reactive_target_fps)))
                target_ms = 1000.0 / target_fps
                if reactive_ema_ms <= 0.0:
                    reactive_ema_ms = elapsed_ms
                else:
                    reactive_ema_ms = (reactive_ema_ms * 0.85) + (elapsed_ms * 0.15)

                if (frame_idx - reactive_last_adjust_frame) >= 4:
                    new_scale = float(reactive_scale_cur)
                    if reactive_ema_ms > target_ms * 1.08:
                        new_scale -= 0.08
                    elif reactive_ema_ms < target_ms * 0.90:
                        new_scale += 0.04
                    new_scale = max(0.50, min(2.0, new_scale))
                    if abs(new_scale - reactive_scale_cur) >= 0.009:
                        reactive_scale_cur = new_scale
                        reactive_last_adjust_frame = frame_idx

            overlay_rect = src_rect
            if cfg.overlay_fullscreen_upscale and use_sr:
                monitor_rect = get_monitor_rect_for_window(cfg.hwnd)
                if monitor_rect is not None:
                    overlay_rect = monitor_rect
            elif use_sr:
                monitor_rect = get_monitor_rect_for_window(cfg.hwnd)
                overlay_rect = build_centered_overlay_rect(
                    src_rect=src_rect,
                    target_w=target_w,
                    target_h=target_h,
                    bounds_rect=monitor_rect,
                )

            ow = max(1, int(overlay_rect[2] - overlay_rect[0]))
            oh = max(1, int(overlay_rect[3] - overlay_rect[1]))
            if out.shape[1] != ow or out.shape[0] != oh:
                interp = cv2.INTER_AREA if (out.shape[1] > ow or out.shape[0] > oh) else cv2.INTER_CUBIC
                overlay_frame = resize_bgr(
                    out,
                    (ow, oh),
                    interp,
                    use_cuda=use_gpu_video,
                )
            else:
                overlay_frame = out
            self._push_overlay(
                {
                    "frame": overlay_frame,
                    "rect": overlay_rect,
                    "alpha": float(cfg.overlay_alpha),
                    "click_through": True,
                    "exclude_from_capture": bool(cfg.overlay_exclude_from_capture),
                    "target_hwnd": int(cfg.hwnd),
                }
            )

            ov_scale_now = reactive_scale_cur if bool(cfg.ov_reactive_scale) else float(cfg.ov_internal_scale)
            ov_parallel_reqs_now = int(ov_runtime.get("ov_parallel_reqs", 1)) if ov_runtime is not None else 1
            if bool(cfg.benchmark_enabled):
                try:
                    self._write_benchmark_row(
                        cfg=cfg,
                        frame_idx=frame_idx,
                        use_sr=use_sr,
                        frame=frame,
                        out=out,
                        fps=fps,
                        loop_ms=loop_ms,
                        capture_age_ms=capture_age_ms,
                        elapsed_ms=elapsed_ms,
                        infer_ms=infer_ms,
                        fg_ms=fg_ms,
                        temporal_ms=temporal_ms,
                        tile_ms=tile_ms,
                        ov_scale=ov_scale_now,
                        ov_parallel_reqs=ov_parallel_reqs_now,
                    )
                except Exception as e:
                    self._close_benchmark_writer()
                    if not bench_failure_note_shown:
                        self._push_status({"warning": f"Benchmark logger write failed: {e}"})
                        bench_failure_note_shown = True

            self._push_status(
                {
                    "fps": fps,
                    "frame_ms": elapsed_ms,
                    "infer_ms": infer_ms,
                    "fg_ms": fg_ms,
                    "capture_age_ms": capture_age_ms,
                    "temporal_ms": temporal_ms,
                    "tile_ms": tile_ms,
                    "backend": cfg.backend,
                    "device": cfg.device,
                    "ov_scale": ov_scale_now,
                    "ov_reactive": bool(cfg.ov_reactive_scale),
                    "ov_target_fps": float(cfg.ov_reactive_target_fps),
                    "temporal_on": bool(cfg.temporal_restore),
                    "temporal_strength": float(cfg.temporal_strength),
                    "fg_enabled": bool(cfg.fg_enabled),
                    "input_shape": f"{frame.shape[1]}x{frame.shape[0]}",
                    "output_shape": f"{out.shape[1]}x{out.shape[0]}",
                }
            )
            frame_idx += 1
        capture_pump.join(timeout=1.0)
        self._close_benchmark_writer()
        self._push_status({"overlay_hide": True})
        self._push_status({"stopped": True})


class LunaRealtimeGui:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("LunaSR Realtime GUI")
        self.root.geometry("1120x400")

        self.config_lock = threading.Lock()
        self.config_holder: Dict[str, RuntimeConfig] = {
            "cfg": RuntimeConfig(
                hwnd=0,
                backend="openvino_sr",
                capture_backend=DEFAULT_CAPTURE_BACKEND,
                preset="max_performance",
                output_preset=DEFAULT_OUTPUT_PRESET,
                gpu_video_path=DEFAULT_GPU_VIDEO_PATH,
                upscale_enabled=True,
                frame_budget_ms=100.0,
                device="NPU",
                ov_model=DEFAULT_OV_MODEL,
                ov_preset=DEFAULT_OV_PRESET,
                ov_internal_scale=DEFAULT_OV_INTERNAL_SCALE,
                ov_reactive_scale=True,
                ov_reactive_target_fps=60.0,
                temporal_restore=DEFAULT_TEMPORAL_RESTORE,
                temporal_strength=DEFAULT_TEMPORAL_STRENGTH,
                benchmark_enabled=DEFAULT_BENCHMARK_ENABLED,
                benchmark_csv=DEFAULT_BENCHMARK_CSV,
                strict_npu_only=DEFAULT_STRICT_NPU_ONLY,
                strict_gpu_only=DEFAULT_STRICT_GPU_ONLY,
                ov_cache_dir=DEFAULT_OV_CACHE_DIR,
                allow_cpu_fallback=False,
                overlay_alpha=1.0,
                overlay_click_through=True,
                overlay_exclude_from_capture=True,
                overlay_fullscreen_upscale=False,
                fg_enabled=False,
                fg_interp_only=False,
                fg_model=DEFAULT_FG_MODEL,
                fg_precision="f16",
                fg_size=256,
                fg_timestep=0.50,
            )
        }
        self.status_q: queue.Queue = queue.Queue(maxsize=64)
        self.overlay_q: queue.Queue = queue.Queue(maxsize=1)
        self.worker: Optional[LunaRealtimeWorker] = None
        self.overlay_window = OverlayWindow(self.root)
        self.window_map: Dict[str, int] = {}

        self.target_var = tk.StringVar()
        self.backend_var = tk.StringVar(value="openvino_sr")
        self.capture_backend_var = tk.StringVar(value=DEFAULT_CAPTURE_BACKEND)
        self.preset_var = tk.StringVar(value="max_performance")
        self.output_preset_var = tk.StringVar(value=DEFAULT_OUTPUT_PRESET)
        self.gpu_video_path_var = tk.BooleanVar(value=DEFAULT_GPU_VIDEO_PATH)
        self.upscale_var = tk.BooleanVar(value=True)
        self.frame_budget_var = tk.StringVar(value="100")

        self.device_var = tk.StringVar(value="NPU")
        self.ov_model_var = tk.StringVar(value=DEFAULT_OV_MODEL)
        self.ov_preset_var = tk.StringVar(value=DEFAULT_OV_PRESET)
        self.ov_internal_scale_var = tk.StringVar(value=f"{DEFAULT_OV_INTERNAL_SCALE:.2f}")
        self.ov_reactive_var = tk.BooleanVar(value=True)
        self.ov_target_fps_var = tk.StringVar(value="60")
        self.temporal_restore_var = tk.BooleanVar(value=DEFAULT_TEMPORAL_RESTORE)
        self.temporal_strength_var = tk.StringVar(value=f"{DEFAULT_TEMPORAL_STRENGTH:.2f}")
        self.benchmark_var = tk.BooleanVar(value=DEFAULT_BENCHMARK_ENABLED)
        self.benchmark_csv_var = tk.StringVar(value=DEFAULT_BENCHMARK_CSV)
        self.strict_npu_only_var = tk.BooleanVar(value=DEFAULT_STRICT_NPU_ONLY)
        self.strict_gpu_only_var = tk.BooleanVar(value=DEFAULT_STRICT_GPU_ONLY)
        self.ov_cache_var = tk.StringVar(value=DEFAULT_OV_CACHE_DIR)
        self.allow_cpu_fallback_var = tk.BooleanVar(value=False)
        self.overlay_alpha_var = tk.StringVar(value="1.00")
        self.overlay_click_var = tk.BooleanVar(value=True)
        self.overlay_exclude_var = tk.BooleanVar(value=True)
        self.overlay_fullscreen_var = tk.BooleanVar(value=False)
        self.fg_var = tk.BooleanVar(value=False)
        self.fg_interp_only_var = tk.BooleanVar(value=False)
        self.fg_model_var = tk.StringVar(value=DEFAULT_FG_MODEL)
        self.fg_precision_var = tk.StringVar(value="f16")
        self.fg_size_var = tk.StringVar(value="256")
        self.fg_timestep_var = tk.StringVar(value="0.50")

        self.status_main_var = tk.StringVar(value="Select target window and click Start.")
        self.status_perf_var = tk.StringVar(
            value="fps=0.00 | cap_age_ms=0.0 | frame_ms=0.0 | infer_ms=0.0 | fg_ms=0.0 | temporal_ms=0.0 | tile_ms=0.0"
        )
        self.status_shape_var = tk.StringVar(value="input=- | output=-")
        self.profile_note_var = tk.StringVar(
            value=f"backend=openvino_sr | dpi={DPI_AWARENESS_MODE} | cuda={'on' if CUDA_AVAILABLE else 'off'}"
        )

        self._build_widgets()
        self.root.bind_all("<Alt-p>", self._on_alt_p_hotkey)
        self.root.bind_all("<Alt-P>", self._on_alt_p_hotkey)
        self.refresh_window_list()
        self._sync_runtime_config()
        self._poll_status_queue()
        self._poll_overlay_queue()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_widgets(self):
        frame = ttk.Frame(self.root, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)

        row0 = ttk.Frame(frame)
        row0.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(row0, text="Target Window").pack(side=tk.LEFT)
        self.target_combo = ttk.Combobox(row0, textvariable=self.target_var, state="readonly", width=92)
        self.target_combo.pack(side=tk.LEFT, padx=(8, 8), fill=tk.X, expand=True)
        self.target_combo.bind("<<ComboboxSelected>>", self._on_control_changed)
        ttk.Button(row0, text="Refresh", command=self.refresh_window_list).pack(side=tk.LEFT)

        row1 = ttk.Frame(frame)
        row1.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(row1, text="Output").pack(side=tk.LEFT)
        self.output_preset_combo = ttk.Combobox(
            row1,
            textvariable=self.output_preset_var,
            values=["AUTO", "HD", "FHD", "QHD", "4K"],
            state="readonly",
            width=7,
        )
        self.output_preset_combo.pack(side=tk.LEFT, padx=(8, 14))
        self.output_preset_combo.bind("<<ComboboxSelected>>", self._on_control_changed)

        self.toggle_btn = ttk.Checkbutton(
            row1,
            text="Upscale ON",
            variable=self.upscale_var,
            command=self._on_toggle_clicked,
        )
        self.toggle_btn.pack(side=tk.LEFT, padx=(0, 14))

        self.fg_toggle_btn = ttk.Checkbutton(
            row1,
            text="FG OFF",
            variable=self.fg_var,
            command=self._on_fg_toggle_clicked,
        )
        self.fg_toggle_btn.pack(side=tk.LEFT, padx=(0, 14))

        self.fg_only_check = ttk.Checkbutton(
            row1,
            text="FG Interp Only",
            variable=self.fg_interp_only_var,
            command=self._on_control_changed,
        )
        self.fg_only_check.pack(side=tk.LEFT, padx=(0, 14))

        ttk.Label(row1, text="Frame Budget ms").pack(side=tk.LEFT)
        self.frame_budget_entry = ttk.Entry(row1, textvariable=self.frame_budget_var, width=8)
        self.frame_budget_entry.pack(side=tk.LEFT, padx=(8, 0))
        self.frame_budget_entry.bind("<FocusOut>", self._on_control_changed)

        row2 = ttk.Frame(frame)
        row2.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(row2, text="Device").pack(side=tk.LEFT)
        self.device_combo = ttk.Combobox(
            row2,
            textvariable=self.device_var,
            values=["AUTO", "NPU", "GPU", "CPU"],
            state="readonly",
            width=8,
        )
        self.device_combo.pack(side=tk.LEFT, padx=(8, 14))
        self.device_combo.bind("<<ComboboxSelected>>", self._on_control_changed)

        ttk.Label(row2, text="Capture").pack(side=tk.LEFT)
        self.capture_backend_combo = ttk.Combobox(
            row2,
            textvariable=self.capture_backend_var,
            values=["gdi", "pil"],
            state="readonly",
            width=6,
        )
        self.capture_backend_combo.pack(side=tk.LEFT, padx=(8, 14))
        self.capture_backend_combo.bind("<<ComboboxSelected>>", self._on_control_changed)

        self.gpu_video_check = ttk.Checkbutton(
            row2,
            text="GPU Video Path",
            variable=self.gpu_video_path_var,
            command=self._on_control_changed,
        )
        self.gpu_video_check.pack(side=tk.LEFT, padx=(0, 14))

        ttk.Label(row2, text="OV Preset").pack(side=tk.LEFT)
        self.ov_preset_combo = ttk.Combobox(
            row2,
            textvariable=self.ov_preset_var,
            values=["speed", "quality"],
            state="readonly",
            width=9,
        )
        self.ov_preset_combo.pack(side=tk.LEFT, padx=(8, 14))
        self.ov_preset_combo.bind("<<ComboboxSelected>>", self._on_control_changed)

        ttk.Label(row2, text="OV Internal Scale").pack(side=tk.LEFT)
        self.ov_internal_entry = ttk.Entry(row2, textvariable=self.ov_internal_scale_var, width=7)
        self.ov_internal_entry.pack(side=tk.LEFT, padx=(8, 14))
        self.ov_internal_entry.bind("<FocusOut>", self._on_control_changed)

        self.ov_reactive_check = ttk.Checkbutton(
            row2,
            text="Reactive Scale",
            variable=self.ov_reactive_var,
            command=self._on_control_changed,
        )
        self.ov_reactive_check.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(row2, text="Target FPS").pack(side=tk.LEFT)
        self.ov_target_fps_entry = ttk.Entry(row2, textvariable=self.ov_target_fps_var, width=6)
        self.ov_target_fps_entry.pack(side=tk.LEFT, padx=(8, 14))
        self.ov_target_fps_entry.bind("<FocusOut>", self._on_control_changed)

        self.temporal_restore_check = ttk.Checkbutton(
            row2,
            text="Temporal Restore",
            variable=self.temporal_restore_var,
            command=self._on_control_changed,
        )
        self.temporal_restore_check.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(row2, text="TR Strength").pack(side=tk.LEFT)
        self.temporal_strength_entry = ttk.Entry(row2, textvariable=self.temporal_strength_var, width=6)
        self.temporal_strength_entry.pack(side=tk.LEFT, padx=(8, 14))
        self.temporal_strength_entry.bind("<FocusOut>", self._on_control_changed)

        ttk.Label(row2, text="OV Cache").pack(side=tk.LEFT)
        self.ov_cache_entry = ttk.Entry(row2, textvariable=self.ov_cache_var, width=12)
        self.ov_cache_entry.pack(side=tk.LEFT, padx=(8, 14))
        self.ov_cache_entry.bind("<FocusOut>", self._on_control_changed)

        self.strict_npu_check = ttk.Checkbutton(
            row2,
            text="Strict NPU only (auto)",
            variable=self.strict_npu_only_var,
            command=self._on_control_changed,
        )
        self.strict_npu_check.pack(side=tk.LEFT, padx=(0, 10))
        self.strict_npu_check.config(state=tk.DISABLED)

        self.strict_gpu_check = ttk.Checkbutton(
            row2,
            text="Strict GPU only (auto)",
            variable=self.strict_gpu_only_var,
            command=self._on_control_changed,
        )
        self.strict_gpu_check.pack(side=tk.LEFT, padx=(0, 14))
        self.strict_gpu_check.config(state=tk.DISABLED)

        ttk.Label(row2, text="OV Model").pack(side=tk.LEFT)
        self.ov_model_entry = ttk.Entry(row2, textvariable=self.ov_model_var, width=46)
        self.ov_model_entry.pack(side=tk.LEFT, padx=(8, 0), fill=tk.X, expand=True)
        self.ov_model_entry.bind("<FocusOut>", self._on_control_changed)

        row3 = ttk.Frame(frame)
        row3.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(row3, text="Overlay Alpha").pack(side=tk.LEFT)
        self.overlay_alpha_entry = ttk.Entry(row3, textvariable=self.overlay_alpha_var, width=6)
        self.overlay_alpha_entry.pack(side=tk.LEFT, padx=(8, 14))
        self.overlay_alpha_entry.bind("<FocusOut>", self._on_control_changed)

        self.overlay_exclude_check = ttk.Checkbutton(
            row3,
            text="Exclude from capture",
            variable=self.overlay_exclude_var,
            command=self._on_control_changed,
        )
        self.overlay_exclude_check.pack(side=tk.LEFT, padx=(0, 14))

        self.overlay_fullscreen_check = ttk.Checkbutton(
            row3,
            text="SR to Fullscreen",
            variable=self.overlay_fullscreen_var,
            command=self._on_control_changed,
        )
        self.overlay_fullscreen_check.pack(side=tk.LEFT, padx=(0, 14))

        self.start_btn = ttk.Button(row3, text="Start", command=self.start_worker)
        self.start_btn.pack(side=tk.LEFT)
        self.stop_btn = ttk.Button(row3, text="Stop", command=self.stop_worker, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=(8, 0))
        self.help_btn = ttk.Button(row3, text="Help", command=self.show_help)
        self.help_btn.pack(side=tk.LEFT, padx=(8, 0))

        row4 = ttk.Frame(frame)
        row4.pack(fill=tk.X, pady=(0, 8))
        self.benchmark_check = ttk.Checkbutton(
            row4,
            text="Benchmark Log",
            variable=self.benchmark_var,
            command=self._on_control_changed,
        )
        self.benchmark_check.pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(row4, text="CSV Path").pack(side=tk.LEFT)
        self.benchmark_csv_entry = ttk.Entry(row4, textvariable=self.benchmark_csv_var, width=98)
        self.benchmark_csv_entry.pack(side=tk.LEFT, padx=(8, 0), fill=tk.X, expand=True)
        self.benchmark_csv_entry.bind("<FocusOut>", self._on_control_changed)

        if LUNASR_ONLY_MODE:
            self.output_preset_combo.config(state=tk.DISABLED)
            self.device_combo.config(state=tk.DISABLED)
            self.capture_backend_combo.config(state=tk.DISABLED)
            self.gpu_video_check.config(state=tk.DISABLED)
            self.ov_preset_combo.config(state=tk.DISABLED)
            self.ov_internal_entry.config(state=tk.DISABLED)
            self.ov_reactive_check.config(state=tk.DISABLED)
            self.ov_target_fps_entry.config(state=tk.DISABLED)
            self.temporal_restore_check.config(state=tk.DISABLED)
            self.temporal_strength_entry.config(state=tk.DISABLED)
            self.ov_cache_entry.config(state=tk.DISABLED)
            self.strict_npu_check.config(state=tk.DISABLED)
            self.strict_gpu_check.config(state=tk.DISABLED)
            self.ov_model_entry.config(state=tk.DISABLED)
            self.fg_toggle_btn.config(state=tk.DISABLED)
            self.fg_only_check.config(state=tk.DISABLED)

        ttk.Label(frame, textvariable=self.status_main_var).pack(anchor=tk.W, pady=(4, 2))
        ttk.Label(frame, textvariable=self.profile_note_var).pack(anchor=tk.W, pady=(0, 2))
        ttk.Label(frame, textvariable=self.status_perf_var).pack(anchor=tk.W, pady=(0, 2))
        ttk.Label(frame, textvariable=self.status_shape_var).pack(anchor=tk.W, pady=(0, 2))

    def refresh_window_list(self):
        windows = enumerate_windows()
        labels = [w.label for w in windows]
        self.window_map = {w.label: w.hwnd for w in windows}
        self.target_combo["values"] = labels
        if not labels:
            self.target_var.set("")
            self.status_main_var.set("No visible windows found.")
            return

        current = self.target_var.get()
        if current in self.window_map:
            return

        preferred = None
        for label in labels:
            lower = label.lower()
            if "minecraft" in lower:
                preferred = label
                break
        self.target_var.set(preferred or labels[0])
        self.status_main_var.set(f"Detected {len(labels)} windows.")
        self._sync_runtime_config()

    def _on_toggle_clicked(self):
        self.toggle_btn.config(text="Upscale ON" if self.upscale_var.get() else "Upscale OFF")
        self._sync_runtime_config()

    def _on_fg_toggle_clicked(self):
        if LUNASR_ONLY_MODE:
            self.fg_var.set(False)
            self.fg_toggle_btn.config(text="FG OFF")
            self._sync_runtime_config()
            return
        self.fg_toggle_btn.config(text="FG ON" if self.fg_var.get() else "FG OFF")
        self._sync_runtime_config()

    def _on_control_changed(self, _event=None):
        self._sync_runtime_config()

    def _on_alt_p_hotkey(self, _event=None):
        self.stop_worker(reason="Stopped by Alt+P.")
        return "break"

    def show_help(self):
        messagebox.showinfo(
            "LunaSR Help",
            "Controls:\n"
            "- Start: begin realtime overlay processing.\n"
            "- Stop: stop processing.\n"
            "- Alt+P: stop hotkey (works while this GUI is active).\n\n"
            "Modes:\n"
            "- Backend: realtime is fixed to openvino_sr (strict NPU compile path).\n"
            "- Upscale ON: SR pipeline active.\n"
            "- Output preset: AUTO(default, source->next target), HD(720p), FHD(1080p), QHD(1440p), 4K(2160p).\n"
            "- AUTO mapping: <720->HD, <1080->FHD, <1440->QHD, >=1440->4K.\n"
            "- Capture backend: gdi (default) or pil.\n"
            f"- GPU Video Path: {'available' if CUDA_AVAILABLE else 'fallback to CPU'}.\n"
            "- Capture is pipelined on a dedicated thread (latest-frame mode).\n"
            f"- Capture frame buffer: ring={CAPTURE_FRAME_BUFFER_SIZE}, newest-frame consume.\n"
            "- NPU compile: INT8 hint is forced on NPU path.\n"
            "- Device policy: Device=GPU -> GPU only, Device=NPU -> NPU only.\n"
            "- OV Internal Scale range: 0.50 ~ 2.00.\n"
            "- GPU compile path uses 720p base cap (max 1280x720) before tiled SR infer.\n"
            "- Reactive Scale: adjust OV internal scale automatically to chase target FPS.\n"
            "- Auto SR model by output preset: HD/4K -> t256, FHD/QHD -> t192 (custom model path is respected).\n"
            "- Temporal Restore: motion-compensated history blend to reduce flicker.\n"
            "- Two-input OV model(input_curr,input_prev) runs temporal blend on NPU and skips CPU temporal step.\n"
            "- FG default model: fixed_fg_algo_mid (algorithmic midpoint interpolation, no training).\n"
            "- OV tile infer uses async request pool (NPU=auto max, GPU=optimal auto, CPU=2).\n"
            "- Benchmark Log: write per-frame metrics to CSV while running.\n"
            "- SR to Fullscreen: stretch SR output to monitor size.\n"
            "- CPU fallback: disabled in current GUI profile.",
        )

    def _sync_runtime_config(self):
        hwnd = self.window_map.get(self.target_var.get(), 0)
        backend = "openvino_sr"
        if self.backend_var.get() != "openvino_sr":
            self.backend_var.set("openvino_sr")
        capture_backend = (self.capture_backend_var.get().strip().lower() or DEFAULT_CAPTURE_BACKEND)
        if capture_backend not in ("gdi", "pil"):
            capture_backend = DEFAULT_CAPTURE_BACKEND
        strict_npu_only = bool(self.strict_npu_only_var.get())
        strict_gpu_only = bool(self.strict_gpu_only_var.get())
        gpu_video_path = bool(self.gpu_video_path_var.get())
        preset = self.preset_var.get().strip() or "max_performance"
        output_preset = normalize_output_preset(self.output_preset_var.get())
        if self.output_preset_var.get().strip().upper() != output_preset:
            self.output_preset_var.set(output_preset)
        frame_budget_ms = max(0.0, parse_float(self.frame_budget_var.get(), 100.0))
        device = (self.device_var.get().strip() or "AUTO").upper()
        if device not in ("AUTO", "NPU", "GPU", "CPU"):
            device = "AUTO"
            self.device_var.set("AUTO")

        # Enforce strict execution by selected device family.
        if device == "NPU":
            strict_npu_only = True
            strict_gpu_only = False
            if not self.strict_npu_only_var.get():
                self.strict_npu_only_var.set(True)
            if self.strict_gpu_only_var.get():
                self.strict_gpu_only_var.set(False)
        elif device == "GPU":
            strict_npu_only = False
            strict_gpu_only = True
            if self.strict_npu_only_var.get():
                self.strict_npu_only_var.set(False)
            if not self.strict_gpu_only_var.get():
                self.strict_gpu_only_var.set(True)
        else:
            strict_npu_only = False
            strict_gpu_only = False
            if self.strict_npu_only_var.get():
                self.strict_npu_only_var.set(False)
            if self.strict_gpu_only_var.get():
                self.strict_gpu_only_var.set(False)

        if strict_npu_only:
            device = "NPU"
            if self.device_var.get().strip().upper() != "NPU":
                self.device_var.set("NPU")
            gpu_video_path = False
            if self.gpu_video_path_var.get():
                self.gpu_video_path_var.set(False)
        elif strict_gpu_only:
            device = "GPU"
            if self.device_var.get().strip().upper() != "GPU":
                self.device_var.set("GPU")
        ov_model_raw = self.ov_model_var.get().strip() or DEFAULT_OV_MODEL
        ov_model = resolve_auto_ov_model_for_output(
            selected_model=ov_model_raw,
            output_preset=output_preset,
        )
        if ov_model != ov_model_raw:
            self.ov_model_var.set(ov_model)
        ov_preset = self.ov_preset_var.get().strip() or DEFAULT_OV_PRESET
        ov_internal_scale = parse_float(self.ov_internal_scale_var.get(), DEFAULT_OV_INTERNAL_SCALE)
        ov_internal_scale = max(0.50, min(2.0, ov_internal_scale))
        ov_reactive_scale = bool(self.ov_reactive_var.get())
        ov_reactive_target_fps = parse_float(self.ov_target_fps_var.get(), 60.0)
        ov_reactive_target_fps = max(20.0, min(240.0, ov_reactive_target_fps))
        temporal_restore = bool(self.temporal_restore_var.get())
        temporal_strength = parse_float(self.temporal_strength_var.get(), DEFAULT_TEMPORAL_STRENGTH)
        temporal_strength = max(0.0, min(1.0, temporal_strength))
        benchmark_enabled = bool(self.benchmark_var.get())
        benchmark_csv = (self.benchmark_csv_var.get() or "").strip() or DEFAULT_BENCHMARK_CSV
        ov_cache_dir = self.ov_cache_var.get().strip() or DEFAULT_OV_CACHE_DIR
        allow_cpu_fallback = False
        if self.allow_cpu_fallback_var.get():
            self.allow_cpu_fallback_var.set(False)
        overlay_alpha = parse_float(self.overlay_alpha_var.get(), 1.0)
        overlay_alpha = max(0.2, min(1.0, overlay_alpha))
        overlay_click = True
        if not self.overlay_click_var.get():
            self.overlay_click_var.set(True)
        overlay_exclude = bool(self.overlay_exclude_var.get())
        overlay_fullscreen = bool(self.overlay_fullscreen_var.get())
        fg_enabled = bool(self.fg_var.get())
        fg_interp_only = bool(self.fg_interp_only_var.get())
        if LUNASR_ONLY_MODE:
            fg_enabled = False
            fg_interp_only = False
            if self.fg_var.get():
                self.fg_var.set(False)
            if self.fg_interp_only_var.get():
                self.fg_interp_only_var.set(False)
            self.fg_toggle_btn.config(text="FG OFF")
        if fg_interp_only and not fg_enabled:
            fg_enabled = True
            self.fg_var.set(True)
            self.fg_toggle_btn.config(text="FG ON")
        fg_model = self.fg_model_var.get().strip() or DEFAULT_FG_MODEL
        fg_precision = (self.fg_precision_var.get().strip() or "f16").lower()
        if fg_precision not in ("f16", "i8"):
            fg_precision = "f16"
        fg_size = max(16, parse_int(self.fg_size_var.get(), 256))
        fg_timestep = parse_float(self.fg_timestep_var.get(), 0.50)
        fg_timestep = max(0.0, min(1.0, fg_timestep))

        with self.config_lock:
            self.config_holder["cfg"] = RuntimeConfig(
                hwnd=hwnd,
                backend=backend,
                capture_backend=capture_backend,
                preset=preset,
                output_preset=output_preset,
                gpu_video_path=gpu_video_path,
                upscale_enabled=bool(self.upscale_var.get()),
                frame_budget_ms=frame_budget_ms,
                device=device,
                ov_model=ov_model,
                ov_preset=ov_preset,
                ov_internal_scale=ov_internal_scale,
                ov_reactive_scale=ov_reactive_scale,
                ov_reactive_target_fps=ov_reactive_target_fps,
                temporal_restore=temporal_restore,
                temporal_strength=temporal_strength,
                benchmark_enabled=benchmark_enabled,
                benchmark_csv=benchmark_csv,
                strict_npu_only=strict_npu_only,
                strict_gpu_only=strict_gpu_only,
                ov_cache_dir=ov_cache_dir,
                allow_cpu_fallback=allow_cpu_fallback,
                overlay_alpha=overlay_alpha,
                overlay_click_through=overlay_click,
                overlay_exclude_from_capture=overlay_exclude,
                overlay_fullscreen_upscale=overlay_fullscreen,
                fg_enabled=fg_enabled,
                fg_interp_only=fg_interp_only,
                fg_model=fg_model,
                fg_precision=fg_precision,
                fg_size=fg_size,
                fg_timestep=fg_timestep,
            )

    def start_worker(self):
        self._sync_runtime_config()
        with self.config_lock:
            cfg = self.config_holder["cfg"]

        if cfg.hwnd <= 0:
            messagebox.showerror("LunaSR GUI", "Select a valid target window.")
            return
        if cfg.backend == "openvino_sr":
            model_path = resolve_model_path(cfg.ov_model)
            if not model_path.exists():
                messagebox.showerror("LunaSR GUI", f"OpenVINO model not found:\n{model_path}")
                return
        if cfg.fg_enabled:
            fg_model_path = resolve_model_path(cfg.fg_model)
            if not fg_model_path.exists():
                messagebox.showerror("LunaSR GUI", f"FG model not found:\n{fg_model_path}")
                return

        if self.worker is not None and self.worker.is_alive():
            return

        self.worker = LunaRealtimeWorker(self.config_lock, self.config_holder, self.status_q, self.overlay_q)
        self.worker.start()
        try:
            user32.SetForegroundWindow(int(cfg.hwnd))
        except Exception:
            pass
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        fg_state = "on" if cfg.fg_enabled else "off"
        fg_mode = "interp_only" if cfg.fg_interp_only else "normal"
        self.status_main_var.set(
            f"Worker running in overlay mode (fg={fg_state}, fg_mode={fg_mode}). Stop or Alt+P."
        )

    def stop_worker(self, reason: Optional[str] = None):
        if self.worker is not None:
            self.worker.stop()
            self.worker.join(timeout=2.0)
            self.worker = None
        self.overlay_window.hide()
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        if reason:
            self.status_main_var.set(reason)
        else:
            self.status_main_var.set("Worker stopped.")

    def _poll_status_queue(self):
        latest: Optional[Dict[str, object]] = None
        while True:
            try:
                latest = self.status_q.get_nowait()
            except queue.Empty:
                break
        if latest is not None:
            if "error" in latest:
                self.status_main_var.set(str(latest["error"]))
            if "warning" in latest:
                self.status_main_var.set(str(latest["warning"]))
            if "profile_note" in latest:
                self.profile_note_var.set(str(latest["profile_note"]))
            if "benchmark_note" in latest:
                self.status_main_var.set(str(latest["benchmark_note"]))
            if "fallback" in latest:
                self.status_main_var.set(str(latest["fallback"]))
            if "fps" in latest:
                capture_age_ms = float(latest.get("capture_age_ms", 0.0))
                fg_ms = float(latest.get("fg_ms", 0.0))
                temporal_ms = float(latest.get("temporal_ms", 0.0))
                ov_scale = float(latest.get("ov_scale", 0.0))
                ov_reactive = bool(latest.get("ov_reactive", False))
                ov_target = float(latest.get("ov_target_fps", 0.0))
                temporal_on = bool(latest.get("temporal_on", False))
                temporal_strength = float(latest.get("temporal_strength", 0.0))
                reactive_tag = f"ov_scale={ov_scale:.2f}" + (
                    f"(reactive@{ov_target:.0f})" if ov_reactive else "(fixed)"
                )
                temporal_tag = f"temporal={'on' if temporal_on else 'off'}@{temporal_strength:.2f}"
                self.status_perf_var.set(
                    f"fps={float(latest['fps']):.2f} | cap_age_ms={capture_age_ms:.1f} | "
                    f"frame_ms={float(latest['frame_ms']):.1f} | "
                    f"infer_ms={float(latest['infer_ms']):.1f} | fg_ms={fg_ms:.1f} | "
                    f"temporal_ms={temporal_ms:.1f} | tile_ms={float(latest['tile_ms']):.1f} | "
                    f"{reactive_tag} | {temporal_tag}"
                )
            if "input_shape" in latest:
                self.status_shape_var.set(
                    f"backend={latest['backend']} device={latest['device']} | "
                    f"input={latest['input_shape']} | output={latest['output_shape']}"
                )
            if "stopped" in latest:
                self.start_btn.config(state=tk.NORMAL)
                self.stop_btn.config(state=tk.DISABLED)
                self.status_main_var.set("Worker stopped.")
            if "overlay_hide" in latest:
                self.overlay_window.hide()
        self.root.after(STATUS_POLL_INTERVAL_MS, self._poll_status_queue)

    def _poll_overlay_queue(self):
        payload: Optional[Dict[str, object]] = None
        while True:
            try:
                payload = self.overlay_q.get_nowait()
            except queue.Empty:
                break

        if payload is None:
            self.root.after(OVERLAY_POLL_INTERVAL_MS, self._poll_overlay_queue)
            return

        frame = payload.get("frame")
        rect = payload.get("rect")
        if not isinstance(frame, np.ndarray) or not isinstance(rect, tuple) or len(rect) != 4:
            self.root.after(OVERLAY_POLL_INTERVAL_MS, self._poll_overlay_queue)
            return

        try:
            self.overlay_window.show_frame(
                frame_bgr=frame,
                rect=rect,  # type: ignore[arg-type]
                alpha=float(payload.get("alpha", 1.0)),
                click_through=bool(payload.get("click_through", True)),
                exclude_from_capture=bool(payload.get("exclude_from_capture", True)),
                target_hwnd=int(payload.get("target_hwnd", 0)),
            )
        except Exception as exc:
            self.status_main_var.set(f"Overlay render error: {exc}")
        finally:
            self.root.after(OVERLAY_POLL_INTERVAL_MS, self._poll_overlay_queue)

    def on_close(self):
        self.stop_worker()
        self.root.destroy()


def main() -> int:
    root = tk.Tk()
    app = LunaRealtimeGui(root)
    app._on_toggle_clicked()
    app._on_fg_toggle_clicked()
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
