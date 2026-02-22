import ctypes
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import ImageGrab

DEFAULT_CAPTURE_BACKEND = "dxgi"
SUPPORTED_CAPTURE_BACKENDS = ("dxgi", "gdi", "pil")
CAPTURE_FRAME_BUFFER_SIZE = 4

WINDOW_TITLE_MAX = 512
GWL_EXSTYLE = -20
WS_EX_TOOLWINDOW = 0x00000080
MONITOR_DEFAULTTONEAREST = 0x00000002
PW_RENDERFULLCONTENT = 0x00000002
DIB_RGB_COLORS = 0
BI_RGB = 0

user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)

user32.MonitorFromWindow.argtypes = [ctypes.c_void_p, ctypes.c_uint]
user32.MonitorFromWindow.restype = ctypes.c_void_p
user32.GetMonitorInfoW.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
user32.GetMonitorInfoW.restype = ctypes.c_bool
user32.GetWindowDC.argtypes = [ctypes.c_void_p]
user32.GetWindowDC.restype = ctypes.c_void_p
user32.ReleaseDC.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
user32.ReleaseDC.restype = ctypes.c_int
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


_MONITOR_ENUM_PROC = ctypes.WINFUNCTYPE(
    ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(RECT), ctypes.c_void_p
)
user32.EnumDisplayMonitors.argtypes = [ctypes.c_void_p, ctypes.c_void_p, _MONITOR_ENUM_PROC, ctypes.c_void_p]
user32.EnumDisplayMonitors.restype = ctypes.c_bool


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


def enumerate_monitor_rects() -> List[Tuple[int, int, int, int]]:
    monitors: List[Tuple[int, int, int, int]] = []

    @_MONITOR_ENUM_PROC
    def enum_proc(_hmon, _hdc, lprc, _lparam):
        if not lprc:
            return True
        r = lprc.contents
        left = int(r.left)
        top = int(r.top)
        right = int(r.right)
        bottom = int(r.bottom)
        if right > left and bottom > top:
            monitors.append((left, top, right, bottom))
        return True

    try:
        user32.EnumDisplayMonitors(None, None, enum_proc, None)
    except Exception:
        return []
    return monitors


def enumerate_windows(skip_prefixes: Tuple[str, ...] = ("lunasr realtime gui",)) -> List[WindowInfo]:
    result: List[WindowInfo] = []
    prefix_tokens = tuple(str(x).strip().lower() for x in skip_prefixes if str(x).strip())

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def enum_cb(hwnd, _lparam):
        hwnd_i = int(hwnd)
        if not user32.IsWindowVisible(hwnd_i):
            return True
        title = get_window_text(hwnd_i)
        if not title:
            return True
        lower = title.lower()
        if any(lower.startswith(tok) for tok in prefix_tokens):
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

        ok = bool(user32.PrintWindow(int(hwnd), mem_dc, PW_RENDERFULLCONTENT))
        if not ok:
            ok = bool(user32.PrintWindow(int(hwnd), mem_dc, 0))
        if not ok:
            return None

        bmi = BITMAPINFO()
        bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.bmiHeader.biWidth = w
        bmi.bmiHeader.biHeight = -h
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

    frame_rgb = np.asarray(img)
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


class _DxgiCaptureSession:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._dxcam = None
        self._dxcam_failed = False
        self._dxcam_error = ""
        self._cameras: Dict[int, Any] = {}
        self._camera_is_bgr: Dict[int, bool] = {}
        self._output_count_hint = 1
        self._last_error = ""

    def _set_error(self, text: str) -> None:
        self._last_error = str(text or "").strip()

    def last_error(self) -> str:
        return self._last_error

    def _estimate_output_count(self, dxcam_mod: Any) -> int:
        count = max(1, len(enumerate_monitor_rects()))
        try:
            info = dxcam_mod.output_info()
            token = str(info)
            hit = token.count("Output[")
            if hit > 0:
                count = max(count, int(hit))
        except Exception:
            pass
        return max(1, min(8, int(count)))

    def _import_dxcam_module(self) -> Optional[Any]:
        try:
            import dxcam  # type: ignore

            return dxcam
        except Exception:
            pass

        here = Path(__file__).resolve()
        candidate_roots = [
            here.parent.parent / "third_party",
            here.parent / "third_party",
            Path.cwd() / "third_party",
        ]
        for root in candidate_roots:
            pkg = root / "dxcam" / "__init__.py"
            if not pkg.exists():
                continue
            try:
                root_str = str(root.resolve())
            except Exception:
                root_str = str(root)
            if root_str not in sys.path:
                sys.path.insert(0, root_str)
            try:
                import dxcam  # type: ignore

                return dxcam
            except Exception:
                continue
        return None

    def _ensure_dxcam(self) -> Optional[Any]:
        if self._dxcam is not None:
            return self._dxcam
        if self._dxcam_failed:
            return None
        with self._lock:
            if self._dxcam is not None:
                return self._dxcam
            if self._dxcam_failed:
                return None
            dxcam = self._import_dxcam_module()
            if dxcam is None:
                e = ModuleNotFoundError("dxcam")
                self._dxcam_failed = True
                self._dxcam_error = f"dxcam_import_failed:{type(e).__name__}"
                self._set_error(self._dxcam_error)
                return None
            self._dxcam = dxcam
            self._output_count_hint = self._estimate_output_count(dxcam)
            return self._dxcam

    def support_status(self) -> Tuple[bool, str]:
        mod = self._ensure_dxcam()
        if mod is None:
            return False, self._dxcam_error or self._last_error or "dxcam_unavailable"
        return True, "ok"

    def _get_camera(self, output_idx: int) -> Optional[Any]:
        idx = int(max(0, output_idx))
        cam = self._cameras.get(idx)
        if cam is not None:
            return cam
        dxcam_mod = self._ensure_dxcam()
        if dxcam_mod is None:
            return None
        with self._lock:
            cam = self._cameras.get(idx)
            if cam is not None:
                return cam
            try:
                cam = dxcam_mod.create(output_idx=idx, output_color="BGR")
                is_bgr = True
            except TypeError:
                try:
                    cam = dxcam_mod.create(output_idx=idx)
                    is_bgr = False
                except Exception as e:
                    self._set_error(f"dxcam_create_failed(output={idx},{type(e).__name__})")
                    return None
            except Exception as e:
                self._set_error(f"dxcam_create_failed(output={idx},{type(e).__name__})")
                return None
            self._cameras[idx] = cam
            self._camera_is_bgr[idx] = bool(is_bgr)
            return cam

    def _candidate_outputs(
        self,
        rect: Tuple[int, int, int, int],
        monitor_rect: Optional[Tuple[int, int, int, int]],
    ) -> Tuple[List[int], List[Tuple[int, int, int, int]]]:
        monitors = enumerate_monitor_rects()
        target_idx: Optional[int] = None
        if monitor_rect is not None:
            for i, mr in enumerate(monitors):
                if tuple(mr) == tuple(monitor_rect):
                    target_idx = i
                    break
        if target_idx is None:
            cx = int((int(rect[0]) + int(rect[2])) * 0.5)
            cy = int((int(rect[1]) + int(rect[3])) * 0.5)
            for i, mr in enumerate(monitors):
                if mr[0] <= cx < mr[2] and mr[1] <= cy < mr[3]:
                    target_idx = i
                    break

        max_outputs = max(1, int(self._output_count_hint), len(monitors))
        max_outputs = min(8, max_outputs)
        candidates: List[int] = []
        if target_idx is not None and 0 <= int(target_idx) < max_outputs:
            # Prefer target monitor first, then others as fallback.
            candidates.append(int(target_idx))
        for i in range(max_outputs):
            if i not in candidates:
                candidates.append(i)
        return candidates, monitors

    def _regions_for_output(
        self,
        rect: Tuple[int, int, int, int],
        output_idx: int,
        monitors: List[Tuple[int, int, int, int]],
    ) -> List[Tuple[int, int, int, int]]:
        left, top, right, bottom = (int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))
        regions: List[Tuple[int, int, int, int]] = [(left, top, right, bottom)]
        if 0 <= int(output_idx) < len(monitors):
            ml, mt, _mr, _mb = monitors[int(output_idx)]
            local = (left - ml, top - mt, right - ml, bottom - mt)
            if local[2] > local[0] and local[3] > local[1] and local != regions[0]:
                regions.append(local)
        return regions

    def _grab(self, cam: Any, region: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        try:
            frame = cam.grab(region=region)
            if frame is None:
                return None
            return np.asarray(frame)
        except TypeError:
            try:
                frame_full = cam.grab()
            except Exception:
                return None
            if frame_full is None:
                return None
            arr_full = np.asarray(frame_full)
            if arr_full.ndim != 3:
                return arr_full
            x0, y0, x1, y1 = (int(region[0]), int(region[1]), int(region[2]), int(region[3]))
            h, w = arr_full.shape[:2]
            if 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h:
                return arr_full[y0:y1, x0:x1]
            return None
        except Exception:
            return None

    def capture_rect(
        self,
        rect: Tuple[int, int, int, int],
        monitor_rect: Optional[Tuple[int, int, int, int]] = None,
    ) -> Optional[np.ndarray]:
        dxcam_mod = self._ensure_dxcam()
        if dxcam_mod is None:
            return None

        left, top, right, bottom = (int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))
        w = max(1, right - left)
        h = max(1, bottom - top)

        candidates, monitors = self._candidate_outputs(rect=rect, monitor_rect=monitor_rect)
        for output_idx in candidates:
            cam = self._get_camera(output_idx)
            if cam is None:
                continue
            regions = self._regions_for_output(rect=rect, output_idx=output_idx, monitors=monitors)
            for region in regions:
                arr = self._grab(cam, region=region)
                if arr is None:
                    continue
                if arr.ndim != 3 or arr.shape[2] < 3:
                    continue
                if arr.shape[2] == 4:
                    arr = arr[:, :, :3]
                if not bool(self._camera_is_bgr.get(int(output_idx), True)):
                    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                if arr.shape[1] != w or arr.shape[0] != h:
                    interp = cv2.INTER_AREA if (arr.shape[1] > w or arr.shape[0] > h) else cv2.INTER_CUBIC
                    arr = cv2.resize(arr, (w, h), interpolation=interp)
                self._last_error = ""
                return np.ascontiguousarray(arr)

        self._set_error("dxgi_grab_failed(all_outputs)")
        return None


_DXGI_CAPTURE = _DxgiCaptureSession()


def dxgi_capture_status() -> Tuple[bool, str]:
    return _DXGI_CAPTURE.support_status()


def dxgi_capture_last_error() -> str:
    return _DXGI_CAPTURE.last_error()


def capture_window_bgr_dxgi(hwnd: int) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    rect = get_window_rect(hwnd)
    if rect is None:
        return None
    monitor_rect = get_monitor_rect_for_window(hwnd)
    frame = _DXGI_CAPTURE.capture_rect(rect, monitor_rect=monitor_rect)
    if frame is None:
        return None
    return frame, rect


def capture_window_bgr(
    hwnd: int,
    backend: str = DEFAULT_CAPTURE_BACKEND,
    allow_fallback: bool = True,
) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    token = (backend or DEFAULT_CAPTURE_BACKEND).strip().lower()
    if token not in SUPPORTED_CAPTURE_BACKENDS:
        token = DEFAULT_CAPTURE_BACKEND

    if not bool(allow_fallback):
        if token == "dxgi":
            return capture_window_bgr_dxgi(hwnd)
        if token == "pil":
            return capture_window_bgr_pil(hwnd)
        return capture_window_bgr_gdi(hwnd)

    if token == "dxgi":
        frame = capture_window_bgr_dxgi(hwnd)
        if frame is not None:
            return frame
        frame = capture_window_bgr_gdi(hwnd)
        if frame is not None:
            return frame
        return capture_window_bgr_pil(hwnd)

    if token == "pil":
        frame = capture_window_bgr_pil(hwnd)
        if frame is not None:
            return frame
        frame = capture_window_bgr_dxgi(hwnd)
        if frame is not None:
            return frame
        return capture_window_bgr_gdi(hwnd)

    frame = capture_window_bgr_gdi(hwnd)
    if frame is not None:
        return frame
    frame = capture_window_bgr_dxgi(hwnd)
    if frame is not None:
        return frame
    return capture_window_bgr_pil(hwnd)


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
        self.buffer: Deque[Tuple[int, float, np.ndarray, Tuple[int, int, int, int]]] = deque(
            maxlen=max(1, int(buffer_size))
        )
        self.latest_id: int = 0
        self.last_error: str = ""
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
                strict_npu_only = bool(getattr(cfg, "strict_npu_only", False))
                strict_gpu_only = bool(getattr(cfg, "strict_gpu_only", False))
                device = str(getattr(cfg, "device", "AUTO")).upper()
            except Exception:
                time.sleep(0.01)
                continue

            if hwnd <= 0:
                time.sleep(0.01)
                continue

            lock_accel_path = bool(strict_npu_only or strict_gpu_only or device in ("NPU", "GPU"))
            if lock_accel_path:
                backend = "dxgi"
            captured = capture_window_bgr(hwnd, backend=backend, allow_fallback=(not lock_accel_path))
            now = time.perf_counter()
            if captured is None:
                mode = "strict_no_cpu_fallback" if lock_accel_path else "auto_fallback"
                detail = ""
                if str(backend).strip().lower() == "dxgi":
                    detail = dxgi_capture_last_error()
                if detail:
                    self.last_error = f"capture_failed({backend},{mode},{detail})"
                else:
                    self.last_error = f"capture_failed({backend},{mode})"
                time.sleep(0.003)
                continue

            frame, rect = captured
            with self.cond:
                self.latest_id += 1
                frame_id = int(self.latest_id)
                self.buffer.append((frame_id, float(now), frame, rect))
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
