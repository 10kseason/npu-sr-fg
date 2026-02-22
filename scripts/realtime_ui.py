import ctypes
import tkinter as tk
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

user32 = ctypes.WinDLL("user32", use_last_error=True)
user32.GetWindowLongW.argtypes = [ctypes.c_void_p, ctypes.c_int]
user32.GetWindowLongW.restype = ctypes.c_long
user32.SetWindowLongW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_long]
user32.SetWindowLongW.restype = ctypes.c_long
user32.SetWindowPos.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_uint,
]
user32.SetWindowPos.restype = ctypes.c_bool
user32.SetLayeredWindowAttributes.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_ubyte, ctypes.c_uint]
user32.SetLayeredWindowAttributes.restype = ctypes.c_bool
user32.SetWindowDisplayAffinity.argtypes = [ctypes.c_void_p, ctypes.c_uint]
user32.SetWindowDisplayAffinity.restype = ctypes.c_bool
user32.EnableWindow.argtypes = [ctypes.c_void_p, ctypes.c_bool]
user32.EnableWindow.restype = ctypes.c_bool
user32.GetForegroundWindow.argtypes = []
user32.GetForegroundWindow.restype = ctypes.c_void_p
user32.SetForegroundWindow.argtypes = [ctypes.c_void_p]
user32.SetForegroundWindow.restype = ctypes.c_bool

GWL_EXSTYLE = -20
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_NOACTIVATE = 0x08000000
LWA_ALPHA = 0x00000002
SWP_NOACTIVATE = 0x0010
SWP_FRAMECHANGED = 0x0020
HWND_TOPMOST = -1
WDA_NONE = 0x00000000
WDA_EXCLUDEFROMCAPTURE = 0x00000011


def build_help_text(
    *,
    cuda_available: bool,
    capture_backends: Tuple[str, ...],
    capture_frame_buffer_size: int,
) -> str:
    backends = ", ".join(capture_backends)
    return (
        "Controls:\n"
        "- Start: begin realtime overlay processing.\n"
        "- Stop: stop processing.\n"
        "- Alt+P: stop hotkey (works while this GUI is active).\n"
        "- F8: global start/stop toggle (works while GUI is in tray).\n"
        "- F9: global GUI show/hide toggle.\n"
        "- Close button: hide to tray (does not terminate).\n\n"
        "Modes:\n"
        "- Backend: realtime is fixed to openvino_sr (strict NPU compile path).\n"
        "- Upscale ON: SR pipeline active.\n"
        "- Output preset: AUTO(default, source->next target), HD(720p), FHD(1080p), QHD(1440p), 4K(2160p).\n"
        "- AUTO mapping: <720->HD, <1080->FHD, <1440->QHD, >=1440->4K.\n"
        f"- Capture backend: {backends}.\n"
        "- Strict NPU/GPU mode uses DXGI-only capture (requires dxcam module in current Python env).\n"
        "- In strict DXGI mode, SR preview is detached outside target window by default to avoid self-capture.\n"
        "- Enable 'Force in-place overlay' (or set LUNASR_FORCE_INPLACE_OVERLAY=1) to draw directly over the target window.\n"
        f"- GPU Video Path: {'available' if cuda_available else 'fallback to CPU'}.\n"
        "- Capture is pipelined on a dedicated thread (latest-frame mode).\n"
        f"- Capture frame buffer: ring={capture_frame_buffer_size}, newest-frame consume.\n"
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
        "- CPU fallback: disabled in current GUI profile."
    )


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
            try:
                user32.EnableWindow(self.hwnd, ctypes.c_bool(not click_through))
            except Exception:
                pass
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

            # Do not force-focus target window here.
            # Reacquiring focus every frame can block user interaction on other windows.
        else:
            self.win.attributes("-alpha", a)

        if not self.visible:
            self.win.deiconify()
        self.visible = True
