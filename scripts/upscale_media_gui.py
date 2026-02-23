import math
import queue
import shutil
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import openvino as ov
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    from upscale_png_x2_npu import (
        build_compile_cfg,
        get_downsample_interp,
        get_final_upsample_interp,
        get_model_scale,
        np_dtype_from_ov_type,
        postprocess_tensor,
        resize_to_x2,
        run_infer_tiled,
        to_nchw_tensor,
    )
except ImportError:
    from scripts.upscale_png_x2_npu import (
        build_compile_cfg,
        get_downsample_interp,
        get_final_upsample_interp,
        get_model_scale,
        np_dtype_from_ov_type,
        postprocess_tensor,
        resize_to_x2,
        run_infer_tiled,
        to_nchw_tensor,
    )

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

DEFAULT_MODEL = ROOT_DIR / "model" / "ir" / "fixed_sr_algo_x2_quality_plus_aa_anime.xml"
DEFAULT_CACHE_DIR = ROOT_DIR / ".ov_cache"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}
SCALE_MIN = 2
SCALE_MAX = 10
SOFT_POST_BLEND = 0.14
SOFT_POST_SIGMA = 0.85
SOFT_POST_MAX_EST_MEMORY_BYTES = 1_200_000_000
TEMP_MEDIA_DIR = ROOT_DIR / ".tmp_media_io"


class JobCancelledError(RuntimeError):
    pass


@dataclass
class UpscaleJobConfig:
    input_path: Path
    output_path: Path
    model_path: Path
    scale: int
    device: str
    preset: str
    cache_dir: str
    tile_h: int
    tile_w: int
    tile_overlap: int
    final_upsample: str
    soft_postprocess: bool


def detect_media_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTS:
        return "image"
    if suffix in VIDEO_EXTS:
        return "video"
    raise RuntimeError(f"Unsupported file extension: {path.suffix}")


def suggest_output_path(input_path: Path, scale: int) -> Path:
    suffix = input_path.suffix.lower()
    if suffix in VIDEO_EXTS:
        out_suffix = ".mp4"
    elif suffix in IMAGE_EXTS:
        out_suffix = suffix
    else:
        out_suffix = ".mp4"
    return input_path.with_name(f"{input_path.stem}_x{scale}{out_suffix}")


def _largest_power_of_two_leq(n: int) -> int:
    factor = 1
    while factor * 2 <= n:
        factor *= 2
    return factor


def _strip_quotes(raw: str) -> str:
    s = raw.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in {"'", '"'}:
        return s[1:-1]
    return s


def read_image_unicode(path: Path) -> Optional[np.ndarray]:
    # cv2.imread can fail on non-ASCII paths depending on OpenCV build; use imdecode workaround.
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        if buf.size == 0:
            return None
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def write_image_unicode(path: Path, image: np.ndarray) -> bool:
    ext = path.suffix.lower()
    if not ext:
        ext = ".png"
    try:
        ok, encoded = cv2.imencode(ext, image)
        if not ok:
            return False
        encoded.tofile(str(path))
        return True
    except Exception:
        return False


def try_apply_soft_postprocess(image: np.ndarray) -> Tuple[np.ndarray, bool, str]:
    # Mild blur-blend to reduce residual aliasing/ringing while keeping edge detail.
    # Skip when estimated temporary memory is too large to avoid OOM/SystemError.
    est_bytes = int(image.nbytes) * 3
    if est_bytes > SOFT_POST_MAX_EST_MEMORY_BYTES:
        need_gib = float(est_bytes) / float(1024**3)
        lim_gib = float(SOFT_POST_MAX_EST_MEMORY_BYTES) / float(1024**3)
        return (
            image,
            False,
            f"soft_post=skip (estimated temp memory {need_gib:.2f} GiB > limit {lim_gib:.2f} GiB)",
        )
    try:
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=SOFT_POST_SIGMA, sigmaY=SOFT_POST_SIGMA)
        mixed = cv2.addWeighted(image, 1.0 - SOFT_POST_BLEND, blurred, SOFT_POST_BLEND, 0.0)
        return mixed, True, ""
    except Exception as exc:
        return image, False, f"soft_post=skip ({type(exc).__name__}: {exc})"


def _path_is_ascii(path: Path) -> bool:
    try:
        str(path).encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def _make_temp_media_path(prefix: str, suffix: str) -> Path:
    sfx = suffix if suffix.startswith(".") else f".{suffix}" if suffix else ".mp4"
    TEMP_MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:10]
    return TEMP_MEDIA_DIR / f"{prefix}_{token}{sfx}"


def open_video_capture_with_fallback(
    input_path: Path,
    log: Callable[[str], None],
) -> Tuple[cv2.VideoCapture, Optional[Path]]:
    cap = cv2.VideoCapture(str(input_path))
    if cap.isOpened():
        return cap, None
    cap.release()

    if _path_is_ascii(input_path):
        raise FileNotFoundError(f"Failed to open video: {input_path}")

    temp_copy = _make_temp_media_path("input_copy", input_path.suffix.lower() or ".mp4")
    shutil.copy2(input_path, temp_copy)
    cap2 = cv2.VideoCapture(str(temp_copy))
    if cap2.isOpened():
        log(f"video_open_fallback=copy_to_ascii_temp ({temp_copy.name})")
        return cap2, temp_copy
    cap2.release()
    try:
        temp_copy.unlink(missing_ok=True)
    except Exception:
        pass
    raise FileNotFoundError(f"Failed to open video (including fallback copy): {input_path}")


def open_video_writer_with_fallback(
    output_path: Path,
    fourcc: int,
    fps: float,
    frame_size: Tuple[int, int],
    log: Callable[[str], None],
) -> Tuple[cv2.VideoWriter, Optional[Path]]:
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)
    if writer.isOpened():
        return writer, None
    writer.release()

    if _path_is_ascii(output_path):
        raise RuntimeError(f"Failed to open output video writer: {output_path}")

    temp_out = _make_temp_media_path("output_tmp", output_path.suffix.lower() or ".mp4")
    writer2 = cv2.VideoWriter(str(temp_out), fourcc, fps, frame_size)
    if writer2.isOpened():
        log(f"video_write_fallback=write_to_ascii_temp ({temp_out.name})")
        return writer2, temp_out
    writer2.release()
    try:
        temp_out.unlink(missing_ok=True)
    except Exception:
        pass
    raise RuntimeError(f"Failed to open output video writer (including fallback): {output_path}")


def run_infer_tiled_multi(
    image: np.ndarray,
    req,
    input_metas: List[Dict[str, Any]],
    out_name: str,
    in_h: int,
    in_w: int,
    model_scale_h: float,
    model_scale_w: float,
    tile_h: int,
    tile_w: int,
    tile_overlap: int,
    downsample_interp: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    src_h, src_w = image.shape[:2]
    dst = np.zeros((src_h * 2, src_w * 2, 3), dtype=np.uint8)

    tiles_x = math.ceil(src_w / tile_w)
    tiles_y = math.ceil(src_h / tile_h)
    total_tiles = tiles_x * tiles_y

    infer_latencies_ms: List[float] = []
    t_total0 = time.perf_counter()

    for ty in range(tiles_y):
        for tx in range(tiles_x):
            x0 = tx * tile_w
            y0 = ty * tile_h
            x1 = min(x0 + tile_w, src_w)
            y1 = min(y0 + tile_h, src_h)

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
            core_h = y1 - y0
            core_w = x1 - x0

            if valid_h != in_h or valid_w != in_w:
                bottom = in_h - valid_h
                right = in_w - valid_w
                patch = cv2.copyMakeBorder(
                    patch,
                    0,
                    max(0, bottom),
                    0,
                    max(0, right),
                    borderType=cv2.BORDER_REFLECT_101,
                )

            feed: Dict[str, np.ndarray] = {}
            for meta in input_metas:
                feed[meta["name"]] = to_nchw_tensor(
                    patch,
                    int(meta["c"]),
                    in_h,
                    in_w,
                    meta["dtype"],
                )

            t0 = time.perf_counter()
            outputs = req.infer(feed)
            t1 = time.perf_counter()
            infer_latencies_ms.append((t1 - t0) * 1000.0)

            y = outputs[out_name] if out_name in outputs else next(iter(outputs.values()))
            out_tile = postprocess_tensor(np.asarray(y))
            out_x2 = resize_to_x2(
                out_tile,
                valid_h,
                valid_w,
                model_scale_h,
                model_scale_w,
                downsample_interp,
            )
            cx0 = core_x0 * 2
            cy0 = core_y0 * 2
            cx1 = cx0 + core_w * 2
            cy1 = cy0 + core_h * 2
            out_core = out_x2[cy0:cy1, cx0:cx1]
            dst[y0 * 2 : y1 * 2, x0 * 2 : x1 * 2] = out_core

    t_total1 = time.perf_counter()
    avg_ms = float(np.mean(infer_latencies_ms)) if infer_latencies_ms else 0.0
    return dst, {
        "tiles": float(total_tiles),
        "tile_infer_avg_ms": avg_ms,
        "elapsed_ms": (t_total1 - t_total0) * 1000.0,
    }


class OpenVinoX2Pipeline:
    def __init__(self, cfg: UpscaleJobConfig, log: Callable[[str], None]):
        self.cfg = cfg
        self.log = log

        core = ov.Core()
        model = core.read_model(str(cfg.model_path))
        image_inputs: List[Dict[str, Any]] = []
        for idx, in_port in enumerate(model.inputs):
            shape = [int(x) for x in in_port.shape]
            if len(shape) != 4:
                continue
            if int(shape[1]) not in (1, 3, 4):
                continue
            image_inputs.append({"index": idx, "shape": shape})

        if not image_inputs:
            raise RuntimeError("No image-like 4D input found in model.")
        if len(image_inputs) > 4:
            raise RuntimeError(f"At most 4 image inputs are supported. got={len(image_inputs)}")

        image_outputs: List[Dict[str, Any]] = []
        for idx, out_port in enumerate(model.outputs):
            shape = [int(x) for x in out_port.shape]
            if len(shape) != 4:
                continue
            if int(shape[1]) not in (1, 3, 4):
                continue
            image_outputs.append({"index": idx, "shape": shape})

        if not image_outputs:
            raise RuntimeError("No image-like 4D output found in model.")

        in_shape = list(image_inputs[0]["shape"])
        out_shape = list(image_outputs[0]["shape"])
        in_n, in_c, in_h, in_w = in_shape
        if in_n != 1:
            raise RuntimeError(f"Only batch=1 is supported. got N={in_n}")
        for meta in image_inputs[1:]:
            shp = list(meta["shape"])
            if shp != in_shape:
                raise RuntimeError(
                    f"All image inputs must match first input shape. first={in_shape}, got={shp}"
                )

        tile_overlap = max(0, int(cfg.tile_overlap))
        if tile_overlap * 2 >= in_h or tile_overlap * 2 >= in_w:
            raise RuntimeError(
                f"--tile-overlap is too large for model input {in_w}x{in_h}. got overlap={tile_overlap}"
            )

        tile_h_default = in_h - (tile_overlap * 2)
        tile_w_default = in_w - (tile_overlap * 2)
        tile_h = cfg.tile_h if cfg.tile_h > 0 else tile_h_default
        tile_w = cfg.tile_w if cfg.tile_w > 0 else tile_w_default
        if tile_h <= 0 or tile_w <= 0:
            raise RuntimeError(
                f"Computed tile size is invalid. tile={tile_w}x{tile_h}, overlap={tile_overlap}, model={in_w}x{in_h}"
            )
        if tile_h + (tile_overlap * 2) > in_h or tile_w + (tile_overlap * 2) > in_w:
            raise RuntimeError(
                f"tile+2*overlap must be <= model input. got tile={tile_w}x{tile_h}, "
                f"overlap={tile_overlap}, model={in_w}x{in_h}"
            )

        model_scale = get_model_scale(in_shape, out_shape)
        compile_cfg = build_compile_cfg(cfg.cache_dir, cfg.preset)
        downsample_interp = get_downsample_interp(cfg.preset)

        t0 = time.perf_counter()
        compiled = core.compile_model(model, cfg.device, compile_cfg)
        req = compiled.create_infer_request()
        t1 = time.perf_counter()

        input_metas: List[Dict[str, Any]] = []
        for meta in image_inputs:
            idx = int(meta["index"])
            cport = compiled.inputs[idx]
            shp = list(meta["shape"])
            input_metas.append(
                {
                    "index": idx,
                    "name": cport.any_name,
                    "shape": shp,
                    "c": int(shp[1]),
                    "dtype": np_dtype_from_ov_type(cport.element_type),
                }
            )

        out_index = int(image_outputs[0]["index"])
        out_name = compiled.outputs[out_index].any_name

        self.input_metas = input_metas
        self.multi_input = len(input_metas) > 1
        self.in_name = input_metas[0]["name"]
        self.in_dtype = input_metas[0]["dtype"]
        self.out_name = out_name
        self.in_c = in_c
        self.in_h = in_h
        self.in_w = in_w
        self.model_scale_h = model_scale["h"]
        self.model_scale_w = model_scale["w"]
        self.tile_h = tile_h
        self.tile_w = tile_w
        self.tile_overlap = tile_overlap
        self.downsample_interp = downsample_interp
        self.req = req

        self.log(
            f"model_input={in_w}x{in_h}, image_inputs={len(input_metas)}, tile={tile_w}x{tile_h}, "
            f"overlap={tile_overlap}, compile_ms={(t1 - t0) * 1000.0:.1f}"
        )

    def upscale_x2(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        if self.multi_input:
            return run_infer_tiled_multi(
                image=image_bgr,
                req=self.req,
                input_metas=self.input_metas,
                out_name=self.out_name,
                in_h=self.in_h,
                in_w=self.in_w,
                model_scale_h=self.model_scale_h,
                model_scale_w=self.model_scale_w,
                tile_h=self.tile_h,
                tile_w=self.tile_w,
                tile_overlap=self.tile_overlap,
                downsample_interp=self.downsample_interp,
            )

        out, stats = run_infer_tiled(
            image=image_bgr,
            req=self.req,
            in_name=self.in_name,
            out_name=self.out_name,
            in_c=self.in_c,
            in_h=self.in_h,
            in_w=self.in_w,
            in_dtype=self.in_dtype,
            model_scale_h=self.model_scale_h,
            model_scale_w=self.model_scale_w,
            tile_h=self.tile_h,
            tile_w=self.tile_w,
            tile_overlap=self.tile_overlap,
            downsample_interp=self.downsample_interp,
            progress_every=0,
        )
        return out, stats


def upscale_frame_to_scale(
    image_bgr: np.ndarray,
    target_scale: int,
    pipeline: OpenVinoX2Pipeline,
    final_interp: int,
    cancel_event: threading.Event,
    log: Callable[[str], None],
    stage_progress_cb: Optional[Callable[[int, int], None]] = None,
    verbose: bool = True,
) -> np.ndarray:
    if target_scale < SCALE_MIN or target_scale > SCALE_MAX:
        raise RuntimeError(f"scale must be in [{SCALE_MIN},{SCALE_MAX}]. got {target_scale}")

    src_h, src_w = image_bgr.shape[:2]
    ai_factor = _largest_power_of_two_leq(target_scale)
    ai_stages = int(math.log2(ai_factor))
    current = image_bgr

    if verbose:
        log(
            f"target=x{target_scale}, ai_factor=x{ai_factor}, "
            f"final_interp={'none' if ai_factor == target_scale else 'on'}"
        )

    for stage in range(ai_stages):
        if cancel_event.is_set():
            raise JobCancelledError("Cancelled by user.")
        current, stats = pipeline.upscale_x2(current)
        if verbose:
            log(
                f"  stage {stage + 1}/{ai_stages}: x{2 ** (stage + 1)}, "
                f"tile_avg_ms={stats['tile_infer_avg_ms']:.2f}, elapsed_ms={stats['elapsed_ms']:.1f}"
            )
        if stage_progress_cb is not None:
            stage_progress_cb(stage + 1, ai_stages)

    target_w = src_w * target_scale
    target_h = src_h * target_scale
    if current.shape[1] != target_w or current.shape[0] != target_h:
        if cancel_event.is_set():
            raise JobCancelledError("Cancelled by user.")
        current = cv2.resize(current, (target_w, target_h), interpolation=final_interp)
        if verbose:
            log(f"  final_resize -> {target_w}x{target_h}")

    return current


def _video_fourcc_for_suffix(suffix: str) -> int:
    token = suffix.lower()
    if token in {".avi"}:
        return cv2.VideoWriter_fourcc(*"XVID")
    return cv2.VideoWriter_fourcc(*"mp4v")


class UpscaleMediaGUI:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title(f"LunaSR Media Upscale GUI (x{SCALE_MIN}~x{SCALE_MAX})")
        self.root.geometry("980x700")
        self.root.minsize(920, 620)

        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.scale_var = tk.StringVar(value="2")
        self.device_var = tk.StringVar(value="NPU")
        self.preset_var = tk.StringVar(value="quality")
        self.model_var = tk.StringVar(value=str(DEFAULT_MODEL))
        self.cache_var = tk.StringVar(value=str(DEFAULT_CACHE_DIR))
        self.tile_h_var = tk.StringVar(value="0")
        self.tile_w_var = tk.StringVar(value="0")
        self.overlap_var = tk.StringVar(value="16")
        self.final_var = tk.StringVar(value="auto")
        self.soft_post_var = tk.BooleanVar(value=False)
        self.progress_var = tk.DoubleVar(value=0.0)

        self._queue: "queue.Queue[Tuple[str, object]]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._cancel_event = threading.Event()

        self.start_button: Optional[ttk.Button] = None
        self.cancel_button: Optional[ttk.Button] = None
        self.log_text: Optional[tk.Text] = None

        self._build_ui()
        self.root.after(80, self._drain_queue)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=12)
        container.pack(fill="both", expand=True)
        container.columnconfigure(1, weight=1)
        container.rowconfigure(7, weight=1)

        ttk.Label(container, text="Input").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        input_entry = ttk.Entry(container, textvariable=self.input_var)
        input_entry.grid(row=0, column=1, sticky="ew", pady=4)
        ttk.Button(container, text="Browse...", command=self._browse_input).grid(row=0, column=2, pady=4)

        ttk.Label(container, text="Output").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
        output_entry = ttk.Entry(container, textvariable=self.output_var)
        output_entry.grid(row=1, column=1, sticky="ew", pady=4)
        ttk.Button(container, text="Browse...", command=self._browse_output).grid(row=1, column=2, pady=4)

        opts = ttk.Frame(container)
        opts.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(10, 6))
        for i in range(10):
            opts.columnconfigure(i, weight=1)

        ttk.Label(opts, text="Scale").grid(row=0, column=0, sticky="w")
        scale_combo = ttk.Combobox(
            opts,
            textvariable=self.scale_var,
            values=[str(x) for x in range(SCALE_MIN, SCALE_MAX + 1)],
            state="readonly",
            width=8,
        )
        scale_combo.grid(row=1, column=0, sticky="ew", padx=(0, 8))
        scale_combo.bind("<<ComboboxSelected>>", self._on_scale_changed)

        ttk.Label(opts, text="Device").grid(row=0, column=1, sticky="w")
        ttk.Combobox(
            opts,
            textvariable=self.device_var,
            values=["NPU", "GPU", "CPU", "AUTO"],
            state="readonly",
            width=10,
        ).grid(row=1, column=1, sticky="ew", padx=(0, 8))

        ttk.Label(opts, text="Preset").grid(row=0, column=2, sticky="w")
        ttk.Combobox(
            opts,
            textvariable=self.preset_var,
            values=["quality", "speed"],
            state="readonly",
            width=10,
        ).grid(row=1, column=2, sticky="ew", padx=(0, 8))

        ttk.Label(opts, text="Final Upsample").grid(row=0, column=3, sticky="w")
        ttk.Combobox(
            opts,
            textvariable=self.final_var,
            values=["auto", "bilinear", "bicubic", "lanczos"],
            state="readonly",
            width=12,
        ).grid(row=1, column=3, sticky="ew", padx=(0, 8))

        ttk.Label(opts, text="Tile H (0=auto)").grid(row=0, column=4, sticky="w")
        ttk.Entry(opts, textvariable=self.tile_h_var, width=10).grid(row=1, column=4, sticky="ew", padx=(0, 8))

        ttk.Label(opts, text="Tile W (0=auto)").grid(row=0, column=5, sticky="w")
        ttk.Entry(opts, textvariable=self.tile_w_var, width=10).grid(row=1, column=5, sticky="ew", padx=(0, 8))

        ttk.Label(opts, text="Overlap").grid(row=0, column=6, sticky="w")
        ttk.Entry(opts, textvariable=self.overlap_var, width=10).grid(row=1, column=6, sticky="ew", padx=(0, 8))

        ttk.Checkbutton(
            opts,
            text="Soft Postprocess",
            variable=self.soft_post_var,
            onvalue=True,
            offvalue=False,
        ).grid(row=1, column=7, sticky="w", padx=(4, 0))

        ttk.Label(container, text="Model (x2 IR XML)").grid(row=3, column=0, sticky="w", padx=(0, 8), pady=4)
        model_entry = ttk.Entry(container, textvariable=self.model_var)
        model_entry.grid(row=3, column=1, sticky="ew", pady=4)
        ttk.Button(container, text="Browse...", command=self._browse_model).grid(row=3, column=2, pady=4)

        ttk.Label(container, text="Cache Dir").grid(row=4, column=0, sticky="w", padx=(0, 8), pady=4)
        cache_entry = ttk.Entry(container, textvariable=self.cache_var)
        cache_entry.grid(row=4, column=1, sticky="ew", pady=4)
        ttk.Button(container, text="Browse...", command=self._browse_cache).grid(row=4, column=2, pady=4)

        button_bar = ttk.Frame(container)
        button_bar.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(10, 4))
        button_bar.columnconfigure(0, weight=1)
        button_bar.columnconfigure(1, weight=1)
        self.start_button = ttk.Button(button_bar, text="Start", command=self._start_job)
        self.start_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.cancel_button = ttk.Button(button_bar, text="Cancel", command=self._cancel_job, state="disabled")
        self.cancel_button.grid(row=0, column=1, sticky="ew", padx=(6, 0))

        progress = ttk.Progressbar(container, variable=self.progress_var, maximum=100.0)
        progress.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(4, 8))

        log_frame = ttk.LabelFrame(container, text="Log")
        log_frame.grid(row=7, column=0, columnspan=3, sticky="nsew")
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)

        self.log_text = tk.Text(log_frame, height=20, wrap="word", state="disabled")
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scroll.set)

        hint = (
            f"Supports image/video input and x{SCALE_MIN}~x{SCALE_MAX} scaling.\n"
            "For non-power-of-two scales, it runs AI x2 stages then final interpolation.\n"
            "Optional Soft Postprocess applies mild smoothing to reduce alias/ringing."
        )
        ttk.Label(container, text=hint, foreground="#333333").grid(
            row=8, column=0, columnspan=3, sticky="w", pady=(8, 0)
        )

    def _append_log(self, msg: str) -> None:
        if self.log_text is None:
            return
        self.log_text.configure(state="normal")
        self.log_text.insert("end", msg.rstrip() + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _set_busy(self, busy: bool) -> None:
        if self.start_button is not None:
            self.start_button.configure(state="disabled" if busy else "normal")
        if self.cancel_button is not None:
            self.cancel_button.configure(state="normal" if busy else "disabled")

    def _browse_input(self) -> None:
        path = filedialog.askopenfilename(
            title="Select input media",
            filetypes=[
                (
                    "Media files",
                    "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.mp4 *.avi *.mov *.mkv *.wmv *.m4v",
                ),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self.input_var.set(path)
        try:
            scale = int(self.scale_var.get())
        except Exception:
            scale = 2
        self.output_var.set(str(suggest_output_path(Path(path), scale)))

    def _browse_output(self) -> None:
        in_path_raw = _strip_quotes(self.input_var.get())
        in_path = Path(in_path_raw) if in_path_raw else None

        default_ext = ".mp4"
        if in_path is not None and in_path.suffix.lower() in IMAGE_EXTS:
            default_ext = in_path.suffix.lower()

        path = filedialog.asksaveasfilename(
            title="Select output path",
            defaultextension=default_ext,
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.output_var.set(path)

    def _browse_model(self) -> None:
        path = filedialog.askopenfilename(
            title="Select OpenVINO IR XML model",
            filetypes=[("IR XML", "*.xml"), ("All files", "*.*")],
        )
        if path:
            self.model_var.set(path)

    def _browse_cache(self) -> None:
        path = filedialog.askdirectory(title="Select cache directory")
        if path:
            self.cache_var.set(path)

    def _on_scale_changed(self, _event=None) -> None:
        in_path_raw = _strip_quotes(self.input_var.get())
        if not in_path_raw:
            return
        in_path = Path(in_path_raw)
        if not in_path.exists():
            return
        try:
            scale = int(self.scale_var.get())
        except Exception:
            return
        if not self.output_var.get().strip():
            self.output_var.set(str(suggest_output_path(in_path, scale)))

    def _parse_config(self) -> UpscaleJobConfig:
        input_raw = _strip_quotes(self.input_var.get())
        output_raw = _strip_quotes(self.output_var.get())
        model_raw = _strip_quotes(self.model_var.get())
        cache_raw = _strip_quotes(self.cache_var.get())

        if not input_raw:
            raise RuntimeError("Input path is required.")
        input_path = Path(input_raw)
        if not input_path.exists():
            raise RuntimeError(f"Input path not found: {input_path}")
        media_kind = detect_media_kind(input_path)

        scale = int(self.scale_var.get())
        if scale < SCALE_MIN or scale > SCALE_MAX:
            raise RuntimeError(f"Scale must be between {SCALE_MIN} and {SCALE_MAX}.")

        output_path = Path(output_raw) if output_raw else suggest_output_path(input_path, scale)
        if not output_path.suffix:
            if media_kind == "image":
                output_path = output_path.with_suffix(input_path.suffix)
            else:
                output_path = output_path.with_suffix(".mp4")
        out_suffix = output_path.suffix.lower()
        if media_kind == "image" and out_suffix not in IMAGE_EXTS:
            raise RuntimeError(f"Image output must use image extension: {sorted(IMAGE_EXTS)}")
        if media_kind == "video" and out_suffix not in VIDEO_EXTS:
            raise RuntimeError(f"Video output must use video extension: {sorted(VIDEO_EXTS)}")
        if output_path.resolve(strict=False) == input_path.resolve(strict=False):
            raise RuntimeError("Output path must be different from input path.")
        model_path = Path(model_raw) if model_raw else DEFAULT_MODEL
        if not model_path.exists():
            raise RuntimeError(f"Model path not found: {model_path}")

        tile_h = int(self.tile_h_var.get())
        tile_w = int(self.tile_w_var.get())
        overlap = int(self.overlap_var.get())
        if tile_h < 0 or tile_w < 0 or overlap < 0:
            raise RuntimeError("Tile H/W and overlap must be >= 0.")

        cfg = UpscaleJobConfig(
            input_path=input_path,
            output_path=output_path,
            model_path=model_path,
            scale=scale,
            device=self.device_var.get().strip() or "NPU",
            preset=self.preset_var.get().strip() or "quality",
            cache_dir=cache_raw,
            tile_h=tile_h,
            tile_w=tile_w,
            tile_overlap=overlap,
            final_upsample=self.final_var.get().strip() or "auto",
            soft_postprocess=bool(self.soft_post_var.get()),
        )
        return cfg

    def _start_job(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            messagebox.showwarning("Busy", "A job is already running.")
            return

        try:
            cfg = self._parse_config()
        except Exception as exc:
            messagebox.showerror("Invalid input", str(exc))
            return

        self.progress_var.set(0.0)
        self._append_log(
            f"start: input={cfg.input_path}, output={cfg.output_path}, "
            f"scale=x{cfg.scale}, device={cfg.device}, preset={cfg.preset}, "
            f"soft_post={'on' if cfg.soft_postprocess else 'off'}"
        )

        self._set_busy(True)
        self._cancel_event.clear()
        self._worker = threading.Thread(target=self._run_job, args=(cfg,), daemon=True)
        self._worker.start()

    def _cancel_job(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            self._cancel_event.set()
            self._append_log("cancel requested...")

    def _run_job(self, cfg: UpscaleJobConfig) -> None:
        def log(msg: str) -> None:
            self._queue.put(("log", msg))

        def set_progress(value: float) -> None:
            self._queue.put(("progress", float(max(0.0, min(100.0, value)))))

        try:
            t0 = time.perf_counter()
            final_interp = get_final_upsample_interp(cfg.final_upsample, cfg.preset)
            pipeline = OpenVinoX2Pipeline(cfg, log)
            kind = detect_media_kind(cfg.input_path)
            log(f"media={kind}")

            if kind == "image":
                self._process_image(cfg, pipeline, final_interp, log, set_progress)
            else:
                self._process_video(cfg, pipeline, final_interp, log, set_progress)

            elapsed = time.perf_counter() - t0
            set_progress(100.0)
            self._queue.put(
                (
                    "done",
                    {
                        "ok": True,
                        "message": f"Done in {elapsed:.2f}s",
                        "output": str(cfg.output_path),
                    },
                )
            )
        except JobCancelledError:
            self._queue.put(("done", {"ok": False, "message": "Cancelled by user."}))
        except Exception as exc:
            self._queue.put(("log", traceback.format_exc()))
            self._queue.put(("done", {"ok": False, "message": str(exc)}))

    def _process_image(
        self,
        cfg: UpscaleJobConfig,
        pipeline: OpenVinoX2Pipeline,
        final_interp: int,
        log: Callable[[str], None],
        set_progress: Callable[[float], None],
    ) -> None:
        image = read_image_unicode(cfg.input_path)
        if image is None:
            raise RuntimeError(f"Failed to read image: {cfg.input_path}")

        src_h, src_w = image.shape[:2]
        log(f"image_input={src_w}x{src_h}")
        set_progress(5.0)

        def on_stage(stage: int, total: int) -> None:
            if total <= 0:
                return
            pct = 5.0 + (80.0 * float(stage) / float(total))
            set_progress(pct)

        out = upscale_frame_to_scale(
            image_bgr=image,
            target_scale=cfg.scale,
            pipeline=pipeline,
            final_interp=final_interp,
            cancel_event=self._cancel_event,
            log=log,
            stage_progress_cb=on_stage,
            verbose=True,
        )
        if cfg.soft_postprocess:
            out, applied, msg = try_apply_soft_postprocess(out)
            if applied:
                log(
                    f"soft_post=on (blend={SOFT_POST_BLEND:.2f}, sigma={SOFT_POST_SIGMA:.2f})"
                )
            else:
                log(msg)

        cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
        ok = write_image_unicode(cfg.output_path, out)
        if not ok:
            raise RuntimeError(f"Failed to write image: {cfg.output_path}")
        set_progress(100.0)
        log(f"saved_image={cfg.output_path}")

    def _process_video(
        self,
        cfg: UpscaleJobConfig,
        pipeline: OpenVinoX2Pipeline,
        final_interp: int,
        log: Callable[[str], None],
        set_progress: Callable[[float], None],
    ) -> None:
        cap, temp_input_copy = open_video_capture_with_fallback(cfg.input_path, log)

        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        src_fps = float(cap.get(cv2.CAP_PROP_FPS))
        src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if src_fps <= 0:
            src_fps = 30.0

        out_w = src_w * cfg.scale
        out_h = src_h * cfg.scale

        cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = _video_fourcc_for_suffix(cfg.output_path.suffix)
        try:
            writer, temp_output_path = open_video_writer_with_fallback(
                cfg.output_path,
                fourcc,
                src_fps,
                (out_w, out_h),
                log,
            )
        except Exception:
            cap.release()
            if temp_input_copy is not None:
                try:
                    temp_input_copy.unlink(missing_ok=True)
                except Exception:
                    pass
            raise

        log(
            f"video_input={src_w}x{src_h}, fps={src_fps:.3f}, frames={src_frames}, "
            f"video_output={out_w}x{out_h}"
        )
        if cfg.soft_postprocess:
            log(f"soft_post=on (blend={SOFT_POST_BLEND:.2f}, sigma={SOFT_POST_SIGMA:.2f})")
        set_progress(1.0)

        processed = 0
        t0 = time.perf_counter()
        soft_post_runtime_on = bool(cfg.soft_postprocess)
        soft_post_skip_logged = False
        job_ok = False

        try:
            while True:
                if self._cancel_event.is_set():
                    raise JobCancelledError("Cancelled by user.")

                ok, frame = cap.read()
                if not ok or frame is None:
                    break

                out = upscale_frame_to_scale(
                    image_bgr=frame,
                    target_scale=cfg.scale,
                    pipeline=pipeline,
                    final_interp=final_interp,
                    cancel_event=self._cancel_event,
                    log=lambda _: None,
                    stage_progress_cb=None,
                    verbose=False,
                )
                if soft_post_runtime_on:
                    out, applied, msg = try_apply_soft_postprocess(out)
                    if not applied:
                        soft_post_runtime_on = False
                        if not soft_post_skip_logged:
                            log(msg)
                            soft_post_skip_logged = True
                writer.write(out)
                processed += 1

                if src_frames > 0:
                    set_progress((processed * 100.0) / src_frames)
                elif processed % 10 == 0:
                    set_progress(min(99.0, 1.0 + processed * 0.1))

                if processed % 5 == 0:
                    elapsed = time.perf_counter() - t0
                    fps_now = processed / elapsed if elapsed > 0 else 0.0
                    total_txt = str(src_frames) if src_frames > 0 else "?"
                    log(f"[frame {processed}/{total_txt}] avg_fps={fps_now:.2f}")
            job_ok = processed > 0
        finally:
            cap.release()
            writer.release()
            if temp_input_copy is not None:
                try:
                    temp_input_copy.unlink(missing_ok=True)
                except Exception:
                    pass
            if temp_output_path is not None:
                if job_ok:
                    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        if cfg.output_path.exists():
                            cfg.output_path.unlink()
                    except Exception:
                        pass
                    shutil.move(str(temp_output_path), str(cfg.output_path))
                else:
                    try:
                        temp_output_path.unlink(missing_ok=True)
                    except Exception:
                        pass

        if processed < 1:
            raise RuntimeError("No frame was processed from the input video.")

        elapsed = time.perf_counter() - t0
        avg_fps = processed / elapsed if elapsed > 0 else 0.0
        set_progress(100.0)
        log(f"saved_video={cfg.output_path}")
        log(f"processed_frames={processed}, elapsed_s={elapsed:.2f}, avg_fps={avg_fps:.2f}")

    def _drain_queue(self) -> None:
        while True:
            try:
                kind, payload = self._queue.get_nowait()
            except queue.Empty:
                break

            if kind == "log":
                self._append_log(str(payload))
            elif kind == "progress":
                self.progress_var.set(float(payload))
            elif kind == "done":
                info = payload if isinstance(payload, dict) else {"ok": False, "message": str(payload)}
                self._set_busy(False)
                ok = bool(info.get("ok", False))
                msg = str(info.get("message", ""))
                self._append_log(msg)
                if ok:
                    out = str(info.get("output", ""))
                    messagebox.showinfo("Done", f"{msg}\n\nOutput:\n{out}")
                else:
                    messagebox.showerror("Failed", msg)

        self.root.after(80, self._drain_queue)

    def _on_close(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            if not messagebox.askyesno("Exit", "A job is running. Cancel and exit?"):
                return
            self._cancel_event.set()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> int:
    app = UpscaleMediaGUI()
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
