import argparse
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import openvino as ov

GPU_TILE_BASE_MAX_W = 1280
GPU_TILE_BASE_MAX_H = 720


def np_dtype_from_ov_type(etype: ov.Type):
    if etype in (ov.Type.f32,):
        return np.float32
    if etype in (ov.Type.f16, ov.Type.bf16):
        return np.float16
    if etype in (ov.Type.i64,):
        return np.int64
    if etype in (ov.Type.i32,):
        return np.int32
    if etype in (ov.Type.i16,):
        return np.int16
    if etype in (ov.Type.i8,):
        return np.int8
    if etype in (ov.Type.u8,):
        return np.uint8
    if etype in (ov.Type.boolean,):
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
        raise RuntimeError(f"Unsupported channel count: C={c}")

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


def get_model_scale(input_shape: List[int], output_shape: List[int]) -> Dict[str, float]:
    in_h, in_w = int(input_shape[2]), int(input_shape[3])
    out_h, out_w = int(output_shape[2]), int(output_shape[3])
    return {"h": out_h / in_h, "w": out_w / in_w}


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


def build_sr_compile_cfg(cache_dir: str, preset: str) -> Dict[str, object]:
    cfg: Dict[str, object] = {}
    if cache_dir:
        cfg["CACHE_DIR"] = cache_dir

    if preset == "quality":
        cfg["INFERENCE_PRECISION_HINT"] = "f16"
    elif preset == "speed":
        cfg["INFERENCE_PRECISION_HINT"] = "i8"
        cfg["NPU_COMPILER_DYNAMIC_QUANTIZATION"] = True
        cfg["NPU_QDQ_OPTIMIZATION"] = True
        cfg["NPU_QDQ_OPTIMIZATION_AGGRESSIVE"] = True
    else:
        raise RuntimeError(f"Unsupported preset: {preset}")
    return cfg


def build_fg_compile_cfg(cache_dir: str, fg_precision: str) -> Dict[str, object]:
    cfg: Dict[str, object] = {}
    if cache_dir:
        cfg["CACHE_DIR"] = cache_dir

    cfg["INFERENCE_PRECISION_HINT"] = fg_precision
    if fg_precision == "i8":
        cfg["NPU_COMPILER_DYNAMIC_QUANTIZATION"] = True
        cfg["NPU_QDQ_OPTIMIZATION"] = True
        cfg["NPU_QDQ_OPTIMIZATION_AGGRESSIVE"] = True
    return cfg


def get_downsample_interp(preset: str) -> int:
    return cv2.INTER_LANCZOS4 if preset == "quality" else cv2.INTER_AREA


def get_final_upsample_interp(name: str, preset: str) -> int:
    if name == "auto":
        return cv2.INTER_LANCZOS4 if preset == "quality" else cv2.INTER_CUBIC
    if name == "bilinear":
        return cv2.INTER_LINEAR
    if name == "bicubic":
        return cv2.INTER_CUBIC
    if name == "lanczos":
        return cv2.INTER_LANCZOS4
    return cv2.INTER_CUBIC


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


def run_sr_tiled(
    image: np.ndarray,
    req,
    in_name: str,
    out_name: str,
    in_c: int,
    in_h: int,
    in_w: int,
    in_dtype,
    model_scale_h: float,
    model_scale_w: float,
    tile_h: int,
    tile_w: int,
    tile_overlap: int,
    downsample_interp: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    src_h, src_w = image.shape[:2]
    dst = np.zeros((src_h * 2, src_w * 2, 3), dtype=np.uint8)
    core_h = max(1, tile_h - tile_overlap * 2)
    core_w = max(1, tile_w - tile_overlap * 2)
    tiles_x = (src_w + core_w - 1) // core_w
    tiles_y = (src_h + core_h - 1) // core_h
    total_tiles = tiles_x * tiles_y

    infer_latencies_ms = []
    t0 = time.perf_counter()

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

            x = to_nchw_tensor(patch, in_c, in_h, in_w, in_dtype)
            ti0 = time.perf_counter()
            outputs = req.infer({in_name: x})
            ti1 = time.perf_counter()
            infer_latencies_ms.append((ti1 - ti0) * 1000.0)

            y = outputs[out_name] if out_name in outputs else next(iter(outputs.values()))
            out_tile = postprocess_tensor(np.asarray(y))
            out_x2 = resize_to_x2(
                out_tile, valid_h, valid_w, model_scale_h, model_scale_w, downsample_interp
            )
            cx0 = core_x0 * 2
            cy0 = core_y0 * 2
            cx1 = cx0 + core_out_w
            cy1 = cy0 + core_out_h
            out_core = out_x2[cy0:cy1, cx0:cx1]
            dst[y0 * 2 : y1 * 2, x0 * 2 : x1 * 2] = out_core

    t1 = time.perf_counter()
    return dst, {
        "tiles": float(total_tiles),
        "tile_infer_avg_ms": float(np.mean(infer_latencies_ms)) if infer_latencies_ms else 0.0,
        "elapsed_ms": (t1 - t0) * 1000.0,
    }


def reduce_fg_ghosting(
    mid: np.ndarray,
    frame0: np.ndarray,
    frame1: np.ndarray,
    strength: float = 0.20,
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


def compile_fg_model(
    core: ov.Core,
    fg_model_path: str,
    device: str,
    cache_dir: str,
    fg_precision: str,
    fg_size: int,
):
    model = core.read_model(fg_model_path)
    reshape_map = {}
    for inp in model.inputs:
        rank = len(inp.partial_shape)
        if rank == 4:
            reshape_map[inp.any_name] = [1, 3, fg_size, fg_size]
        elif rank == 1:
            reshape_map[inp.any_name] = [1]
    if reshape_map:
        model.reshape(reshape_map)

    cfg = build_fg_compile_cfg(cache_dir, fg_precision)
    t0 = time.perf_counter()
    compiled = core.compile_model(model, device, cfg)
    t1 = time.perf_counter()
    return compiled, cfg, (t1 - t0) * 1000.0


def run_fg_mid(
    req,
    inputs,
    out_name: str,
    frame0: np.ndarray,
    frame1: np.ndarray,
    timestep: float,
    fg_size: int,
    anti_ghost: float = 0.20,
) -> Tuple[np.ndarray, float]:
    feed = {}
    image_input_names = []
    image_input_shapes = []

    for inp in inputs:
        name = inp.any_name
        lname = name.lower()
        shape = [int(x) for x in inp.shape]
        dtype = np_dtype_from_ov_type(inp.element_type)

        if "timestep" in lname and int(np.prod(shape)) == 1:
            feed[name] = np.array([timestep], dtype=dtype).reshape(shape)
        elif len(shape) == 4 and shape[0] == 1 and shape[1] in (1, 3, 4):
            image_input_names.append(name)
            image_input_shapes.append((shape, dtype))
        else:
            if np.issubdtype(dtype, np.floating):
                feed[name] = np.zeros(shape, dtype=dtype)
            elif np.issubdtype(dtype, np.integer):
                feed[name] = np.zeros(shape, dtype=dtype)
            else:
                feed[name] = np.zeros(shape, dtype=np.float32)

    if len(image_input_names) < 2:
        raise RuntimeError("FG model needs at least two image inputs (img0/img1).")

    mapped = {}
    for i, name in enumerate(image_input_names):
        lname = name.lower()
        if "img0" in lname:
            mapped[name] = frame0
        elif "img1" in lname:
            mapped[name] = frame1
        else:
            mapped[name] = frame0 if i == 0 else frame1

    for i, name in enumerate(image_input_names):
        shape, dtype = image_input_shapes[i]
        _, c, h, w = shape
        src = mapped[name]
        feed[name] = to_nchw_tensor(src, c, h, w, dtype)

    t0 = time.perf_counter()
    outputs = req.infer(feed)
    t1 = time.perf_counter()
    y = outputs[out_name] if out_name in outputs else next(iter(outputs.values()))
    mid = postprocess_tensor(np.asarray(y))
    if mid.shape[0] != fg_size or mid.shape[1] != fg_size:
        mid = cv2.resize(mid, (fg_size, fg_size), interpolation=cv2.INTER_CUBIC)
    mid = reduce_fg_ghosting(mid=mid, frame0=frame0, frame1=frame1, strength=float(anti_ghost))
    return mid, (t1 - t0) * 1000.0


def main() -> int:
    parser = argparse.ArgumentParser(description="NPU video x2 upscale with FG frame interpolation")
    parser.add_argument("--input", required=True, help="Input MP4 path")
    parser.add_argument("--output", required=True, help="Output MP4 path")
    parser.add_argument("--sr-model", default="model/ir/fixed_sr_algo_x2_sharp067.xml")
    parser.add_argument("--fg-model", default="model/ir/fixed_fg_algo_mid.xml")
    parser.add_argument("--preset", choices=["quality", "speed"], default="speed")
    parser.add_argument("--fg-precision", choices=["f16", "i8"], default="f16")
    parser.add_argument("--device", default="NPU")
    parser.add_argument("--cache-dir", default=".ov_cache")
    parser.add_argument("--tile-h", type=int, default=0)
    parser.add_argument("--tile-w", type=int, default=0)
    parser.add_argument("--tile-overlap", type=int, default=8, help="SR tile overlap in input pixels.")
    parser.add_argument("--internal-scale", type=float, default=0.5)
    parser.add_argument("--fg-size", type=int, default=256, help="FG input size (square)")
    parser.add_argument("--timestep", type=float, default=0.5)
    parser.add_argument("--fg-anti-ghost", type=float, default=0.20, help="Motion-adaptive anti-ghost blend (0~1).")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means full video")
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument(
        "--final-upsample",
        choices=["auto", "bilinear", "bicubic", "lanczos"],
        default="auto",
    )
    args = parser.parse_args()

    if not (0.0 < args.internal_scale <= 1.0):
        raise RuntimeError(f"--internal-scale must be in (0, 1]. got {args.internal_scale}")
    if args.fg_size < 16:
        raise RuntimeError(f"--fg-size is too small: {args.fg_size}")

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {args.input}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = float(cap.get(cv2.CAP_PROP_FPS))
    src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if src_fps <= 0:
        src_fps = 30.0
    target_in_frames = src_frames if args.max_frames <= 0 else min(src_frames, args.max_frames)

    out_w = src_w * 2
    out_h = src_h * 2
    out_fps = src_fps * 2.0

    print(f"input_video={args.input}")
    print(f"input_shape={src_w}x{src_h}, fps={src_fps:.3f}, frames={src_frames}")
    print(f"target_input_frames={target_in_frames}")
    print(f"output_shape={out_w}x{out_h}, output_fps={out_fps:.3f}")

    core = ov.Core()

    sr_model = core.read_model(args.sr_model)
    if len(sr_model.inputs) != 1 or len(sr_model.outputs) != 1:
        cap.release()
        raise RuntimeError("SR model must be single-input and single-output.")
    sr_in_shape = [int(x) for x in sr_model.inputs[0].shape]
    sr_out_shape = [int(x) for x in sr_model.outputs[0].shape]
    if len(sr_in_shape) != 4 or len(sr_out_shape) != 4:
        cap.release()
        raise RuntimeError(f"SR model must be 4D NCHW. got {sr_in_shape} -> {sr_out_shape}")

    sr_n, sr_c, sr_h, sr_w = sr_in_shape
    if sr_n != 1:
        cap.release()
        raise RuntimeError(f"SR model must use batch=1. got N={sr_n}")

    tile_h = args.tile_h if args.tile_h > 0 else sr_h
    tile_w = args.tile_w if args.tile_w > 0 else sr_w
    if tile_h > sr_h or tile_w > sr_w:
        cap.release()
        raise RuntimeError(
            f"tile size must be <= SR model input. got tile={tile_h}x{tile_w}, model={sr_h}x{sr_w}"
        )
    tile_overlap = max(0, int(args.tile_overlap))
    tile_overlap = min(tile_overlap, (tile_h - 1) // 2, (tile_w - 1) // 2)

    sr_scale = get_model_scale(sr_in_shape, sr_out_shape)
    sr_cfg = build_sr_compile_cfg(args.cache_dir, args.preset)
    downsample_interp = get_downsample_interp(args.preset)
    final_upsample_interp = get_final_upsample_interp(args.final_upsample, args.preset)

    print(f"sr_model={args.sr_model}")
    print(f"sr_input={sr_in_shape}, sr_output={sr_out_shape}, tile={tile_w}x{tile_h}, overlap={tile_overlap}")
    print(f"sr_cfg={sr_cfg}")

    t_sr0 = time.perf_counter()
    compiled_sr = core.compile_model(sr_model, args.device, sr_cfg)
    sr_req = compiled_sr.create_infer_request()
    t_sr1 = time.perf_counter()
    print(f"sr_compile_ms={(t_sr1 - t_sr0) * 1000.0:.3f}")
    sr_exec_devices = "unknown"
    try:
        v = compiled_sr.get_property("EXECUTION_DEVICES")
        if isinstance(v, (list, tuple)):
            sr_exec_devices = ",".join(str(x) for x in v)
        else:
            sr_exec_devices = str(v)
    except Exception:
        pass

    sr_in_name = compiled_sr.inputs[0].any_name
    sr_out_name = compiled_sr.outputs[0].any_name
    sr_in_dtype = np_dtype_from_ov_type(compiled_sr.inputs[0].element_type)
    gpu_tile_720p_on = ("GPU" in str(args.device).upper()) or ("GPU" in str(sr_exec_devices).upper())
    print(f"sr_exec_devices={sr_exec_devices}, gpu_tile_base_720p={'on' if gpu_tile_720p_on else 'off'}")

    print(f"fg_model={args.fg_model}, fg_size={args.fg_size}, fg_precision={args.fg_precision}")
    print(f"fg_anti_ghost={float(args.fg_anti_ghost):.3f}")
    compiled_fg, fg_cfg, fg_compile_ms = compile_fg_model(
        core=core,
        fg_model_path=args.fg_model,
        device=args.device,
        cache_dir=args.cache_dir,
        fg_precision=args.fg_precision,
        fg_size=args.fg_size,
    )
    fg_req = compiled_fg.create_infer_request()
    fg_inputs = compiled_fg.inputs
    fg_out_name = compiled_fg.outputs[0].any_name
    print(f"fg_cfg={fg_cfg}")
    print(f"fg_compile_ms={fg_compile_ms:.3f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, out_fps, (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to create output writer: {out_path}")

    ok, prev = cap.read()
    if not ok or prev is None:
        cap.release()
        writer.release()
        raise RuntimeError("Input video has no readable frame.")

    prev = cv2.resize(prev, (src_w, src_h), interpolation=cv2.INTER_CUBIC)
    processed_input_frames = 1
    output_frames = 0
    sr_frame_ms = []
    fg_ms = []

    def upscale_to_x2(frame_bgr: np.ndarray) -> Tuple[np.ndarray, float]:
        internal_h = max(1, int(round(frame_bgr.shape[0] * args.internal_scale)))
        internal_w = max(1, int(round(frame_bgr.shape[1] * args.internal_scale)))
        if gpu_tile_720p_on:
            internal_w, internal_h = clamp_size_preserve_aspect(
                src_w=internal_w,
                src_h=internal_h,
                max_w=GPU_TILE_BASE_MAX_W,
                max_h=GPU_TILE_BASE_MAX_H,
            )
        internal = (
            cv2.resize(frame_bgr, (internal_w, internal_h), interpolation=cv2.INTER_AREA)
            if (internal_w != frame_bgr.shape[1] or internal_h != frame_bgr.shape[0])
            else frame_bgr
        )
        up_internal, stats = run_sr_tiled(
            image=internal,
            req=sr_req,
            in_name=sr_in_name,
            out_name=sr_out_name,
            in_c=sr_c,
            in_h=sr_h,
            in_w=sr_w,
            in_dtype=sr_in_dtype,
            model_scale_h=sr_scale["h"],
            model_scale_w=sr_scale["w"],
            tile_h=tile_h,
            tile_w=tile_w,
            tile_overlap=tile_overlap,
            downsample_interp=downsample_interp,
        )
        up_final = (
            cv2.resize(up_internal, (out_w, out_h), interpolation=final_upsample_interp)
            if args.internal_scale < 1.0
            else up_internal
        )
        return up_final, float(stats["elapsed_ms"])

    t_total0 = time.perf_counter()
    sr_prev, sr_prev_ms = upscale_to_x2(prev)
    sr_frame_ms.append(sr_prev_ms)

    while processed_input_frames < target_in_frames:
        ok, curr = cap.read()
        if not ok or curr is None:
            break
        curr = cv2.resize(curr, (src_w, src_h), interpolation=cv2.INTER_CUBIC)

        fg0 = cv2.resize(prev, (args.fg_size, args.fg_size), interpolation=cv2.INTER_AREA)
        fg1 = cv2.resize(curr, (args.fg_size, args.fg_size), interpolation=cv2.INTER_AREA)
        mid_fg, fg_elapsed_ms = run_fg_mid(
            req=fg_req,
            inputs=fg_inputs,
            out_name=fg_out_name,
            frame0=fg0,
            frame1=fg1,
            timestep=args.timestep,
            fg_size=args.fg_size,
            anti_ghost=float(args.fg_anti_ghost),
        )
        fg_ms.append(fg_elapsed_ms)
        mid = cv2.resize(mid_fg, (src_w, src_h), interpolation=cv2.INTER_CUBIC)

        sr_mid, sr_mid_ms = upscale_to_x2(mid)
        sr_curr, sr_curr_ms = upscale_to_x2(curr)
        sr_frame_ms.append(sr_mid_ms)
        sr_frame_ms.append(sr_curr_ms)

        writer.write(sr_prev)
        writer.write(sr_mid)
        output_frames += 2

        prev = curr
        sr_prev = sr_curr
        processed_input_frames += 1

        if processed_input_frames % max(1, args.progress_every) == 0 or processed_input_frames == target_in_frames:
            print(
                f"[in_frame {processed_input_frames}/{target_in_frames}] "
                f"sr_prev_ms={sr_curr_ms:.3f}, fg_ms={fg_elapsed_ms:.3f}, out_frames={output_frames}"
            )

    writer.write(sr_prev)
    output_frames += 1

    t_total1 = time.perf_counter()
    cap.release()
    writer.release()

    total_elapsed_ms = (t_total1 - t_total0) * 1000.0
    effective_in_fps = (1000.0 * processed_input_frames / total_elapsed_ms) if total_elapsed_ms > 0 else 0.0
    effective_out_fps = (1000.0 * output_frames / total_elapsed_ms) if total_elapsed_ms > 0 else 0.0
    avg_sr_ms = float(np.mean(sr_frame_ms)) if sr_frame_ms else 0.0
    avg_fg_ms = float(np.mean(fg_ms)) if fg_ms else 0.0

    print(f"saved_video={out_path}")
    print(f"processed_input_frames={processed_input_frames}, output_frames={output_frames}")
    print(f"avg_fg_ms={avg_fg_ms:.3f}, avg_sr_frame_ms={avg_sr_ms:.3f}")
    print(f"effective_input_fps={effective_in_fps:.3f}, effective_output_fps={effective_out_fps:.3f}")
    print(f"total_elapsed_ms={total_elapsed_ms:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
