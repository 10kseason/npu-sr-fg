import argparse
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import openvino as ov


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
        raise RuntimeError(f"Output tensor is not an image-like tensor: shape={y.shape}")

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


def build_compile_cfg(cache_dir: str, preset: str) -> Dict[str, object]:
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


def get_skip_interp(name: str) -> int:
    if name == "bilinear":
        return cv2.INTER_LINEAR
    if name == "bicubic":
        return cv2.INTER_CUBIC
    raise RuntimeError(f"Unsupported skip mode: {name}")


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


def run_infer_tiled(
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
    tiles_x = math.ceil(src_w / core_w)
    tiles_y = math.ceil(src_h / core_h)
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
            core_h_cur = y1 - y0
            core_w_cur = x1 - x0

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
            cx1 = cx0 + core_w_cur * 2
            cy1 = cy0 + core_h_cur * 2
            out_core = out_x2[cy0:cy1, cx0:cx1]
            dst[y0 * 2 : y1 * 2, x0 * 2 : x1 * 2] = out_core

    t1 = time.perf_counter()
    return dst, {
        "tiles": float(total_tiles),
        "tile_infer_avg_ms": float(np.mean(infer_latencies_ms)) if infer_latencies_ms else 0.0,
        "elapsed_ms": (t1 - t0) * 1000.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Upscale MP4 by x2 with NPU and save MP4")
    parser.add_argument("--input", required=True, help="Input MP4 path")
    parser.add_argument("--output", required=True, help="Output MP4 path")
    parser.add_argument(
        "--model",
        default="model/ir/fixed_sr_algo_x2_sharp067.xml",
        help="OpenVINO IR XML model path",
    )
    parser.add_argument("--preset", choices=["quality", "speed"], default="speed")
    parser.add_argument("--device", default="NPU")
    parser.add_argument("--cache-dir", default=".ov_cache")
    parser.add_argument("--tile-h", type=int, default=0)
    parser.add_argument("--tile-w", type=int, default=0)
    parser.add_argument(
        "--tile-overlap",
        type=int,
        default=16,
        help="Tile overlap (input pixels) to reduce grid seams. 0 disables overlap.",
    )
    parser.add_argument("--internal-scale", type=float, default=0.5, help="(0,1] internal processing scale")
    parser.add_argument(
        "--final-upsample",
        choices=["auto", "bilinear", "bicubic", "lanczos"],
        default="auto",
    )
    parser.add_argument("--process-every", type=int, default=1, help="Run AI every K frames")
    parser.add_argument("--skip-mode", choices=["reuse", "bilinear", "bicubic"], default="reuse")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means full video")
    parser.add_argument("--progress-every", type=int, default=5, help="Print every N frames")
    args = parser.parse_args()

    if not (0.0 < args.internal_scale <= 1.0):
        raise RuntimeError(f"--internal-scale must be in (0, 1]. got {args.internal_scale}")
    if args.process_every < 1:
        raise RuntimeError(f"--process-every must be >= 1. got {args.process_every}")

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {args.input}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = float(cap.get(cv2.CAP_PROP_FPS))
    src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if src_fps <= 0:
        src_fps = 30.0

    out_w = src_w * 2
    out_h = src_h * 2
    target_frames = src_frames if args.max_frames <= 0 else min(src_frames, args.max_frames)

    print(f"input_video={args.input}")
    print(f"input_shape={src_w}x{src_h}, fps={src_fps:.3f}, frames={src_frames}")
    print(f"target_output={out_w}x{out_h}, target_frames={target_frames}")
    print(
        f"preset={args.preset}, internal_scale={args.internal_scale:.3f}, "
        f"process_every={args.process_every}, skip_mode={args.skip_mode}"
    )

    core = ov.Core()
    model = core.read_model(args.model)
    if len(model.inputs) != 1 or len(model.outputs) != 1:
        raise RuntimeError("This script supports single-input, single-output image models.")

    in_port = model.inputs[0]
    out_port = model.outputs[0]
    in_shape = [int(x) for x in in_port.shape]
    out_shape = [int(x) for x in out_port.shape]
    if len(in_shape) != 4 or len(out_shape) != 4:
        raise RuntimeError(f"Expected 4D NCHW model. got input={in_shape}, output={out_shape}")
    in_n, in_c, in_h, in_w = in_shape
    if in_n != 1:
        raise RuntimeError(f"Only batch=1 is supported. got N={in_n}")

    tile_overlap = max(0, int(args.tile_overlap))
    if tile_overlap * 2 >= in_h or tile_overlap * 2 >= in_w:
        raise RuntimeError(
            f"--tile-overlap is too large for model input {in_h}x{in_w}. got overlap={tile_overlap}"
        )

    tile_h_default = in_h - (tile_overlap * 2)
    tile_w_default = in_w - (tile_overlap * 2)
    tile_h = args.tile_h if args.tile_h > 0 else tile_h_default
    tile_w = args.tile_w if args.tile_w > 0 else tile_w_default
    if tile_h <= 0 or tile_w <= 0:
        raise RuntimeError(
            f"Computed tile size is invalid. tile={tile_w}x{tile_h}, overlap={tile_overlap}, model={in_w}x{in_h}"
        )
    if tile_h + (tile_overlap * 2) > in_h or tile_w + (tile_overlap * 2) > in_w:
        raise RuntimeError(
            f"tile+2*overlap must be <= model input. got tile={tile_w}x{tile_h}, "
            f"overlap={tile_overlap}, model={in_w}x{in_h}"
        )
    if tile_h > in_h or tile_w > in_w:
        raise RuntimeError(
            f"tile size must be <= model input size. got tile={tile_h}x{tile_w}, model={in_h}x{in_w}"
        )

    model_scale = get_model_scale(in_shape, out_shape)
    compile_cfg = build_compile_cfg(args.cache_dir, args.preset)
    downsample_interp = get_downsample_interp(args.preset)
    final_upsample_interp = get_final_upsample_interp(args.final_upsample, args.preset)

    print(
        f"model_input={in_shape}, model_output={out_shape}, "
        f"tile_input={tile_w}x{tile_h}, tile_overlap={tile_overlap}"
    )
    print(f"compile_cfg={compile_cfg}")

    t_compile0 = time.perf_counter()
    compiled = core.compile_model(model, args.device, compile_cfg)
    req = compiled.create_infer_request()
    t_compile1 = time.perf_counter()
    print(f"compile_ms={(t_compile1 - t_compile0) * 1000.0:.3f}")

    in_name = compiled.inputs[0].any_name
    out_name = compiled.outputs[0].any_name
    in_dtype = np_dtype_from_ov_type(compiled.inputs[0].element_type)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, src_fps, (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open output video writer: {out_path}")

    processed = 0
    ai_frames = 0
    skipped_frames = 0
    frame_ms_list = []
    ai_frame_ms_list = []
    ai_tile_ms_list = []
    last_ai_frame: Optional[np.ndarray] = None

    t_total0 = time.perf_counter()
    while processed < target_frames:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        frame_t0 = time.perf_counter()
        do_infer = (processed % args.process_every == 0) or (last_ai_frame is None)

        if do_infer:
            internal_h = max(1, int(round(frame.shape[0] * args.internal_scale)))
            internal_w = max(1, int(round(frame.shape[1] * args.internal_scale)))
            internal_frame = (
                cv2.resize(frame, (internal_w, internal_h), interpolation=cv2.INTER_AREA)
                if args.internal_scale < 1.0
                else frame
            )

            out_internal, stats = run_infer_tiled(
                image=internal_frame,
                req=req,
                in_name=in_name,
                out_name=out_name,
                in_c=in_c,
                in_h=in_h,
                in_w=in_w,
                in_dtype=in_dtype,
                model_scale_h=model_scale["h"],
                model_scale_w=model_scale["w"],
                tile_h=tile_h,
                tile_w=tile_w,
                tile_overlap=tile_overlap,
                downsample_interp=downsample_interp,
            )
            out_final = (
                cv2.resize(out_internal, (out_w, out_h), interpolation=final_upsample_interp)
                if args.internal_scale < 1.0
                else out_internal
            )

            ai_frames += 1
            ai_frame_ms_list.append(float(stats["elapsed_ms"]))
            ai_tile_ms_list.append(float(stats["tile_infer_avg_ms"]))
            last_ai_frame = out_final
        else:
            skipped_frames += 1
            if args.skip_mode == "reuse":
                out_final = last_ai_frame
            else:
                out_final = cv2.resize(frame, (out_w, out_h), interpolation=get_skip_interp(args.skip_mode))

        writer.write(out_final)
        processed += 1
        frame_t1 = time.perf_counter()
        frame_ms = (frame_t1 - frame_t0) * 1000.0
        frame_ms_list.append(frame_ms)

        if processed % max(1, args.progress_every) == 0 or processed == target_frames:
            print(f"[frame {processed}/{target_frames}] frame_ms={frame_ms:.3f}")

    t_total1 = time.perf_counter()
    cap.release()
    writer.release()

    total_elapsed_ms = (t_total1 - t_total0) * 1000.0
    effective_avg_ms = total_elapsed_ms / max(1, processed)
    effective_fps = 1000.0 / effective_avg_ms if effective_avg_ms > 0 else 0.0
    ai_frame_avg_ms = float(np.mean(ai_frame_ms_list)) if ai_frame_ms_list else 0.0
    ai_tile_avg_ms = float(np.mean(ai_tile_ms_list)) if ai_tile_ms_list else 0.0

    print(f"saved_video={out_path}")
    print(f"processed_frames={processed}, ai_frames={ai_frames}, skipped_frames={skipped_frames}")
    print(f"ai_tile_infer_avg_ms={ai_tile_avg_ms:.3f}")
    print(f"ai_frame_avg_ms={ai_frame_avg_ms:.3f}")
    print(f"effective_frame_avg_ms={effective_avg_ms:.3f}, effective_fps={effective_fps:.3f}")
    print(f"total_elapsed_ms={total_elapsed_ms:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
