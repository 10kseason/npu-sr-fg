import argparse
import csv
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
    scale_h = out_h / in_h
    scale_w = out_w / in_w
    return {"h": scale_h, "w": scale_w}


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
    if preset == "quality":
        return cv2.INTER_LANCZOS4
    if preset == "speed":
        return cv2.INTER_AREA
    return cv2.INTER_AREA


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
):
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
    progress_every: int,
    frame_label: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    src_h, src_w = image.shape[:2]
    dst = np.zeros((src_h * 2, src_w * 2, 3), dtype=np.uint8)

    tiles_x = math.ceil(src_w / tile_w)
    tiles_y = math.ceil(src_h / tile_h)
    total_tiles = tiles_x * tiles_y

    infer_latencies_ms = []
    tile_idx = 0
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

            x = to_nchw_tensor(patch, in_c, in_h, in_w, in_dtype)

            t0 = time.perf_counter()
            outputs = req.infer({in_name: x})
            t1 = time.perf_counter()
            infer_latencies_ms.append((t1 - t0) * 1000.0)

            y = outputs[out_name] if out_name in outputs else next(iter(outputs.values()))
            y = np.asarray(y)
            out_tile = postprocess_tensor(y)
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

            tile_idx += 1
            if progress_every > 0 and (tile_idx % progress_every == 0 or tile_idx == total_tiles):
                prefix = f"[{frame_label}] " if frame_label else ""
                print(
                    f"{prefix}[tile {tile_idx}/{total_tiles}] "
                    f"last_infer_ms={infer_latencies_ms[-1]:.3f}"
                )

    t_total1 = time.perf_counter()
    avg_ms = float(np.mean(infer_latencies_ms)) if infer_latencies_ms else 0.0
    return dst, {
        "tiles": float(total_tiles),
        "tile_infer_avg_ms": avg_ms,
        "elapsed_ms": (t_total1 - t_total0) * 1000.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Upscale PNG with NPU and save exact x2 output")
    parser.add_argument("--input", required=True, help="Input PNG path")
    parser.add_argument("--output", required=True, help="Output PNG path (x2 size)")
    parser.add_argument(
        "--model",
        default="model/ir/fixed_sr_algo_x2_sharp067.xml",
        help="OpenVINO IR XML model path",
    )
    parser.add_argument(
        "--preset",
        choices=["quality", "speed"],
        default="quality",
        help="x4plus preset: quality=f16, speed=i8cfg",
    )
    parser.add_argument("--device", default="NPU", help="OpenVINO device (default: NPU)")
    parser.add_argument("--cache-dir", default=".ov_cache", help="OpenVINO CACHE_DIR")
    parser.add_argument("--tile-h", type=int, default=0, help="Input tile height (0 uses model input height)")
    parser.add_argument("--tile-w", type=int, default=0, help="Input tile width (0 uses model input width)")
    parser.add_argument(
        "--tile-overlap",
        type=int,
        default=16,
        help="Tile overlap (input pixels) to reduce grid seams. 0 disables overlap.",
    )
    parser.add_argument("--progress-every", type=int, default=10, help="Print progress every N tiles (0 disables)")
    parser.add_argument(
        "--internal-scale",
        type=float,
        default=1.0,
        help="Process on downscaled input internally (0<scale<=1.0), then resize back to exact x2 output",
    )
    parser.add_argument(
        "--final-upsample",
        choices=["auto", "bilinear", "bicubic", "lanczos"],
        default="auto",
        help="Interpolation when internal-scale < 1 and result is resized back to exact x2 output size",
    )
    parser.add_argument(
        "--benchmark-frames",
        type=int,
        default=1,
        help="Run N frames to estimate effective FPS (single PNG reused as stream input)",
    )
    parser.add_argument(
        "--process-every",
        type=int,
        default=1,
        help="Run AI inference every K frames (others are skip frames)",
    )
    parser.add_argument(
        "--skip-mode",
        choices=["reuse", "bilinear", "bicubic"],
        default="reuse",
        help="For skipped frames: reuse last AI frame or use cheap interpolation",
    )
    parser.add_argument("--bench-csv", default="", help="Optional CSV path for per-frame benchmark rows")
    args = parser.parse_args()

    image = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read input image: {args.input}")
    src_h, src_w = image.shape[:2]
    print(f"input_shape={src_w}x{src_h}")
    print(f"target_output_shape={src_w * 2}x{src_h * 2}")

    if not (0.0 < args.internal_scale <= 1.0):
        raise RuntimeError(f"--internal-scale must be in (0, 1]. got {args.internal_scale}")
    if args.benchmark_frames < 1:
        raise RuntimeError(f"--benchmark-frames must be >= 1. got {args.benchmark_frames}")
    if args.process_every < 1:
        raise RuntimeError(f"--process-every must be >= 1. got {args.process_every}")

    core = ov.Core()
    model = core.read_model(args.model)
    if len(model.inputs) != 1 or len(model.outputs) != 1:
        raise RuntimeError("This script currently supports single-input, single-output image models.")

    in_port = model.inputs[0]
    out_port = model.outputs[0]
    in_shape = [int(x) for x in in_port.shape]
    out_shape = [int(x) for x in out_port.shape]
    if len(in_shape) != 4 or len(out_shape) != 4:
        raise RuntimeError(f"Expected 4D NCHW shapes. got input={in_shape}, output={out_shape}")

    in_n, in_c, in_h, in_w = in_shape
    if in_n != 1:
        raise RuntimeError(f"Only batch=1 is supported. got N={in_n}")

    model_scale = get_model_scale(in_shape, out_shape)
    print(
        f"model_input={in_shape}, model_output={out_shape}, "
        f"model_scale_h={model_scale['h']:.3f}, model_scale_w={model_scale['w']:.3f}"
    )
    print(f"preset={args.preset}")

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
    print(f"tile_input={tile_w}x{tile_h}, tile_overlap={tile_overlap}")

    compile_cfg = build_compile_cfg(args.cache_dir, args.preset)
    print(f"compile_cfg={compile_cfg}")
    downsample_interp = get_downsample_interp(args.preset)
    final_upsample_interp = get_final_upsample_interp(args.final_upsample, args.preset)

    internal_h = max(1, int(round(src_h * args.internal_scale)))
    internal_w = max(1, int(round(src_w * args.internal_scale)))
    if args.internal_scale < 1.0:
        internal_image = cv2.resize(image, (internal_w, internal_h), interpolation=cv2.INTER_AREA)
    else:
        internal_image = image
    print(
        f"internal_scale={args.internal_scale:.3f}, "
        f"internal_shape={internal_w}x{internal_h}"
    )
    print(
        f"benchmark_frames={args.benchmark_frames}, process_every={args.process_every}, "
        f"skip_mode={args.skip_mode}"
    )

    t_compile0 = time.perf_counter()
    compiled = core.compile_model(model, args.device, compile_cfg)
    req = compiled.create_infer_request()
    t_compile1 = time.perf_counter()
    print(f"compile_ms={(t_compile1 - t_compile0) * 1000.0:.3f}")

    in_name = compiled.inputs[0].any_name
    out_name = compiled.outputs[0].any_name
    in_dtype = np_dtype_from_ov_type(compiled.inputs[0].element_type)
    print(f"input_name={in_name}, output_name={out_name}, input_dtype={in_dtype.__name__}")

    tiles_x = math.ceil(internal_w / tile_w)
    tiles_y = math.ceil(internal_h / tile_h)
    total_tiles = tiles_x * tiles_y
    print(f"internal_tiles={tiles_x}x{tiles_y} (total={total_tiles})")

    t_total0 = time.perf_counter()
    frame_rows = []
    last_ai_frame = None
    last_out_frame = None

    for fi in range(args.benchmark_frames):
        frame_t0 = time.perf_counter()
        do_infer = (fi % args.process_every == 0) or (last_ai_frame is None)

        ai_elapsed_ms = 0.0
        ai_tile_avg_ms = 0.0
        ai_tiles = 0

        if do_infer:
            frame_label = f"frame {fi + 1}/{args.benchmark_frames}" if args.benchmark_frames > 1 else None
            out_internal, stats = run_infer_tiled(
                image=internal_image,
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
                progress_every=args.progress_every,
                frame_label=frame_label,
            )
            if args.internal_scale < 1.0:
                out_final = cv2.resize(
                    out_internal,
                    (src_w * 2, src_h * 2),
                    interpolation=final_upsample_interp,
                )
            else:
                out_final = out_internal
            last_ai_frame = out_final
            last_out_frame = out_final

            ai_elapsed_ms = float(stats["elapsed_ms"])
            ai_tile_avg_ms = float(stats["tile_infer_avg_ms"])
            ai_tiles = int(stats["tiles"])
            mode = "ai"
        else:
            if args.skip_mode == "reuse":
                out_final = last_ai_frame
            else:
                skip_interp = get_skip_interp(args.skip_mode)
                out_final = cv2.resize(image, (src_w * 2, src_h * 2), interpolation=skip_interp)
            last_out_frame = out_final
            mode = "skip"

        frame_t1 = time.perf_counter()
        frame_ms = (frame_t1 - frame_t0) * 1000.0
        frame_rows.append(
            {
                "frame": fi + 1,
                "mode": mode,
                "frame_ms": frame_ms,
                "ai_elapsed_ms": ai_elapsed_ms,
                "ai_tile_avg_ms": ai_tile_avg_ms,
                "ai_tiles": ai_tiles,
            }
        )
        print(f"[frame {fi + 1}/{args.benchmark_frames}] mode={mode}, frame_ms={frame_ms:.3f}")

    t_total1 = time.perf_counter()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if last_out_frame is None:
        raise RuntimeError("No output frame was produced.")
    ok = cv2.imwrite(str(out_path), last_out_frame)
    if not ok:
        raise RuntimeError(f"Failed to write output image: {out_path}")

    total_elapsed_ms = (t_total1 - t_total0) * 1000.0
    effective_avg_ms = total_elapsed_ms / max(1, args.benchmark_frames)
    effective_fps = 1000.0 / effective_avg_ms if effective_avg_ms > 0 else 0.0

    ai_rows = [r for r in frame_rows if r["mode"] == "ai"]
    ai_avg_frame_ms = (
        float(np.mean([r["frame_ms"] for r in ai_rows])) if ai_rows else 0.0
    )
    ai_frame_fps = (1000.0 / ai_avg_frame_ms) if ai_avg_frame_ms > 0 else 0.0
    ai_tile_avg_ms = (
        float(np.mean([r["ai_tile_avg_ms"] for r in ai_rows])) if ai_rows else 0.0
    )

    if args.bench_csv:
        csv_path = Path(args.bench_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["frame", "mode", "frame_ms", "ai_elapsed_ms", "ai_tile_avg_ms", "ai_tiles"],
            )
            writer.writeheader()
            writer.writerows(frame_rows)
        print(f"bench_csv={csv_path}")

    print(f"saved={out_path}")
    print(f"output_shape={last_out_frame.shape[1]}x{last_out_frame.shape[0]}")
    print(f"ai_frames={len(ai_rows)}, skipped_frames={args.benchmark_frames - len(ai_rows)}")
    print(f"ai_tile_infer_avg_ms={ai_tile_avg_ms:.3f}")
    print(f"ai_frame_avg_ms={ai_avg_frame_ms:.3f}, ai_frame_fps={ai_frame_fps:.3f}")
    print(f"effective_frame_avg_ms={effective_avg_ms:.3f}, effective_fps={effective_fps:.3f}")
    print(f"total_elapsed_ms={total_elapsed_ms:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
