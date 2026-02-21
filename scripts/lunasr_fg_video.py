import argparse
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import openvino as ov

from lunasr_universal import (
    apply_performance_fallback_step,
    auto_tune,
    legacy_mode_to_tier,
    lunasr_upscale_bgr_internal,
    mode_defaults,
    resolve_quality_tier,
    resolve_tune_strategy,
)
from upscale_video_x2_with_fg_npu import compile_fg_model, run_fg_mid


def main() -> int:
    parser = argparse.ArgumentParser(description="LunaSR + FG pipeline (video)")
    parser.add_argument("--input", required=True, help="Input MP4 path")
    parser.add_argument("--output", required=True, help="Output MP4 path")
    parser.add_argument("--scale", type=float, default=2.0, help="LunaSR upscale factor")
    parser.add_argument(
        "--internal-scale",
        type=float,
        default=1.0,
        help="LunaSR internal processing scale (0,1], then resized to exact output size",
    )
    parser.add_argument(
        "--quality-tier",
        default=None,
        help="balanced | quality | ultra_quality (supports Korean aliases)",
    )
    parser.add_argument(
        "--quality-mode",
        type=int,
        choices=[0, 1, 2],
        default=None,
        help="Legacy mode. 0=balanced, 1=quality, 2=ultra_quality",
    )
    parser.add_argument("--auto-tune", action="store_true", help="Auto tune LunaSR params from first frame")
    parser.add_argument(
        "--auto-tune-strategy",
        default="default",
        help="Auto tune strategy: default | c",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Optional profile. universal_balanced maps to strategy c",
    )
    parser.add_argument("--seed", type=int, default=1234, help="LunaSR noise seed")

    parser.add_argument("--fg-model", default="model/model.onnx", help="FG model path")
    parser.add_argument("--fg-size", type=int, default=256, help="FG input size (square)")
    parser.add_argument("--fg-precision", choices=["f16", "i8"], default="f16")
    parser.add_argument("--timestep", type=float, default=0.5, help="FG timestep")
    parser.add_argument("--fg-every", type=int, default=1, help="Run FG every K frame-pairs")
    parser.add_argument("--fg-fallback", choices=["blend", "duplicate"], default="blend")
    parser.add_argument(
        "--mid-upscale-mode",
        choices=["infer", "blend"],
        default="infer",
        help="infer: upscale mid-frame with LunaSR, blend: blend up_prev/up_curr for speed",
    )
    parser.add_argument("--device", default="NPU", help="OpenVINO device for FG")
    parser.add_argument("--cache-dir", default=".ov_cache", help="OpenVINO cache dir")

    parser.add_argument("--max-frames", type=int, default=0, help="0 means full input")
    parser.add_argument("--progress-every", type=int, default=5, help="Print every N input frames")
    parser.add_argument(
        "--frame-budget-ms",
        type=float,
        default=0.0,
        help="If >0, apply staged LunaSR fallback when frame time exceeds budget",
    )
    args = parser.parse_args()

    if args.scale <= 0:
        raise RuntimeError(f"--scale must be > 0. got {args.scale}")
    if not (0.0 < args.internal_scale <= 1.0):
        raise RuntimeError(f"--internal-scale must be in (0,1]. got {args.internal_scale}")
    if args.fg_size < 16:
        raise RuntimeError(f"--fg-size is too small: {args.fg_size}")
    if args.fg_every < 1:
        raise RuntimeError(f"--fg-every must be >= 1. got {args.fg_every}")

    tier = "quality"
    if args.quality_mode is not None:
        tier = legacy_mode_to_tier(args.quality_mode)
    if args.quality_tier is not None:
        tier = resolve_quality_tier(args.quality_tier)
    tune_strategy = resolve_tune_strategy(args.auto_tune_strategy, args.profile)

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

    out_w = int(round(src_w * args.scale))
    out_h = int(round(src_h * args.scale))
    out_fps = src_fps * 2.0

    print(f"input_video={args.input}")
    print(f"input_shape={src_w}x{src_h}, fps={src_fps:.3f}, frames={src_frames}")
    print(f"target_input_frames={target_in_frames}")
    print(f"output_shape={out_w}x{out_h}, output_fps={out_fps:.3f}")
    print(
        f"lunasr_scale={args.scale}, internal_scale={args.internal_scale:.3f}, quality_tier={tier}, "
        f"auto_tune={args.auto_tune}, auto_tune_strategy={tune_strategy}"
    )
    print(
        f"fg_model={args.fg_model}, fg_size={args.fg_size}, fg_precision={args.fg_precision}, "
        f"fg_every={args.fg_every}, fg_fallback={args.fg_fallback}, mid_upscale_mode={args.mid_upscale_mode}"
    )
    print(f"frame_budget_ms={args.frame_budget_ms:.3f}")

    core = ov.Core()
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
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        out_fps,
        (out_w, out_h),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to create output writer: {out_path}")

    ok, prev = cap.read()
    if not ok or prev is None:
        cap.release()
        writer.release()
        raise RuntimeError("No readable frame in input video.")
    prev = cv2.resize(prev, (src_w, src_h), interpolation=cv2.INTER_CUBIC)

    luna_params = mode_defaults(tier)
    if args.auto_tune:
        y = cv2.cvtColor(prev, cv2.COLOR_BGR2YCrCb).astype(np.float32)[:, :, 0] / 255.0
        luna_params, metrics = auto_tune(y, luna_params, strategy=tune_strategy, is_video=True)
        print(f"auto_tune_metrics={metrics}")
        print(f"auto_tuned_params={vars(luna_params)}")

    luna_ms = []
    fg_ms = []
    fg_calls = 0
    fg_skips = 0
    fallback_stage = 0

    t_l0 = time.perf_counter()
    seed_prev = args.seed if tune_strategy == "c" else args.seed
    up_prev = lunasr_upscale_bgr_internal(
        prev,
        args.scale,
        luna_params,
        seed_prev,
        internal_scale=args.internal_scale,
    )
    t_l1 = time.perf_counter()
    luna_ms.append((t_l1 - t_l0) * 1000.0)

    processed_input_frames = 1
    output_frames = 0
    t_total0 = time.perf_counter()

    while processed_input_frames < target_in_frames:
        ok, curr = cap.read()
        if not ok or curr is None:
            break
        curr = cv2.resize(curr, (src_w, src_h), interpolation=cv2.INTER_CUBIC)

        pair_idx = processed_input_frames
        do_fg = (pair_idx % args.fg_every == 0)

        if do_fg:
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
            )
            fg_ms.append(fg_elapsed_ms)
            fg_calls += 1
            mid = cv2.resize(mid_fg, (src_w, src_h), interpolation=cv2.INTER_CUBIC)
        else:
            fg_skips += 1
            if args.fg_fallback == "blend":
                mid = cv2.addWeighted(prev, 0.5, curr, 0.5, 0.0)
            else:
                mid = prev

        t_m0 = time.perf_counter()
        if tune_strategy == "c":
            seed_mid = args.seed + ((processed_input_frames * 2) // 120)
        else:
            seed_mid = args.seed + output_frames + 1
        if args.mid_upscale_mode == "infer":
            up_mid = lunasr_upscale_bgr_internal(
                mid,
                args.scale,
                luna_params,
                seed_mid,
                internal_scale=args.internal_scale,
            )
            t_m1 = time.perf_counter()
            luna_ms.append((t_m1 - t_m0) * 1000.0)
        else:
            # In speed mode, skip one LunaSR call and synthesize mid at output scale.
            t_m1 = time.perf_counter()
            luna_ms.append((t_m1 - t_m0) * 1000.0)

        t_c0 = time.perf_counter()
        if tune_strategy == "c":
            seed_curr = args.seed + ((processed_input_frames * 2 + 1) // 120)
        else:
            seed_curr = args.seed + output_frames + 2
        up_curr = lunasr_upscale_bgr_internal(
            curr,
            args.scale,
            luna_params,
            seed_curr,
            internal_scale=args.internal_scale,
        )
        t_c1 = time.perf_counter()
        luna_ms.append((t_c1 - t_c0) * 1000.0)
        if args.frame_budget_ms > 0 and luna_ms[-1] > args.frame_budget_ms:
            luna_params, fallback_stage, msg = apply_performance_fallback_step(
                luna_params, fallback_stage
            )
            if msg != "fallback_step_done":
                print(f"[in_frame {processed_input_frames}] {msg}")

        if args.mid_upscale_mode == "blend":
            up_mid = cv2.addWeighted(up_prev, 0.5, up_curr, 0.5, 0.0)

        writer.write(up_prev)
        writer.write(up_mid)
        output_frames += 2

        prev = curr
        up_prev = up_curr
        processed_input_frames += 1

        if processed_input_frames % max(1, args.progress_every) == 0 or processed_input_frames == target_in_frames:
            last_fg = fg_ms[-1] if fg_ms else 0.0
            print(
                f"[in_frame {processed_input_frames}/{target_in_frames}] "
                f"luna_last_ms={luna_ms[-1]:.3f}, fg_last_ms={last_fg:.3f}, out_frames={output_frames}"
            )

    writer.write(up_prev)
    output_frames += 1

    t_total1 = time.perf_counter()
    cap.release()
    writer.release()

    total_elapsed_ms = (t_total1 - t_total0) * 1000.0
    avg_luna_ms = float(np.mean(luna_ms)) if luna_ms else 0.0
    avg_fg_ms = float(np.mean(fg_ms)) if fg_ms else 0.0
    effective_in_fps = (1000.0 * processed_input_frames / total_elapsed_ms) if total_elapsed_ms > 0 else 0.0
    effective_out_fps = (1000.0 * output_frames / total_elapsed_ms) if total_elapsed_ms > 0 else 0.0

    print(f"saved_video={out_path}")
    print(f"processed_input_frames={processed_input_frames}, output_frames={output_frames}")
    print(f"fg_calls={fg_calls}, fg_skips={fg_skips}")
    print(f"avg_fg_ms={avg_fg_ms:.3f}, avg_lunasr_frame_ms={avg_luna_ms:.3f}")
    print(f"effective_input_fps={effective_in_fps:.3f}, effective_output_fps={effective_out_fps:.3f}")
    print(f"total_elapsed_ms={total_elapsed_ms:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
