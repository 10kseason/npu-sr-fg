import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import sys

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from lunasr_realtime_gui import compile_ov_runtime, run_ov_infer_tiled


TARGET_PRESETS: Dict[str, int] = {
    "HD": 720,
    "FHD": 1080,
    "QHD": 1440,
    "4K": 2160,
}


def even_int(v: int) -> int:
    x = max(2, int(v))
    return x - (x % 2)


def make_target_size(height: int, aspect: float) -> Tuple[int, int]:
    h = even_int(height)
    w = even_int(int(round(float(h) * float(aspect))))
    return w, h


def make_infer_size(target_w: int, target_h: int, ov_scale: float) -> Tuple[int, int]:
    iw = even_int(int(round(float(target_w) * 0.5 * float(ov_scale))))
    ih = even_int(int(round(float(target_h) * 0.5 * float(ov_scale))))
    return iw, ih


def load_source_image(path: str, fallback_shape: Tuple[int, int] = (720, 1280)) -> np.ndarray:
    p = Path(path)
    if p.exists():
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is not None:
            return img
    h, w = fallback_shape
    rng = np.random.default_rng(1234)
    img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    return img


def build_temporal_inputs(
    hist: List[np.ndarray],
    cur: np.ndarray,
    prev_depth: int,
) -> Optional[List[np.ndarray]]:
    if prev_depth <= 0:
        return None
    out: List[np.ndarray] = []
    for i in range(prev_depth):
        if i < len(hist):
            out.append(hist[i])
        elif hist:
            out.append(hist[-1])
        else:
            out.append(cur)
    return out


def run_case(
    ov_ctx: Dict[str, object],
    source_img: np.ndarray,
    preset: str,
    target_h: int,
    ov_scale: float,
    warmup: int,
    runs: int,
    out_dir: Path,
    save_samples: bool,
) -> Dict[str, object]:
    aspect = float(source_img.shape[1]) / float(max(1, int(source_img.shape[0])))
    target_w, target_h2 = make_target_size(target_h, aspect)
    infer_w, infer_h = make_infer_size(target_w, target_h2, ov_scale)
    base = cv2.resize(source_img, (infer_w, infer_h), interpolation=cv2.INTER_AREA)

    temporal_prev_depth = int(ov_ctx.get("temporal_prev_depth", 0))
    temporal_model = bool(ov_ctx.get("temporal_model", False))
    hist: List[np.ndarray] = []
    frame_ms: List[float] = []
    tile_ms: List[float] = []
    last_out = None

    total_iters = max(1, int(warmup) + int(runs))
    for i in range(total_iters):
        sx = (i % 3) - 1
        sy = ((i // 3) % 3) - 1
        cur = np.roll(base, shift=(sy, sx), axis=(0, 1))
        temporal_inputs = None
        if temporal_model:
            temporal_inputs = build_temporal_inputs(hist=hist, cur=cur, prev_depth=temporal_prev_depth)
        t0 = time.perf_counter()
        out, tms = run_ov_infer_tiled(cur, ov_ctx, temporal_images=temporal_inputs)
        t1 = time.perf_counter()
        last_out = out
        hist.insert(0, cur.copy())
        if len(hist) > max(1, temporal_prev_depth):
            del hist[max(1, temporal_prev_depth) :]
        if i >= warmup:
            frame_ms.append((t1 - t0) * 1000.0)
            tile_ms.append(float(tms))

    avg_frame_ms = float(np.mean(frame_ms)) if frame_ms else 0.0
    p95_frame_ms = float(np.percentile(frame_ms, 95)) if frame_ms else 0.0
    avg_tile_ms = float(np.mean(tile_ms)) if tile_ms else 0.0
    fps_est = (1000.0 / avg_frame_ms) if avg_frame_ms > 0 else 0.0

    sample_path = ""
    if save_samples and last_out is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        sample = cv2.resize(last_out, (target_w, target_h2), interpolation=cv2.INTER_CUBIC)
        sample_path = str(out_dir / f"bench_tiling_{preset.lower()}_{target_w}x{target_h2}.png")
        cv2.imwrite(sample_path, sample)

    return {
        "preset": preset,
        "target_w": target_w,
        "target_h": target_h2,
        "infer_w": infer_w,
        "infer_h": infer_h,
        "avg_frame_ms": avg_frame_ms,
        "p95_frame_ms": p95_frame_ms,
        "avg_tile_ms": avg_tile_ms,
        "fps_est": fps_est,
        "sample_path": sample_path,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="NPU tiling benchmark across HD/FHD/QHD/4K presets.")
    parser.add_argument("--model", default="model/ir/fixed_sr_algo_x2_temporal.xml")
    parser.add_argument("--device", default="NPU")
    parser.add_argument("--ov-preset", choices=["speed", "quality"], default="speed")
    parser.add_argument("--cache-dir", default=".ov_cache")
    parser.add_argument("--source-image", default="test mp4 and jpg/input_720p.png")
    parser.add_argument("--ov-internal-scale", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=12)
    parser.add_argument("--no-samples", action="store_true")
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--strict-npu-only", action="store_true", default=True)
    parser.add_argument(
        "--out-csv",
        default="test mp4 and jpg/results/bench_npu_tiling_hd_fhd_qhd_4k.csv",
    )
    args = parser.parse_args()

    ov_scale = max(0.50, min(2.0, float(args.ov_internal_scale)))
    source = load_source_image(args.source_image)
    aspect = float(source.shape[1]) / float(max(1, int(source.shape[0])))

    ov_ctx = compile_ov_runtime(
        model_path=args.model,
        device=args.device,
        preset=args.ov_preset,
        cache_dir=args.cache_dir,
        allow_cpu_fallback=bool(args.allow_cpu_fallback),
        strict_npu_only=bool(args.strict_npu_only),
    )
    compile_device = str(ov_ctx.get("compile_device", "unknown"))
    exec_devices = str(ov_ctx.get("exec_devices", "unknown"))
    req_pool = int(ov_ctx.get("ov_parallel_reqs", 1))
    temporal_model = bool(ov_ctx.get("temporal_model", False))
    temporal_prev_depth = int(ov_ctx.get("temporal_prev_depth", 0))

    out_csv = Path(args.out_csv)
    out_dir = out_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for p, h in TARGET_PRESETS.items():
        r = run_case(
            ov_ctx=ov_ctx,
            source_img=source,
            preset=p,
            target_h=h,
            ov_scale=ov_scale,
            warmup=max(0, int(args.warmup)),
            runs=max(1, int(args.runs)),
            out_dir=out_dir,
            save_samples=not bool(args.no_samples),
        )
        r["model"] = str(args.model)
        r["device"] = str(args.device)
        r["compile_device"] = compile_device
        r["exec_devices"] = exec_devices
        r["req_pool"] = req_pool
        r["ov_preset"] = str(args.ov_preset)
        r["ov_scale"] = ov_scale
        r["temporal_model"] = int(temporal_model)
        r["temporal_prev_depth"] = temporal_prev_depth
        r["source_w"] = int(source.shape[1])
        r["source_h"] = int(source.shape[0])
        r["source_aspect"] = aspect
        rows.append(r)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "preset",
            "target_w",
            "target_h",
            "infer_w",
            "infer_h",
            "avg_frame_ms",
            "p95_frame_ms",
            "avg_tile_ms",
            "fps_est",
            "model",
            "device",
            "compile_device",
            "exec_devices",
            "req_pool",
            "ov_preset",
            "ov_scale",
            "temporal_model",
            "temporal_prev_depth",
            "source_w",
            "source_h",
            "source_aspect",
            "sample_path",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"model={args.model}")
    print(f"source={args.source_image} ({source.shape[1]}x{source.shape[0]})")
    print(
        f"compile_device={compile_device}, exec_devices={exec_devices}, req_pool={req_pool}, "
        f"temporal_model={int(temporal_model)}, temporal_prev_depth={temporal_prev_depth}"
    )
    print(f"ov_scale={ov_scale:.3f}, ov_preset={args.ov_preset}")
    for r in rows:
        print(
            f"{r['preset']}: target={r['target_w']}x{r['target_h']}, infer={r['infer_w']}x{r['infer_h']}, "
            f"avg_frame_ms={float(r['avg_frame_ms']):.3f}, p95={float(r['p95_frame_ms']):.3f}, "
            f"avg_tile_ms={float(r['avg_tile_ms']):.3f}, fps_est={float(r['fps_est']):.2f}"
        )
    print(f"saved_csv={out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
