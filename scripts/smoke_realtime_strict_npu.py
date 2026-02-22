import argparse
import queue
import threading
import time
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Optional

from lunasr_realtime_gui import (
    DEFAULT_BENCHMARK_CSV,
    DEFAULT_FG_MODEL,
    DEFAULT_OV_CACHE_DIR,
    DEFAULT_OV_INTERNAL_SCALE,
    DEFAULT_OV_MODEL,
    DEFAULT_OV_PRESET,
    DEFAULT_OUTPUT_PRESET,
    LunaRealtimeWorker,
    RuntimeConfig,
    enumerate_windows,
)


def pick_hwnd(title_contains: str, hwnd_hex: str) -> int:
    if hwnd_hex:
        return int(str(hwnd_hex), 0)

    wins = enumerate_windows()
    if not wins:
        raise RuntimeError("No capturable windows found.")
    token = (title_contains or "").strip().lower()
    if token:
        for w in wins:
            if token in w.title.lower():
                return int(w.hwnd)
    return int(wins[0].hwnd)


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    s = sorted(float(v) for v in values)
    idx = int(round((len(s) - 1) * float(p)))
    idx = max(0, min(len(s) - 1, idx))
    return float(s[idx])


def summarize_ms(name: str, samples: List[float]) -> str:
    if not samples:
        return f"{name}=n/a"
    return (
        f"{name} avg={mean(samples):.2f} "
        f"p50={median(samples):.2f} "
        f"p90={percentile(samples, 0.90):.2f} "
        f"max={max(samples):.2f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Strict NPU realtime smoke benchmark (DXGI + OV).")
    parser.add_argument("--title-contains", type=str, default="", help="Window title contains token.")
    parser.add_argument("--hwnd", type=str, default="", help="Window handle (hex/int). Overrides title token.")
    parser.add_argument("--seconds", type=float, default=12.0, help="Run duration in seconds.")
    parser.add_argument("--max-samples", type=int, default=160, help="Maximum status samples with fps payload.")
    parser.add_argument("--model", type=str, default=DEFAULT_OV_MODEL, help="OpenVINO model path.")
    parser.add_argument("--output-preset", type=str, default=DEFAULT_OUTPUT_PRESET, help="AUTO/HD/FHD/QHD/4K.")
    parser.add_argument("--ov-preset", type=str, default=DEFAULT_OV_PRESET, help="speed/quality.")
    parser.add_argument("--ov-scale", type=float, default=DEFAULT_OV_INTERNAL_SCALE, help="Fixed OV scale.")
    parser.add_argument("--bench-csv", type=str, default="bench_realtime_strict_npu_smoke.csv", help="CSV output.")
    args = parser.parse_args()

    hwnd = pick_hwnd(args.title_contains, args.hwnd)
    cfg = RuntimeConfig(
        hwnd=int(hwnd),
        backend="openvino_sr",
        capture_backend="dxgi",
        preset="max_performance",
        output_preset=str(args.output_preset),
        gpu_video_path=False,
        upscale_enabled=True,
        frame_budget_ms=100.0,
        device="NPU",
        ov_model=str(args.model),
        ov_preset=str(args.ov_preset),
        ov_internal_scale=max(0.50, min(2.0, float(args.ov_scale))),
        ov_reactive_scale=False,
        ov_reactive_target_fps=60.0,
        temporal_restore=False,
        temporal_strength=0.0,
        benchmark_enabled=True,
        benchmark_csv=str(args.bench_csv or DEFAULT_BENCHMARK_CSV),
        strict_npu_only=True,
        strict_gpu_only=False,
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

    config_lock = threading.Lock()
    config_holder: Dict[str, RuntimeConfig] = {"cfg": cfg}
    status_q: queue.Queue = queue.Queue(maxsize=512)
    overlay_q: queue.Queue = queue.Queue(maxsize=2)
    worker = LunaRealtimeWorker(config_lock, config_holder, status_q, overlay_q)

    worker.start()
    t0 = time.perf_counter()
    deadline = t0 + max(2.0, float(args.seconds))
    max_samples = max(10, int(args.max_samples))

    fps_samples: List[float] = []
    frame_ms_samples: List[float] = []
    infer_ms_samples: List[float] = []
    capture_age_ms_samples: List[float] = []
    tile_ms_samples: List[float] = []
    fg_ms_samples: List[float] = []
    last_profile_note: Optional[str] = None
    last_warning: Optional[str] = None
    last_error: Optional[str] = None

    try:
        while time.perf_counter() < deadline and len(fps_samples) < max_samples:
            try:
                payload = status_q.get(timeout=0.25)
            except queue.Empty:
                continue
            if "profile_note" in payload:
                last_profile_note = str(payload.get("profile_note", ""))
            if "warning" in payload:
                last_warning = str(payload.get("warning", ""))
            if "error" in payload:
                last_error = str(payload.get("error", ""))
            if "fps" not in payload:
                continue
            fps_samples.append(float(payload.get("fps", 0.0)))
            frame_ms_samples.append(float(payload.get("frame_ms", 0.0)))
            infer_ms_samples.append(float(payload.get("infer_ms", 0.0)))
            capture_age_ms_samples.append(float(payload.get("capture_age_ms", 0.0)))
            tile_ms_samples.append(float(payload.get("tile_ms", 0.0)))
            fg_ms_samples.append(float(payload.get("fg_ms", 0.0)))
    finally:
        worker.stop()
        worker.join(timeout=3.0)

    elapsed = (time.perf_counter() - t0) * 1000.0
    print(f"hwnd=0x{int(hwnd):X}")
    print(f"elapsed_ms={elapsed:.1f}")
    if last_profile_note:
        print(f"profile_note={last_profile_note}")
    if last_warning:
        print(f"last_warning={last_warning}")
    if last_error:
        print(f"last_error={last_error}")

    print(f"samples={len(fps_samples)}")
    if not fps_samples:
        print("status=failed_no_fps_samples")
        return 2

    print(
        f"fps avg={mean(fps_samples):.2f} "
        f"p50={median(fps_samples):.2f} "
        f"p90={percentile(fps_samples, 0.90):.2f} "
        f"max={max(fps_samples):.2f}"
    )
    print(summarize_ms("frame_ms", frame_ms_samples))
    print(summarize_ms("infer_ms", infer_ms_samples))
    print(summarize_ms("capture_age_ms", capture_age_ms_samples))
    print(summarize_ms("tile_ms", tile_ms_samples))
    print(summarize_ms("fg_ms", fg_ms_samples))
    print(f"bench_csv={Path(cfg.benchmark_csv).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
