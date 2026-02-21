import argparse
import os
import statistics
import time
from pathlib import Path
from typing import Dict, List

import numpy as np


def parse_shape(shape_text: str) -> List[int]:
    parts = [p.strip() for p in shape_text.split(",") if p.strip()]
    if not parts:
        raise ValueError("input-shape is empty")
    return [int(p) for p in parts]


def ensure_openvino_runtime_on_path() -> None:
    # onnxruntime-openvino may fail to load OpenVINO EP if openvino.dll
    # is not discoverable from PATH in global Python installs.
    try:
        import openvino

        libs_dir = Path(openvino.__file__).resolve().parent / "libs"
        if libs_dir.is_dir():
            os.environ["PATH"] = str(libs_dir) + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(str(libs_dir))
    except Exception as exc:
        print(f"WARN: Could not preconfigure OpenVINO runtime path: {exc}")


def resolve_input_shape(model_shape, fallback_shape: List[int]) -> List[int]:
    resolved = []
    for idx, dim in enumerate(model_shape):
        if isinstance(dim, int) and dim > 0:
            resolved.append(dim)
        else:
            if idx < len(fallback_shape):
                resolved.append(fallback_shape[idx])
            else:
                resolved.append(1)
    return resolved


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    idx = int(round((len(values) - 1) * p))
    return sorted(values)[idx]


def np_dtype_from_ort_type(ort_type: str):
    mapping = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(double)": np.float64,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
        "tensor(int16)": np.int16,
        "tensor(int8)": np.int8,
        "tensor(uint8)": np.uint8,
        "tensor(bool)": np.bool_,
    }
    return mapping.get(ort_type, np.float32)


def build_feed(inputs, fallback_shape: List[int], timestep: float) -> Dict[str, np.ndarray]:
    feed: Dict[str, np.ndarray] = {}
    for inp in inputs:
        resolved_shape = resolve_input_shape(inp.shape, fallback_shape)
        dtype = np_dtype_from_ort_type(inp.type)
        name_lower = inp.name.lower()

        if "timestep" in name_lower and np.prod(resolved_shape) == 1:
            x = np.array([timestep], dtype=dtype).reshape(resolved_shape)
        elif np.issubdtype(dtype, np.floating):
            x = np.random.rand(*resolved_shape).astype(dtype)
        elif np.issubdtype(dtype, np.integer):
            x = np.random.randint(0, 255, size=resolved_shape, dtype=dtype)
        else:
            x = np.zeros(resolved_shape, dtype=np.float32)

        feed[inp.name] = x
        print(
            f"Input: name={inp.name}, type={inp.type}, model_shape={inp.shape}, "
            f"resolved_shape={resolved_shape}"
        )

    return feed


def main() -> int:
    parser = argparse.ArgumentParser(description="ONNX Runtime NPU probe")
    parser.add_argument("--model", required=True, help="Path to ONNX model")
    parser.add_argument(
        "--input-shape",
        default="1,3,540,960",
        help="Fallback input shape for dynamic dims, e.g. 1,3,540,960",
    )
    parser.add_argument("--runs", type=int, default=80, help="Benchmark runs")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup runs")
    parser.add_argument("--device", default="NPU", help="OpenVINO device type")
    parser.add_argument(
        "--timestep",
        type=float,
        default=0.5,
        help="Value used when a scalar timestep input exists",
    )
    parser.add_argument(
        "--strict-openvino",
        action="store_true",
        help="Fail if OpenVINOExecutionProvider is not active",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Print per-step progress during warmup and benchmark runs",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Progress print frequency in runs (default: 1)",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional OpenVINO cache directory (sets OV_CACHE_DIR)",
    )
    args = parser.parse_args()

    if args.cache_dir:
        os.environ["OV_CACHE_DIR"] = args.cache_dir
        print(f"OV_CACHE_DIR={args.cache_dir}")

    ensure_openvino_runtime_on_path()
    import onnxruntime as ort

    fallback_shape = parse_shape(args.input_shape)

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers = [
        ("OpenVINOExecutionProvider", {"device_type": args.device}),
        "CPUExecutionProvider",
    ]

    print(f"Loading model: {args.model}")
    print(f"Requested providers: {providers}")
    load_t0 = time.perf_counter()
    session = ort.InferenceSession(args.model, sess_options=so, providers=providers)
    load_t1 = time.perf_counter()
    print(f"Session init+compile time: {(load_t1 - load_t0) * 1000.0:.3f} ms")
    actual_providers = session.get_providers()
    print(f"Actual providers: {actual_providers}")
    if "OpenVINOExecutionProvider" not in actual_providers:
        print("WARN: OpenVINOExecutionProvider is not active; benchmark is likely CPU fallback.")
        if args.strict_openvino:
            return 2

    inputs = session.get_inputs()
    if not inputs:
        raise RuntimeError("Model has no inputs")

    feed = build_feed(inputs, fallback_shape, args.timestep)

    warmup_runs = max(0, args.warmup)
    bench_runs = max(1, args.runs)
    progress_every = max(1, args.progress_every)

    for i in range(warmup_runs):
        session.run(None, feed)
        if args.show_progress and ((i + 1) % progress_every == 0 or (i + 1) == warmup_runs):
            pct = ((i + 1) / warmup_runs) * 100.0 if warmup_runs else 100.0
            print(f"[warmup {i + 1}/{warmup_runs}] {pct:.1f}%")

    latencies_ms = []
    for i in range(bench_runs):
        t0 = time.perf_counter()
        session.run(None, feed)
        t1 = time.perf_counter()
        lat = (t1 - t0) * 1000.0
        latencies_ms.append(lat)
        if args.show_progress and ((i + 1) % progress_every == 0 or (i + 1) == bench_runs):
            pct = ((i + 1) / bench_runs) * 100.0
            print(f"[bench {i + 1}/{bench_runs}] {pct:.1f}% | last={lat:.3f} ms")

    avg_ms = statistics.fmean(latencies_ms)
    p50 = percentile(latencies_ms, 0.50)
    p90 = percentile(latencies_ms, 0.90)
    p99 = percentile(latencies_ms, 0.99)

    print("\nBenchmark results (ms):")
    print(f"avg: {avg_ms:.3f}")
    print(f"p50: {p50:.3f}")
    print(f"p90: {p90:.3f}")
    print(f"p99: {p99:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
