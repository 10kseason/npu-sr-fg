import argparse
import threading
import time
from typing import Dict, List

import numpy as np
import openvino as ov


def parse_shape_text(shape_text: str) -> List[int]:
    return [int(x.strip()) for x in shape_text.split(",") if x.strip()]


def build_shape_map(shape_args: List[str]) -> Dict[str, List[int]]:
    shape_map: Dict[str, List[int]] = {}
    for item in shape_args:
        if "=" not in item:
            raise ValueError(
                f"Invalid --shape entry: '{item}'. Use name=1,3,64,64 format."
            )
        name, shape_text = item.split("=", 1)
        name = name.strip()
        shape = parse_shape_text(shape_text)
        shape_map[name] = shape
    return shape_map


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    idx = int(round((len(values) - 1) * p))
    return sorted(values)[idx]


def numpy_dtype_from_ov_type(etype: ov.Type):
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


def compile_with_progress(
    core: ov.Core,
    model: ov.Model,
    device: str,
    interval_sec: float,
    config: Dict[str, str],
):
    out = {}
    err = {}

    def worker():
        try:
            out["compiled"] = core.compile_model(model, device, config)
        except Exception as exc:
            err["exc"] = exc

    t0 = time.perf_counter()
    th = threading.Thread(target=worker, daemon=True)
    th.start()

    while th.is_alive():
        elapsed = time.perf_counter() - t0
        print(f"[compile] elapsed={elapsed:.1f}s")
        th.join(timeout=interval_sec)

    if "exc" in err:
        raise err["exc"]

    elapsed = time.perf_counter() - t0
    return out["compiled"], elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description="OpenVINO NPU benchmark with compile progress")
    parser.add_argument("--model", required=True, help="Path to ONNX or OpenVINO IR XML model")
    parser.add_argument("--device", default="NPU", help="OpenVINO device: NPU/CPU/GPU/AUTO")
    parser.add_argument(
        "--shape",
        action="append",
        default=[],
        help="Optional reshape per input. Repeatable. Format: name=1,3,64,64",
    )
    parser.add_argument("--warmup", type=int, default=3, help="Warmup runs")
    parser.add_argument("--runs", type=int, default=30, help="Benchmark runs")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print run progress every N iterations",
    )
    parser.add_argument(
        "--compile-progress-sec",
        type=float,
        default=1.0,
        help="Compile progress print interval in seconds",
    )
    parser.add_argument("--timestep", type=float, default=0.5, help="Value for timestep-like scalar input")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional OpenVINO cache directory (sets CACHE_DIR property)",
    )
    args = parser.parse_args()

    shape_map = build_shape_map(args.shape)
    core = ov.Core()

    read_t0 = time.perf_counter()
    model = core.read_model(args.model)
    read_t1 = time.perf_counter()
    print(f"read_model_ms={(read_t1 - read_t0) * 1000.0:.3f}")

    if shape_map:
        model.reshape(shape_map)
        print(f"reshape_applied={shape_map}")

    compile_cfg = {}
    if args.cache_dir:
        compile_cfg["CACHE_DIR"] = args.cache_dir
        print(f"CACHE_DIR={args.cache_dir}")

    compiled, compile_elapsed = compile_with_progress(
        core, model, args.device, args.compile_progress_sec, compile_cfg
    )
    print(f"compile_done_s={compile_elapsed:.3f}")
    print(f"compiled_device={args.device}")

    req = compiled.create_infer_request()

    feed = {}
    for inp in compiled.inputs:
        name = inp.any_name
        shape = list(inp.shape)
        np_dtype = numpy_dtype_from_ov_type(inp.element_type)
        if "timestep" in name.lower() and int(np.prod(shape)) == 1:
            arr = np.array([args.timestep], dtype=np_dtype).reshape(shape)
        elif np.issubdtype(np_dtype, np.floating):
            arr = np.random.rand(*shape).astype(np_dtype)
        elif np.issubdtype(np_dtype, np.integer):
            arr = np.random.randint(0, 255, size=shape, dtype=np_dtype)
        else:
            arr = np.zeros(shape, dtype=np.float32)
        feed[name] = arr
        print(f"input name={name}, shape={shape}, dtype={np_dtype.__name__}")

    for _ in range(max(0, args.warmup)):
        req.infer(feed)

    latencies = []
    runs = max(1, args.runs)
    progress_every = max(1, args.progress_every)
    for i in range(runs):
        t0 = time.perf_counter()
        _ = req.infer(feed)
        t1 = time.perf_counter()
        lat = (t1 - t0) * 1000.0
        latencies.append(lat)
        if (i + 1) % progress_every == 0 or (i + 1) == runs:
            print(f"[bench {i + 1}/{runs}] last_ms={lat:.3f}")

    print("\nBenchmark results (ms):")
    print(f"avg: {float(np.mean(latencies)):.3f}")
    print(f"p50: {percentile(latencies, 0.50):.3f}")
    print(f"p90: {percentile(latencies, 0.90):.3f}")
    print(f"p99: {percentile(latencies, 0.99):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
