import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple


def parse_resolution(text: str) -> Tuple[int, int]:
    token = text.strip().lower().replace(" ", "")
    if "x" not in token:
        raise ValueError(f"Invalid resolution '{text}'. Use HxW format (example: 544x960).")
    h_text, w_text = token.split("x", 1)
    return int(h_text), int(w_text)


def parse_candidates(text: str) -> List[Tuple[int, int]]:
    items = [x.strip() for x in text.split(",") if x.strip()]
    if not items:
        raise ValueError("No candidates provided.")
    return [parse_resolution(x) for x in items]


def np_dtype_from_ov(ov_type_name: str):
    import numpy as np

    mapping = {
        "f32": np.float32,
        "f16": np.float16,
        "bf16": np.float16,
        "i64": np.int64,
        "i32": np.int32,
        "i16": np.int16,
        "i8": np.int8,
        "u8": np.uint8,
        "boolean": np.bool_,
    }
    return mapping.get(ov_type_name, np.float32)


def choose_shape_for_input(inp, h: int, w: int) -> List[int]:
    rank = len(inp.partial_shape)
    if rank == 4:
        return [1, 3, h, w]
    if rank == 1:
        return [1]
    resolved = []
    for d in inp.partial_shape:
        resolved.append(d.get_length() if d.is_static else 1)
    return resolved


def build_compile_config(
    cache_dir: str,
    precision_hint: str,
    dynamic_quantization: bool,
    qdq_optimization: bool,
    qdq_optimization_aggressive: bool,
    npu_compilation_mode_params: str,
) -> Dict[str, object]:
    cfg: Dict[str, object] = {}
    if cache_dir:
        cfg["CACHE_DIR"] = cache_dir
    if precision_hint:
        cfg["INFERENCE_PRECISION_HINT"] = precision_hint
    if dynamic_quantization:
        cfg["NPU_COMPILER_DYNAMIC_QUANTIZATION"] = True
    if qdq_optimization:
        cfg["NPU_QDQ_OPTIMIZATION"] = True
    if qdq_optimization_aggressive:
        cfg["NPU_QDQ_OPTIMIZATION_AGGRESSIVE"] = True
    if npu_compilation_mode_params:
        cfg["NPU_COMPILATION_MODE_PARAMS"] = npu_compilation_mode_params
    return cfg


def worker_run(
    model_path: str,
    h: int,
    w: int,
    device: str,
    cache_dir: str,
    precision_hint: str,
    dynamic_quantization: bool,
    qdq_optimization: bool,
    qdq_optimization_aggressive: bool,
    npu_compilation_mode_params: str,
    warmup_runs: int,
    infer_runs: int,
    timestep: float,
) -> Dict[str, object]:
    import numpy as np
    import openvino as ov

    result: Dict[str, object] = {
        "shape": f"{h}x{w}",
        "h": h,
        "w": w,
        "compile_ok": False,
        "infer_ok": False,
        "infer_status": "not_run",
        "error": "",
    }

    try:
        core = ov.Core()

        t0 = time.perf_counter()
        model = core.read_model(model_path)
        result["read_ms"] = (time.perf_counter() - t0) * 1000.0

        reshape_map = {}
        for inp in model.inputs:
            reshape_map[inp.any_name] = choose_shape_for_input(inp, h, w)
        model.reshape(reshape_map)
        result["reshape"] = str(reshape_map)

        compile_cfg = build_compile_config(
            cache_dir=cache_dir,
            precision_hint=precision_hint,
            dynamic_quantization=dynamic_quantization,
            qdq_optimization=qdq_optimization,
            qdq_optimization_aggressive=qdq_optimization_aggressive,
            npu_compilation_mode_params=npu_compilation_mode_params,
        )
        result["compile_cfg"] = str(compile_cfg)

        t1 = time.perf_counter()
        compiled = core.compile_model(model, device, compile_cfg)
        result["compile_ms"] = (time.perf_counter() - t1) * 1000.0
        result["compile_ok"] = True

        if infer_runs > 0:
            req = compiled.create_infer_request()
            feed = {}
            for inp in compiled.inputs:
                name = inp.any_name
                shape = [int(x) for x in inp.shape]
                type_name = inp.element_type.get_type_name()
                dtype = np_dtype_from_ov(type_name)
                if "timestep" in name.lower() and int(np.prod(shape)) == 1:
                    arr = np.array([timestep], dtype=dtype).reshape(shape)
                elif np.issubdtype(dtype, np.floating):
                    arr = np.random.rand(*shape).astype(dtype)
                elif np.issubdtype(dtype, np.integer):
                    arr = np.random.randint(0, 255, size=shape, dtype=dtype)
                else:
                    arr = np.zeros(shape, dtype=np.float32)
                feed[name] = arr

            for _ in range(max(0, warmup_runs)):
                req.infer(feed)

            lats = []
            for _ in range(max(1, infer_runs)):
                ti = time.perf_counter()
                req.infer(feed)
                lats.append((time.perf_counter() - ti) * 1000.0)
            result["infer_avg_ms"] = sum(lats) / len(lats)
            result["infer_ok"] = True
            result["infer_status"] = "ok"
        else:
            result["infer_status"] = "skipped(infer_runs=0)"

    except Exception as exc:
        result["error"] = str(exc)
        if result.get("compile_ok"):
            result["infer_status"] = "failed"

    return result


def run_one_with_timeout(
    script_path: str,
    model_path: str,
    h: int,
    w: int,
    device: str,
    cache_dir: str,
    precision_hint: str,
    dynamic_quantization: bool,
    qdq_optimization: bool,
    qdq_optimization_aggressive: bool,
    npu_compilation_mode_params: str,
    warmup_runs: int,
    infer_runs: int,
    timestep: float,
    timeout_sec: float,
) -> Dict[str, object]:
    cmd = [
        sys.executable,
        script_path,
        "--worker",
        "--model",
        model_path,
        "--shape",
        f"{h}x{w}",
        "--device",
        device,
        "--precision-hint",
        precision_hint or "",
        "--dynamic-quantization",
        str(dynamic_quantization).lower(),
        "--qdq-optimization",
        str(qdq_optimization).lower(),
        "--qdq-optimization-aggressive",
        str(qdq_optimization_aggressive).lower(),
        "--npu-compilation-mode-params",
        npu_compilation_mode_params or "",
        "--warmup-runs",
        str(warmup_runs),
        "--infer-runs",
        str(infer_runs),
        "--timestep",
        str(timestep),
        "--cache-dir",
        cache_dir or "",
    ]
    try:
        cp = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "shape": f"{h}x{w}",
            "h": h,
            "w": w,
            "compile_ok": False,
            "infer_ok": False,
            "error": f"timeout after {timeout_sec}s",
        }

    stdout = cp.stdout.strip().splitlines()
    if not stdout:
        return {
            "shape": f"{h}x{w}",
            "h": h,
            "w": w,
            "compile_ok": False,
            "infer_ok": False,
            "error": f"worker no output (rc={cp.returncode}) stderr={cp.stderr.strip()}",
        }

    last = stdout[-1]
    try:
        result = json.loads(last)
    except Exception:
        return {
            "shape": f"{h}x{w}",
            "h": h,
            "w": w,
            "compile_ok": False,
            "infer_ok": False,
            "error": f"worker bad json (rc={cp.returncode}) stdout_tail={last} stderr={cp.stderr.strip()}",
        }

    return result


def run_worker_mode(args) -> int:
    h, w = parse_resolution(args.shape)
    result = worker_run(
        model_path=args.model,
        h=h,
        w=w,
        device=args.device,
        cache_dir=args.cache_dir,
        precision_hint=args.precision_hint,
        dynamic_quantization=args.dynamic_quantization,
        qdq_optimization=args.qdq_optimization,
        qdq_optimization_aggressive=args.qdq_optimization_aggressive,
        npu_compilation_mode_params=args.npu_compilation_mode_params,
        warmup_runs=args.warmup_runs,
        infer_runs=args.infer_runs,
        timestep=args.timestep,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


def run_main_mode(args) -> int:
    candidates = parse_candidates(args.candidates)
    script_path = str(Path(__file__).resolve())

    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Precision hint: {args.precision_hint}")
    print(
        "Quant config: "
        f"dynamic_quantization={args.dynamic_quantization}, "
        f"qdq_optimization={args.qdq_optimization}, "
        f"qdq_optimization_aggressive={args.qdq_optimization_aggressive}"
    )
    print(f"Candidates: {', '.join(f'{h}x{w}' for h, w in candidates)}")
    print(f"Stride check: {args.stride}")
    print(f"Timeout per candidate: {args.timeout_sec}s")

    rows = []
    for idx, (h, w) in enumerate(candidates, start=1):
        print(f"\n[{idx}/{len(candidates)}] Testing {h}x{w} ...")
        print(
            f"divisible_by_{args.stride}: "
            f"h={'yes' if h % args.stride == 0 else 'no'}, "
            f"w={'yes' if w % args.stride == 0 else 'no'}"
        )
        res = run_one_with_timeout(
            script_path=script_path,
            model_path=args.model,
            h=h,
            w=w,
            device=args.device,
            cache_dir=args.cache_dir,
            precision_hint=args.precision_hint,
            dynamic_quantization=args.dynamic_quantization,
            qdq_optimization=args.qdq_optimization,
            qdq_optimization_aggressive=args.qdq_optimization_aggressive,
            npu_compilation_mode_params=args.npu_compilation_mode_params,
            warmup_runs=args.warmup_runs,
            infer_runs=args.infer_runs,
            timestep=args.timestep,
            timeout_sec=args.timeout_sec,
        )
        rows.append(res)

        if res.get("compile_ok"):
            compile_ms = float(res.get("compile_ms", 0.0))
            print(f"compile: OK ({compile_ms:.2f} ms)")
        else:
            print(f"compile: FAIL ({res.get('error', 'unknown error')})")

        if res.get("infer_ok"):
            infer_ms = float(res.get("infer_avg_ms", 0.0))
            print(f"infer: OK (avg {infer_ms:.2f} ms)")
        else:
            print(f"infer: {res.get('infer_status', 'skipped/failed')}")

    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved CSV: {args.csv_out}")

    ok_rows = [r for r in rows if r.get("compile_ok")]
    print(f"Compile success: {len(ok_rows)}/{len(rows)}")
    if ok_rows:
        fastest = min(ok_rows, key=lambda x: float(x.get("compile_ms", 1e18)))
        print(
            "Best compile candidate: "
            f"{fastest['shape']} ({float(fastest.get('compile_ms', 0.0)):.2f} ms)"
        )
    return 0


def main() -> int:
    def str2bool(v: str) -> bool:
        return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

    parser = argparse.ArgumentParser(
        description="Sweep FG candidate resolutions and test NPU compile/infer stability."
    )
    parser.add_argument("--model", required=True, help="Path to FG ONNX or IR model")
    parser.add_argument("--device", default="NPU", help="OpenVINO device")
    parser.add_argument("--cache-dir", default=".ov_cache", help="OpenVINO CACHE_DIR")
    parser.add_argument("--precision-hint", default="f16", help="OpenVINO INFERENCE_PRECISION_HINT (f16/i8)")
    parser.add_argument(
        "--dynamic-quantization",
        type=str2bool,
        default=False,
        help="Enable NPU_COMPILER_DYNAMIC_QUANTIZATION",
    )
    parser.add_argument(
        "--qdq-optimization",
        type=str2bool,
        default=False,
        help="Enable NPU_QDQ_OPTIMIZATION",
    )
    parser.add_argument(
        "--qdq-optimization-aggressive",
        type=str2bool,
        default=False,
        help="Enable NPU_QDQ_OPTIMIZATION_AGGRESSIVE",
    )
    parser.add_argument(
        "--npu-compilation-mode-params",
        default="",
        help="Pass-through NPU_COMPILATION_MODE_PARAMS",
    )
    parser.add_argument("--warmup-runs", type=int, default=5, help="Warmup runs before timed inference")
    parser.add_argument("--infer-runs", type=int, default=1, help="Infer runs after successful compile")
    parser.add_argument("--timestep", type=float, default=0.5, help="Timestep value for scalar input")

    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--shape", default="", help=argparse.SUPPRESS)

    parser.add_argument(
        "--candidates",
        default="544x960,528x960,512x960,512x896,256x256",
        help="Comma-separated HxW list",
    )
    parser.add_argument("--stride", type=int, default=16, help="Expected divisible stride")
    parser.add_argument("--timeout-sec", type=float, default=180, help="Per-candidate timeout")
    parser.add_argument("--csv-out", default="fg_shape_sweep_results.csv", help="Output CSV path")

    args = parser.parse_args()
    if args.worker:
        return run_worker_mode(args)
    return run_main_mode(args)


if __name__ == "__main__":
    raise SystemExit(main())
