import argparse
import os
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np


def resolve_input_shape(model_shape: List, image_h: int, image_w: int) -> List[int]:
    # Common layout is NCHW. For dynamic dims, fill with image size.
    fallback = [1, 3, image_h, image_w]
    out = []
    for i, dim in enumerate(model_shape):
        if isinstance(dim, int) and dim > 0:
            out.append(dim)
        else:
            out.append(fallback[i] if i < len(fallback) else 1)
    return out


def ensure_openvino_runtime_on_path() -> None:
    try:
        import openvino

        libs_dir = Path(openvino.__file__).resolve().parent / "libs"
        if libs_dir.is_dir():
            os.environ["PATH"] = str(libs_dir) + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(str(libs_dir))
    except Exception as exc:
        print(f"WARN: Could not preconfigure OpenVINO runtime path: {exc}")


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


def to_nchw_tensor(image_bgr: np.ndarray, h: int, w: int, c: int, dtype):
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
        x = x.astype(np.float32) / 255.0
        x = x.astype(dtype)
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


def build_feed(session, image0: np.ndarray, image1: np.ndarray, timestep: float) -> Dict[str, np.ndarray]:
    h0, w0 = image0.shape[:2]
    feed: Dict[str, np.ndarray] = {}
    image_input_count = 0

    for inp in session.get_inputs():
        resolved_shape = resolve_input_shape(inp.shape, h0, w0)
        dtype = np_dtype_from_ort_type(inp.type)
        lower_name = inp.name.lower()

        if "timestep" in lower_name and np.prod(resolved_shape) == 1:
            x = np.array([timestep], dtype=dtype).reshape(resolved_shape)
        elif len(resolved_shape) == 4 and resolved_shape[0] == 1 and resolved_shape[1] in (1, 3, 4):
            _, c, h, w = resolved_shape
            source = image0 if image_input_count == 0 else image1
            x = to_nchw_tensor(source, h, w, c, dtype)
            image_input_count += 1
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
    parser = argparse.ArgumentParser(description="Run ONNX image model (upscale/FG) on NPU")
    parser.add_argument("--model", required=True, help="Path to ONNX model")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument(
        "--input2",
        default=None,
        help="Optional second input image (used for FG/interpolation models)",
    )
    parser.add_argument("--output", required=True, help="Output image path")
    parser.add_argument("--device", default="NPU", help="OpenVINO device type")
    parser.add_argument("--timestep", type=float, default=0.5, help="Interpolation timestep")
    parser.add_argument(
        "--strict-openvino",
        action="store_true",
        help="Fail if OpenVINOExecutionProvider is not active",
    )
    args = parser.parse_args()

    image0 = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if image0 is None:
        raise FileNotFoundError(f"Failed to read input image: {args.input}")

    if args.input2:
        image1 = cv2.imread(args.input2, cv2.IMREAD_COLOR)
        if image1 is None:
            raise FileNotFoundError(f"Failed to read second input image: {args.input2}")
    else:
        image1 = image0

    ensure_openvino_runtime_on_path()
    import onnxruntime as ort

    providers = [
        ("OpenVINOExecutionProvider", {"device_type": args.device}),
        "CPUExecutionProvider",
    ]
    session = ort.InferenceSession(args.model, providers=providers)
    actual_providers = session.get_providers()
    print(f"Actual providers: {actual_providers}")
    if "OpenVINOExecutionProvider" not in actual_providers:
        print("WARN: OpenVINOExecutionProvider is not active; this is likely CPU fallback.")
        if args.strict_openvino:
            return 2

    feed = build_feed(session, image0, image1, args.timestep)

    t0 = time.perf_counter()
    outputs = session.run(None, feed)
    t1 = time.perf_counter()
    print(f"Inference time: {(t1 - t0) * 1000.0:.3f} ms")

    y = outputs[0]
    out_img = postprocess_tensor(y)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), out_img)
    if not ok:
        raise RuntimeError(f"Failed to write output image: {out_path}")

    print(f"Saved output: {out_path}")
    print(f"Output shape: {out_img.shape[1]}x{out_img.shape[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
