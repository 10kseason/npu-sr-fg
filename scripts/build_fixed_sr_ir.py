import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import openvino as ov


def resolve_temporal_preset(name: str) -> Dict[str, float]:
    n = (name or "custom").strip().lower()
    table: Dict[str, Dict[str, float]] = {
        # Baseline temporal-plus behavior.
        "balanced": {
            "temporal_gain": 0.35,
            "temporal_diff_threshold": 0.08,
            "temporal_hf_gain": 1.05,
            "temporal_detail_boost": 0.12,
            "temporal_detail_threshold": 0.05,
        },
        # Sharper details with lower history dependency.
        "crisp": {
            "temporal_gain": 0.26,
            "temporal_diff_threshold": 0.07,
            "temporal_hf_gain": 1.14,
            "temporal_detail_boost": 0.20,
            "temporal_detail_threshold": 0.04,
        },
        # More stable temporal blend with conservative edge boost.
        "stable": {
            "temporal_gain": 0.44,
            "temporal_diff_threshold": 0.10,
            "temporal_hf_gain": 0.98,
            "temporal_detail_boost": 0.07,
            "temporal_detail_threshold": 0.06,
        },
    }
    if n in table:
        return dict(table[n])
    return {}


def parse_shape(text: str) -> Tuple[int, int, int, int]:
    parts = [p.strip() for p in str(text).split(",")]
    if len(parts) != 4:
        raise RuntimeError(f"Expected --input-shape like 1,3,256,256. got: {text}")
    n, c, h, w = [int(x) for x in parts]
    if n != 1 or c != 3 or h <= 0 or w <= 0:
        raise RuntimeError(f"Only static NCHW 1,3,H,W is supported. got: {n},{c},{h},{w}")
    return n, c, h, w


def np_dtype_from_element_type(etype: ov.Type):
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


def build_compile_cfg(device: str, precision_hint: str, cache_dir: str) -> Dict[str, object]:
    cfg: Dict[str, object] = {}
    if cache_dir:
        cfg["CACHE_DIR"] = cache_dir
    hint = (precision_hint or "f16").lower()
    if hint == "i8" and "NPU" in (device or "").upper():
        cfg["INFERENCE_PRECISION_HINT"] = "i8"
        cfg["NPU_COMPILER_DYNAMIC_QUANTIZATION"] = True
        cfg["NPU_QDQ_OPTIMIZATION"] = True
        cfg["NPU_QDQ_OPTIMIZATION_AGGRESSIVE"] = True
    elif hint == "f32":
        cfg["INFERENCE_PRECISION_HINT"] = "f32"
    else:
        cfg["INFERENCE_PRECISION_HINT"] = "f16"
    return cfg


def _depthwise_kernel_3x3_gaussian(channels: int = 3) -> np.ndarray:
    k = np.array(
        [
            [1.0, 2.0, 1.0],
            [2.0, 4.0, 2.0],
            [1.0, 2.0, 1.0],
        ],
        dtype=np.float32,
    )
    k = k / np.sum(k)
    w = np.zeros((channels, 1, 1, 3, 3), dtype=np.float32)
    for c in range(channels):
        w[c, 0, 0, :, :] = k
    return w


def build_fixed_fg_mid_model(
    in_h: int,
    in_w: int,
    sharpen_gain: float = 0.18,
    motion_low: float = 0.03,
    motion_high: float = 0.18,
    motion_smooth: float = 0.12,
) -> ov.Model:
    ops = ov.opset10
    img0 = ops.parameter([1, 3, in_h, in_w], ov.Type.f32, name="img0")
    img1 = ops.parameter([1, 3, in_h, in_w], ov.Type.f32, name="img1")
    timestep = ops.parameter([1], ov.Type.f32, name="timestep")

    img0_01 = ops.clamp(img0, 0.0, 1.0, name="img0_clamp")
    img1_01 = ops.clamp(img1, 0.0, 1.0, name="img1_clamp")
    t4 = ops.reshape(
        timestep,
        ops.constant(np.array([1, 1, 1, 1], dtype=np.int64)),
        False,
    )
    t01 = ops.clamp(t4, 0.0, 1.0, name="timestep_clamp")
    one = ops.constant(np.float32(1.0))
    inv_t = ops.subtract(one, t01)

    base = ops.add(
        ops.multiply(img0_01, inv_t),
        ops.multiply(img1_01, t01),
    )

    k = ops.constant(_depthwise_kernel_3x3_gaussian(3))
    blur0 = ops.group_convolution(
        data=img0_01,
        filters=k,
        strides=[1, 1],
        pads_begin=[1, 1],
        pads_end=[1, 1],
        dilations=[1, 1],
    )
    blur1 = ops.group_convolution(
        data=img1_01,
        filters=k,
        strides=[1, 1],
        pads_begin=[1, 1],
        pads_end=[1, 1],
        dilations=[1, 1],
    )
    detail0 = ops.subtract(img0_01, blur0)
    detail1 = ops.subtract(img1_01, blur1)
    detail_mix = ops.add(
        ops.multiply(detail0, inv_t),
        ops.multiply(detail1, t01),
    )

    diff_abs = ops.abs(ops.subtract(img1_01, img0_01))
    motion = ops.reduce_mean(
        diff_abs,
        ops.constant(np.array([1], dtype=np.int64)),
        True,
    )
    m_low = max(0.0, min(0.99, float(motion_low)))
    m_high = max(m_low + 1e-4, min(1.0, float(motion_high)))
    inv_span = 1.0 / max(1e-6, m_high - m_low)
    motion_gate = ops.clamp(
        ops.multiply(
            ops.subtract(motion, ops.constant(np.float32(m_low))),
            ops.constant(np.float32(inv_span)),
        ),
        0.0,
        1.0,
    )

    sh = max(0.0, float(sharpen_gain))
    sh_high_motion = sh * 0.45
    detail_gain = ops.add(
        ops.multiply(ops.constant(np.float32(sh)), ops.subtract(one, motion_gate)),
        ops.multiply(ops.constant(np.float32(sh_high_motion)), motion_gate),
    )
    detail_term = ops.multiply(detail_mix, detail_gain)
    enhanced = ops.add(base, detail_term)

    smooth = max(0.0, min(1.0, float(motion_smooth)))
    if smooth > 1e-6:
        blur_base = ops.group_convolution(
            data=base,
            filters=k,
            strides=[1, 1],
            pads_begin=[1, 1],
            pads_end=[1, 1],
            dilations=[1, 1],
        )
        smooth_w = ops.multiply(motion_gate, ops.constant(np.float32(smooth)))
        y = ops.add(
            ops.multiply(enhanced, ops.subtract(one, smooth_w)),
            ops.multiply(blur_base, smooth_w),
        )
    else:
        y = enhanced

    y01 = ops.clamp(y, 0.0, 1.0, name="output_clamp")
    result = ops.result(y01, name="output")
    return ov.Model([result], [img0, img1, timestep], "fixed_fg_algo_mid")


def build_fixed_sr_model(
    in_h: int,
    in_w: int,
    upscale: int,
    sharpen_gain: float,
    preserve_gain: float,
    temporal_model: bool = False,
    temporal_gain: float = 0.35,
    temporal_diff_threshold: float = 0.08,
    temporal_depth: int = 2,
    temporal_hf_gain: float = 1.0,
    temporal_detail_boost: float = 0.0,
    temporal_detail_threshold: float = 0.05,
) -> ov.Model:
    ops = ov.opset10
    out_h = int(in_h * upscale)
    out_w = int(in_w * upscale)

    x_curr = ops.parameter([1, 3, in_h, in_w], ov.Type.f32, name="input_curr")
    x_curr01 = ops.clamp(x_curr, 0.0, 1.0, name="input_curr_clamp")
    model_inputs = [x_curr]

    temporal_depth = max(1, int(temporal_depth))
    if not temporal_model:
        temporal_depth = 1
    temporal_prev_inputs: List[Tuple[str, Any]] = []
    for i in range(1, temporal_depth):
        name = "input_prev" if i == 1 else f"input_prev{i}"
        x_prev = ops.parameter([1, 3, in_h, in_w], ov.Type.f32, name=name)
        x_prev01 = ops.clamp(x_prev, 0.0, 1.0, name=f"{name}_clamp")
        temporal_prev_inputs.append((name, x_prev01))
        model_inputs.append(x_prev)

    out_shape = ops.constant(np.array([1, 3, out_h, out_w], dtype=np.int64))
    scales = ops.constant(np.array([1.0, 1.0, float(upscale), float(upscale)], dtype=np.float32))
    up_curr = ops.interpolate(
        image=x_curr01,
        output_shape=out_shape,
        scales=scales,
        mode="linear",
        shape_calculation_mode="sizes",
        coordinate_transformation_mode="half_pixel",
        nearest_mode="round_prefer_floor",
        antialias=False,
    )

    k = ops.constant(_depthwise_kernel_3x3_gaussian(3))
    blur0 = ops.group_convolution(
        data=up_curr,
        filters=k,
        strides=[1, 1],
        pads_begin=[1, 1],
        pads_end=[1, 1],
        dilations=[1, 1],
    )

    detail = ops.subtract(up_curr, blur0)
    sharpened = ops.add(up_curr, ops.multiply(detail, ops.constant(np.float32(sharpen_gain))))

    y_pre = sharpened
    if temporal_model and temporal_prev_inputs:
        # Temporal-plus algorithm:
        # 1) blend only low-frequency component from previous frames
        # 2) keep current high-frequency branch to preserve detail at low internal scale
        # 3) confidence-gated weights to avoid ghosting on motion/disocclusion
        low_curr = blur0
        high_curr = detail
        weighted_low = low_curr
        weighted_sum = ops.constant(np.float32(1.0))
        base_gain = max(0.0, min(1.0, float(temporal_gain)))
        base_th = max(1e-6, float(temporal_diff_threshold))
        for idx, (_name, prev01) in enumerate(temporal_prev_inputs, start=1):
            up_prev = ops.interpolate(
                image=prev01,
                output_shape=out_shape,
                scales=scales,
                mode="linear",
                shape_calculation_mode="sizes",
                coordinate_transformation_mode="half_pixel",
                nearest_mode="round_prefer_floor",
                antialias=False,
            )
            # Older history gets lower gain/threshold.
            gain_i = base_gain * (0.60 ** (idx - 1))
            th_i = max(0.02, base_th * (0.92 ** (idx - 1)))
            inv_th = 1.0 / max(1e-6, th_i)
            diff_abs = ops.abs(ops.subtract(sharpened, up_prev))
            diff_mean = ops.reduce_mean(
                diff_abs,
                ops.constant(np.array([1], dtype=np.int64)),
                True,
            )
            conf_i = ops.clamp(
                ops.multiply(
                    ops.subtract(ops.constant(np.float32(th_i)), diff_mean),
                    ops.constant(np.float32(inv_th)),
                ),
                0.0,
                1.0,
            )
            w_i = ops.multiply(conf_i, ops.constant(np.float32(gain_i)))
            low_prev = ops.group_convolution(
                data=up_prev,
                filters=k,
                strides=[1, 1],
                pads_begin=[1, 1],
                pads_end=[1, 1],
                dilations=[1, 1],
            )
            weighted_low = ops.add(weighted_low, ops.multiply(low_prev, w_i))
            weighted_sum = ops.add(weighted_sum, w_i)

        low_blend = ops.divide(weighted_low, weighted_sum)
        hf_base_gain = max(0.0, float(temporal_hf_gain))
        detail_boost = max(0.0, float(temporal_detail_boost))
        detail_th = max(0.0, min(0.99, float(temporal_detail_threshold)))
        if detail_boost > 1e-6:
            # Edge-aware high-frequency gain keeps perceived detail at lower OV internal scales.
            high_abs = ops.abs(high_curr)
            high_mag = ops.reduce_mean(
                high_abs,
                ops.constant(np.array([1], dtype=np.int64)),
                True,
            )
            inv_th = 1.0 / max(1e-6, 1.0 - detail_th)
            edge_conf = ops.clamp(
                ops.multiply(
                    ops.subtract(high_mag, ops.constant(np.float32(detail_th))),
                    ops.constant(np.float32(inv_th)),
                ),
                0.0,
                1.0,
            )
            hf_gain_map = ops.add(
                ops.constant(np.float32(hf_base_gain)),
                ops.multiply(edge_conf, ops.constant(np.float32(detail_boost))),
            )
            hf_term = ops.multiply(high_curr, hf_gain_map)
        else:
            hf_term = ops.multiply(high_curr, ops.constant(np.float32(hf_base_gain)))
        y_pre = ops.add(low_blend, hf_term)
        y_pre = ops.clamp(y_pre, 0.0, 1.0)

    blur1 = ops.group_convolution(
        data=y_pre,
        filters=k,
        strides=[1, 1],
        pads_begin=[1, 1],
        pads_end=[1, 1],
        dilations=[1, 1],
    )

    # Blend sharpened and smoothed output: preserve_gain=1 keeps detail, 0 keeps smooth.
    mix_a = ops.multiply(y_pre, ops.constant(np.float32(preserve_gain)))
    mix_b = ops.multiply(blur1, ops.constant(np.float32(1.0 - preserve_gain)))
    y = ops.add(mix_a, mix_b)
    y01 = ops.clamp(y, 0.0, 1.0, name="output_clamp")

    result = ops.result(y01, name="output")
    name = "fixed_sr_algo_x2_temporal" if temporal_model else "fixed_sr_algo_x2"
    return ov.Model([result], model_inputs, name)


def _to_nchw01_image(image_path: Path, h: int, w: int) -> np.ndarray:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    if img.shape[0] != h or img.shape[1] != w:
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = np.transpose(rgb, (2, 0, 1))[None].astype(np.float32) / 255.0
    return x


def _save_nchw01_image(y: np.ndarray, out_path: Path) -> None:
    if y.ndim == 4:
        y = y[0]
    if y.shape[0] == 1:
        y = np.repeat(y, 3, axis=0)
    rgb = np.transpose(y, (1, 2, 0))
    rgb = np.clip(rgb, 0.0, 1.0)
    bgr = cv2.cvtColor((rgb * 255.0 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), bgr):
        raise RuntimeError(f"Failed to write image: {out_path}")


def _cast_nchw01_to_dtype(x: np.ndarray, dtype) -> np.ndarray:
    if np.issubdtype(dtype, np.floating):
        return x.astype(dtype)
    if np.issubdtype(dtype, np.integer):
        return np.clip(x * 255.0 + 0.5, 0.0, 255.0).astype(dtype)
    return x.astype(np.float32)


def run(args) -> int:
    model_type = str(args.model_type).strip().lower()
    if model_type not in ("sr_x2", "fg_mid"):
        raise RuntimeError(f"Unknown --model-type: {args.model_type}")

    _, _, in_h, in_w = parse_shape(args.input_shape)

    if model_type == "fg_mid":
        fg_sharpen_gain = max(0.0, float(args.fg_sharpen_gain))
        fg_motion_low = max(0.0, min(0.99, float(args.fg_motion_low)))
        fg_motion_high = max(fg_motion_low + 1e-4, min(1.0, float(args.fg_motion_high)))
        fg_motion_smooth = max(0.0, min(1.0, float(args.fg_motion_smooth)))
        model = build_fixed_fg_mid_model(
            in_h=in_h,
            in_w=in_w,
            sharpen_gain=fg_sharpen_gain,
            motion_low=fg_motion_low,
            motion_high=fg_motion_high,
            motion_smooth=fg_motion_smooth,
        )
        upscale = 1
        temporal_model = False
        temporal_depth = 1
        temporal_preset = "custom"
        temporal_gain = 0.0
        temporal_diff_threshold = 0.0
        temporal_hf_gain = 0.0
        temporal_detail_boost = 0.0
        temporal_detail_threshold = 0.0
        sharpen_gain = 0.0
        preserve_gain = 1.0
    else:
        upscale = max(1, int(args.upscale))
        sharpen_gain = float(args.sharpen_gain)
        preserve_gain = float(args.preserve_gain)
        preserve_gain = max(0.0, min(1.0, preserve_gain))
        temporal_model = bool(args.temporal_model)
        temporal_gain = max(0.0, min(1.0, float(args.temporal_gain)))
        temporal_diff_threshold = max(1e-4, float(args.temporal_diff_threshold))
        temporal_depth = max(2, int(args.temporal_depth))
        temporal_hf_gain = max(0.0, float(args.temporal_hf_gain))
        temporal_detail_boost = max(0.0, float(args.temporal_detail_boost))
        temporal_detail_threshold = max(0.0, min(0.99, float(args.temporal_detail_threshold)))
        temporal_preset = str(args.temporal_preset).strip().lower()
        if temporal_model and temporal_preset != "custom":
            preset_vals = resolve_temporal_preset(temporal_preset)
            if not preset_vals:
                raise RuntimeError(
                    f"Unknown --temporal-preset: {args.temporal_preset} "
                    f"(expected one of: custom, balanced, crisp, stable)"
                )
            temporal_gain = float(preset_vals["temporal_gain"])
            temporal_diff_threshold = float(preset_vals["temporal_diff_threshold"])
            temporal_hf_gain = float(preset_vals["temporal_hf_gain"])
            temporal_detail_boost = float(preset_vals["temporal_detail_boost"])
            temporal_detail_threshold = float(preset_vals["temporal_detail_threshold"])

        model = build_fixed_sr_model(
            in_h=in_h,
            in_w=in_w,
            upscale=upscale,
            sharpen_gain=sharpen_gain,
            preserve_gain=preserve_gain,
            temporal_model=temporal_model,
            temporal_gain=temporal_gain,
            temporal_diff_threshold=temporal_diff_threshold,
            temporal_depth=temporal_depth,
            temporal_hf_gain=temporal_hf_gain,
            temporal_detail_boost=temporal_detail_boost,
            temporal_detail_threshold=temporal_detail_threshold,
        )

    out_xml = Path(args.output_xml)
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(model, str(out_xml))

    print(f"saved_ir={out_xml}")
    print(f"model_type={model_type}")
    if model_type == "fg_mid":
        print(f"input_shape=img0:1,3,{in_h},{in_w}; img1:1,3,{in_h},{in_w}; timestep:1")
    elif temporal_model:
        input_desc = [f"input_curr:1,3,{in_h},{in_w}"]
        for i in range(1, temporal_depth):
            nm = "input_prev" if i == 1 else f"input_prev{i}"
            input_desc.append(f"{nm}:1,3,{in_h},{in_w}")
        print(f"input_shape={'; '.join(input_desc)}")
    else:
        print(f"input_shape=1,3,{in_h},{in_w}")
    print(f"output_shape=1,3,{in_h * upscale},{in_w * upscale}")
    if model_type == "fg_mid":
        algo = "fg_mid(blend+detail_mix+motion_gate+motion_smooth)"
    else:
        algo = "interpolate+depthwise_gaussian+unsharp+blend"
        if temporal_model:
            algo += "+temporal_plus(lowfreq_accum+hf_preserve)"
            if temporal_detail_boost > 1e-6:
                algo += "+edge_aware_hf_gain"
    print(f"algo={algo}")
    if model_type == "fg_mid":
        print(
            f"fg_sharpen_gain={fg_sharpen_gain:.3f}, fg_motion_low={fg_motion_low:.3f}, "
            f"fg_motion_high={fg_motion_high:.3f}, fg_motion_smooth={fg_motion_smooth:.3f}"
        )
    else:
        print(f"sharpen_gain={sharpen_gain:.3f}, preserve_gain={preserve_gain:.3f}")
    if model_type == "sr_x2" and temporal_model:
        print(f"temporal_preset={temporal_preset}")
        print(
            f"temporal_depth={temporal_depth}, temporal_gain={temporal_gain:.3f}, "
            f"temporal_diff_threshold={temporal_diff_threshold:.4f}, temporal_hf_gain={temporal_hf_gain:.3f}, "
            f"temporal_detail_boost={temporal_detail_boost:.3f}, "
            f"temporal_detail_threshold={temporal_detail_threshold:.3f}"
        )

    if not args.smoke:
        return 0

    core = ov.Core()
    cfg = build_compile_cfg(args.device, args.precision_hint, args.cache_dir)
    t0 = time.perf_counter()
    compiled = core.compile_model(model, args.device, cfg)
    t1 = time.perf_counter()
    req = compiled.create_infer_request()

    in_name = compiled.inputs[0].any_name
    out_name = compiled.outputs[0].any_name

    if args.image:
        x = _to_nchw01_image(Path(args.image), in_h, in_w)
    else:
        x = np.random.rand(1, 3, in_h, in_w).astype(np.float32)
    if args.image_prev:
        x_prev = _to_nchw01_image(Path(args.image_prev), in_h, in_w)
    else:
        x_prev = np.roll(x, shift=1, axis=3)

    ti0 = time.perf_counter()
    feed: Dict[str, np.ndarray] = {}
    if model_type == "fg_mid":
        for inp in compiled.inputs:
            name = str(inp.any_name)
            lname = name.lower()
            shape = [int(v) for v in inp.shape]
            dtype = np_dtype_from_element_type(inp.element_type)
            if "img0" in lname:
                feed[name] = _cast_nchw01_to_dtype(x, dtype)
            elif "img1" in lname:
                feed[name] = _cast_nchw01_to_dtype(x_prev, dtype)
            elif "timestep" in lname and int(np.prod(shape)) == 1:
                feed[name] = np.full(shape, float(args.fg_timestep), dtype=dtype)
            else:
                feed[name] = np.zeros(shape, dtype=dtype)
    else:
        feed[in_name] = x
        if temporal_model and len(compiled.inputs) >= 2:
            for inp in compiled.inputs[1:]:
                feed[str(inp.any_name)] = x
    outputs = req.infer(feed)
    ti1 = time.perf_counter()

    y = outputs[out_name] if out_name in outputs else next(iter(outputs.values()))
    y_np = np.asarray(y)

    exec_devices = "unknown"
    try:
        v = compiled.get_property("EXECUTION_DEVICES")
        if isinstance(v, (list, tuple)):
            exec_devices = ",".join(str(x) for x in v)
        else:
            exec_devices = str(v)
    except Exception:
        pass

    print(f"smoke_device={args.device}")
    print(f"smoke_exec_devices={exec_devices}")
    print(f"compile_cfg={cfg}")
    print(f"compile_ms={(t1 - t0) * 1000.0:.3f}")
    print(f"infer_ms={(ti1 - ti0) * 1000.0:.3f}")
    print(f"infer_output_shape={tuple(int(x) for x in y_np.shape)}")

    if args.out_image:
        _save_nchw01_image(y_np, Path(args.out_image))
        print(f"saved_smoke_image={args.out_image}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Build fixed (no-train) SR algorithm IR for OpenVINO/NPU.")
    parser.add_argument(
        "--model-type",
        choices=["sr_x2", "fg_mid"],
        default="sr_x2",
        help="Model type: SR x2 or FG midpoint interpolation.",
    )
    parser.add_argument("--input-shape", default="1,3,256,256", help="Static NCHW shape for model input.")
    parser.add_argument("--upscale", type=int, default=2, help="Upscale factor.")
    parser.add_argument("--sharpen-gain", type=float, default=0.67, help="Unsharp detail gain.")
    parser.add_argument("--preserve-gain", type=float, default=0.90, help="Blend ratio between sharpened/smoothed.")
    parser.add_argument("--temporal-model", action="store_true", help="Build temporal-plus multi-input model.")
    parser.add_argument(
        "--temporal-preset",
        choices=["custom", "balanced", "crisp", "stable"],
        default="custom",
        help="Temporal-plus preset. custom keeps explicit --temporal-* values.",
    )
    parser.add_argument("--temporal-gain", type=float, default=0.35, help="Temporal blend max gain (0~1).")
    parser.add_argument("--temporal-depth", type=int, default=3, help="Temporal input depth (2=prev, 3=prev+prev2).")
    parser.add_argument("--temporal-hf-gain", type=float, default=1.05, help="Current high-frequency preserve gain.")
    parser.add_argument("--temporal-detail-boost", type=float, default=0.12, help="Extra HF gain on edge regions.")
    parser.add_argument(
        "--temporal-detail-threshold",
        type=float,
        default=0.05,
        help="HF magnitude threshold for edge-aware gain map.",
    )
    parser.add_argument(
        "--temporal-diff-threshold",
        type=float,
        default=0.08,
        help="Difference threshold for temporal confidence (0~1 scale).",
    )
    parser.add_argument(
        "--output-xml",
        default="model/ir/fixed_sr_algo_x2_sharp067.xml",
        help="Output IR xml path.",
    )
    parser.add_argument("--fg-sharpen-gain", type=float, default=0.18, help="FG detail preserve gain.")
    parser.add_argument("--fg-motion-low", type=float, default=0.03, help="FG motion gate low threshold.")
    parser.add_argument("--fg-motion-high", type=float, default=0.18, help="FG motion gate high threshold.")
    parser.add_argument(
        "--fg-motion-smooth",
        type=float,
        default=0.12,
        help="FG smoothing amount on high motion.",
    )
    parser.add_argument("--fg-timestep", type=float, default=0.5, help="FG smoke inference timestep.")

    parser.add_argument("--smoke", action="store_true", help="Compile and infer once after saving.")
    parser.add_argument("--device", default="NPU", help="Smoke compile device.")
    parser.add_argument("--precision-hint", choices=["f16", "f32", "i8"], default="f16")
    parser.add_argument("--cache-dir", default=".ov_cache")
    parser.add_argument("--image", default="", help="Optional input image for smoke test.")
    parser.add_argument("--image-prev", default="", help="Optional previous/next image for fg_mid smoke test.")
    parser.add_argument("--out-image", default="", help="Optional output image path for smoke inference.")
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
