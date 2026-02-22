from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class BackendBundle:
    luna_runtime: Optional[Dict[str, Any]]
    ov_runtime: Optional[Dict[str, Any]]
    fg_runtime: Optional[Dict[str, Any]]
    profile_note: str
    warnings: List[str]


def build_backend_bundle(
    cfg: Any,
    *,
    build_luna_runtime_fn: Callable[[str, float], Dict[str, Any]],
    compile_ov_runtime_fn: Callable[..., Dict[str, Any]],
    compile_fg_runtime_fn: Callable[..., Dict[str, Any]],
    cuda_available: bool,
    dpi_awareness_mode: str,
) -> BackendBundle:
    warnings: List[str] = []
    luna_runtime: Optional[Dict[str, Any]] = None
    ov_runtime: Optional[Dict[str, Any]] = None
    strict_npu_only = bool(getattr(cfg, "strict_npu_only", False))
    strict_gpu_only = bool(getattr(cfg, "strict_gpu_only", False))
    allow_cpu_fallback = bool(getattr(cfg, "allow_cpu_fallback", False)) and (not strict_npu_only) and (
        not strict_gpu_only
    )

    backend = str(getattr(cfg, "backend", "")).strip()
    if strict_npu_only and backend != "openvino_sr":
        raise RuntimeError("strict_npu_only requires backend=openvino_sr.")
    if backend == "lunasr":
        luna_runtime = build_luna_runtime_fn(str(getattr(cfg, "preset", "max_performance")), float(cfg.frame_budget_ms))
        backend_note = str(luna_runtime["note"])
    elif backend == "openvino_sr":
        ov_runtime = compile_ov_runtime_fn(
            model_path=str(getattr(cfg, "ov_model", "")),
            device=str(getattr(cfg, "device", "AUTO")),
            preset=str(getattr(cfg, "ov_preset", "speed")),
            cache_dir=str(getattr(cfg, "ov_cache_dir", "")),
            allow_cpu_fallback=allow_cpu_fallback,
            strict_npu_only=strict_npu_only,
            strict_gpu_only=strict_gpu_only,
        )
        backend_note = (
            f"backend=openvino_sr, device={cfg.device}, preset={cfg.ov_preset}, "
            f"capture={cfg.capture_backend}, output={cfg.output_preset}, "
            f"gpu_video={'cuda' if (cfg.gpu_video_path and cuda_available) else 'off'}, "
            f"reactive={'on' if cfg.ov_reactive_scale else 'off'}@{cfg.ov_reactive_target_fps:.0f}fps, "
            f"temporal={'on' if cfg.temporal_restore else 'off'}@{cfg.temporal_strength:.2f}, "
            f"dpi={dpi_awareness_mode}, "
            f"npu_i8={'on' if 'NPU' in str(ov_runtime['compile_device']).upper() else 'off'}, "
            f"gpu_tile_base_720p={'on' if 'GPU' in str(ov_runtime['compile_device']).upper() else 'off'}, "
            f"cpu_fallback={'on' if allow_cpu_fallback else 'off'}, "
            f"strict_npu={'on' if strict_npu_only else 'off'}, "
            f"strict_gpu={'on' if strict_gpu_only else 'off'}, "
            f"temporal_model={'on' if ov_runtime.get('temporal_model', False) else 'off'}, "
            f"model={ov_runtime['model_path']}, compile_device={ov_runtime['compile_device']}, "
            f"exec={ov_runtime['exec_devices']}, "
            f"req_pool={ov_runtime.get('ov_parallel_reqs', 1)}, "
            f"compile_ms={ov_runtime['compile_ms']:.1f}"
        )
        if ov_runtime.get("compile_note"):
            backend_note = backend_note + f" | {ov_runtime['compile_note']}"
    else:
        raise RuntimeError(f"Unknown backend: {backend}")

    fg_runtime: Optional[Dict[str, Any]] = None
    fg_note = "fg=off"
    if bool(getattr(cfg, "fg_enabled", False)):
        try:
            fg_runtime = compile_fg_runtime_fn(
                model_path=str(getattr(cfg, "fg_model", "")),
                device=str(getattr(cfg, "device", "AUTO")),
                cache_dir=str(getattr(cfg, "ov_cache_dir", "")),
                precision=str(getattr(cfg, "fg_precision", "f16")),
                fg_size=int(getattr(cfg, "fg_size", 256)),
                allow_cpu_fallback=allow_cpu_fallback,
                strict_npu_only=strict_npu_only,
                strict_gpu_only=strict_gpu_only,
            )
            fg_note = (
                f"fg=on model={fg_runtime['model_path']} "
                f"prec={fg_runtime['precision']} size={fg_runtime['fg_size']} "
                f"compile_device={fg_runtime['compile_device']} "
                f"exec={fg_runtime['exec_devices']} "
                f"compile_ms={fg_runtime['compile_ms']:.1f}"
            )
            if fg_runtime.get("compile_note"):
                fg_note = fg_note + f" | {fg_runtime['compile_note']}"
            if bool(getattr(cfg, "fg_interp_only", False)):
                fg_note = fg_note + " mode=interp_only"
        except Exception as exc:
            fg_runtime = None
            fg_note = f"fg=on setup_failed: {exc}"
            warnings.append(f"FG setup failed, fallback=blend: {exc}")

    return BackendBundle(
        luna_runtime=luna_runtime,
        ov_runtime=ov_runtime,
        fg_runtime=fg_runtime,
        profile_note=f"{backend_note} | {fg_note}",
        warnings=warnings,
    )
