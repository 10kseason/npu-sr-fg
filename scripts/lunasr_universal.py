import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


@dataclass
class LunaParams:
    quality_tier: str = "quality"
    m0: float = 0.03
    m1: float = 0.14
    alpha_base: float = 40.0
    beta_base: float = 2.5
    block_strength: float = 0.20
    sharp_base: float = 0.14
    clamp_delta: float = 1.0 / 255.0
    dither_amp: float = 0.6 / 255.0
    text_t0: float = 0.04
    text_t1: float = 0.20
    var_v0: float = 0.0004
    var_v1: float = 0.0030
    artifact_passes: int = 1
    refine_gain: float = 0.06
    source_detail_gain: float = 0.03
    preblur_sigma: float = 0.0
    deblock_softness: float = 1.0
    dither_layers: int = 1
    resample_interp: int = cv2.INTER_CUBIC
    chroma_interp: int = cv2.INTER_LINEAR


def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def legacy_mode_to_tier(mode: int) -> str:
    if mode == 0:
        return "balanced"
    if mode == 2:
        return "ultra_quality"
    return "quality"


def resolve_quality_tier(raw: Optional[str]) -> str:
    if raw is None:
        return "quality"
    token = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "0": "balanced",
        "q0": "balanced",
        "balanced": "balanced",
        "balance": "balanced",
        "1": "quality",
        "q1": "quality",
        "quality": "quality",
        "2": "ultra_quality",
        "q2": "ultra_quality",
        "ultra": "ultra_quality",
        "ultra_quality": "ultra_quality",
        "ultraquality": "ultra_quality",
        "highest": "ultra_quality",
    }
    if token not in mapping:
        raise RuntimeError(
            f"Unknown quality tier: {raw}. Use balanced/quality/ultra_quality."
        )
    return mapping[token]


def resolve_tune_strategy(raw: Optional[str], profile: Optional[str]) -> str:
    if profile:
        p = str(profile).strip().lower().replace("-", "_")
        if p in {"universal_balanced", "c"}:
            return "c"
    if raw is None:
        return "default"
    token = str(raw).strip().lower().replace("-", "_")
    if token in {"default", "base"}:
        return "default"
    if token in {"c", "universal_balanced"}:
        return "c"
    raise RuntimeError(
        f"Unknown auto-tune strategy: {raw}. Use default or c."
    )


def resolve_luna_profile(raw: Optional[str]) -> str:
    if raw is None:
        return "custom"
    token = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "custom": "custom",
        "default": "custom",
        "max_performance": "max_performance",
        "maxperf": "max_performance",
        "performance": "max_performance",
        "quality_up": "quality_up",
        "qualityup": "quality_up",
        "quality": "quality_up",
    }
    if token not in mapping:
        raise RuntimeError(
            f"Unknown luna profile: {raw}. Use custom|max_performance|quality_up."
        )
    return mapping[token]


def apply_luna_profile_overrides(
    profile_name: str,
    tier: str,
    tune_strategy: str,
    internal_scale: float,
    auto_tune_on: bool,
    frame_budget_ms: float,
) -> Tuple[str, str, float, bool, float, str]:
    note = "profile=custom (manual settings)"
    if profile_name == "max_performance":
        # User requirement: in max performance profile, internal-scale is fixed to 0.66.
        tier = "balanced"
        tune_strategy = "c"
        internal_scale = 0.66
        auto_tune_on = True
        if frame_budget_ms <= 0:
            frame_budget_ms = 0.0
        note = (
            "profile=max_performance: tier=balanced, strategy=c, "
            "internal_scale=0.66(fixed), auto_tune=on"
        )
    elif profile_name == "quality_up":
        tier = "ultra_quality"
        tune_strategy = "default"
        internal_scale = 1.0
        auto_tune_on = True
        frame_budget_ms = 0.0
        note = (
            "profile=quality_up: tier=ultra_quality, strategy=default, "
            "internal_scale=1.0(fixed), auto_tune=on"
        )
    return tier, tune_strategy, internal_scale, auto_tune_on, frame_budget_ms, note


def mode_defaults(tier: str) -> LunaParams:
    t = resolve_quality_tier(tier)
    if t == "balanced":
        return LunaParams(
            quality_tier="balanced",
            alpha_base=26.0,
            beta_base=1.1,
            block_strength=0.14,
            sharp_base=0.10,
            clamp_delta=1.0 / 255.0,
            dither_amp=0.35 / 255.0,
            artifact_passes=1,
            refine_gain=0.20,
            source_detail_gain=0.25,
            preblur_sigma=0.0,
            deblock_softness=0.85,
            dither_layers=1,
            resample_interp=cv2.INTER_CUBIC,
            chroma_interp=cv2.INTER_LINEAR,
        )
    if t == "ultra_quality":
        return LunaParams(
            quality_tier="ultra_quality",
            alpha_base=58.0,
            beta_base=4.5,
            block_strength=0.28,
            sharp_base=0.16,
            clamp_delta=0.8 / 255.0,
            dither_amp=0.72 / 255.0,
            artifact_passes=3,
            refine_gain=0.70,
            source_detail_gain=0.75,
            preblur_sigma=0.45,
            deblock_softness=1.2,
            dither_layers=3,
            resample_interp=cv2.INTER_LANCZOS4,
            chroma_interp=cv2.INTER_LANCZOS4,
        )
    return LunaParams(
        quality_tier="quality",
        alpha_base=44.0,
        beta_base=2.8,
        block_strength=0.22,
        sharp_base=0.14,
        clamp_delta=0.9 / 255.0,
        dither_amp=0.55 / 255.0,
        artifact_passes=2,
        refine_gain=0.55,
        source_detail_gain=0.60,
        preblur_sigma=0.35,
        deblock_softness=1.0,
        dither_layers=2,
        resample_interp=cv2.INTER_LANCZOS4,
        chroma_interp=cv2.INTER_CUBIC,
    )


def grad_and_var(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3, scale=1 / 8.0)
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3, scale=1 / 8.0)
    grad = np.sqrt(gx * gx + gy * gy)
    mean = cv2.blur(y, (3, 3))
    mean2 = cv2.blur(y * y, (3, 3))
    var = np.maximum(mean2 - mean * mean, 0.0)
    return grad, var


def text_mask_from_grad_var(grad: np.ndarray, var: np.ndarray, p: LunaParams) -> np.ndarray:
    return clamp01((grad - p.text_t0) / max(1e-6, (p.text_t1 - p.text_t0))) * clamp01(
        (var - p.var_v0) / max(1e-6, (p.var_v1 - p.var_v0))
    )


def blockiness_metric(y: np.ndarray) -> float:
    h, w = y.shape
    if h < 16 or w < 16:
        return 0.0
    dx = np.abs(y[:, 1:] - y[:, :-1])
    dy = np.abs(y[1:, :] - y[:-1, :])

    cols = np.arange(w - 1)
    rows = np.arange(h - 1)
    bx = cols % 8 == 7
    by = rows % 8 == 7

    v_boundary = float(dx[:, bx].mean()) if np.any(bx) else 0.0
    v_non = float(dx[:, ~bx].mean()) if np.any(~bx) else 0.0
    h_boundary = float(dy[by, :].mean()) if np.any(by) else 0.0
    h_non = float(dy[~by, :].mean()) if np.any(~by) else 0.0

    boundary = 0.5 * (v_boundary + h_boundary)
    non_boundary = 0.5 * (v_non + h_non)
    if non_boundary <= 1e-6:
        return 0.0
    return max(0.0, boundary / non_boundary - 1.0)


def banding_risk(y: np.ndarray, grad: np.ndarray, m0: float) -> float:
    if y.shape[0] < 2 or y.shape[1] < 2:
        return 0.0
    d = np.abs(y[:, 1:] - y[:, :-1])
    g = grad[:, 1:]
    flat = g < max(1e-6, m0 * 0.8)
    stair = (d > (0.8 / 255.0)) & (d < (6.0 / 255.0))
    if not np.any(flat):
        return 0.0
    score = float(stair[flat].mean())
    return float(np.clip(score / 0.35, 0.0, 1.0))


def analyze_metrics(y: np.ndarray, p: LunaParams) -> Dict[str, float]:
    grad, var = grad_and_var(y)
    text_mask = text_mask_from_grad_var(grad, var, p)
    text_ratio = float(text_mask.mean())
    block_raw = float(blockiness_metric(y))
    noise_raw = float(np.mean(np.abs(cv2.Laplacian(y, cv2.CV_32F, ksize=3))))
    band_raw = float(banding_risk(y, grad, p.m0))

    # Normalize to 0..1 ranges for strategy rules.
    b = float(np.clip(block_raw / 0.70, 0.0, 1.0))
    t = float(np.clip(text_ratio / 0.20, 0.0, 1.0))
    n = float(np.clip(noise_raw / 0.25, 0.0, 1.0))
    g = float(np.clip(band_raw, 0.0, 1.0))

    return {
        "B": b,
        "T": t,
        "N": n,
        "G": g,
        "text_ratio_raw": text_ratio,
        "blockiness_raw": block_raw,
        "noise_raw": noise_raw,
        "banding_raw": band_raw,
    }


def apply_strategy_default(base: LunaParams, metrics: Dict[str, float]) -> LunaParams:
    text_ratio = metrics["text_ratio_raw"]
    block_raw = metrics["blockiness_raw"]
    noise_raw = metrics["noise_raw"]

    p = LunaParams(**vars(base))
    p.block_strength = float(np.clip(base.block_strength + 0.08 * block_raw, 0.10, 0.35))
    p.alpha_base = float(np.clip(base.alpha_base + 18.0 * block_raw + 280.0 * noise_raw, 20.0, 90.0))
    p.beta_base = float(np.clip(base.beta_base + 1.3 * text_ratio, 1.0, 7.0))
    p.sharp_base = float(np.clip(base.sharp_base - 0.10 * block_raw - 0.20 * noise_raw, 0.08, 0.25))
    p.clamp_delta = float(np.clip(base.clamp_delta * (1.0 - 0.6 * text_ratio), 0.0, 1.0 / 255.0))
    p.dither_amp = float(np.clip(base.dither_amp * (1.0 + 0.7 * block_raw), 0.3 / 255.0, 0.9 / 255.0))
    p.refine_gain = float(np.clip(base.refine_gain + 0.05 * text_ratio - 0.08 * noise_raw, 0.03, 0.90))
    p.source_detail_gain = float(np.clip(base.source_detail_gain + 0.04 * text_ratio - 0.06 * block_raw, 0.02, 0.90))
    p.artifact_passes = int(np.clip(base.artifact_passes + (1 if block_raw > 0.18 else 0), 1, 4))
    return p


def apply_strategy_c(base: LunaParams, metrics: Dict[str, float], is_video: bool) -> LunaParams:
    p = LunaParams(**vars(base))
    b = metrics["B"]
    t = metrics["T"]
    n = metrics["N"]
    g = metrics["G"]

    passes_cap = {"balanced": 1, "quality": 2, "ultra_quality": 2}
    cap = passes_cap.get(p.quality_tier, 2)
    p.artifact_passes = 1 + int(round(np.clip(b * 1.6, 0.0, cap - 1)))

    sigma_base = {"balanced": 0.0, "quality": 0.35, "ultra_quality": 0.45}
    p.preblur_sigma = sigma_base.get(p.quality_tier, 0.35) * n * ((1.0 - t) ** 2.0)
    if t > 0.45:
        p.preblur_sigma = 0.0
    p.preblur_sigma = float(np.clip(p.preblur_sigma, 0.0, 0.25))

    if p.quality_tier in {"quality", "ultra_quality"}:
        if (t > 0.45) or (b > 0.65) or (n > 0.70):
            p.resample_interp = cv2.INTER_CUBIC
        else:
            p.resample_interp = cv2.INTER_LANCZOS4
    else:
        p.resample_interp = cv2.INTER_CUBIC

    if t > 0.45:
        p.chroma_interp = cv2.INTER_LINEAR
    elif p.quality_tier == "ultra_quality":
        p.chroma_interp = cv2.INTER_LANCZOS4
    elif p.quality_tier == "quality":
        p.chroma_interp = cv2.INTER_CUBIC
    else:
        p.chroma_interp = cv2.INTER_LINEAR

    source_base = {"balanced": 0.25, "quality": 0.60, "ultra_quality": 0.75}
    refine_base = {"balanced": 0.20, "quality": 0.55, "ultra_quality": 0.70}
    sharp_base = {"balanced": 0.10, "quality": 0.14, "ultra_quality": 0.16}

    source_eff = source_base.get(p.quality_tier, 0.60) * ((1.0 - b) ** 1.2) * (1.0 - 0.6 * t)
    refine_eff = refine_base.get(p.quality_tier, 0.55) * (1.0 - 0.7 * b) * (1.0 - 0.7 * t)

    # Required guardrails from update text.
    source_eff *= (1.0 - 0.5 * t)
    refine_eff *= (1.0 - 0.7 * t)
    if b > 0.70:
        source_eff = min(source_eff, 0.22)

    p.source_detail_gain = float(np.clip(source_eff, 0.0, 1.0))
    p.refine_gain = float(np.clip(refine_eff, 0.0, 1.0))

    sharp_eff = sharp_base.get(p.quality_tier, 0.14) * (1.0 - 0.5 * b) * (1.0 - 0.7 * t) * (1.0 - 0.5 * n)
    p.sharp_base = float(np.clip(sharp_eff, 0.0, 1.0))

    p.clamp_delta = float((1.0 / 255.0) * (0.2 + 0.8 * (1.0 - t)))

    # C profile: keep dithering conservative, prioritize temporal stability on video.
    if is_video:
        p.dither_layers = 1
    else:
        p.dither_layers = 2 if (g > 0.5 and t < 0.3 and p.quality_tier in {"quality", "ultra_quality"}) else 1
    p.dither_amp = float((0.6 / 255.0) * g * (1.0 - 0.7 * t))

    return p


def auto_tune(
    y: np.ndarray,
    base: LunaParams,
    strategy: str = "default",
    is_video: bool = False,
) -> Tuple[LunaParams, Dict[str, float]]:
    metrics = analyze_metrics(y, base)
    if strategy == "c":
        tuned = apply_strategy_c(base, metrics, is_video=is_video)
    else:
        tuned = apply_strategy_default(base, metrics)
    return tuned, metrics


def apply_performance_fallback_step(p: LunaParams, stage: int) -> Tuple[LunaParams, int, str]:
    q = LunaParams(**vars(p))
    if stage == 0:
        q.chroma_interp = cv2.INTER_LINEAR
        return q, 1, "fallback_step1: chroma -> LINEAR"
    if stage == 1:
        q.resample_interp = cv2.INTER_CUBIC
        return q, 2, "fallback_step2: luma -> CUBIC"
    if stage == 2:
        q.artifact_passes = 1
        return q, 3, "fallback_step3: artifact_passes -> 1"
    if stage == 3:
        q.preblur_sigma = 0.0
        return q, 4, "fallback_step4: preblur_sigma -> 0"
    if stage == 4:
        q.dither_layers = 1
        return q, 5, "fallback_step5: dither_layers -> 1"
    return q, stage, "fallback_step_done"


def noise2d(h: int, w: int, seed: int) -> np.ndarray:
    ys = np.arange(h, dtype=np.uint32)[:, None]
    xs = np.arange(w, dtype=np.uint32)[None, :]
    s = np.uint32(seed & 0xFFFFFFFF)
    n = xs * np.uint32(1973) + ys * np.uint32(9277) + s * np.uint32(26699) + np.uint32(0x68BC21EB)
    n = (n << np.uint32(13)) ^ n
    hashed = n * (n * n * np.uint32(15731) + np.uint32(789221)) + np.uint32(1376312589)
    return (hashed & np.uint32(0x7FFFFFFF)).astype(np.float32) / float(0x7FFFFFFF)


def build_masks(y: np.ndarray, p: LunaParams) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grad, var = grad_and_var(y)
    a = clamp01((grad - p.m0) / max(1e-6, (p.m1 - p.m0)))
    text = text_mask_from_grad_var(grad, var, p)

    h, w = y.shape
    xs = np.arange(w, dtype=np.float32)[None, :]
    ys = np.arange(h, dtype=np.float32)[:, None]
    fracx = np.mod(xs / 8.0, 1.0)
    fracy = np.mod(ys / 8.0, 1.0)
    bx = 1.0 - clamp01(np.abs(fracx - 0.5) / 0.5)
    by = 1.0 - clamp01(np.abs(fracy - 0.5) / 0.5)
    b = np.maximum(bx, by)
    gamma_block = b * (1.0 - a) * p.block_strength * (1.0 - text)
    return grad, a, text, gamma_block


def guided_filter_gray(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    r = max(1, int(radius))
    k = (r * 2 + 1, r * 2 + 1)
    e = max(1e-6, float(eps))

    mean_i = cv2.boxFilter(guide, cv2.CV_32F, k, borderType=cv2.BORDER_REFLECT101)
    mean_p = cv2.boxFilter(src, cv2.CV_32F, k, borderType=cv2.BORDER_REFLECT101)
    mean_ip = cv2.boxFilter(guide * src, cv2.CV_32F, k, borderType=cv2.BORDER_REFLECT101)
    cov_ip = mean_ip - mean_i * mean_p

    mean_ii = cv2.boxFilter(guide * guide, cv2.CV_32F, k, borderType=cv2.BORDER_REFLECT101)
    var_i = mean_ii - mean_i * mean_i

    a = cov_ip / (var_i + e)
    b = mean_p - a * mean_i
    mean_a = cv2.boxFilter(a, cv2.CV_32F, k, borderType=cv2.BORDER_REFLECT101)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, k, borderType=cv2.BORDER_REFLECT101)
    return mean_a * guide + mean_b


def run_artifact_fix(y: np.ndarray, gamma_block: np.ndarray, p: LunaParams) -> np.ndarray:
    y_work = clamp01(y.astype(np.float32))
    for pi in range(p.artifact_passes):
        pass_ratio = float(pi + 1) / float(max(1, p.artifact_passes))
        radius = int(np.clip(round(1.5 + p.beta_base * (1.0 + 0.20 * pi)), 1, 16))
        eps = float(np.clip((0.0012 + p.alpha_base * 0.00005) * (1.0 + 0.25 * pi), 1e-5, 0.08))
        y_guided = guided_filter_gray(y_work, y_work, radius=radius, eps=eps)
        mix = clamp01(gamma_block * (0.55 + 0.35 * pass_ratio) * p.deblock_softness)
        y_work = (1.0 - mix) * y_work + mix * y_guided
        y_work = clamp01(y_work)
    return y_work


def apply_lunasr_luma(y: np.ndarray, out_h: int, out_w: int, p: LunaParams, seed: int) -> np.ndarray:
    grad, a, text, gamma_block = build_masks(y, p)
    y_fix = run_artifact_fix(y, gamma_block, p)

    if p.preblur_sigma > 0.0:
        y_src = cv2.GaussianBlur(y_fix, (0, 0), sigmaX=p.preblur_sigma, sigmaY=p.preblur_sigma)
    else:
        y_src = y_fix

    y0 = cv2.resize(y_src, (out_w, out_h), interpolation=p.resample_interp)
    low_src = cv2.GaussianBlur(y_src, (0, 0), sigmaX=1.0, sigmaY=1.0)
    ymean = cv2.resize(low_src, (out_w, out_h), interpolation=p.resample_interp)

    a_up = cv2.resize(a, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    text_up = cv2.resize(text, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    grad_up = cv2.resize(grad, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    source_hi = y_fix - cv2.GaussianBlur(y_fix, (0, 0), sigmaX=1.1, sigmaY=1.1)
    source_hi_up = cv2.resize(source_hi, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    edge_gate = clamp01(a_up)
    refine_eff = p.refine_gain * (1.0 - 0.7 * text_up)
    source_eff = p.source_detail_gain * (1.0 - 0.5 * text_up)
    y_refine = y0 + (refine_eff * edge_gate) * (y0 - ymean) + (source_eff * edge_gate) * source_hi_up

    sharp_eff = p.sharp_base * (1.0 - 0.7 * text_up)
    y_sharp = y_refine + sharp_eff * (y_refine - ymean)

    min_s = cv2.resize(cv2.erode(y_fix, np.ones((3, 3), np.uint8)), (out_w, out_h), interpolation=p.resample_interp)
    max_s = cv2.resize(cv2.dilate(y_fix, np.ones((3, 3), np.uint8)), (out_w, out_h), interpolation=p.resample_interp)
    y_clamped = np.clip(y_sharp, min_s - p.clamp_delta, max_s + p.clamp_delta)

    band = (1.0 - a_up) * clamp01((p.m0 - grad_up) / max(1e-6, p.m0)) * (1.0 - text_up)

    dither_total = np.zeros((out_h, out_w), dtype=np.float32)
    for li in range(max(1, p.dither_layers)):
        n = noise2d(out_h, out_w, seed + li * 7919).astype(np.float32) - 0.5
        if li > 0:
            n = cv2.GaussianBlur(n, (0, 0), sigmaX=0.6 * li, sigmaY=0.6 * li)
        dither_total += n * (p.dither_amp * (0.7 ** li))

    return clamp01(y_clamped + dither_total * band)


def lunasr_upscale_bgr(frame_bgr: np.ndarray, scale: float, p: LunaParams, seed: int) -> np.ndarray:
    return lunasr_upscale_bgr_internal(
        frame_bgr=frame_bgr,
        scale=scale,
        p=p,
        seed=seed,
        internal_scale=1.0,
    )


def lunasr_upscale_bgr_internal(
    frame_bgr: np.ndarray,
    scale: float,
    p: LunaParams,
    seed: int,
    internal_scale: float = 1.0,
) -> np.ndarray:
    src_h, src_w = frame_bgr.shape[:2]
    target_h = int(round(src_h * scale))
    target_w = int(round(src_w * scale))

    if not (0.0 < internal_scale <= 1.0):
        raise RuntimeError(f"internal_scale must be in (0,1]. got {internal_scale}")

    if internal_scale < 1.0:
        work_h = max(1, int(round(src_h * internal_scale)))
        work_w = max(1, int(round(src_w * internal_scale)))
        work = cv2.resize(frame_bgr, (work_w, work_h), interpolation=cv2.INTER_AREA)
    else:
        work = frame_bgr
        work_h, work_w = src_h, src_w

    out_h = int(round(work_h * scale))
    out_w = int(round(work_w * scale))

    yuv = cv2.cvtColor(work, cv2.COLOR_BGR2YCrCb).astype(np.float32) / 255.0
    y = yuv[:, :, 0]
    cr = yuv[:, :, 1]
    cb = yuv[:, :, 2]

    y_up = apply_lunasr_luma(y, out_h, out_w, p, seed)
    cr_up = cv2.resize(cr, (out_w, out_h), interpolation=p.chroma_interp)
    cb_up = cv2.resize(cb, (out_w, out_h), interpolation=p.chroma_interp)

    yuv_up = np.stack([y_up, cr_up, cb_up], axis=2)
    out = cv2.cvtColor(np.clip(yuv_up * 255.0 + 0.5, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR)

    if out.shape[0] != target_h or out.shape[1] != target_w:
        out = cv2.resize(out, (target_w, target_h), interpolation=p.resample_interp)
    return out


def process_image(
    input_path: str,
    output_path: str,
    scale: float,
    p: LunaParams,
    auto_tune_on: bool,
    seed: int,
    tune_strategy: str,
    internal_scale: float,
) -> None:
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {input_path}")

    y = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb).astype(np.float32)[:, :, 0] / 255.0
    if auto_tune_on:
        p, metrics = auto_tune(y, p, strategy=tune_strategy, is_video=False)
        print(f"auto_tune_strategy={tune_strategy}")
        print(f"auto_tune_metrics={metrics}")
        print(f"auto_tuned_params={vars(p)}")

    t0 = time.perf_counter()
    out = lunasr_upscale_bgr_internal(img, scale, p, seed, internal_scale=internal_scale)
    t1 = time.perf_counter()

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), out):
        raise RuntimeError(f"Failed to write image: {out_path}")

    print(f"quality_tier={p.quality_tier}")
    print(f"saved={out_path}")
    print(f"input_shape={img.shape[1]}x{img.shape[0]}, output_shape={out.shape[1]}x{out.shape[0]}")
    print(f"elapsed_ms={(t1 - t0) * 1000.0:.3f}")


def process_video(
    input_path: str,
    output_path: str,
    scale: float,
    p: LunaParams,
    auto_tune_on: bool,
    seed: int,
    max_frames: int,
    progress_every: int,
    tune_strategy: str,
    frame_budget_ms: float,
    internal_scale: float,
) -> None:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {input_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        fps = 30.0
    target = frames if max_frames <= 0 else min(frames, max_frames)

    out_w = int(round(src_w * scale))
    out_h = int(round(src_h * scale))
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (out_w, out_h),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open writer: {out_path}")

    print(f"input_video={input_path}")
    print(f"quality_tier={p.quality_tier}")
    print(f"auto_tune_strategy={tune_strategy}")
    print(f"internal_scale={internal_scale:.3f}")
    print(f"input_shape={src_w}x{src_h}, fps={fps:.3f}, frames={frames}")
    print(f"output_shape={out_w}x{out_h}, target_frames={target}")
    print(f"frame_budget_ms={frame_budget_ms:.3f}")

    tuned = False
    fallback_stage = 0
    processed = 0
    elapsed = []
    t0 = time.perf_counter()

    while processed < target:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        if auto_tune_on and not tuned:
            y = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb).astype(np.float32)[:, :, 0] / 255.0
            p, metrics = auto_tune(y, p, strategy=tune_strategy, is_video=True)
            print(f"auto_tune_metrics={metrics}")
            print(f"auto_tuned_params={vars(p)}")
            tuned = True

        # C strategy: deterministic pattern for temporal stability.
        if tune_strategy == "c":
            frame_seed = seed + (processed // 120)
        else:
            frame_seed = seed + processed

        fi0 = time.perf_counter()
        out = lunasr_upscale_bgr_internal(frame, scale, p, frame_seed, internal_scale=internal_scale)
        fi1 = time.perf_counter()
        writer.write(out)
        processed += 1
        frame_ms = (fi1 - fi0) * 1000.0
        elapsed.append(frame_ms)

        if frame_budget_ms > 0 and frame_ms > frame_budget_ms:
            p, fallback_stage, msg = apply_performance_fallback_step(p, fallback_stage)
            if msg != "fallback_step_done":
                print(f"[frame {processed}] {msg}")

        if processed % max(1, progress_every) == 0 or processed == target:
            print(f"[frame {processed}/{target}] frame_ms={elapsed[-1]:.3f}")

    t1 = time.perf_counter()
    cap.release()
    writer.release()

    avg_ms = float(np.mean(elapsed)) if elapsed else 0.0
    eff_fps = (1000.0 / avg_ms) if avg_ms > 0 else 0.0
    total_ms = (t1 - t0) * 1000.0

    print(f"saved_video={out_path}")
    print(f"processed_frames={processed}")
    print(f"avg_frame_ms={avg_ms:.3f}, effective_fps={eff_fps:.3f}")
    print(f"total_elapsed_ms={total_ms:.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="LunaSR-Universal v0.3 (tier + strategy)")
    parser.add_argument("--input", required=True, help="Input image or video path")
    parser.add_argument("--output", required=True, help="Output path (.png/.mp4)")
    parser.add_argument("--scale", type=float, default=2.0, help="Upscale factor (recommended <= 2.0)")
    parser.add_argument(
        "--quality-tier",
        default=None,
        help="balanced | quality | ultra_quality",
    )
    parser.add_argument(
        "--quality-mode",
        type=int,
        choices=[0, 1, 2],
        default=None,
        help="Legacy mode. 0=balanced, 1=quality, 2=ultra_quality",
    )
    parser.add_argument("--auto-tune", action="store_true", help="Auto tune params from content")
    parser.add_argument(
        "--luna-profile",
        default="custom",
        help="custom | max_performance | quality_up",
    )
    parser.add_argument(
        "--auto-tune-strategy",
        default="default",
        help="Auto tune strategy: default | c",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Optional named profile. universal_balanced maps to strategy c",
    )
    parser.add_argument(
        "--frame-budget-ms",
        type=float,
        default=0.0,
        help="Video only: if >0, apply staged fallback when frame time exceeds budget",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Noise seed")
    parser.add_argument(
        "--internal-scale",
        type=float,
        default=1.0,
        help="Process at reduced internal resolution, then resize to exact output size",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="Video only: limit frames, 0=all")
    parser.add_argument("--progress-every", type=int, default=5, help="Video only: print every N frames")
    args = parser.parse_args()

    if args.scale <= 0:
        raise RuntimeError(f"--scale must be > 0. got {args.scale}")
    if not (0.0 < args.internal_scale <= 1.0):
        raise RuntimeError(f"--internal-scale must be in (0,1]. got {args.internal_scale}")
    if args.scale > 2.0:
        print("WARN: LunaSR recommends scale <= 2.0 for quality/perf balance.")

    profile_name = resolve_luna_profile(args.luna_profile)
    tier = "quality"
    if args.quality_mode is not None:
        tier = legacy_mode_to_tier(args.quality_mode)
    if args.quality_tier is not None:
        tier = resolve_quality_tier(args.quality_tier)
    tune_strategy = resolve_tune_strategy(args.auto_tune_strategy, args.profile)
    (
        tier,
        tune_strategy,
        args.internal_scale,
        args.auto_tune,
        args.frame_budget_ms,
        profile_note,
    ) = apply_luna_profile_overrides(
        profile_name=profile_name,
        tier=tier,
        tune_strategy=tune_strategy,
        internal_scale=args.internal_scale,
        auto_tune_on=args.auto_tune,
        frame_budget_ms=args.frame_budget_ms,
    )
    print(profile_note)

    p = mode_defaults(tier)

    ext = Path(args.input).suffix.lower()
    if ext in {".mp4", ".avi", ".mov", ".mkv"}:
        process_video(
            input_path=args.input,
            output_path=args.output,
            scale=args.scale,
            p=p,
            auto_tune_on=args.auto_tune,
            seed=args.seed,
            max_frames=args.max_frames,
            progress_every=args.progress_every,
            tune_strategy=tune_strategy,
            frame_budget_ms=args.frame_budget_ms,
            internal_scale=args.internal_scale,
        )
    else:
        process_image(
            input_path=args.input,
            output_path=args.output,
            scale=args.scale,
            p=p,
            auto_tune_on=args.auto_tune,
            seed=args.seed,
            tune_strategy=tune_strategy,
            internal_scale=args.internal_scale,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
