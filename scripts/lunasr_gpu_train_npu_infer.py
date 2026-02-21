import argparse
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from lunasr_universal import (
    apply_luna_profile_overrides,
    lunasr_upscale_bgr_internal,
    mode_defaults,
    resolve_luna_profile,
)


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_shape(text: str) -> Tuple[int, int, int, int]:
    parts = [p.strip() for p in str(text).split(",")]
    if len(parts) != 4:
        raise RuntimeError(f"Expected shape like 1,3,256,256. got {text}")
    n, c, h, w = [int(x) for x in parts]
    if n != 1 or c not in (1, 3, 4) or h <= 0 or w <= 0:
        raise RuntimeError(f"Unsupported input shape: {n},{c},{h},{w}")
    return n, c, h, w


def list_images(root: Path) -> List[Path]:
    if not root.exists():
        return []
    items: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            items.append(p)
    items.sort()
    return items


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def bgr_to_nchw01(bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x = np.transpose(rgb, (2, 0, 1))[None].astype(np.float32) / 255.0
    return x


def nchw01_to_bgr(x: np.ndarray) -> np.ndarray:
    if x.ndim == 4:
        x = x[0]
    x = np.transpose(x, (1, 2, 0))
    x = np.clip(x, 0.0, 1.0)
    rgb = (x * 255.0 + 0.5).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def run_make_dataset(args) -> int:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    lr_root = output_dir / "lr"
    hr_root = output_dir / "hr"
    lr_root.mkdir(parents=True, exist_ok=True)
    hr_root.mkdir(parents=True, exist_ok=True)

    images = list_images(input_dir)
    if not images:
        raise RuntimeError(f"No images found under: {input_dir}")
    if args.max_images > 0:
        images = images[: args.max_images]

    profile_name = resolve_luna_profile(args.luna_profile)
    tier = "quality"
    tune_strategy = "default"
    internal_scale = float(args.internal_scale)
    auto_tune_on = False
    frame_budget_ms = 0.0
    (
        tier,
        tune_strategy,
        internal_scale,
        auto_tune_on,
        frame_budget_ms,
        profile_note,
    ) = apply_luna_profile_overrides(
        profile_name=profile_name,
        tier=tier,
        tune_strategy=tune_strategy,
        internal_scale=internal_scale,
        auto_tune_on=auto_tune_on,
        frame_budget_ms=frame_budget_ms,
    )
    p = mode_defaults(tier)

    print(f"teacher_profile={profile_name}")
    print(f"teacher_note={profile_note}")
    print(f"teacher_tier={tier}, internal_scale={internal_scale:.3f}, scale={args.scale}")
    print(f"input_images={len(images)}")

    t0 = time.perf_counter()
    saved = 0
    for idx, src in enumerate(images):
        rel = src.relative_to(input_dir)
        lr_out = lr_root / rel
        hr_out = hr_root / rel
        ensure_parent(lr_out)
        ensure_parent(hr_out)

        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[skip] read failed: {src}")
            continue
        if args.max_edge > 0:
            h, w = img.shape[:2]
            m = max(h, w)
            if m > args.max_edge:
                s = float(args.max_edge) / float(m)
                nw = max(1, int(round(w * s)))
                nh = max(1, int(round(h * s)))
                img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

        seed = int(args.seed + idx)
        teacher = lunasr_upscale_bgr_internal(
            frame_bgr=img,
            scale=float(args.scale),
            p=p,
            seed=seed,
            internal_scale=float(internal_scale),
        )
        if not cv2.imwrite(str(lr_out), img):
            print(f"[skip] write failed: {lr_out}")
            continue
        if not cv2.imwrite(str(hr_out), teacher):
            print(f"[skip] write failed: {hr_out}")
            continue
        saved += 1
        if saved % max(1, args.progress_every) == 0 or saved == len(images):
            print(f"[{saved}/{len(images)}] {rel}")

    t1 = time.perf_counter()
    print(f"saved_pairs={saved}")
    print(f"lr_dir={lr_root}")
    print(f"hr_dir={hr_root}")
    print(f"elapsed_ms={(t1 - t0) * 1000.0:.1f}")
    return 0


def _lazy_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, Dataset
    except Exception as e:
        raise RuntimeError(
            "PyTorch is required for training/export. install torch first."
        ) from e
    return torch, nn, F, DataLoader, Dataset


def _build_espcn(nn, scale: int):
    class ESPCN(nn.Module):
        def __init__(self, s: int):
            super().__init__()
            self.s = int(s)
            self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
            self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(32, 3 * self.s * self.s, kernel_size=3, padding=1)
            self.ps = nn.PixelShuffle(self.s)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = self.ps(self.conv3(x))
            x = torch.clamp(x, 0.0, 1.0)
            return x

    return ESPCN(scale)


@dataclass
class PairItem:
    lr: Path
    hr: Path


def _collect_pairs(dataset_dir: Path) -> List[PairItem]:
    lr_root = dataset_dir / "lr"
    hr_root = dataset_dir / "hr"
    if not lr_root.exists() or not hr_root.exists():
        raise RuntimeError(f"Dataset must contain lr/ and hr/: {dataset_dir}")
    lr_files = list_images(lr_root)
    pairs: List[PairItem] = []
    for lr in lr_files:
        rel = lr.relative_to(lr_root)
        hr = hr_root / rel
        if hr.exists():
            pairs.append(PairItem(lr=lr, hr=hr))
    if not pairs:
        raise RuntimeError(f"No matched lr/hr pairs in: {dataset_dir}")
    return pairs


def run_train(args) -> int:
    torch, nn, F, DataLoader, Dataset = _lazy_torch()

    class PairDataset(Dataset):
        def __init__(self, pairs: Sequence[PairItem], patch: int, scale: int, augment: bool):
            self.pairs = list(pairs)
            self.patch = int(patch)
            self.scale = int(scale)
            self.augment = bool(augment)

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, i: int):
            item = self.pairs[i]
            lr = cv2.imread(str(item.lr), cv2.IMREAD_COLOR)
            hr = cv2.imread(str(item.hr), cv2.IMREAD_COLOR)
            if lr is None or hr is None:
                raise RuntimeError(f"Failed to read pair: {item.lr} / {item.hr}")

            h, w = lr.shape[:2]
            th = h * self.scale
            tw = w * self.scale
            if hr.shape[0] != th or hr.shape[1] != tw:
                hr = cv2.resize(hr, (tw, th), interpolation=cv2.INTER_CUBIC)

            if self.patch > 0 and h > self.patch and w > self.patch:
                x = random.randint(0, w - self.patch)
                y = random.randint(0, h - self.patch)
                lr = lr[y : y + self.patch, x : x + self.patch]
                y2 = y * self.scale
                x2 = x * self.scale
                p2 = self.patch * self.scale
                hr = hr[y2 : y2 + p2, x2 : x2 + p2]

            if self.augment:
                if random.random() < 0.5:
                    lr = cv2.flip(lr, 1)
                    hr = cv2.flip(hr, 1)
                if random.random() < 0.5:
                    lr = cv2.flip(lr, 0)
                    hr = cv2.flip(hr, 0)

            x = torch.from_numpy(np.transpose(cv2.cvtColor(lr, cv2.COLOR_BGR2RGB), (2, 0, 1))).float() / 255.0
            y = torch.from_numpy(np.transpose(cv2.cvtColor(hr, cv2.COLOR_BGR2RGB), (2, 0, 1))).float() / 255.0
            return x, y

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = _collect_pairs(dataset_dir)
    if args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]
    random.shuffle(pairs)

    if args.val_split > 0.0:
        n_val = max(1, int(round(len(pairs) * args.val_split)))
        val_pairs = pairs[:n_val]
        train_pairs = pairs[n_val:]
    else:
        val_pairs = []
        train_pairs = pairs
    if not train_pairs:
        raise RuntimeError("No train pairs after split.")

    scale = int(args.scale)
    train_ds = PairDataset(train_pairs, patch=args.patch, scale=scale, augment=bool(args.augment))
    val_ds = PairDataset(val_pairs, patch=0, scale=scale, augment=False) if val_pairs else None

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=False,
    )
    val_loader = (
        DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True) if val_ds is not None else None
    )

    device = str(args.device).strip().lower()
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available. Install/enable GPU stack.")
        torch_device = torch.device("cuda")
    else:
        torch_device = torch.device(device)

    model = _build_espcn(nn, scale=scale).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    criterion = nn.L1Loss()

    best_val = float("inf")
    t0 = time.perf_counter()
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses: List[float] = []
        for x, y in train_loader:
            x = x.to(torch_device, non_blocking=True)
            y = y.to(torch_device, non_blocking=True)
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        train_loss = float(np.mean(losses)) if losses else 0.0
        val_loss = train_loss
        if val_loader is not None:
            model.eval()
            vals: List[float] = []
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(torch_device, non_blocking=True)
                    y = y.to(torch_device, non_blocking=True)
                    pred = model(x)
                    vals.append(float(criterion(pred, y).item()))
            val_loss = float(np.mean(vals)) if vals else train_loss

        ckpt = {
            "model_state": model.state_dict(),
            "scale": scale,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "arch": "espcn_x2",
        }
        last_path = out_dir / "student_last.pt"
        torch.save(ckpt, str(last_path))
        if val_loss <= best_val:
            best_val = val_loss
            torch.save(ckpt, str(out_dir / "student_best.pt"))

        print(
            f"[epoch {epoch}/{args.epochs}] train_l1={train_loss:.6f} "
            f"val_l1={val_loss:.6f} device={torch_device}"
        )

    t1 = time.perf_counter()
    print(f"train_pairs={len(train_pairs)}, val_pairs={len(val_pairs)}")
    print(f"saved_ckpt={out_dir / 'student_best.pt'}")
    print(f"elapsed_ms={(t1 - t0) * 1000.0:.1f}")
    return 0


def run_export_onnx(args) -> int:
    torch, nn, F, DataLoader, Dataset = _lazy_torch()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    scale = int(ckpt.get("scale", args.scale))
    model = _build_espcn(nn, scale=scale)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    n, c, h, w = parse_shape(args.input_shape)
    dummy = torch.zeros((n, c, h, w), dtype=torch.float32)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        export_params=True,
        opset_version=int(args.opset),
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,
    )
    print(f"saved_onnx={out_path}")
    print(f"input_shape={n},{c},{h},{w}")
    return 0


def _lazy_ov_nncf():
    try:
        import openvino as ov
    except Exception as e:
        raise RuntimeError("OpenVINO is required.") from e
    try:
        import nncf
    except Exception as e:
        raise RuntimeError("NNCF is required for INT8 PTQ. install nncf.") from e
    return ov, nncf


def _make_calibration_samples(
    img_paths: Sequence[Path],
    in_c: int,
    in_h: int,
    in_w: int,
    limit: int,
) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for p in img_paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if rgb.shape[0] != in_h or rgb.shape[1] != in_w:
            rgb = cv2.resize(rgb, (in_w, in_h), interpolation=cv2.INTER_AREA)
        if in_c == 1:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)[:, :, None]
        elif in_c == 4:
            a = np.full((in_h, in_w, 1), 255, dtype=np.uint8)
            rgb = np.concatenate([rgb, a], axis=2)
        x = np.transpose(rgb, (2, 0, 1))[None].astype(np.float32) / 255.0
        out.append(x)
        if 0 < limit <= len(out):
            break
    return out


def run_ptq_int8(args) -> int:
    ov, nncf = _lazy_ov_nncf()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = ov.Core().read_model(str(model_path))
    if len(model.inputs) != 1:
        raise RuntimeError("PTQ script expects single-input model.")

    in_shape = [int(x) for x in model.inputs[0].shape]
    if len(in_shape) != 4:
        raise RuntimeError(f"Expected 4D NCHW input. got {in_shape}")
    n, c, h, w = in_shape
    if n != 1 or c not in (1, 3, 4):
        raise RuntimeError(f"Unsupported model input shape: {in_shape}")
    in_name = model.inputs[0].any_name

    calib_dir = Path(args.calib_dir)
    imgs = list_images(calib_dir)
    if not imgs:
        raise RuntimeError(f"No calibration images found: {calib_dir}")
    samples = _make_calibration_samples(imgs, in_c=c, in_h=h, in_w=w, limit=int(args.calib_size))
    if not samples:
        raise RuntimeError("No usable calibration samples.")

    def transform_fn(sample):
        return {in_name: sample}

    dataset = nncf.Dataset(samples, transform_fn)
    preset = nncf.QuantizationPreset.PERFORMANCE if args.preset == "performance" else nncf.QuantizationPreset.MIXED

    t0 = time.perf_counter()
    q_model = nncf.quantize(model, dataset, preset=preset)
    t1 = time.perf_counter()

    out_xml = Path(args.output_xml)
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(q_model, str(out_xml))

    print(f"saved_int8_ir={out_xml}")
    print(f"calibration_samples={len(samples)}")
    print(f"quantize_ms={(t1 - t0) * 1000.0:.1f}")
    return 0


def run_smoke_npu(args) -> int:
    ov, nncf = _lazy_ov_nncf()
    model_xml = Path(args.model_xml)
    image_path = Path(args.image)
    if not model_xml.exists():
        raise FileNotFoundError(f"Model not found: {model_xml}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    core = ov.Core()
    model = core.read_model(str(model_xml))
    if len(model.inputs) != 1:
        raise RuntimeError("Smoke test expects single-input model.")
    in_shape = [int(x) for x in model.inputs[0].shape]
    n, c, h, w = in_shape
    if n != 1:
        raise RuntimeError(f"Expected batch=1. got {in_shape}")
    in_name = model.inputs[0].any_name
    out_name = model.outputs[0].any_name

    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if rgb.shape[0] != h or rgb.shape[1] != w:
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    if c == 1:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)[:, :, None]
    elif c == 4:
        a = np.full((h, w, 1), 255, dtype=np.uint8)
        rgb = np.concatenate([rgb, a], axis=2)
    x = np.transpose(rgb, (2, 0, 1))[None].astype(np.float32) / 255.0

    compiled = core.compile_model(model, args.device)
    req = compiled.create_infer_request()
    try:
        exec_devices = compiled.get_property("EXECUTION_DEVICES")
    except Exception:
        exec_devices = "unknown"

    t0 = time.perf_counter()
    y = req.infer({in_name: x})
    t1 = time.perf_counter()
    out = y[out_name] if out_name in y else next(iter(y.values()))

    print(f"device_request={args.device}")
    print(f"execution_devices={exec_devices}")
    print(f"output_shape={np.asarray(out).shape}")
    print(f"infer_ms={(t1 - t0) * 1000.0:.3f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train on GPU, infer on NPU pipeline for LunaSR distillation.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ds = sub.add_parser("make-dataset", help="Generate LR/HR teacher pairs using LunaSR.")
    p_ds.add_argument("--input-dir", required=True)
    p_ds.add_argument("--output-dir", required=True)
    p_ds.add_argument("--luna-profile", default="max_performance", help="custom|max_performance|quality_up")
    p_ds.add_argument("--scale", type=float, default=2.0)
    p_ds.add_argument("--internal-scale", type=float, default=1.0)
    p_ds.add_argument("--seed", type=int, default=1234)
    p_ds.add_argument("--max-images", type=int, default=0, help="0=all")
    p_ds.add_argument("--max-edge", type=int, default=0, help="Downscale long edge before teacher if >0")
    p_ds.add_argument("--progress-every", type=int, default=20)

    p_tr = sub.add_parser("train", help="Train student model on GPU.")
    p_tr.add_argument("--dataset-dir", required=True, help="Folder with lr/ and hr/")
    p_tr.add_argument("--output-dir", required=True)
    p_tr.add_argument("--scale", type=int, default=2)
    p_tr.add_argument("--epochs", type=int, default=30)
    p_tr.add_argument("--batch-size", type=int, default=8)
    p_tr.add_argument("--lr", type=float, default=1e-4)
    p_tr.add_argument("--patch", type=int, default=96, help="LR patch size")
    p_tr.add_argument("--augment", action="store_true")
    p_tr.add_argument("--num-workers", type=int, default=0)
    p_tr.add_argument("--val-split", type=float, default=0.05)
    p_tr.add_argument("--max-pairs", type=int, default=0, help="0=all")
    p_tr.add_argument("--device", default="cuda", help="Use cuda for training")

    p_ex = sub.add_parser("export-onnx", help="Export trained checkpoint to ONNX.")
    p_ex.add_argument("--checkpoint", required=True)
    p_ex.add_argument("--output", required=True)
    p_ex.add_argument("--input-shape", default="1,3,256,256")
    p_ex.add_argument("--scale", type=int, default=2)
    p_ex.add_argument("--opset", type=int, default=17)

    p_q = sub.add_parser("ptq-int8", help="Run NNCF PTQ and save OpenVINO INT8 IR.")
    p_q.add_argument("--model", required=True, help="Input ONNX or IR XML")
    p_q.add_argument("--calib-dir", required=True)
    p_q.add_argument("--calib-size", type=int, default=100)
    p_q.add_argument("--preset", choices=["performance", "mixed"], default="performance")
    p_q.add_argument("--output-xml", required=True)

    p_sm = sub.add_parser("smoke-npu", help="Compile/infer once and print execution devices.")
    p_sm.add_argument("--model-xml", required=True)
    p_sm.add_argument("--image", required=True)
    p_sm.add_argument("--device", default="NPU")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.cmd == "make-dataset":
        return run_make_dataset(args)
    if args.cmd == "train":
        return run_train(args)
    if args.cmd == "export-onnx":
        return run_export_onnx(args)
    if args.cmd == "ptq-int8":
        return run_ptq_int8(args)
    if args.cmd == "smoke-npu":
        return run_smoke_npu(args)
    raise RuntimeError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
