# LunaSR Upscale Lab (OpenVINO NPU/GPU)

[Korean README](README.ko.md)

Windows-focused OpenVINO lab for:
- Realtime overlay upscaling experiments
- Offline image/video upscaling GUI
- Algorithmic SR/FG IR model builders

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Python Dependencies

Runtime dependencies are in:
- `requirements.txt`

Build dependency for EXE packaging:
- `requirements-build.txt` (`pyinstaller`)

Install build dependency only when needed:

```powershell
pip install -r requirements-build.txt
```

## Offline Upscale GUI (Image/Video)

Run:

```powershell
python scripts\upscale_media_gui.py
```

Current behavior:
- Supports images and videos
- Scale range: `x2 ~ x10`
- Model profiles: `anime`, `photo`, `custom`
- Optional `Soft Postprocess` toggle
- Non-ASCII path fallback for image/video I/O

Default profile models:
- `model/ir/fixed_sr_algo_x2_quality_plus_aa_anime.xml`
- `model/ir/fixed_sr_algo_x2_quality_plus_aa_photo.xml`

Important:
- OpenVINO IR must be provided as XML+BIN pair.

## Build EXE (Core-only, with anime/photo models)

Onedir build:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_upscale_gui_exe.ps1
```

Onefile build:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_upscale_gui_exe.ps1 -OneFile
```

Outputs:
- Onedir release: `release/LunaSR-Upscale-GUI/`
- Onefile exe: `release/LunaSR-Upscale-GUI/LunaSR-Upscale-GUI.exe`
- Zip: `release/LunaSR-Upscale-GUI.zip`

The packaging script includes only core app/runtime assets and anime/photo model files (not test media).

## Realtime Overlay GUI

Run:

```powershell
python scripts\lunasr_realtime_gui.py
```

## Build/Model Scripts

- `scripts/build_fixed_sr_ir.py`: Build fixed SR/FG IR models
  - Supports SR presets: `quality_plus`, `quality_plus_aa`, `quality_plus_aa_anime`, `quality_plus_aa_photo`
- `scripts/upscale_media_gui.py`: Offline image/video upscale GUI
- `scripts/build_upscale_gui_exe.ps1`: EXE packaging script

## Other Key Scripts

- `scripts/check_npu_stack.py`: Verify OpenVINO/ORT/NPU stack
- `scripts/ov_npu_bench.py`: OpenVINO ONNX/IR benchmark
- `scripts/upscale_png_x2_npu.py`: Offline image x2 upscale
- `scripts/upscale_video_x2_npu.py`: Offline video x2 upscale
- `scripts/upscale_video_x2_with_fg_npu.py`: Video upscale + FG pipeline
