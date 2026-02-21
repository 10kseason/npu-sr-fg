# LunaSR Realtime Upscale Lab (OpenVINO NPU/GPU)

[Korean README](README.ko.md)

This repository is a Windows-focused lab for realtime window upscaling/frame-generation experiments using OpenVINO.

## Scope

- Realtime overlay pipeline: capture -> upscale/infer -> overlay.
- OpenVINO-first runtime for Intel NPU/GPU.
- Algorithmic fixed SR/FG IR builders (no dataset required).
- Benchmarking and utility scripts for NPU/GPU profiling.

## Current Realtime Defaults

- Backend in GUI is fixed to `openvino_sr`.
- Default SR model is `model/ir/fixed_sr_algo_x2_temporal.xml`.
- Default output preset is `AUTO`.
- `AUTO` output mapping:
- `<720` -> `HD (720)`
- `<1080` -> `FHD (1080)`
- `<1440` -> `QHD (1440)`
- `>=1440` -> `4K (2160)`
- Device policy is enforced:
- `Device=GPU` -> GPU only
- `Device=NPU` -> NPU only
- CPU fallback is disabled in GUI profile.
- Overlay click-through is always enabled in overlay mode.
- `Alt+P` stops realtime processing while GUI is active.
- Capture uses a ring frame buffer (`size=4`), and overlay polling is high-frequency (`8ms`).
- GPU path uses throughput-oriented compile hints and parallel infer requests.
- NPU path applies INT8-oriented compile hints where available.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run Realtime GUI

```powershell
python scripts\lunasr_realtime_gui.py
```

Basic flow:

- Select target window.
- Select `Device` (`NPU` or `GPU`).
- Keep `Output` at `AUTO` unless you need a fixed target.
- Click `Start`.
- Use `Alt+P` (or `Stop`) to stop.

## Model Files (Important)

- OpenVINO IR models require both files:
- `*.xml`
- `*.bin`
- If you share/deploy the model for OpenVINO, upload the XML+BIN pair together.
- ONNX is optional and only needed for ONNX-based runtimes.

## Benchmark Logging

- Enable `Benchmark Log` in GUI to write per-frame CSV metrics.
- Default file: `bench_realtime_overlay.csv`.

## Key Scripts

- `scripts/check_npu_stack.py`: Verify OpenVINO/ORT/NPU stack visibility.
- `scripts/ov_npu_bench.py`: Compile + inference benchmark for ONNX/IR on OpenVINO.
- `scripts/build_fixed_sr_ir.py`: Build fixed algorithmic SR/FG IR models.
- `scripts/upscale_png_x2_npu.py`: Offline image x2 upscale with OpenVINO.
- `scripts/upscale_video_x2_npu.py`: Offline video x2 upscale with OpenVINO.
- `scripts/upscale_video_x2_with_fg_npu.py`: Video upscale + FG flow.
- `scripts/bench_npu_tiling_presets.py`: Preset-based tiling benchmark.
- `scripts/lunasr_gpu_train_npu_infer.py`: Train on GPU, export, and run NPU infer workflow.

## Optional: Train on GPU, Infer on NPU

Install extra packages first:

```powershell
pip install torch nncf
```

Then use:

- `scripts/lunasr_gpu_train_npu_infer.py make-dataset`
- `scripts/lunasr_gpu_train_npu_infer.py train`
- `scripts/lunasr_gpu_train_npu_infer.py export-onnx`
- `scripts/lunasr_gpu_train_npu_infer.py ptq-int8`
- `scripts/lunasr_gpu_train_npu_infer.py smoke-npu`
