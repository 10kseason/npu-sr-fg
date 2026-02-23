# LunaSR 업스케일 랩 (OpenVINO NPU/GPU)

[English README](README.md)

이 저장소는 Windows 환경에서 OpenVINO 기반 업스케일 실험을 위한 프로젝트입니다.

- 실시간 오버레이 업스케일 GUI
- 오프라인 이미지/영상 업스케일 GUI
- 학습 없이 쓰는 알고리즘 기반 SR/FG IR 모델 빌더

## 설치

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 파이썬 디펜던시

실행용:
- `requirements.txt`

EXE 빌드용:
- `requirements-build.txt` (`pyinstaller`)

빌드할 때만 추가 설치:

```powershell
pip install -r requirements-build.txt
```

## 오프라인 업스케일 GUI (이미지/영상)

실행:

```powershell
python scripts\upscale_media_gui.py
```

현재 기능:
- 이미지/영상 모두 지원
- 배율 `x2 ~ x10`
- 모델 프로필: `anime`, `photo`, `custom`
- `Soft Postprocess` 토글 지원
- 한글/특수문자 경로 I/O fallback 적용

기본 모델:
- `model/ir/fixed_sr_algo_x2_quality_plus_aa_anime.xml`
- `model/ir/fixed_sr_algo_x2_quality_plus_aa_photo.xml`

중요:
- OpenVINO IR은 `XML + BIN` 쌍이 함께 있어야 동작합니다.

## EXE 빌드 (핵심만 패키징, anime/photo 모델 포함)

`onedir`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_upscale_gui_exe.ps1
```

`onefile`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_upscale_gui_exe.ps1 -OneFile
```

출력:
- onedir 배포 폴더: `release/LunaSR-Upscale-GUI/`
- onefile 단일 exe: `release/LunaSR-Upscale-GUI/LunaSR-Upscale-GUI.exe`
- zip: `release/LunaSR-Upscale-GUI.zip`

패키징 시 테스트 이미지/벤치 파일은 포함하지 않고, 핵심 앱/런타임과 anime/photo 모델만 포함합니다.

## 실시간 오버레이 GUI

실행:

```powershell
python scripts\lunasr_realtime_gui.py
```

## 빌드/모델 스크립트

- `scripts/build_fixed_sr_ir.py`: 고정형 SR/FG IR 모델 생성
  - SR 프리셋: `quality_plus`, `quality_plus_aa`, `quality_plus_aa_anime`, `quality_plus_aa_photo`
- `scripts/upscale_media_gui.py`: 오프라인 이미지/영상 업스케일 GUI
- `scripts/build_upscale_gui_exe.ps1`: EXE 빌드 스크립트

## 기타 주요 스크립트

- `scripts/check_npu_stack.py`: OpenVINO/ORT/NPU 스택 확인
- `scripts/ov_npu_bench.py`: OpenVINO ONNX/IR 벤치
- `scripts/upscale_png_x2_npu.py`: 이미지 x2 업스케일
- `scripts/upscale_video_x2_npu.py`: 영상 x2 업스케일
- `scripts/upscale_video_x2_with_fg_npu.py`: 영상 업스케일 + FG 파이프라인
