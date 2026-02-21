# LunaSR 실시간 업스케일 실험실 (OpenVINO NPU/GPU)

[English README](README.md)

이 저장소는 Windows 환경에서 OpenVINO 기반 실시간 창 업스케일/프레임 생성 실험을 위한 워크스페이스입니다.

## 범위

- 실시간 오버레이 파이프라인: 캡처 -> 업스케일/추론 -> 오버레이.
- Intel NPU/GPU 대상 OpenVINO 우선 런타임.
- 데이터셋 없이 사용하는 고정형 SR/FG IR 빌더 제공.
- NPU/GPU 프로파일링용 벤치마크/유틸 스크립트 제공.

## 현재 실시간 기본 동작

- GUI 백엔드는 `openvino_sr`로 고정됩니다.
- 기본 SR 모델은 `model/ir/fixed_sr_algo_x2_temporal.xml` 입니다.
- 기본 출력 프리셋은 `AUTO` 입니다.
- `AUTO` 출력 매핑:
- `<720` -> `HD (720)`
- `<1080` -> `FHD (1080)`
- `<1440` -> `QHD (1440)`
- `>=1440` -> `4K (2160)`
- 디바이스 정책은 자동 강제됩니다:
- `Device=GPU` -> GPU 전용 실행
- `Device=NPU` -> NPU 전용 실행
- GUI 프로필에서 CPU 폴백은 비활성화되어 있습니다.
- 오버레이 모드에서 클릭 통과(click-through)는 항상 켜집니다.
- GUI가 활성 상태일 때 `Alt+P`로 실시간 처리를 중지할 수 있습니다.
- 캡처는 링 프레임 버퍼(`size=4`)를 사용하고, 오버레이 폴링은 고주기(`8ms`)로 동작합니다.
- GPU 경로는 처리량(throughput) 위주 컴파일 힌트와 병렬 요청 풀을 사용합니다.
- NPU 경로는 가능한 경우 INT8 지향 컴파일 힌트를 사용합니다.

## 설치

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 실시간 GUI 실행

```powershell
python scripts\lunasr_realtime_gui.py
```

기본 사용 순서:

- 대상 창 선택
- `Device`를 `NPU` 또는 `GPU`로 선택
- 특별한 이유가 없으면 `Output`은 `AUTO` 유지
- `Start` 클릭
- 중지는 `Alt+P` 또는 `Stop` 사용

## 모델 파일(중요)

- OpenVINO IR 모델은 아래 2개가 한 쌍입니다:
- `*.xml`
- `*.bin`
- OpenVINO용으로 모델을 공유/배포할 때는 XML+BIN을 함께 올려야 합니다.
- ONNX는 ONNX 런타임을 쓸 때만 필요하며 필수는 아닙니다.

## 벤치마크 로그

- GUI의 `Benchmark Log`를 켜면 프레임 단위 CSV 로그를 저장합니다.
- 기본 파일명: `bench_realtime_overlay.csv`.

## 주요 스크립트

- `scripts/check_npu_stack.py`: OpenVINO/ORT/NPU 스택 인식 확인.
- `scripts/ov_npu_bench.py`: OpenVINO에서 ONNX/IR 컴파일+추론 벤치.
- `scripts/build_fixed_sr_ir.py`: 고정형 알고리즘 SR/FG IR 생성.
- `scripts/upscale_png_x2_npu.py`: OpenVINO 기반 오프라인 이미지 x2 업스케일.
- `scripts/upscale_video_x2_npu.py`: OpenVINO 기반 오프라인 영상 x2 업스케일.
- `scripts/upscale_video_x2_with_fg_npu.py`: 영상 업스케일 + FG 경로.
- `scripts/bench_npu_tiling_presets.py`: 프리셋 기반 타일링 벤치마크.
- `scripts/lunasr_gpu_train_npu_infer.py`: GPU 학습 -> 내보내기 -> NPU 추론 워크플로.

## 선택 사항: GPU 학습, NPU 추론

추가 패키지 설치:

```powershell
pip install torch nncf
```

이후 사용 명령:

- `scripts/lunasr_gpu_train_npu_infer.py make-dataset`
- `scripts/lunasr_gpu_train_npu_infer.py train`
- `scripts/lunasr_gpu_train_npu_infer.py export-onnx`
- `scripts/lunasr_gpu_train_npu_infer.py ptq-int8`
- `scripts/lunasr_gpu_train_npu_infer.py smoke-npu`
