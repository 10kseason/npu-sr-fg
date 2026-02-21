import platform
import sys


def print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> int:
    print_header("System")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"Machine: {platform.machine()}")

    print_header("ONNX Runtime")
    try:
        import onnxruntime as ort

        print(f"onnxruntime: {ort.__version__}")
        providers = ort.get_available_providers()
        print(f"Available providers: {providers}")
        if "OpenVINOExecutionProvider" not in providers:
            print("WARN: OpenVINOExecutionProvider is not available.")
    except Exception as exc:
        print(f"ERROR: Failed to import onnxruntime: {exc}")
        return 1

    print_header("OpenVINO")
    try:
        import openvino as ov

        print(f"openvino: {ov.__version__}")
        core = ov.Core()
        devices = list(core.available_devices)
        print(f"Available devices: {devices}")
        npu_like = [d for d in devices if "NPU" in d.upper()]
        if not npu_like:
            print("WARN: No NPU device was reported by OpenVINO.")
        else:
            for dev in npu_like:
                try:
                    full_name = core.get_property(dev, "FULL_DEVICE_NAME")
                except Exception:
                    full_name = "<unknown>"
                print(f"NPU device: {dev} ({full_name})")
    except Exception as exc:
        print(f"ERROR: Failed to import/use OpenVINO: {exc}")
        return 1

    print_header("Summary")
    print("If there are WARN/ERROR lines, share them and we will debug next.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
