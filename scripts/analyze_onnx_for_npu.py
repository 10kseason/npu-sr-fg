import argparse
from collections import Counter

import openvino as ov


def shape_to_text(pshape) -> str:
    parts = []
    for d in pshape:
        parts.append(str(d.get_length()) if d.is_static else "?")
    return "[" + ",".join(parts) + "]"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze ONNX/OpenVINO model for common NPU blockers (dynamic shape, dynamic pad)."
    )
    parser.add_argument("--model", required=True, help="Path to ONNX or IR XML model")
    args = parser.parse_args()

    core = ov.Core()
    model = core.read_model(args.model)

    print("=== Inputs ===")
    any_dynamic = False
    for inp in model.inputs:
        pshape = inp.partial_shape
        dynamic_dims = [idx for idx, d in enumerate(pshape) if not d.is_static]
        if dynamic_dims:
            any_dynamic = True
        print(
            f"name={inp.any_name}, shape={shape_to_text(pshape)}, "
            f"dynamic_dims={dynamic_dims if dynamic_dims else 'none'}"
        )

    print("\n=== Op Summary ===")
    type_count = Counter(op.get_type_name() for op in model.get_ordered_ops())
    for name in ("Pad", "ShapeOf", "Reshape", "Slice", "Gather", "Concat", "Interpolate"):
        print(f"{name}: {type_count.get(name, 0)}")

    print("\n=== Pad Analysis ===")
    pad_ops = [op for op in model.get_ordered_ops() if op.get_type_name() == "Pad"]
    if not pad_ops:
        print("No Pad ops found.")
    else:
        for op in pad_ops:
            pads_begin_src = op.input(1).get_source_output().get_node()
            pads_end_src = op.input(2).get_source_output().get_node()
            is_begin_const = pads_begin_src.get_type_name() == "Constant"
            is_end_const = pads_end_src.get_type_name() == "Constant"
            print(
                f"pad={op.get_friendly_name()}, "
                f"pads_begin_src={pads_begin_src.get_friendly_name()}({pads_begin_src.get_type_name()}), "
                f"pads_end_src={pads_end_src.get_friendly_name()}({pads_end_src.get_type_name()}), "
                f"pads_constant={is_begin_const and is_end_const}"
            )

    print("\n=== Verdict ===")
    if any_dynamic:
        print("- Dynamic input shape detected: static reshape/export recommended.")
    else:
        print("- Input shapes are static.")

    if any(
        (
            op.input(1).get_source_output().get_node().get_type_name() != "Constant"
            or op.input(2).get_source_output().get_node().get_type_name() != "Constant"
        )
        for op in pad_ops
    ):
        print("- Dynamic pad detected: this is a frequent NPU compile blocker.")
    else:
        print("- Pad constants look static.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
