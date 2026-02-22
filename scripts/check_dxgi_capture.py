import argparse
import sys
from pathlib import Path

import cv2

from realtime_capture import capture_window_bgr_dxgi, dxgi_capture_status, enumerate_monitor_rects


def main() -> int:
    parser = argparse.ArgumentParser(description="Quick DXGI capture preflight/check.")
    parser.add_argument("--hwnd", type=lambda x: int(str(x), 0), default=0, help="Target window handle (hex or int).")
    parser.add_argument("--save", type=str, default="", help="Optional output image path.")
    args = parser.parse_args()

    ok, note = dxgi_capture_status()
    print(f"python={sys.executable}")
    print(f"dxgi_status={ok} note={note}")
    print(f"monitors={enumerate_monitor_rects()}")

    if not ok:
        return 2
    if int(args.hwnd) <= 0:
        return 0

    cap = capture_window_bgr_dxgi(int(args.hwnd))
    if cap is None:
        print("capture=failed")
        return 3
    frame, rect = cap
    print(f"capture=ok shape={frame.shape} rect={rect}")

    if args.save:
        out = Path(args.save).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), frame)
        print(f"saved={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
