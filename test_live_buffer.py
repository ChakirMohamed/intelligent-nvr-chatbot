"""Validate the RTSPReader live-frame buffer without the full API stack.

Generates a tiny synthetic video, feeds it through RTSPReader (which treats a
file path like a stream), and checks that get_latest_jpeg() returns a decodable
JPEG and that the connection/segment state is populated.
"""
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from src.ingestion.rtsp_reader import RTSPReader


def main() -> int:
    tmp = Path(tempfile.mkdtemp())
    video = tmp / "synthetic.mp4"
    RTSPReader.generate_test_video(str(video), duration=6, fps=15)

    reader = RTSPReader(
        camera_id="cam_test",
        rtsp_url=str(video),
        storage_path=str(tmp / "rec"),
        sampling_fps=2.0,
        segment_duration=300,
    )
    reader.start()

    # Give the capture thread a moment to open the file and decode a frame.
    jpeg = None
    for _ in range(20):
        time.sleep(0.1)
        jpeg = reader.get_latest_jpeg()
        if jpeg:
            break

    ok = True
    if not jpeg:
        print("[FAIL] get_latest_jpeg() returned None — no frame buffered.")
        ok = False
    else:
        arr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            print("[FAIL] Returned bytes are not a decodable JPEG.")
            ok = False
        else:
            print(f"[OK]   JPEG decoded: {arr.shape[1]}x{arr.shape[0]}, {len(jpeg)} bytes")

    print(f"[OK]   is_connected   = {reader.is_connected}")
    print(f"[OK]   current_segment= {reader.current_segment}")
    if not reader.is_connected:
        print("[FAIL] Reader should report connected while streaming.")
        ok = False

    reader.stop()
    print(f"[OK]   after stop, is_connected = {reader.is_connected}")

    print("\n" + ("[SUCCESS] Live buffer works." if ok else "[FAILURE] See above."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
