"""
Quick RTSP camera connectivity test (standalone, no project deps beyond OpenCV).

Usage
-----
    python test_camera.py "rtsp://user:pass@ip:554/path"

Tries TCP transport first (more reliable than UDP for most NVRs), opens the
stream, reads a few frames, prints resolution/FPS, and saves a snapshot so you
can confirm the picture visually.
"""
import os
import sys
import time

# Force RTSP over TCP + a connection timeout so we don't hang forever on a bad URL.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|stimeout;5000000",  # stimeout is in microseconds (5 s)
)

import cv2

DEFAULT_URL = "rtsp://admin:Sofyan94@192.168.1.86:554/h264/ch1/main/av_stream"


def test(url: str) -> int:
    # Hide password when echoing the URL.
    shown = url
    if "@" in url and "//" in url:
        head, tail = url.split("//", 1)
        if "@" in tail:
            creds, host = tail.split("@", 1)
            user = creds.split(":", 1)[0]
            shown = f"{head}//{user}:****@{host}"

    print(f"Connecting to: {shown}")
    print("Transport    : TCP (OPENCV_FFMPEG_CAPTURE_OPTIONS)")
    t0 = time.time()

    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"\n[FAIL] Cannot open stream (after {time.time() - t0:.1f}s).")
        print("Possible causes: wrong IP/port, bad credentials, wrong stream path,")
        print("camera unreachable on the network, or firewall blocking 554.")
        return 1

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    print(f"[OK]   Stream opened in {time.time() - t0:.1f}s")
    print(f"       Resolution : {width}x{height}")
    print(f"       FPS (meta) : {fps:.1f}")

    # Read a handful of frames to confirm decoding actually works.
    grabbed = 0
    last_frame = None
    for i in range(10):
        ret, frame = cap.read()
        if not ret:
            print(f"       Frame {i}: read FAILED")
            continue
        grabbed += 1
        last_frame = frame

    print(f"       Frames read: {grabbed}/10")

    if last_frame is not None:
        snap = "camera_snapshot.jpg"
        cv2.imwrite(snap, last_frame)
        print(f"[OK]   Snapshot saved: {snap}")

    cap.release()

    if grabbed == 0:
        print("\n[FAIL] Stream opened but no frames could be decoded.")
        return 1

    print("\n[SUCCESS] Camera is reachable and streaming.")
    return 0


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    sys.exit(test(url))
