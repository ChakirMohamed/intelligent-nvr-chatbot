"""Validate the event-generation fixes without loading YOLO/CLIP.

Fix 1 — file playback is throttled to source fps (no flood).
Fix 2 — frames with no detected object are dropped.
Fix 3 — sustained activity collapses into one event; new event on object
        change or after EVENT_GAP_SECONDS of silence.
"""
import os
import sys
import tempfile
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

os.environ["INDEX_REQUIRE_OBJECT"] = "true"
os.environ["EVENT_GAP_SECONDS"] = "10"

import src.api.api as api
from src.ingestion.rtsp_reader import RTSPReader

ok = True


def check(cond: bool, label: str) -> None:
    global ok
    print(f"  [{'OK' if cond else 'FAIL'}] {label}")
    ok = ok and cond


# ── Fix 1: real-time pacing for a file source ───────────────────────────────────
print("Fix 1 — file playback pacing")
tmp = Path(tempfile.mkdtemp())
video = tmp / "v.mp4"
RTSPReader.generate_test_video(str(video), duration=4, fps=15)

hits = {"n": 0}
reader = RTSPReader(
    camera_id="cam_pace",
    rtsp_url=str(video),
    storage_path=str(tmp / "rec"),
    sampling_fps=4.0,
    on_motion_frame=lambda *_: hits.__setitem__("n", hits["n"] + 1),
)
reader.start()
time.sleep(1.0)
reader.stop()
# At 4 sampling fps, ~1 s of paced playback yields a handful of callbacks.
# Without pacing the whole file would replay repeatedly → dozens/hundreds.
check(hits["n"] <= 10, f"paced ~{hits['n']} motion frames in 1s (unpaced would be >>10)")
check(reader.is_connected is False, "reader disconnected after stop")


# ── Fix 2 + Fix 3: worker grouping ──────────────────────────────────────────────
print("Fix 2 + Fix 3 — object filter + event grouping")


class _Det:
    def __init__(self, name): self.class_name = name


class _FakeDetector:
    """Returns a scripted object list per call."""
    def __init__(self, script): self.script, self.i = script, 0

    def detect(self, _frame):
        objs = self.script[self.i] if self.i < len(self.script) else []
        self.i += 1
        return [_Det(o) for o in objs]


class _FakeIndexer:
    def __init__(self): self.rows = []
    def index_frame(self, **kw): self.rows.append(kw); return len(self.rows)
    def save(self): pass


# Scripted detections aligned with the timestamps below.
script = [
    ["person"],          # t=0   → new event #1
    ["person"],          # t=1   → continuation
    ["person"],          # t=2   → continuation
    [],                  # t=3   → no object, dropped (Fix 2)
    ["person"],          # t=15  → gap >10s → new event #2
    ["person", "car"],   # t=16  → objects changed → new event #3
    ["person", "car"],   # t=16.5→ continuation
]
times = [0, 1, 2, 3, 15, 16, 16.5]

fake_idx = _FakeIndexer()
api._detector = _FakeDetector(script)
api._indexer = fake_idx
api._ingest_running = True

worker = threading.Thread(target=api._indexer_worker, daemon=True)
worker.start()

base = datetime(2026, 6, 18, 12, 0, 0)
for dt in times:
    api._index_queue.put((
        "cam_x", None, base + timedelta(seconds=dt), str(video),
    ))

# Let the worker drain the queue.
for _ in range(50):
    if api._index_queue.empty():
        break
    time.sleep(0.05)
time.sleep(0.2)
api._ingest_running = False
worker.join(timeout=3)

n = len(fake_idx.rows)
check(n == 3, f"3 events from 7 frames (got {n})")
if n == 3:
    check(fake_idx.rows[0]["detected_objects"] == ["person"], "event #1 = [person]")
    check(fake_idx.rows[2]["detected_objects"] == ["car", "person"], "event #3 = [car, person]")

print("\n" + ("[SUCCESS] Event fixes work." if ok else "[FAILURE] See above."))
sys.exit(0 if ok else 1)
