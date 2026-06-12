"""
End-to-end demo of the Intelligent NVR Chatbot — no LLM API required.

Steps
-----
1. Generate a 60-second synthetic test video (OpenCV, black bg, moving shapes)
2. Index all frames: YOLO detection + CLIP embeddings → FAISS + SQLite
3. Run 4 chatbot queries via the local intent classifier + template responses
4. Print pipeline status summary

Usage
-----
    python demo.py

Prerequisites
-------------
    pip install ultralytics opencv-python faiss-cpu open-clip-torch torch \
                scikit-learn fastapi uvicorn python-dotenv
    python src/ml/train_intent_classifier.py   # train intent model once
"""
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
for noisy in ("PIL", "timm", "open_clip", "ultralytics", "faiss"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# ── demo constants ─────────────────────────────────────────────────────────────
DEMO_DIR       = Path("./data/demo")
VIDEO_PATH     = DEMO_DIR / "test_footage.mp4"
FAISS_PATH     = DEMO_DIR / "demo_index.faiss"
DB_PATH        = DEMO_DIR / "demo.db"
CLIPS_DIR      = DEMO_DIR / "clips"
DEMO_CAMERA    = "cam_demo"
VIDEO_DURATION = 60    # seconds
VIDEO_FPS      = 25
INDEX_EVERY_S  = 2     # sample 1 frame every N seconds

DEMO_DIR.mkdir(parents=True, exist_ok=True)
CLIPS_DIR.mkdir(parents=True, exist_ok=True)


# ── helpers ────────────────────────────────────────────────────────────────────

def banner(title: str, char: str = "=") -> None:
    width = 62
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def ok(msg: str)   -> None: print(f"  [OK]  {msg}")
def info(msg: str) -> None: print(f"        {msg}")
def warn(msg: str) -> None: print(f"  [!!]  {msg}")


# ── Step 1: generate synthetic test video ──────────────────────────────────────

def generate_video() -> bool:
    banner("STEP 1 — Generate synthetic test video")

    try:
        import cv2
        import numpy as np
    except ImportError:
        warn("opencv-python not installed.  Run: pip install opencv-python")
        return False

    if VIDEO_PATH.exists():
        ok(f"Already exists: {VIDEO_PATH}  ({VIDEO_PATH.stat().st_size // 1024} KB)")
        return True

    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(VIDEO_PATH), fourcc, VIDEO_FPS, (width, height))

    # White rectangle → simulates a person moving horizontally
    rx, ry, rdx = 50, height // 2, 5

    # White circle → simulates a vehicle moving on a different path
    cx, cy, cdx, cdy = width - 100, 100, -3, 2

    total = VIDEO_DURATION * VIDEO_FPS
    for i in range(total):
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # person shape
        cv2.rectangle(frame, (rx, ry - 40), (rx + 30, ry + 40), (255, 255, 255), -1)
        cv2.putText(
            frame, "person", (rx, ry - 46),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1,
        )

        # vehicle shape
        cv2.circle(frame, (cx, cy), 25, (255, 255, 255), -1)
        cv2.putText(
            frame, "vehicle", (cx - 22, cy - 31),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1,
        )

        cv2.putText(
            frame, f"frame {i:04d}", (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (70, 70, 70), 1,
        )
        writer.write(frame)

        # update positions
        rx = (rx + rdx) % (width - 60)
        cx = max(30, min(width - 30, cx + cdx))
        cy = max(30, min(height - 30, cy + cdy))
        if cx in (30, width - 30):
            cdx = -cdx
        if cy in (30, height - 30):
            cdy = -cdy

    writer.release()
    size_kb = VIDEO_PATH.stat().st_size // 1024
    ok(f"Generated: {VIDEO_PATH}  ({size_kb} KB, {VIDEO_DURATION}s @ {VIDEO_FPS}fps)")
    return True


# ── Step 2: YOLO detection + CLIP indexing ─────────────────────────────────────

def index_frames() -> bool:
    banner("STEP 2 — Object detection + CLIP indexing")

    if not VIDEO_PATH.exists():
        warn("Video not found — run step 1 first.")
        return False

    # Clean previous run artefacts for reproducibility
    DB_PATH.unlink(missing_ok=True)
    FAISS_PATH.unlink(missing_ok=True)
    FAISS_PATH.with_suffix(".pkl").unlink(missing_ok=True)

    try:
        import cv2
        from src.detection.yolo_detector import YOLODetector
        from src.indexing.frame_indexer import FrameIndexer
    except ImportError as exc:
        warn(f"Missing dependency: {exc}")
        return False

    print("  Loading CLIP ViT-B/32 (first run downloads ~600 MB) ...")
    indexer = FrameIndexer(
        faiss_index_path=str(FAISS_PATH),
        db_path=str(DB_PATH),
        nlist=10,        # small nlist is fine for a tiny demo set
        device="cpu",
    )

    print("  Loading YOLOv8n (CPU) ...")
    detector = YOLODetector(use_openvino=False, confidence_threshold=0.35)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    source_fps = cap.get(cv2.CAP_PROP_FPS) or VIDEO_FPS
    skip = max(1, int(source_fps * INDEX_EVERY_S))
    base_time = datetime(2026, 5, 1, 10, 0, 0)

    frame_idx = 0
    indexed   = 0
    print(f"  Sampling 1 frame every {INDEX_EVERY_S}s from {VIDEO_DURATION}s video ...\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % skip != 0:
            continue

        detections = detector.detect(frame)
        objects    = list({d.class_name for d in detections})
        timestamp  = base_time + timedelta(seconds=frame_idx / source_fps)

        row_id = indexer.index_frame(
            frame=frame,
            camera_id=DEMO_CAMERA,
            video_path=str(VIDEO_PATH),
            timestamp=timestamp,
            detected_objects=objects,
            zone="entrance",
        )
        indexed += 1
        det_str = ", ".join(objects) if objects else "none"
        print(
            f"  Frame #{indexed:02d}  id={row_id}  "
            f"t={timestamp.strftime('%H:%M:%S')}  objects=[{det_str}]"
        )

    cap.release()
    indexer.save()
    ok(
        f"Indexed {indexed} frames — "
        f"FAISS({indexer.total_indexed} vectors) + "
        f"SQLite({indexer.get_frame_count()} rows)"
    )
    return True


# ── Step 3: chatbot queries ────────────────────────────────────────────────────

def run_chatbot(idx_ok: bool) -> bool:
    banner("STEP 3 — Chatbot (local template-based, no LLM API)")

    try:
        from src.search.retrospective_search import RetrospectiveSearch
        from src.agent.chatbot_agent import ChatbotAgent, AgentContext
    except ImportError as exc:
        warn(f"Import error: {exc}")
        return False

    if idx_ok and FAISS_PATH.exists():
        search = RetrospectiveSearch(
            faiss_index_path=str(FAISS_PATH),
            db_path=str(DB_PATH),
            clips_dir=str(CLIPS_DIR),
            top_k=3,
            device="cpu",
        )
    else:
        from unittest.mock import MagicMock
        search = MagicMock()
        search.search.return_value = []
        search.get_summary.return_value = {
            "total_events": 0, "object_counts": {}, "hourly_distribution": {}
        }
        warn("FAISS index unavailable — chatbot running with stub search engine.")

    try:
        agent = ChatbotAgent(
            search_engine=search,
            ml_model_dir="src/ml/models",
        )
    except FileNotFoundError as exc:
        warn(f"Intent model not found: {exc}")
        warn("Run:  python src/ml/train_intent_classifier.py")
        return False

    ctx = AgentContext(session_id="demo")

    queries = [
        "bonjour",
        "montre-moi les événements détectés hier",
        "y a-t-il eu une personne dans la vidéo ?",
        "fais-moi un résumé de l'activité",
    ]

    for msg in queries:
        print(f"\n  User : {msg}")
        t0 = time.time()
        result = agent.chat(msg, ctx)
        elapsed = time.time() - t0

        print(f"  Bot  : {result.get('response', '')}")

        clf = result.get("classification_info", {})
        ml  = clf.get("ml", {})
        if ml:
            print(
                f"         [intent={ml['intent']}  "
                f"conf={ml['confidence']:.0%}  {elapsed:.2f}s]"
            )

        for e in (result.get("events") or [])[:2]:
            info(f"  -> #{e['id']}  {e['timestamp']}  objects={e['detected_objects']}")

        if result.get("summary"):
            s = result["summary"]
            info(
                f"  -> total={s.get('total_events', 0)}  "
                f"objects={s.get('object_counts', {})}"
            )

    return True


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 62)
    print("  Intelligent NVR Chatbot — End-to-End Demo (No LLM API)")
    print("  Universite Mohammed V | Master IT | ML & Deep Learning")
    print("=" * 62)
    print(f"  Date  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Python: {sys.version.split()[0]}")

    vid_ok  = generate_video()
    idx_ok  = index_frames() if vid_ok else False
    chat_ok = run_chatbot(idx_ok)

    banner("DEMO COMPLETE", char="-")
    def tick(flag: bool) -> str:
        return "OK" if flag else "--"

    print(
        f"  Demo complete. Pipeline: "
        f"YOLO [{tick(idx_ok)}]  "
        f"CLIP [{tick(idx_ok)}]  "
        f"FAISS [{tick(idx_ok)}]  "
        f"SQLite [{tick(idx_ok)}]  "
        f"Intent Classifier [{tick(chat_ok)}]  "
        f"Chatbot [{tick(chat_ok)}]"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDemo interrupted.")
