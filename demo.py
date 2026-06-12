"""
End-to-end demo of the Intelligent NVR Chatbot system.
No real camera required — uses a synthetically generated test video.

Usage:
    python demo.py

Prerequisites:
    pip install -r requirements.txt
    cp .env.example .env   # then fill in ANTHROPIC_API_KEY
"""
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# ── logging: show only WARNING+ from third-party libs ─────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
for noisy in ("PIL", "timm", "open_clip", "ultralytics", "faiss"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

from dotenv import load_dotenv
load_dotenv()

# ── demo constants ────────────────────────────────────────────────────────────
DEMO_DIR      = Path("./data/demo")
VIDEO_PATH    = DEMO_DIR / "test_footage.mp4"
FAISS_PATH    = DEMO_DIR / "demo_index.faiss"
DB_PATH       = DEMO_DIR / "demo.db"
CLIPS_DIR     = DEMO_DIR / "clips"
DEMO_CAMERA   = "cam_demo"
VIDEO_DURATION = 60    # seconds
VIDEO_FPS      = 25
INDEX_EVERY_S  = 2     # index 1 frame every N seconds (= 0.5 fps)

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


def _check_dep(package: str) -> bool:
    import importlib
    try:
        importlib.import_module(package.replace("-", "_"))
        return True
    except ImportError:
        return False


# ── Step 0: dependency check ───────────────────────────────────────────────────

def check_dependencies() -> dict:
    banner("STEP 0 — Dependency check")
    deps = {
        "cv2":        "opencv-python",
        "torch":      "torch",
        "open_clip":  "open-clip-torch",
        "faiss":      "faiss-cpu",
        "ultralytics":"ultralytics",
    }
    status = {}
    for mod, pkg in deps.items():
        available = _check_dep(mod)
        status[mod] = available
        mark = "[OK]" if available else "[--]"
        print(f"  {mark}  {pkg}")

    missing = [pkg for mod, pkg in deps.items() if not status[mod]]
    if missing:
        print(f"\n  Missing: {', '.join(missing)}")
        print("  Run:  pip install -r requirements.txt\n")
    else:
        ok("All dependencies available.")
    return status


# ── Step 1: generate test video ────────────────────────────────────────────────

def generate_video(deps: dict) -> bool:
    banner("STEP 1 — Generate synthetic test video")
    if not deps.get("cv2"):
        warn("opencv-python not installed — skipping video generation.")
        return False

    if VIDEO_PATH.exists():
        ok(f"Video already exists: {VIDEO_PATH}  ({VIDEO_PATH.stat().st_size // 1024} KB)")
        return True

    from src.ingestion.rtsp_reader import RTSPReader
    t0 = time.time()
    RTSPReader.generate_test_video(
        str(VIDEO_PATH),
        duration=VIDEO_DURATION,
        fps=VIDEO_FPS,
        width=640,
        height=480,
    )
    ok(f"Generated: {VIDEO_PATH}  ({VIDEO_PATH.stat().st_size // 1024} KB, {time.time()-t0:.1f}s)")
    return True


# ── Step 2: index frames ───────────────────────────────────────────────────────

def index_frames(deps: dict) -> bool:
    banner("STEP 2 — Object detection + CLIP indexing")

    if not all(deps.get(d) for d in ("cv2", "torch", "open_clip", "faiss")):
        warn("Missing deps (cv2 / torch / open-clip-torch / faiss-cpu) — skipping indexing.")
        warn("Run  pip install -r requirements.txt  then re-run this demo.")
        return False

    if not VIDEO_PATH.exists():
        warn(f"Video not found: {VIDEO_PATH} — run step 1 first.")
        return False

    # Reset DB + index for a clean run
    DB_PATH.unlink(missing_ok=True)
    FAISS_PATH.unlink(missing_ok=True)
    FAISS_PATH.with_suffix(".pkl").unlink(missing_ok=True)

    import cv2
    from src.detection.yolo_detector import YOLODetector
    from src.indexing.frame_indexer import FrameIndexer

    print("  Loading CLIP ViT-B/32 (first run downloads ~600 MB) ...")
    indexer = FrameIndexer(
        faiss_index_path=str(FAISS_PATH),
        db_path=str(DB_PATH),
        nlist=10,        # small nlist for tiny demo dataset
        device="cpu",
    )

    print("  Loading YOLOv8n (CPU mode) ...")
    detector = YOLODetector(use_openvino=False, confidence_threshold=0.35)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    source_fps = cap.get(cv2.CAP_PROP_FPS) or VIDEO_FPS
    skip = max(1, int(source_fps * INDEX_EVERY_S))
    # Anchor demo timestamps starting from "2026-05-01 10:00:00"
    base_time = datetime(2026, 5, 1, 10, 0, 0)

    frame_idx, indexed = 0, 0
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
        print(f"  Frame #{indexed:02d}  id={row_id}  t={timestamp.strftime('%H:%M:%S')}  "
              f"objects=[{det_str}]")

    cap.release()
    indexer.save()
    ok(f"Indexed {indexed} frames — FAISS({indexer.total_indexed} vectors) + SQLite({indexer.get_frame_count()} rows)")
    return True


# ── Step 3: semantic search ────────────────────────────────────────────────────

def run_search(deps: dict):
    banner("STEP 3 — Semantic search (CLIP text-to-video)")

    if not all(deps.get(d) for d in ("torch", "open_clip", "faiss")):
        warn("Missing search deps — skipping.")
        return None

    if not FAISS_PATH.exists():
        warn("FAISS index not found — run step 2 first.")
        return None

    from src.search.retrospective_search import RetrospectiveSearch

    search = RetrospectiveSearch(
        faiss_index_path=str(FAISS_PATH),
        db_path=str(DB_PATH),
        clips_dir=str(CLIPS_DIR),
        top_k=3,
        device="cpu",
    )

    queries = [
        # The demo scenario from the project brief
        ("Quelqu'un a tagué mon mur le mois dernier",
         "graffiti person tagging wall spray paint"),
        ("Montre les personnes détectées ce matin",
         "person walking entering"),
        ("Y a-t-il eu des véhicules suspects ?",
         "car truck vehicle suspicious"),
    ]

    for user_q, clip_q in queries:
        print(f"\n  User query : \"{user_q}\"")
        print(f"  CLIP query : \"{clip_q}\"")
        results = search.search(clip_q, top_k=3)
        if not results:
            info("No results.")
        for i, e in enumerate(results, 1):
            ts = e.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            info(f"[{i}] id={e.id}  t={ts}  cam={e.camera_id}  "
                 f"score={e.score:.4f}  objects={e.detected_objects}")

    return search


# ── Step 4: ML intent classifier ──────────────────────────────────────────────

def run_ml_classifier():
    banner("STEP 4 — ML intent classifier (local, no LLM)")

    try:
        from src.ml.intent_classifier import IntentClassifierInference
        clf = IntentClassifierInference("src/ml/models")
    except FileNotFoundError:
        warn("Model not found. Run:  python src/ml/train_intent_classifier.py")
        return

    test_sentences = [
        "Quelqu'un a tagué mon mur le mois dernier",
        "Montre-moi les événements d'hier soir",
        "Envoie-moi le clip du premier résultat",
        "Fais-moi un résumé de la semaine",
        "Bonjour, je commence mon service",
        "Quelle est la résolution des caméras ?",
    ]

    header = f"  {'Sentence':<46}  {'Intent':<13}  {'Conf':>6}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for text in test_sentences:
        r = clf.predict(text)
        conf_bar = "#" * int(r["confidence"] * 10)
        print(f"  {text:<46}  {r['intent']:<13}  {r['confidence']:>5.1%}  {conf_bar}")

    ok("ML classifier running — no LLM API call needed.")


# ── Step 5: chatbot demo ───────────────────────────────────────────────────────

def run_chatbot(search_engine):
    banner("STEP 5 — Chatbot (hybrid ML + LLM)")

    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        warn("No LLM API key found in .env — skipping chatbot demo.")
        info("Set ANTHROPIC_API_KEY=... in your .env file to enable this step.")
        return

    provider = "anthropic" if os.getenv("ANTHROPIC_API_KEY") else "openai"

    # If search engine wasn't built (deps missing), use a stub
    if search_engine is None:
        from unittest.mock import MagicMock
        search_engine = MagicMock()
        search_engine.search.return_value = []
        search_engine.get_summary.return_value = {
            "total_events": 0, "object_counts": {}, "hourly_distribution": {}
        }

    from src.agent.chatbot_agent import ChatbotAgent, AgentContext
    agent = ChatbotAgent(
        search_engine=search_engine,
        llm_provider=provider,
        api_key=api_key,
        ml_model_dir="src/ml/models",
        ml_confidence_threshold=0.85,
    )
    ctx = AgentContext(session_id="demo")

    demo_turns = [
        "Bonjour, je suis l'agent de sécurité de nuit.",
        "Quelqu'un a tagué mon mur le mois dernier, t'as quelque chose ?",
        "Donne-moi le résumé d'activité de la semaine.",
    ]

    for msg in demo_turns:
        print(f"\n  User : {msg}")
        t0 = time.time()
        result = agent.chat(msg, ctx)
        elapsed = time.time() - t0

        print(f"  Bot  : {result.get('response', '')}")

        clf_info = result.get("classification_info", {})
        ml  = clf_info.get("ml")
        src = clf_info.get("source", "?")

        if ml:
            print(f"         [ML: {ml['intent']} {ml['confidence']:.0%} | "
                  f"LLM: {clf_info.get('llm_intent','—')} | "
                  f"source: {src} | {elapsed:.1f}s]")

        if result.get("events"):
            for e in result["events"][:2]:
                info(f"  -> Event #{e['id']}  {e['timestamp']}  score={e['score']:.3f}  "
                     f"clip={e.get('clip_path') or 'none'}")

        if result.get("summary"):
            s = result["summary"]
            info(f"  -> Total events: {s.get('total_events',0)}  "
                 f"Objects: {s.get('object_counts',{})}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 62)
    print("  Intelligent NVR Chatbot — End-to-End Demo")
    print("  Universite Mohammed V | Master IT | ML & Deep Learning")
    print("=" * 62)
    print(f"  Date  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Python: {sys.version.split()[0]}")

    deps   = check_dependencies()
    vid_ok = generate_video(deps)
    idx_ok = index_frames(deps) if vid_ok else False
    search = run_search(deps)   if idx_ok else None
    run_ml_classifier()
    run_chatbot(search)

    banner("DEMO COMPLETE", char="-")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
