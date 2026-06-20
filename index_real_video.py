"""
Full indexing of a REAL surveillance video (e.g. VIRAT) for the demo.

Unlike demo.py (which samples 1 frame every N seconds of a synthetic clip),
this reads EVERY frame of the source video and keeps the ones that contain
motion (MOG2 background subtraction), so nothing meaningful is missed while
long static stretches are skipped.

For each kept frame:
    YOLO detection (fine-tuned VisDrone model) → CLIP embedding
    → FAISS vector + SQLite metadata row (camera_id="virat_cam").

The index is written to the SAME paths the API reads (DB_PATH / FAISS_INDEX_PATH
from .env), so the running API serves exactly this data. The existing index is
cleared first so we start fresh with only the VIRAT footage.

Usage
-----
    # replace the index with one video (default: VIRAT, fine-tuned model)
    python index_real_video.py [path/to/video.mp4]

    # add a second camera without wiping the existing index, using COCO weights
    python index_real_video.py data/demo/youtube_demo.mp4 \
        --no-reset --camera youtube_cam --model coco

Run from the project root (paths in .env are relative to it).
"""
import argparse
import logging
import os
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
for noisy in ("PIL", "timm", "open_clip", "ultralytics", "faiss"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# ── config (mirrors src/api/api.py defaults so the API serves this data) ────────
DEFAULT_VIDEO   = "data/demo/VIRAT_S_010204_05_000856_000890.mp4"
DEFAULT_CAMERA  = "virat_cam"
DEFAULT_ZONE    = "surveillance"

DB_PATH         = os.getenv("DB_PATH", "./data/metadata.db")
FAISS_PATH      = os.getenv("FAISS_INDEX_PATH", "./data/faiss_index/index.faiss")
FAISS_NLIST     = int(os.getenv("FAISS_NLIST", "100"))
FAISS_NPROBE    = int(os.getenv("FAISS_NPROBE", "10"))
MOTION_THRESHOLD = int(os.getenv("MOTION_THRESHOLD", "500"))
DETECTION_CONF  = float(os.getenv("DETECTION_CONFIDENCE", "0.4"))


def banner(title: str, char: str = "=") -> None:
    width = 64
    print(f"\n{char * width}\n  {title}\n{char * width}")


def reset_index(db_path: Path, faiss_path: Path) -> None:
    """Delete the previous SQLite DB + FAISS index so we start fresh."""
    removed = []
    for p in (db_path, faiss_path, faiss_path.with_suffix(".pkl")):
        if p.exists():
            p.unlink()
            removed.append(p.name)
    print(f"  Cleared previous index: {removed or 'nothing to clear'}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full indexing of a real video into the NVR index.")
    p.add_argument("video", nargs="?", default=DEFAULT_VIDEO, help="path to the video file")
    p.add_argument("--camera", default=DEFAULT_CAMERA, help="camera_id for these frames (e.g. youtube_cam)")
    p.add_argument("--zone", default=DEFAULT_ZONE, help="zone label stored with each frame")
    p.add_argument("--no-reset", dest="reset", action="store_false",
                   help="append to the existing index instead of wiping it (add as another camera)")
    p.add_argument("--model", choices=["visdrone", "coco"], default="visdrone",
                   help="'visdrone' = fine-tuned aerial .pt (default); 'coco' = stock yolov8n.pt (better on ground footage)")
    p.add_argument("--conf", type=float, default=DETECTION_CONF, help="YOLO confidence threshold")
    p.add_argument("--fps", type=float, default=0.0,
                   help="index at most this many frames/sec (0 = every frame). "
                        "Use e.g. 2 on long videos to keep CPU indexing fast.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    banner("Full indexing of a real video")

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"  [ERROR] Video not found: {video_path}")
        print( "          Pass a path:  python index_real_video.py path/to/video.mp4")
        return 1

    import cv2
    from src.detection.yolo_detector import YOLODetector
    from src.indexing.frame_indexer import FrameIndexer

    db_path    = Path(DB_PATH)
    faiss_path = Path(FAISS_PATH)
    faiss_path.parent.mkdir(parents=True, exist_ok=True)

    if args.reset:
        reset_index(db_path, faiss_path)   # start fresh
    else:
        print("  Append mode (--no-reset): keeping the existing index, adding to it.")

    # 'coco' → stock yolov8n.pt (good for ground footage); 'visdrone' → fine-tuned .pt.
    model_path = "yolov8n.pt" if args.model == "coco" else None
    print(f"  Video      : {video_path}")
    print(f"  Camera id  : {args.camera}   zone: {args.zone}   model: {args.model}")
    print(f"  DB_PATH    : {db_path}")
    print(f"  FAISS_PATH : {faiss_path}")
    print( "  Loading YOLO + CLIP ViT-B/32 ...")

    # use_openvino=False → loads a .pt directly (fine-tuned, or model_path for COCO).
    detector = YOLODetector(use_openvino=False, model_path=model_path, confidence_threshold=args.conf)
    indexer  = FrameIndexer(
        faiss_index_path=str(faiss_path),
        db_path=str(db_path),
        nlist=FAISS_NLIST,
        nprobe=FAISS_NPROBE,
        device="cpu",
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [ERROR] OpenCV could not open: {video_path}")
        return 1

    source_fps  = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    duration_s  = total_frames / source_fps if total_frames else 0
    # Anchor timestamps so the clip ends ~now → "recent" events for the chatbot.
    base_time = datetime.now().replace(microsecond=0) - timedelta(seconds=duration_s)

    # Sample at most --fps frames/sec (0 = every frame) to bound CPU work.
    stride = max(1, round(source_fps / args.fps)) if args.fps and args.fps > 0 else 1

    # MOG2 motion filter — same params as the live RTSP reader for consistency.
    bg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

    sampling = "every frame" if stride == 1 else f"1 frame every {stride} (~{args.fps:g} fps)"
    print(
        f"  Source     : {total_frames} frames @ {source_fps:.1f} fps "
        f"(~{duration_s:.0f}s).  Sampling: {sampling}.  MOTION_THRESHOLD={MOTION_THRESHOLD}\n"
    )

    frame_idx = 0
    indexed   = 0
    skipped_static = 0
    obj_counter: Counter = Counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Frame sampling — only run YOLO+CLIP on every `stride`-th frame.
        if stride > 1 and frame_idx % stride != 0:
            continue

        # MOG2 motion gate — skip near-static frames.
        mask = bg.apply(frame)
        if int(cv2.countNonZero(mask)) <= MOTION_THRESHOLD:
            skipped_static += 1
            if frame_idx % 250 == 0:
                print(f"  ... scanned {frame_idx}/{total_frames}  (indexed {indexed}, skipped {skipped_static} static)")
            continue

        detections = detector.detect(frame)
        objects    = sorted({d.class_name for d in detections})
        obj_counter.update(objects)
        timestamp  = base_time + timedelta(seconds=frame_idx / source_fps)

        indexer.index_frame(
            frame=frame,
            camera_id=args.camera,
            video_path=str(video_path),
            timestamp=timestamp,
            detected_objects=objects,
            zone=args.zone,
        )
        indexed += 1
        det_str = ", ".join(objects) if objects else "none"
        print(f"  Frame {frame_idx}/{total_frames} - objects: [{det_str}]")

        if indexed % 25 == 0:
            indexer.save()

    cap.release()
    indexer.train_and_flush()   # upgrade Flat→IVF if enough vectors, then persist
    indexer.save()

    banner("INDEXING COMPLETE", char="-")
    print(f"  Frames scanned     : {frame_idx}")
    print(f"  Frames indexed     : {indexed}")
    print(f"  Static frames skipped : {skipped_static}")
    print(f"  Object counts      : {dict(obj_counter) or 'none detected'}")
    print(f"  FAISS vectors      : {indexer.total_indexed}")
    print(f"  SQLite rows        : {indexer.get_frame_count()}")
    print(f"  Camera id          : {args.camera}")
    print(f"  Written to         : {db_path}  +  {faiss_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
