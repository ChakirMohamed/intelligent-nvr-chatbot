"""
FastAPI REST service for the Intelligent NVR system.
Run with:  python -m src.api.api   or   uvicorn src.api.api:app --reload
"""
import asyncio
import logging
import os
import queue
import sqlite3
import threading
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Intelligent NVR API",
    description="AI-powered NVR with semantic video search and chatbot interface",
    version="1.0.0",
)
# CORS — allow the Vite dev frontend (http://localhost:5173) by default.
# Override with a comma-separated CORS_ORIGINS env var if needed.
_cors_origins = [
    o.strip()
    for o in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── singletons ────────────────────────────────────────────────────────────────
_search: Any = None
_chatbot: Any = None
_sessions: Dict[str, Any] = {}

# ── live ingestion state ────────────────────────────────────────────────────────
# One RTSPReader per configured camera (key = camera_id). Each reader records
# MP4 segments + exposes the latest frame for the MJPEG live endpoint, and pushes
# sampled motion frames onto _index_queue for the background indexing worker.
_readers: Dict[str, Any] = {}
_indexer: Any = None
_detector: Any = None
_index_queue: "queue.Queue" = queue.Queue(maxsize=128)
_indexer_thread: Optional[threading.Thread] = None
_ingest_running = False


def _ensure_db(db_path: str) -> None:
    """Create the frames table if it doesn't exist yet (first run, before any ingestion)."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS frames (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id        TEXT    NOT NULL,
                video_path       TEXT    NOT NULL,
                timestamp        TEXT    NOT NULL,
                faiss_id         INTEGER,
                detected_objects TEXT    DEFAULT '',
                zone             TEXT    DEFAULT 'default'
            );
            CREATE INDEX IF NOT EXISTS idx_cam  ON frames(camera_id);
            CREATE INDEX IF NOT EXISTS idx_ts   ON frames(timestamp);
            CREATE INDEX IF NOT EXISTS idx_objs ON frames(detected_objects);
        """)


# ── live ingestion ──────────────────────────────────────────────────────────────

def _mask_url(url: str) -> str:
    """Hide the password when logging an RTSP URL."""
    if "@" in url and "//" in url:
        head, tail = url.split("//", 1)
        if "@" in tail:
            creds, host = tail.split("@", 1)
            user = creds.split(":", 1)[0]
            return f"{head}//{user}:****@{host}"
    return url


def _env_flag(name: str, default: str = "true") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def _make_on_motion(cam_id: str):
    """Build the reader callback that hands sampled motion frames to the worker."""
    def _cb(camera_id: str, frame, ts: datetime) -> None:
        reader = _readers.get(cam_id)
        video_path = reader.current_segment if reader else None
        try:
            _index_queue.put_nowait((camera_id, frame, ts, video_path))
        except queue.Full:
            pass  # worker is behind — drop this frame; recording + live unaffected
    return _cb


def _indexer_worker() -> None:
    """Consume motion frames: YOLO detect → CLIP embed → FAISS/SQLite. Single
    thread, so FAISS is never touched concurrently.

    Two filters keep the event log meaningful instead of one row per frame:
      • Fix 2 — INDEX_REQUIRE_OBJECT: ignore frames where YOLO found nothing.
      • Fix 3 — EVENT_GAP_SECONDS: collapse sustained activity into a single
        event; a new event only starts when the detected objects change or
        after a quiet gap.
    """
    require_object = _env_flag("INDEX_REQUIRE_OBJECT", "true")
    event_gap = float(os.getenv("EVENT_GAP_SECONDS", "10"))
    # per-camera grouping state: camera_id -> {"last_ts": datetime, "objects": frozenset}
    state: Dict[str, Dict[str, Any]] = {}
    saved = 0
    logger.info("Indexer worker started (require_object=%s, event_gap=%.0fs).", require_object, event_gap)
    while _ingest_running:
        try:
            camera_id, frame, ts, video_path = _index_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        try:
            detections = _detector.detect(frame) if _detector else []
            objects = sorted({d.class_name for d in detections})
            obj_set = frozenset(objects)

            # Fix 2 — drop frames with no target object.
            if require_object and not obj_set:
                continue

            # Fix 3 — a new event only when objects change or after a quiet gap.
            prev = state.get(camera_id)
            is_new_event = (
                prev is None
                or obj_set != prev["objects"]
                or (ts - prev["last_ts"]).total_seconds() >= event_gap
            )
            # Advance the "last seen" clock on every qualifying frame so sustained
            # activity stays one event instead of restarting each gap window.
            state[camera_id] = {"last_ts": ts, "objects": obj_set}
            if not is_new_event:
                continue

            _indexer.index_frame(
                frame=frame,
                camera_id=camera_id,
                video_path=video_path or "",
                timestamp=ts,
                detected_objects=objects,
                zone="live",
            )
            saved += 1
            if saved % 20 == 0:
                _indexer.save()
        except Exception as exc:
            logger.error("Indexer worker error: %s", exc)
    if _indexer is not None:
        try:
            _indexer.save()
        except Exception:
            pass
    logger.info("Indexer worker stopped.")


def _start_ingestion(db_path: str) -> None:
    global _indexer, _detector, _indexer_thread, _ingest_running

    if not _env_flag("INGEST_ON_STARTUP"):
        logger.info("Ingestion disabled (INGEST_ON_STARTUP=false).")
        return

    urls = [u.strip() for u in os.getenv("RTSP_URLS", "").split(",") if u.strip()]
    if not urls:
        logger.info("No RTSP_URLS configured — ingestion skipped.")
        return

    from src.ingestion.rtsp_reader import RTSPReader

    # Optional YOLO + CLIP indexing. If it fails to load (missing model, no GPU,
    # etc.) recording + live view still run — indexing is just disabled.
    if _env_flag("INDEX_ON_STARTUP"):
        try:
            from src.indexing.frame_indexer import FrameIndexer
            from src.detection.yolo_detector import YOLODetector
            _indexer = FrameIndexer(
                faiss_index_path=os.getenv("FAISS_INDEX_PATH", "./data/faiss_index/index.faiss"),
                db_path=db_path,
                nlist=int(os.getenv("FAISS_NLIST", "100")),
                nprobe=int(os.getenv("FAISS_NPROBE", "10")),
                device="cpu",
            )
            ov_device = os.getenv("OPENVINO_DEVICE", "GPU")
            _detector = YOLODetector(
                use_openvino=ov_device.upper() != "CPU",
                openvino_device=ov_device,
                confidence_threshold=float(os.getenv("DETECTION_CONFIDENCE", "0.4")),
            )
            logger.info("Live indexing enabled (YOLO + CLIP).")
        except Exception as exc:
            logger.warning("Live indexing disabled (%s). Recording + live view still active.", exc)
            _indexer = None
            _detector = None

    storage          = os.getenv("STORAGE_PATH", "./data/recordings")
    sampling_fps     = float(os.getenv("SAMPLING_FPS", "1"))
    motion_threshold = int(os.getenv("MOTION_THRESHOLD", "500"))
    segment_duration = int(os.getenv("SEGMENT_DURATION", "300"))

    for i, url in enumerate(urls, start=1):
        cam_id = f"cam_{i}"
        reader = RTSPReader(
            camera_id=cam_id,
            rtsp_url=url,
            storage_path=storage,
            sampling_fps=sampling_fps,
            motion_threshold=motion_threshold,
            segment_duration=segment_duration,
            on_motion_frame=_make_on_motion(cam_id) if _indexer else None,
        )
        reader.start()
        _readers[cam_id] = reader
        logger.info("Started %s → %s", cam_id, _mask_url(url))

    _ingest_running = True
    if _indexer is not None:
        _indexer_thread = threading.Thread(target=_indexer_worker, daemon=True, name="indexer-worker")
        _indexer_thread.start()


def _stop_ingestion() -> None:
    global _ingest_running
    _ingest_running = False
    for reader in _readers.values():
        try:
            reader.stop(timeout=5)
        except Exception as exc:
            logger.error("Error stopping reader: %s", exc)


@app.on_event("startup")
async def _startup() -> None:
    global _search, _chatbot

    from src.search.retrospective_search import RetrospectiveSearch
    from src.agent.chatbot_agent import ChatbotAgent

    db_path = os.getenv("DB_PATH", "./data/metadata.db")
    _ensure_db(db_path)

    _search = RetrospectiveSearch(
        faiss_index_path=os.getenv("FAISS_INDEX_PATH", "./data/faiss_index/index.faiss"),
        db_path=db_path,
        clips_dir=os.getenv("CLIPS_DIR", "./data/clips"),
        top_k=int(os.getenv("TOP_K_RESULTS", "10")),
        nprobe=int(os.getenv("FAISS_NPROBE", "10")),
        clip_duration=int(os.getenv("CLIP_DURATION_SECONDS", "15")),
        dedup_seconds=float(os.getenv("EVENT_DEDUP_SECONDS", "15")),
        device="cpu",
    )

    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")
    _chatbot = ChatbotAgent(
        search_engine=_search,
        llm_provider=os.getenv("LLM_PROVIDER", "anthropic"),
        llm_model=os.getenv("LLM_MODEL"),
        api_key=api_key,
    )

    # Start live RTSP readers (recording + live view + indexing) from RTSP_URLS.
    _start_ingestion(db_path)
    logger.info("NVR API ready.")


@app.on_event("shutdown")
async def _shutdown() -> None:
    _stop_ingestion()


# ── helpers ───────────────────────────────────────────────────────────────────

def _require_search():
    if _search is None:
        raise HTTPException(503, detail="Search engine not ready")
    return _search


def _require_chatbot():
    if _chatbot is None:
        raise HTTPException(503, detail="Chatbot not ready")
    return _chatbot


async def _in_thread(fn, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(fn, *args, **kwargs))


# ── request models ────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    camera_id: Optional[str] = None
    start_time: Optional[str] = None   # ISO-8601
    end_time: Optional[str] = None
    object_types: Optional[List[str]] = None
    extract_clips: bool = False


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["system"])
async def health():
    se = _require_search()
    indexed = se._index.ntotal if se._index else 0
    return {"status": "ok", "indexed_frames": indexed, "timestamp": datetime.now().isoformat()}


@app.post("/chat", tags=["chatbot"])
async def chat(req: ChatRequest):
    from src.agent.chatbot_agent import AgentContext
    bot = _require_chatbot()

    if req.session_id not in _sessions:
        _sessions[req.session_id] = AgentContext(session_id=req.session_id)
    ctx = _sessions[req.session_id]

    # The agent's result dict is returned verbatim. For a clip_request intent it
    # also carries clip_url / event_id / timestamp / detected_objects, which the
    # frontend uses to render the "Voir le clip" button and video modal.
    result = await _in_thread(bot.chat, req.message, ctx)
    return result


@app.post("/search", tags=["search"])
async def search(req: SearchRequest):
    se = _require_search()
    start = datetime.fromisoformat(req.start_time) if req.start_time else None
    end   = datetime.fromisoformat(req.end_time)   if req.end_time   else None

    events = await _in_thread(
        se.search,
        query=req.query,
        top_k=req.top_k,
        camera_id=req.camera_id,
        start_time=start,
        end_time=end,
        object_types=req.object_types,
        extract_clips=req.extract_clips,
    )
    return {"query": req.query, "count": len(events), "results": [e.to_dict() for e in events]}


@app.get("/clip/{event_id}", tags=["clips"])
async def get_clip(event_id: int):
    se = _require_search()
    clips_dir = Path(os.getenv("CLIPS_DIR", "./data/clips"))
    matches = list(clips_dir.glob(f"clip_{event_id}_*.mp4"))

    if not matches:
        # No pre-extracted clip — build it on demand from the event's source video.
        from src.search.retrospective_search import Event
        with sqlite3.connect(str(se.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM frames WHERE id=?", (event_id,)).fetchone()
        if row is None:
            raise HTTPException(404, detail=f"No event with id {event_id}")
        path = await _in_thread(se.extract_clip, Event(dict(row)))
        if not path:
            raise HTTPException(404, detail=f"Could not extract clip for event {event_id}")
        matches = [Path(path)]

    return FileResponse(str(matches[0]), media_type="video/mp4", filename=matches[0].name)


@app.get("/events", tags=["events"])
async def list_events(
    camera_id: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    object_type: Optional[str] = None,
    limit: int = Query(50, le=500),
):
    se = _require_search()
    conditions = ["1=1"]
    params: list = []

    if camera_id:
        conditions.append("camera_id=?");        params.append(camera_id)
    if start_time:
        conditions.append("timestamp>=?");        params.append(start_time)
    if end_time:
        conditions.append("timestamp<=?");        params.append(end_time)
    if object_type:
        conditions.append("detected_objects LIKE ?"); params.append(f"%{object_type}%")

    params.append(limit)
    with sqlite3.connect(str(se.db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"SELECT * FROM frames WHERE {' AND '.join(conditions)} ORDER BY timestamp DESC LIMIT ?",
            params,
        ).fetchall()

    # Normalize rows through Event so detected_objects is a list (not the raw
    # comma-joined TEXT column) and the payload matches the search endpoint.
    from src.search.retrospective_search import Event
    events = [Event(dict(r)).to_dict() for r in rows]
    return {"count": len(events), "events": events}


@app.get("/summary", tags=["events"])
async def summary(
    camera_id: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
):
    se = _require_search()
    s = datetime.fromisoformat(start_time) if start_time else None
    e = datetime.fromisoformat(end_time)   if end_time   else None
    return await _in_thread(se.get_summary, camera_id=camera_id, start_time=s, end_time=e)


@app.get("/cameras", tags=["system"])
async def list_cameras():
    se = _require_search()
    with sqlite3.connect(str(se.db_path)) as conn:
        rows = conn.execute(
            "SELECT camera_id, COUNT(*) AS frames, MIN(timestamp) AS first_seen, MAX(timestamp) AS last_seen "
            "FROM frames GROUP BY camera_id ORDER BY camera_id"
        ).fetchall()
    return {"cameras": [{"camera_id": r[0], "frame_count": r[1], "first_seen": r[2], "last_seen": r[3]} for r in rows]}


# ── live view (MJPEG) ─────────────────────────────────────────────────────────
# NOTE: /live/cameras is declared BEFORE /live/{camera_id} so the static path
# isn't captured by the path parameter.

@app.get("/live/cameras", tags=["live"])
async def live_cameras():
    """List cameras with an active live reader and their connection state."""
    return {
        "cameras": [
            {"camera_id": cid, "connected": reader.is_connected}
            for cid, reader in _readers.items()
        ]
    }


@app.get("/live/{camera_id}", tags=["live"])
async def live_stream(camera_id: str):
    """MJPEG live stream (multipart/x-mixed-replace) — embed with <img src=...>."""
    reader = _readers.get(camera_id)
    if reader is None:
        raise HTTPException(404, detail=f"No live camera '{camera_id}'")

    fps = max(1.0, float(os.getenv("LIVE_FPS", "10")))
    boundary = "frame"

    async def gen():
        interval = 1.0 / fps
        while True:
            jpeg = await _in_thread(reader.get_latest_jpeg)
            if jpeg:
                yield (
                    b"--" + boundary.encode() + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                    + jpeg + b"\r\n"
                )
            await asyncio.sleep(interval)

    return StreamingResponse(
        gen(),
        media_type=f"multipart/x-mixed-replace; boundary={boundary}",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.api:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=True,
    )
