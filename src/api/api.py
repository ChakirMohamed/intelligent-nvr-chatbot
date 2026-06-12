"""
FastAPI REST service for the Intelligent NVR system.
Run with:  python -m src.api.api   or   uvicorn src.api.api:app --reload
"""
import asyncio
import logging
import os
import sqlite3
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Intelligent NVR API",
    description="AI-powered NVR with semantic video search and chatbot interface",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── singletons ────────────────────────────────────────────────────────────────
_search: Any = None
_chatbot: Any = None
_sessions: Dict[str, Any] = {}


@app.on_event("startup")
async def _startup() -> None:
    global _search, _chatbot

    from src.search.retrospective_search import RetrospectiveSearch
    from src.agent.chatbot_agent import ChatbotAgent

    _search = RetrospectiveSearch(
        faiss_index_path=os.getenv("FAISS_INDEX_PATH", "./data/faiss_index/index.faiss"),
        db_path=os.getenv("DB_PATH", "./data/metadata.db"),
        clips_dir=os.getenv("CLIPS_DIR", "./data/clips"),
        top_k=int(os.getenv("TOP_K_RESULTS", "10")),
        nprobe=int(os.getenv("FAISS_NPROBE", "10")),
        clip_window=int(os.getenv("CLIP_EXTRACTION_WINDOW", "30")),
        device="cpu",
    )

    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")
    _chatbot = ChatbotAgent(
        search_engine=_search,
        llm_provider=os.getenv("LLM_PROVIDER", "anthropic"),
        llm_model=os.getenv("LLM_MODEL"),
        api_key=api_key,
    )
    logger.info("NVR API ready.")


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
    clips_dir = Path(os.getenv("CLIPS_DIR", "./data/clips"))
    matches = list(clips_dir.glob(f"clip_{event_id}_*.mp4"))
    if not matches:
        raise HTTPException(404, detail=f"No clip found for event {event_id}")
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

    return {"count": len(rows), "events": [dict(r) for r in rows]}


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


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.api:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=True,
    )
