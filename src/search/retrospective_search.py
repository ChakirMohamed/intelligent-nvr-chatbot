"""
Semantic retrospective search over indexed frames.
Text query → CLIP embedding → FAISS ANN → SQLite metadata filter → FFmpeg clips.
"""
import logging
import pickle
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
import torch

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 512


class Event:
    """Represents a matched video frame / event."""

    def __init__(self, row: dict, score: float = 0.0):
        self.id: int = row["id"]
        self.camera_id: str = row["camera_id"]
        self.video_path: str = row["video_path"]
        self.timestamp: datetime = datetime.fromisoformat(row["timestamp"])
        self.detected_objects: List[str] = [
            o.strip() for o in (row.get("detected_objects") or "").split(",") if o.strip()
        ]
        self.zone: str = row.get("zone", "default")
        self.score: float = score
        self.clip_path: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id":               self.id,
            "camera_id":        self.camera_id,
            "timestamp":        self.timestamp.isoformat(),
            "detected_objects": self.detected_objects,
            "zone":             self.zone,
            "score":            round(self.score, 4),
            "clip_path":        self.clip_path,
            "video_path":       self.video_path,
        }


class RetrospectiveSearch:
    def __init__(
        self,
        faiss_index_path: str,
        db_path: str,
        clips_dir: str = "./data/clips",
        top_k: int = 10,
        nprobe: int = 10,
        clip_duration: int = 15,
        dedup_seconds: float = 15.0,
        device: str = "cpu",
    ):
        self.faiss_index_path = Path(faiss_index_path)
        self.db_path = Path(db_path)
        self.clips_dir = Path(clips_dir)
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self.top_k = top_k
        self.clip_duration = clip_duration     # total clip length (seconds)
        self.dedup_seconds = dedup_seconds      # collapse same-camera hits within this gap
        self.device = device

        self._id_map: List[int] = []
        self._index: Optional[faiss.Index] = None
        self._clip_model = None
        self._tokenizer = None

        # Resolved lazily on first clip extraction (see _ffmpeg_exe).
        self._ffmpeg_checked = False
        self._ffmpeg_path: Optional[str] = None

        self._load_clip()
        self._load_index(nprobe)

    # ── CLIP ─────────────────────────────────────────────────────────────────

    def _load_clip(self) -> None:
        import open_clip
        logger.info("Loading CLIP for search …")
        model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        self._clip_model = model.to(self.device).eval()
        self._tokenizer = open_clip.get_tokenizer("ViT-B-32")
        logger.info("CLIP ready.")

    def _embed_text(self, text: str) -> np.ndarray:
        with torch.no_grad():
            tokens = self._tokenizer([text]).to(self.device)
            emb = self._clip_model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().float().numpy()  # (1, 512)

    # ── FAISS ─────────────────────────────────────────────────────────────────

    def _load_index(self, nprobe: int) -> None:
        if not self.faiss_index_path.exists():
            logger.warning("FAISS index not found: %s", self.faiss_index_path)
            return
        self._index = faiss.read_index(str(self.faiss_index_path))
        if isinstance(self._index, faiss.IndexIVFFlat):
            self._index.nprobe = nprobe

        map_path = self.faiss_index_path.with_suffix(".pkl")
        if map_path.exists():
            with open(map_path, "rb") as f:
                data = pickle.load(f)
            self._id_map = data.get("id_map", [])
        logger.info("Search index loaded: %d vectors.", self._index.ntotal)

    # ── SQLite helpers ────────────────────────────────────────────────────────

    def _fetch_by_ids(self, row_ids: List[int], score_map: Dict[int, float]) -> List[Event]:
        if not row_ids:
            return []
        ph = ",".join("?" * len(row_ids))
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"SELECT * FROM frames WHERE id IN ({ph})", row_ids
            ).fetchall()
        return [Event(dict(r), score=score_map.get(r["id"], 0.0)) for r in rows]

    # ── filtering ────────────────────────────────────────────────────────────

    @staticmethod
    def _filter(
        events: List[Event],
        camera_id: Optional[str],
        start_time: Optional[datetime],
        end_time: Optional[datetime],
        object_types: Optional[List[str]],
    ) -> List[Event]:
        out = []
        for e in events:
            if camera_id and e.camera_id != camera_id:
                continue
            if start_time and e.timestamp < start_time:
                continue
            if end_time and e.timestamp > end_time:
                continue
            if object_types and not any(o in e.detected_objects for o in object_types):
                continue
            out.append(e)
        return out

    def _dedup_events(self, events: List[Event]) -> List[Event]:
        """Collapse near-duplicate detections so the user doesn't get several
        result cards for the same moment (overlapping clips). Two events on the
        same camera within dedup_seconds are treated as one; the higher-scoring
        one is kept. Input must already be sorted by score (descending)."""
        if self.dedup_seconds <= 0:
            return events
        kept: List[Event] = []
        for e in events:
            duplicate = any(
                e.camera_id == k.camera_id
                and abs((e.timestamp - k.timestamp).total_seconds()) < self.dedup_seconds
                for k in kept
            )
            if not duplicate:
                kept.append(e)
        return kept

    # ── clip extraction ───────────────────────────────────────────────────────

    def _video_anchor_ts(self, video_path: str) -> Optional[datetime]:
        """Timestamp of the first frame indexed from this video. Frame
        timestamps are linear in frame position, so (event.ts - anchor) is the
        event's true offset (seconds) within the file — unlike wall-clock
        time-of-day, which is meaningless as a seek position."""
        with sqlite3.connect(str(self.db_path)) as conn:
            row = conn.execute(
                "SELECT MIN(timestamp) FROM frames WHERE video_path=?", (video_path,)
            ).fetchone()
        return datetime.fromisoformat(row[0]) if row and row[0] else None

    def _ffmpeg_exe(self) -> Optional[str]:
        """Path to an ffmpeg binary capable of H.264, or None.

        Prefers the pip-bundled imageio-ffmpeg binary (static build with
        libx264, no system install needed); falls back to a system ffmpeg on
        PATH. Result is cached so we only probe once."""
        if self._ffmpeg_checked:
            return self._ffmpeg_path
        self._ffmpeg_checked = True
        exe: Optional[str] = None
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            from shutil import which
            exe = which("ffmpeg")
        self._ffmpeg_path = exe
        if exe:
            logger.info("Clip extraction will use ffmpeg (H.264): %s", exe)
        else:
            logger.warning(
                "ffmpeg not found — clips fall back to OpenCV mp4v, which most "
                "browsers cannot play in <video>. Install imageio-ffmpeg to fix."
            )
        return exe

    def extract_clip(self, event: Event) -> Optional[str]:
        """Cut ±clip_window seconds around the event.

        Prefers ffmpeg → H.264 so the clip is playable in a browser <video>;
        falls back to OpenCV mp4v (MPEG-4 Part 2, download-only) if ffmpeg is
        unavailable."""
        src = Path(event.video_path)
        if not src.exists():
            logger.warning("Source video missing: %s", src)
            return None

        # Offset of this event within its source file (see _video_anchor_ts),
        # then take a clip_duration-second window centered on the event.
        anchor = self._video_anchor_ts(event.video_path)
        rel = (event.timestamp - anchor).total_seconds() if anchor else 0.0
        start = max(0.0, rel - self.clip_duration / 2)
        duration = float(self.clip_duration)

        out_name = f"clip_{event.id}_{event.timestamp.strftime('%Y%m%d_%H%M%S')}.mp4"
        out_path = self.clips_dir / out_name

        if self._extract_with_ffmpeg(src, start, duration, out_path):
            event.clip_path = str(out_path)
            return str(out_path)
        return self._extract_with_opencv(event, src, start, duration, out_path)

    def _extract_with_ffmpeg(
        self, src: Path, start: float, duration: float, out_path: Path
    ) -> bool:
        """Cut the window and re-encode to browser-standard H.264 (yuv420p,
        +faststart). Returns True on a non-empty output file."""
        exe = self._ffmpeg_exe()
        if not exe:
            return False
        import subprocess

        cmd = [
            exe, "-y", "-loglevel", "error",
            "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{duration:.3f}",
            "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", "-an", str(out_path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except Exception as exc:
            logger.error("ffmpeg clip extraction failed to run: %s", exc)
            return False
        if result.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
            logger.info("Extracted H.264 clip → %s", out_path)
            return True
        logger.warning(
            "ffmpeg clip extraction failed (rc=%s): %s",
            result.returncode, (result.stderr or "").strip()[:300],
        )
        return False

    def _extract_with_opencv(
        self, event: Event, src: Path, start: float, duration: float, out_path: Path
    ) -> Optional[str]:
        """Fallback: cut the window with OpenCV. Produces mp4v (MPEG-4 Part 2)
        which downloads fine but is not decodable by browser <video>."""
        import cv2

        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            logger.error("OpenCV could not open source video: %s", src)
            return None
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        start_frame = int(start * fps)
        end_frame   = int((start + duration) * fps)
        if total:
            end_frame = min(end_frame, total)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        written, f = 0, start_frame
        while f < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
            written += 1
            f += 1
        writer.release()
        cap.release()

        if written > 0 and out_path.exists() and out_path.stat().st_size > 0:
            event.clip_path = str(out_path)
            logger.info("Extracted clip (%d frames) → %s", written, out_path)
            return str(out_path)
        logger.error("Clip extraction produced no frames for event %s", event.id)
        return None

    # ── main search API ───────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        camera_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        object_types: Optional[List[str]] = None,
        extract_clips: bool = False,
    ) -> List[Event]:
        if self._index is None or self._index.ntotal == 0:
            logger.warning("FAISS index is empty — no results.")
            return []

        k = top_k or self.top_k
        query_emb = self._embed_text(query)

        # Over-fetch to compensate for post-filtering losses
        fetch_k = min(k * 5, self._index.ntotal)
        distances, indices = self._index.search(query_emb, fetch_k)

        score_map: Dict[int, float] = {}
        row_ids: List[int] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._id_map):
                continue
            rid = self._id_map[idx]
            row_ids.append(rid)
            score_map[rid] = float(dist)

        events = self._fetch_by_ids(row_ids, score_map)
        events = self._filter(events, camera_id, start_time, end_time, object_types)
        events.sort(key=lambda e: e.score, reverse=True)
        # Over-fetch then dedup so near-identical frames don't crowd out distinct
        # moments; only then trim to the requested k.
        events = self._dedup_events(events)
        events = events[:k]

        if extract_clips:
            for e in events:
                self.extract_clip(e)

        return events

    # ── summary ───────────────────────────────────────────────────────────────

    def get_summary(
        self,
        camera_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        conditions = ["1=1"]
        params: List[Any] = []
        if camera_id:
            conditions.append("camera_id=?");  params.append(camera_id)
        if start_time:
            conditions.append("timestamp>=?"); params.append(start_time.isoformat())
        if end_time:
            conditions.append("timestamp<=?"); params.append(end_time.isoformat())

        where = " AND ".join(conditions)
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"SELECT detected_objects, timestamp FROM frames WHERE {where}", params
            ).fetchall()

        obj_counts: Dict[str, int] = {}
        hourly: Dict[int, int] = {}
        for row in rows:
            for obj in (row["detected_objects"] or "").split(","):
                obj = obj.strip()
                if obj:
                    obj_counts[obj] = obj_counts.get(obj, 0) + 1
            h = datetime.fromisoformat(row["timestamp"]).hour
            hourly[h] = hourly.get(h, 0) + 1

        return {
            "total_events":       len(rows),
            "object_counts":      obj_counts,
            "hourly_distribution": hourly,
            "camera_id":          camera_id,
            "start_time":         start_time.isoformat() if start_time else None,
            "end_time":           end_time.isoformat() if end_time else None,
        }
