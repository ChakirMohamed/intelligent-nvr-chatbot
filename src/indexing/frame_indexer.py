"""
CLIP embedding + FAISS IVF index + SQLite metadata store.

Strategy for IVF training:
  Vectors are buffered in RAM until we have >= nlist*40 samples.
  call train_and_flush() once the buffer is full, or call save() to persist
  the current flat buffer index at any time.
"""
import logging
import pickle
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import faiss
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 512   # CLIP ViT-B/32 output


class FrameIndexer:
    def __init__(
        self,
        faiss_index_path: str,
        db_path: str,
        nlist: int = 100,
        nprobe: int = 10,
        device: str = "cpu",
    ):
        self.faiss_index_path = Path(faiss_index_path)
        self.db_path = Path(db_path)
        self.nlist = nlist
        self.nprobe = nprobe
        self.device = device

        # ID map: faiss internal index → SQLite row id
        self._id_map: List[int] = []
        # Buffer for pre-training phase
        self._buffer_embs: List[np.ndarray] = []
        self._buffer_ids: List[int] = []

        self._clip_model = None
        self._clip_preprocess = None
        self._index: Optional[faiss.Index] = None

        self._init_db()
        self._load_or_create_index()
        self._load_clip()

    # ── CLIP ─────────────────────────────────────────────────────────────────

    def _load_clip(self) -> None:
        import open_clip
        logger.info("Loading CLIP ViT-B/32 …")
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        model = model.to(self.device).eval()
        self._clip_model = model
        self._clip_preprocess = preprocess
        logger.info("CLIP ready.")

    def _embed_frame(self, frame: np.ndarray) -> np.ndarray:
        img = Image.fromarray(frame[..., ::-1])   # BGR → RGB
        with torch.no_grad():
            tensor = self._clip_preprocess(img).unsqueeze(0).to(self.device)
            emb = self._clip_model.encode_image(tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().float().numpy()  # shape (1, 512)

    # ── SQLite ────────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.db_path)) as conn:
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

    def _insert_frame_row(
        self,
        camera_id: str,
        video_path: str,
        timestamp: datetime,
        detected_objects: List[str],
        zone: str,
    ) -> int:
        with sqlite3.connect(str(self.db_path)) as conn:
            cur = conn.execute(
                "INSERT INTO frames (camera_id, video_path, timestamp, detected_objects, zone) "
                "VALUES (?, ?, ?, ?, ?)",
                (camera_id, video_path, timestamp.isoformat(), ",".join(detected_objects), zone),
            )
            return cur.lastrowid

    def _set_faiss_id(self, row_id: int, faiss_id: int) -> None:
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("UPDATE frames SET faiss_id=? WHERE id=?", (faiss_id, row_id))

    # ── FAISS ─────────────────────────────────────────────────────────────────

    def _load_or_create_index(self) -> None:
        map_path = self.faiss_index_path.with_suffix(".pkl")
        if self.faiss_index_path.exists() and map_path.exists():
            logger.info("Loading existing FAISS index from %s", self.faiss_index_path)
            self._index = faiss.read_index(str(self.faiss_index_path))
            with open(map_path, "rb") as f:
                data = pickle.load(f)
            self._id_map = data.get("id_map", [])
            self._buffer_ids = data.get("buffer_ids", [])
            self._buffer_embs = data.get("buffer_embs", [])
            logger.info("Index loaded: %d vectors, %d buffered.", len(self._id_map), len(self._buffer_ids))
        else:
            # Start with a flat index; upgrade to IVF once enough data arrives
            self._index = faiss.IndexFlatIP(EMBEDDING_DIM)
            logger.info("New flat FAISS index created (will upgrade to IVF at %d vectors).", self.nlist * 40)

    def _persist(self) -> None:
        self.faiss_index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.faiss_index_path))
        with open(self.faiss_index_path.with_suffix(".pkl"), "wb") as f:
            pickle.dump({
                "id_map":      self._id_map,
                "buffer_ids":  self._buffer_ids,
                "buffer_embs": self._buffer_embs,
            }, f)

    def _maybe_upgrade_to_ivf(self) -> None:
        """Upgrade the flat index to IVFFlat once we have enough training vectors."""
        threshold = self.nlist * 40
        total_in_flat = len(self._id_map) + len(self._buffer_ids)
        if total_in_flat < threshold:
            return
        if isinstance(self._index, faiss.IndexIVFFlat):
            return  # already IVF

        logger.info("Upgrading flat index → IVFFlat (nlist=%d) with %d vectors …", self.nlist, total_in_flat)

        # Gather all embeddings currently in the flat index
        flat_embs = faiss.rev_swig_ptr(self._index.get_xb(), self._index.ntotal * EMBEDDING_DIM)
        flat_embs = flat_embs.reshape(self._index.ntotal, EMBEDDING_DIM).copy()

        # Merge with buffer
        all_ids: List[int] = self._id_map + self._buffer_ids
        buf_matrix = np.vstack(self._buffer_embs).astype(np.float32) if self._buffer_embs else np.empty((0, EMBEDDING_DIM), np.float32)
        all_embs = np.vstack([flat_embs, buf_matrix]) if flat_embs.shape[0] else buf_matrix

        quantizer = faiss.IndexFlatIP(EMBEDDING_DIM)
        ivf = faiss.IndexIVFFlat(quantizer, EMBEDDING_DIM, self.nlist, faiss.METRIC_INNER_PRODUCT)
        ivf.train(all_embs)
        ivf.add(all_embs)
        ivf.nprobe = self.nprobe

        self._index = ivf
        self._id_map = all_ids
        self._buffer_ids = []
        self._buffer_embs = []

        # Refresh faiss_id column in SQLite
        with sqlite3.connect(str(self.db_path)) as conn:
            for fid, rid in enumerate(self._id_map):
                conn.execute("UPDATE frames SET faiss_id=? WHERE id=?", (fid, rid))

        self._persist()
        logger.info("Upgrade complete. IVFFlat index has %d vectors.", self._index.ntotal)

    # ── public API ────────────────────────────────────────────────────────────

    def index_frame(
        self,
        frame: np.ndarray,
        camera_id: str,
        video_path: str,
        timestamp: datetime,
        detected_objects: List[str],
        zone: str = "default",
    ) -> int:
        """Embed and index one frame. Returns the SQLite row id."""
        emb = self._embed_frame(frame)   # (1, 512)

        row_id = self._insert_frame_row(camera_id, video_path, timestamp, detected_objects, zone)

        if isinstance(self._index, faiss.IndexIVFFlat) and self._index.is_trained:
            faiss_id = self._index.ntotal
            self._index.add(emb)
            self._id_map.append(row_id)
            self._set_faiss_id(row_id, faiss_id)
        else:
            # Flat index phase — add directly
            faiss_id = self._index.ntotal
            self._index.add(emb)
            self._id_map.append(row_id)
            self._set_faiss_id(row_id, faiss_id)
            self._maybe_upgrade_to_ivf()

        return row_id

    def save(self) -> None:
        self._persist()
        logger.info("Index saved (%d vectors).", self._index.ntotal)

    def train_and_flush(self) -> None:
        """Force IVF upgrade regardless of threshold (useful at pipeline shutdown)."""
        self._maybe_upgrade_to_ivf()

    def get_frame_count(self) -> int:
        with sqlite3.connect(str(self.db_path)) as conn:
            return conn.execute("SELECT COUNT(*) FROM frames").fetchone()[0]

    @property
    def total_indexed(self) -> int:
        return self._index.ntotal if self._index else 0
