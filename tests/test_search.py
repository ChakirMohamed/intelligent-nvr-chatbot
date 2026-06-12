"""
Tests for RetrospectiveSearch — uses a temporary in-memory SQLite DB and a
mocked FAISS flat index so no CLIP model download is needed during tests.
"""
import sys
import os
import sqlite3
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_engine(tmp_dir: str):
    """Build a RetrospectiveSearch with mocked CLIP and a pre-populated flat index."""
    import faiss
    from src.search.retrospective_search import RetrospectiveSearch, EMBEDDING_DIM

    faiss_path = os.path.join(tmp_dir, "idx.faiss")
    db_path    = os.path.join(tmp_dir, "meta.db")
    clips_dir  = os.path.join(tmp_dir, "clips")

    # Populate SQLite with synthetic rows
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE frames (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id TEXT, video_path TEXT, timestamp TEXT,
                faiss_id INTEGER, detected_objects TEXT, zone TEXT DEFAULT 'default'
            )
        """)
        rows = [
            ("cam1", "/data/rec/cam1/2025-05-01/10-00-00.mp4", "2025-05-01T10:05:00", 0, "person"),
            ("cam1", "/data/rec/cam1/2025-05-01/10-00-00.mp4", "2025-05-01T10:10:00", 1, "car"),
            ("cam2", "/data/rec/cam2/2025-05-01/11-00-00.mp4", "2025-05-01T11:20:00", 2, "person,car"),
        ]
        conn.executemany(
            "INSERT INTO frames (camera_id, video_path, timestamp, faiss_id, detected_objects) VALUES (?,?,?,?,?)",
            rows,
        )

    # Build a flat FAISS index with random embeddings
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    vecs = np.random.randn(3, EMBEDDING_DIM).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    index.add(vecs)
    faiss.write_index(index, faiss_path)

    import pickle
    with open(faiss_path.replace(".faiss", ".pkl"), "wb") as f:
        pickle.dump({"id_map": [1, 2, 3], "buffer_ids": [], "buffer_embs": []}, f)

    # Patch CLIP so no model download happens
    with patch("src.search.retrospective_search.RetrospectiveSearch._load_clip"):
        engine = RetrospectiveSearch(
            faiss_index_path=faiss_path,
            db_path=db_path,
            clips_dir=clips_dir,
            top_k=5,
            device="cpu",
        )

    # Replace embed_text with a function that returns a random unit vector
    def _fake_embed(text: str) -> np.ndarray:
        v = np.random.randn(1, EMBEDDING_DIM).astype(np.float32)
        v /= np.linalg.norm(v)
        return v

    engine._embed_text = _fake_embed
    return engine


class TestSearch(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.engine = _make_engine(self._tmp)

    # ── search ────────────────────────────────────────────────────────────────

    def test_search_returns_list(self):
        results = self.engine.search("person walking")
        self.assertIsInstance(results, list)

    def test_search_top_k_respected(self):
        results = self.engine.search("car", top_k=2)
        self.assertLessEqual(len(results), 2)

    def test_event_has_required_fields(self):
        results = self.engine.search("anything")
        for e in results:
            d = e.to_dict()
            for key in ("id", "camera_id", "timestamp", "detected_objects", "score"):
                self.assertIn(key, d)

    def test_filter_by_camera(self):
        results = self.engine.search("anything", camera_id="cam2")
        for e in results:
            self.assertEqual(e.camera_id, "cam2")

    def test_filter_by_object_type(self):
        results = self.engine.search("anything", object_types=["car"])
        for e in results:
            self.assertIn("car", e.detected_objects)

    def test_filter_by_time_range(self):
        start = datetime(2025, 5, 1, 10, 0, 0)
        end   = datetime(2025, 5, 1, 10, 59, 59)
        results = self.engine.search("person", start_time=start, end_time=end)
        for e in results:
            self.assertGreaterEqual(e.timestamp, start)
            self.assertLessEqual(e.timestamp, end)

    def test_empty_index_returns_empty(self):
        import faiss
        self.engine._index = faiss.IndexFlatIP(512)
        results = self.engine.search("test")
        self.assertEqual(results, [])

    # ── summary ───────────────────────────────────────────────────────────────

    def test_summary_keys(self):
        s = self.engine.get_summary()
        for key in ("total_events", "object_counts", "hourly_distribution"):
            self.assertIn(key, s)

    def test_summary_total_count(self):
        s = self.engine.get_summary()
        self.assertEqual(s["total_events"], 3)

    def test_summary_object_counts(self):
        s = self.engine.get_summary()
        self.assertIn("person", s["object_counts"])
        self.assertIn("car", s["object_counts"])

    def test_summary_camera_filter(self):
        s = self.engine.get_summary(camera_id="cam1")
        self.assertEqual(s["total_events"], 2)

    # ── Event ─────────────────────────────────────────────────────────────────

    def test_event_to_dict_roundtrip(self):
        from src.search.retrospective_search import Event
        row = {
            "id": 7, "camera_id": "cam3",
            "video_path": "/tmp/v.mp4",
            "timestamp": "2025-06-01T08:30:00",
            "detected_objects": "person,truck",
            "zone": "entrance",
        }
        e = Event(row, score=0.85)
        d = e.to_dict()
        self.assertEqual(d["id"], 7)
        self.assertEqual(d["camera_id"], "cam3")
        self.assertIn("person", d["detected_objects"])
        self.assertAlmostEqual(d["score"], 0.85)


if __name__ == "__main__":
    unittest.main(verbosity=2)
