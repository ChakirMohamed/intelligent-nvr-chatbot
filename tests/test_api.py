"""
Integration tests for the FastAPI REST layer.
All heavy dependencies (CLIP, FAISS) are mocked so tests run without GPU/model downloads.
"""
import sys
import os
import sqlite3
import tempfile
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _build_mock_search(db_path: str):
    """Return a RetrospectiveSearch-like mock backed by a real SQLite DB."""
    from src.search.retrospective_search import Event

    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS frames (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id TEXT, video_path TEXT, timestamp TEXT,
                faiss_id INTEGER, detected_objects TEXT, zone TEXT DEFAULT 'default'
            )
        """)
        conn.executemany(
            "INSERT INTO frames (camera_id, video_path, timestamp, faiss_id, detected_objects) VALUES (?,?,?,?,?)",
            [
                ("cam1", "/v.mp4", "2025-05-01T10:00:00", 0, "person"),
                ("cam2", "/v.mp4", "2025-05-01T11:00:00", 1, "car"),
            ],
        )

    mock = MagicMock()
    mock.db_path = db_path
    mock._index = MagicMock()
    mock._index.ntotal = 2

    row1 = {"id": 1, "camera_id": "cam1", "video_path": "/v.mp4",
            "timestamp": "2025-05-01T10:00:00", "detected_objects": "person", "zone": "default"}
    row2 = {"id": 2, "camera_id": "cam2", "video_path": "/v.mp4",
            "timestamp": "2025-05-01T11:00:00", "detected_objects": "car", "zone": "default"}

    e1 = Event(row1, score=0.9)
    e2 = Event(row2, score=0.7)
    mock.search.return_value = [e1, e2]
    mock.get_summary.return_value = {
        "total_events": 2,
        "object_counts": {"person": 1, "car": 1},
        "hourly_distribution": {10: 1, 11: 1},
        "camera_id": None,
        "start_time": None,
        "end_time": None,
    }
    return mock


def _build_mock_chatbot():
    mock = MagicMock()
    mock.chat.return_value = {
        "response": "J'ai trouvé 1 événement.",
        "events": [],
    }
    return mock


class TestAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.mkdtemp()
        db_path  = os.path.join(cls._tmp, "meta.db")
        mock_search  = _build_mock_search(db_path)
        mock_chatbot = _build_mock_chatbot()

        import src.api.api as api_module
        api_module._search  = mock_search
        api_module._chatbot = mock_chatbot

        from fastapi.testclient import TestClient
        cls.client = TestClient(api_module.app)

    # ── health ────────────────────────────────────────────────────────────────

    def test_health_ok(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("indexed_frames", data)

    # ── search ────────────────────────────────────────────────────────────────

    def test_search_returns_results(self):
        r = self.client.post("/search", json={"query": "person walking"})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("results", data)
        self.assertIn("count", data)
        self.assertEqual(data["query"], "person walking")

    def test_search_result_fields(self):
        r = self.client.post("/search", json={"query": "car"})
        self.assertEqual(r.status_code, 200)
        for ev in r.json()["results"]:
            for field in ("id", "camera_id", "timestamp", "detected_objects", "score"):
                self.assertIn(field, ev)

    def test_search_with_filters(self):
        payload = {
            "query": "person",
            "camera_id": "cam1",
            "start_time": "2025-05-01T00:00:00",
            "end_time": "2025-05-01T23:59:59",
            "object_types": ["person"],
        }
        r = self.client.post("/search", json=payload)
        self.assertEqual(r.status_code, 200)

    # ── chat ─────────────────────────────────────────────────────────────────

    def test_chat_basic(self):
        r = self.client.post("/chat", json={"message": "Bonjour", "session_id": "test-session"})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("response", data)

    def test_chat_session_persistence(self):
        self.client.post("/chat", json={"message": "Bonjour", "session_id": "s1"})
        r = self.client.post("/chat", json={"message": "Merci", "session_id": "s1"})
        self.assertEqual(r.status_code, 200)

    def test_chat_different_sessions(self):
        r1 = self.client.post("/chat", json={"message": "Hello", "session_id": "alice"})
        r2 = self.client.post("/chat", json={"message": "Hello", "session_id": "bob"})
        self.assertEqual(r1.status_code, 200)
        self.assertEqual(r2.status_code, 200)

    # ── events ────────────────────────────────────────────────────────────────

    def test_list_events(self):
        r = self.client.get("/events")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("events", data)
        self.assertIn("count", data)

    def test_list_events_camera_filter(self):
        r = self.client.get("/events?camera_id=cam1")
        self.assertEqual(r.status_code, 200)

    def test_list_events_limit(self):
        r = self.client.get("/events?limit=1")
        self.assertEqual(r.status_code, 200)
        self.assertLessEqual(r.json()["count"], 1)

    # ── summary ───────────────────────────────────────────────────────────────

    def test_summary(self):
        r = self.client.get("/summary")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        for key in ("total_events", "object_counts", "hourly_distribution"):
            self.assertIn(key, data)

    def test_summary_with_camera(self):
        r = self.client.get("/summary?camera_id=cam1")
        self.assertEqual(r.status_code, 200)

    # ── cameras ───────────────────────────────────────────────────────────────

    def test_list_cameras(self):
        r = self.client.get("/cameras")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("cameras", data)
        for cam in data["cameras"]:
            self.assertIn("camera_id", cam)
            self.assertIn("frame_count", cam)

    # ── clip (404 expected — no real files) ───────────────────────────────────

    def test_clip_not_found(self):
        r = self.client.get("/clip/9999")
        self.assertEqual(r.status_code, 404)


if __name__ == "__main__":
    unittest.main(verbosity=2)
