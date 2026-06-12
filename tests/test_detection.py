"""
Tests for YOLODetector — runs in CPU mode (no OpenVINO required).
"""
import sys
import os
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _blank(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _noise(h=480, w=640):
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


class TestDetection(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from src.detection.yolo_detector import YOLODetector
        # Always use CPU / plain PyTorch for unit tests
        cls.det = YOLODetector(use_openvino=False, confidence_threshold=0.25)

    # ── Detection class ───────────────────────────────────────────────────────

    def test_detection_to_dict(self):
        from src.detection.yolo_detector import Detection
        d = Detection(class_name="person", confidence=0.91, bbox=(10, 20, 100, 200))
        out = d.to_dict()
        self.assertEqual(out["class"], "person")
        self.assertAlmostEqual(out["confidence"], 0.91)
        self.assertEqual(out["bbox"], [10, 20, 100, 200])

    def test_invalid_class_not_in_target(self):
        from src.detection.yolo_detector import TARGET_CLASSES
        self.assertNotIn("cat", TARGET_CLASSES)
        self.assertIn("person", TARGET_CLASSES)

    # ── inference ─────────────────────────────────────────────────────────────

    def test_detect_returns_list(self):
        result = self.det.detect(_blank())
        self.assertIsInstance(result, list)

    def test_detect_noise_returns_list(self):
        result = self.det.detect(_noise())
        self.assertIsInstance(result, list)

    def test_detect_fields(self):
        from src.detection.yolo_detector import Detection, TARGET_CLASSES
        frame = _noise()
        for det in self.det.detect(frame):
            self.assertIsInstance(det, Detection)
            self.assertIn(det.class_name, TARGET_CLASSES)
            self.assertGreaterEqual(det.confidence, 0.0)
            self.assertLessEqual(det.confidence, 1.0)
            self.assertEqual(len(det.bbox), 4)
            x1, y1, x2, y2 = det.bbox
            self.assertLessEqual(x1, x2)
            self.assertLessEqual(y1, y2)

    def test_detect_various_sizes(self):
        for h, w in [(240, 320), (720, 1280), (64, 64)]:
            frame = _blank(h, w)
            result = self.det.detect(frame)
            self.assertIsInstance(result, list)

    # ── drawing ───────────────────────────────────────────────────────────────

    def test_draw_does_not_modify_original(self):
        from src.detection.yolo_detector import Detection
        frame = _noise()
        original = frame.copy()
        dets = [Detection("car", 0.8, (10, 10, 100, 80))]
        self.det.draw_detections(frame, dets)
        np.testing.assert_array_equal(frame, original)

    def test_draw_output_shape(self):
        from src.detection.yolo_detector import Detection
        frame = _noise()
        dets = [Detection("person", 0.75, (5, 5, 200, 300))]
        out = self.det.draw_detections(frame, dets)
        self.assertEqual(out.shape, frame.shape)

    def test_draw_empty_detections(self):
        frame = _noise()
        out = self.det.draw_detections(frame, [])
        np.testing.assert_array_equal(out, frame)


if __name__ == "__main__":
    unittest.main(verbosity=2)
