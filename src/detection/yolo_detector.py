"""
YOLOv8n object detector with Intel OpenVINO acceleration for Iris Xe iGPU.
Falls back to CPU when the OpenVINO GPU device is unavailable.
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

TARGET_CLASSES = {"person", "car", "motorcycle", "bicycle", "truck"}

_COLOR_MAP = {
    "person":     (0,   255,   0),
    "car":        (255,   0,   0),
    "motorcycle": (0,   165, 255),
    "bicycle":    (255, 255,   0),
    "truck":      (128,   0, 128),
}


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]   # x1, y1, x2, y2

    def to_dict(self) -> dict:
        return {
            "class":      self.class_name,
            "confidence": round(self.confidence, 3),
            "bbox":       list(self.bbox),
        }


class YOLODetector:
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.4,
        use_openvino: bool = True,
        openvino_device: str = "GPU",   # "GPU" = Iris Xe, "CPU" = fallback
    ):
        self.confidence_threshold = confidence_threshold
        self.use_openvino = use_openvino
        self.openvino_device = openvino_device
        self._model = None
        self._infer_device: Optional[str] = None
        self._load(model_path)

    # ── model loading ─────────────────────────────────────────────────────────

    def _load(self, model_path: Optional[str]) -> None:
        from ultralytics import YOLO

        ov_dir = Path(model_path) if model_path else Path("./models/yolov8n_openvino_model")

        if self.use_openvino:
            if not ov_dir.exists():
                logger.info("OpenVINO model not found — exporting from yolov8n.pt …")
                base = YOLO("yolov8n.pt")
                base.export(format="openvino", half=False, dynamic=False)
                # Ultralytics writes to ./yolov8n_openvino_model by default
                exported = Path("yolov8n_openvino_model")
                if exported.exists():
                    import shutil
                    shutil.move(str(exported), str(ov_dir))

            if ov_dir.exists():
                self._model = YOLO(str(ov_dir))
                self._infer_device = self._pick_ov_device()
                logger.info("YOLOv8n OpenVINO model loaded (device=%s).", self._infer_device)
                return

        # plain PyTorch fallback
        self._model = YOLO("yolov8n.pt")
        self._infer_device = "cpu"
        logger.info("YOLOv8n PyTorch model loaded (CPU).")

    def _pick_ov_device(self) -> str:
        """Try requested OpenVINO device; fall back to CPU on error."""
        try:
            from openvino.runtime import Core
            core = Core()
            available = core.available_devices
            if self.openvino_device in available:
                return self.openvino_device
            logger.warning(
                "OpenVINO device '%s' not available (have: %s). Using CPU.",
                self.openvino_device, available,
            )
        except Exception as exc:
            logger.warning("OpenVINO device check failed (%s). Using CPU.", exc)
        return "CPU"

    # ── inference ─────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if self._model is None:
            return []

        results = self._model(
            frame,
            verbose=False,
            conf=self.confidence_threshold,
            device=self._infer_device,
        )

        detections: List[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                name = result.names[cls_id]
                if name not in TARGET_CLASSES:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append(Detection(
                    class_name=name,
                    confidence=float(box.conf[0]),
                    bbox=(x1, y1, x2, y2),
                ))
        return detections

    # ── visualisation ─────────────────────────────────────────────────────────

    def draw_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        out = frame.copy()
        for det in detections:
            color = _COLOR_MAP.get(det.class_name, (255, 255, 255))
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"{det.class_name} {det.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
            cv2.putText(out, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return out
