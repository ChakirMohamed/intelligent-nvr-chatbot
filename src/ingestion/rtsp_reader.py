"""
RTSP stream reader with MOG2 motion filtering and configurable frame sampling.
For testing without a real camera, use RTSPReader.generate_test_video().
"""
import cv2
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


class RTSPReader:
    def __init__(
        self,
        camera_id: str,
        rtsp_url: str,
        storage_path: str,
        sampling_fps: float = 1.0,
        motion_threshold: int = 500,
        segment_duration: int = 300,
        on_motion_frame: Optional[Callable[[str, np.ndarray, datetime], None]] = None,
    ):
        """
        Args:
            camera_id: Unique identifier for this camera.
            rtsp_url: RTSP stream URL or path to a local video file (for testing).
            storage_path: Root directory for .mp4 segment files.
            sampling_fps: Frames per second to keep from the source stream.
            motion_threshold: Minimum foreground pixels to consider a frame as motion.
            segment_duration: Seconds per output .mp4 segment.
            on_motion_frame: Optional callback(camera_id, frame, timestamp) for motion frames.
        """
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.storage_path = Path(storage_path)
        self.sampling_fps = sampling_fps
        self.motion_threshold = motion_threshold
        self.segment_duration = segment_duration
        self.on_motion_frame = on_motion_frame

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=False
        )

        # Live view: most recent decoded frame, published at full source fps so
        # the MJPEG endpoint stays smooth even though recording/indexing run at
        # the lower sampling_fps. Guarded by a lock for cross-thread access.
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_lock = threading.Lock()
        self._connected = False
        self._current_segment: Optional[Path] = None

        # A real network stream delivers frames in real time (cap.read blocks),
        # but a local file is decoded as fast as the CPU allows — which floods
        # the pipeline. For files we throttle playback to the source fps so a
        # recording behaves exactly like a live camera (Fix 1).
        self._is_stream = str(rtsp_url).lower().startswith(
            ("rtsp://", "rtmp://", "rtsps://", "http://", "https://", "udp://", "tcp://")
        )

    # ── internal helpers ──────────────────────────────────────────────────────

    def _segment_path(self, ts: datetime) -> Path:
        cam_dir = self.storage_path / self.camera_id / ts.strftime("%Y-%m-%d")
        cam_dir.mkdir(parents=True, exist_ok=True)
        return cam_dir / f"{ts.strftime('%H-%M-%S')}.mp4"

    def _has_motion(self, frame: np.ndarray) -> bool:
        mask = self._bg_subtractor.apply(frame)
        return int(cv2.countNonZero(mask)) > self.motion_threshold

    def _open_writer(self, path: Path, w: int, h: int) -> cv2.VideoWriter:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter(str(path), fourcc, self.sampling_fps, (w, h))

    # ── main loop ─────────────────────────────────────────────────────────────

    def _stream_loop(self) -> None:
        cap = cv2.VideoCapture(self.rtsp_url)
        if not cap.isOpened():
            logger.error("[%s] Cannot open stream: %s", self.camera_id, self.rtsp_url)
            self._connected = False
            return
        self._connected = True

        source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        skip = max(1, int(source_fps / self.sampling_fps))
        logger.info("[%s] source=%.1f fps  sampling=%.1f fps  skip=%d",
                    self.camera_id, source_fps, self.sampling_fps, skip)

        writer: Optional[cv2.VideoWriter] = None
        segment_start: Optional[datetime] = None
        frame_idx = 0

        # File-playback pacing clock (Fix 1): wall-clock anchor + frames read.
        playback_start = time.time()
        frames_read = 0

        while self._running:
            ret, frame = cap.read()
            if not ret:
                logger.warning("[%s] Frame read failed — reconnecting in 2 s", self.camera_id)
                self._connected = False
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(self.rtsp_url)
                self._connected = cap.isOpened()
                frame_idx = 0
                playback_start = time.time()
                frames_read = 0
                continue

            # For a local file, sleep so playback matches the source fps instead
            # of running at full decode speed (which would flood the pipeline).
            if not self._is_stream:
                frames_read += 1
                target = playback_start + frames_read / source_fps
                delay = target - time.time()
                if delay > 0:
                    time.sleep(delay)

            # Publish for live view at full source fps (before the sampling skip).
            with self._latest_lock:
                self._latest_frame = frame

            frame_idx += 1
            if frame_idx % skip != 0:
                continue

            now = datetime.now()
            h, w = frame.shape[:2]

            # Rotate segment file every segment_duration seconds
            if segment_start is None or (now - segment_start).seconds >= self.segment_duration:
                if writer:
                    writer.release()
                segment_start = now
                seg_path = self._segment_path(now)
                writer = self._open_writer(seg_path, w, h)
                self._current_segment = seg_path
                logger.info("[%s] New segment: %s", self.camera_id, seg_path)

            writer.write(frame)

            if self._has_motion(frame) and self.on_motion_frame:
                try:
                    self.on_motion_frame(self.camera_id, frame.copy(), now)
                except Exception as exc:
                    logger.error("[%s] on_motion_frame callback error: %s", self.camera_id, exc)

        if writer:
            writer.release()
        cap.release()
        self._connected = False
        logger.info("[%s] Stream stopped.", self.camera_id)

    # ── public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._stream_loop, daemon=True, name=f"rtsp-{self.camera_id}")
        self._thread.start()
        logger.info("[%s] Reader started.", self.camera_id)

    def stop(self, timeout: float = 10.0) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=timeout)
        logger.info("[%s] Reader stopped.", self.camera_id)

    # ── live view ─────────────────────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        """True while the stream is open and delivering frames."""
        return self._connected

    @property
    def current_segment(self) -> Optional[str]:
        """Path of the .mp4 segment currently being written (for clip linkage)."""
        return str(self._current_segment) if self._current_segment else None

    def get_latest_jpeg(self, quality: int = 80) -> Optional[bytes]:
        """Return the most recent frame JPEG-encoded, or None if none yet.

        Used by the MJPEG live endpoint. Encoding happens outside the lock so
        we never block the capture thread while serving viewers.
        """
        with self._latest_lock:
            frame = self._latest_frame
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return buf.tobytes() if ok else None

    # ── test utility ──────────────────────────────────────────────────────────

    @staticmethod
    def generate_test_video(
        output_path: str,
        duration: int = 60,
        fps: int = 25,
        width: int = 640,
        height: int = 480,
    ) -> str:
        """Generate a synthetic .mp4 video with a bouncing rectangle for tests."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out), fourcc, fps, (width, height))

        x, y, dx, dy = 50, height // 2, 4, 2
        for i in range(duration * fps):
            frame = np.full((height, width, 3), (20, 20, 40), dtype=np.uint8)
            cv2.rectangle(frame, (x, y - 35), (x + 45, y + 35), (0, 200, 120), -1)
            cv2.putText(frame, f"frame {i}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
            x = (x + dx) % (width - 50)
            y = max(40, min(height - 40, y + dy))
            if y in (40, height - 40):
                dy = -dy
            writer.write(frame)

        writer.release()
        logger.info("Test video written: %s", out)
        return str(out)
