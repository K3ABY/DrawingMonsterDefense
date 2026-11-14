"""
Threaded Camera Stream Ingestion for Drawing Monster Defense.
Optimizes network camera performance by decoupling network frame capture
from MediaPipe processing, completely eliminating buffer lag and stale frames.
"""

import threading
import time
from typing import Optional, Tuple
import cv2
import numpy as np


class ThreadedCameraStream:
    """
    Background worker thread that continually grabs the latest frame from
    the network camera stream (MJPEG), keeping OpenCV's internal buffer at zero.
    """

    def __init__(self, stream_url: str, max_reconnect_retries: int = 5):
        self.stream_url = stream_url
        self.max_reconnect_retries = max_reconnect_retries

        self.cap: Optional[cv2.VideoCapture] = None
        self.latest_frame: Optional[np.ndarray] = None
        self.frame_id: int = 0
        self.last_consumed_id: int = -1

        self.running: bool = False
        self.is_connected: bool = False
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None

        # Camera Metrics
        self.camera_fps: float = 0.0
        self.raw_frame_count: int = 0
        self.dropped_frame_count: int = 0
        self._fps_timer: float = time.time()
        self._fps_counter: int = 0

        self._connect()
        if self.is_connected:
            self.running = True
            self.thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.thread.start()

    def _connect(self) -> bool:
        """Initializes VideoCapture with minimal buffer size."""
        if self.cap is not None:
            self.cap.release()

        self.cap = cv2.VideoCapture(self.stream_url)
        # Attempt to set buffer size to 1 to reduce internal queueing
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.latest_frame = frame
                    self.frame_id += 1
                self.is_connected = True
                return True

        self.is_connected = False
        return False

    def _worker_loop(self) -> None:
        """Continuously pulls frames in background thread to drain buffer."""
        consecutive_failures = 0

        while self.running:
            if not self.is_connected or self.cap is None or not self.cap.isOpened():
                time.sleep(0.5)
                if self.running and self._connect():
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    if consecutive_failures > self.max_reconnect_retries:
                        time.sleep(2.0)
                continue

            ret, frame = self.cap.read()
            curr_time = time.time()

            if ret and frame is not None:
                consecutive_failures = 0
                with self.lock:
                    # If last frame wasn't consumed yet, it's counted as a pipeline drop
                    if self.frame_id > self.last_consumed_id and self.last_consumed_id >= 0:
                        self.dropped_frame_count += 1
                    self.latest_frame = frame
                    self.frame_id += 1
                    self.raw_frame_count += 1

                # Calculate Camera FPS
                self._fps_counter += 1
                dt = curr_time - self._fps_timer
                if dt >= 1.0:
                    self.camera_fps = self._fps_counter / dt
                    self._fps_counter = 0
                    self._fps_timer = curr_time
            else:
                consecutive_failures += 1
                if consecutive_failures > 5:
                    self.is_connected = False
                time.sleep(0.01)

    def read_latest(self) -> Tuple[bool, Optional[np.ndarray], bool]:
        """
        Returns:
            (success, frame_copy, is_new_frame)
            Zero wait time; non-blocking.
        """
        with self.lock:
            if self.latest_frame is None:
                return False, None, False

            is_new = self.frame_id > self.last_consumed_id
            self.last_consumed_id = self.frame_id
            frame_copy = self.latest_frame.copy()
            return True, frame_copy, is_new

    def get_metrics(self) -> dict:
        """Returns camera stream health metrics."""
        with self.lock:
            return {
                "camera_fps": self.camera_fps,
                "is_connected": self.is_connected,
                "raw_frames": self.raw_frame_count,
                "dropped_frames": self.dropped_frame_count,
            }

    def release(self) -> None:
        """Stops thread and releases resources."""
        self.running = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_connected = False

