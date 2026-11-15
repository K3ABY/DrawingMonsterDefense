"""
Threaded Camera Stream Ingestion for Drawing Monster Defense.
Features:
- Automatic Camera Detection: Scans & prioritizes Local Webcam -> Falls back to IP Camera (MJPEG)
- Unified Camera Interface: Seamless frame format for MediaPipe tracking and air drawing
- Zero-Latency Decoupled Ingestion: Minimal OpenCV buffer queue in background thread
- Robust Runtime Reconnection & Fallback: Handles disconnects without freezing or crashing
- Real-time Health Diagnostics: Camera mode, connection status, resolution, FPS, and source info
"""

import os
import sys
import time
import socket
import threading
from enum import Enum
from typing import Optional, Tuple, Dict, Any, List, Union
from urllib.parse import urlparse
import cv2
import numpy as np

try:
    import src.config.camera_config as config
except ImportError:
    import camera_config as config


class CameraMode(Enum):
    LOCAL_WEBCAM = "LOCAL WEBCAM"
    IP_CAMERA = "IP CAMERA"
    NONE = "NO CAMERA"


class CameraStatus(Enum):
    CONNECTED = "CONNECTED"
    CONNECTING = "CONNECTING"
    RECONNECTING = "RECONNECTING"
    FAILED = "FAILED"
    DISCONNECTED = "DISCONNECTED"


def probe_local_camera(
    device_index: int, timeout: float = 1.5
) -> Tuple[bool, Optional[str], Tuple[int, int]]:
    """
    Tests whether a local camera device index can be opened and actually returns valid frames.
    Does not assume a camera is working merely because cap.isOpened() returns True.
    Releases all handles immediately after testing.

    Returns:
        (is_valid, error_reason, (width, height))
    """
    cap = None
    try:
        # On Windows, try DirectShow (DSHOW) first for fast non-blocking probing, then CAP_ANY
        if os.name == "nt":
            cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(device_index, cv2.CAP_ANY)
        else:
            cap = cv2.VideoCapture(device_index, cv2.CAP_ANY)

        if not cap.isOpened():
            return False, f"Device #{device_index} cannot be opened (isOpened=False)", (0, 0)

        # Attempt to grab and read test frames
        ret, frame = cap.read()
        if not ret or frame is None:
            return False, f"Device #{device_index} opened but returned no frame", (0, 0)

        h, w = frame.shape[:2]
        if w <= 0 or h <= 0:
            return False, f"Device #{device_index} returned invalid dimensions {w}x{h}", (0, 0)

        # Validate that the frame contains actual image data (not empty zero buffer)
        if np.all(frame == 0) and np.var(frame) < 0.001:
            # Try a second frame in case first frame was initial driver warmup
            ret2, frame2 = cap.read()
            if not ret2 or frame2 is None or (np.all(frame2 == 0) and np.var(frame2) < 0.001):
                return False, f"Device #{device_index} returned blank/dead sensor data", (0, 0)

        return True, None, (w, h)

    except Exception as e:
        return False, f"Exception probing device #{device_index}: {e}", (0, 0)
    finally:
        if cap is not None:
            cap.release()


def probe_ip_camera(
    stream_url: str, timeout: float = 1.0
) -> Tuple[bool, Optional[str], Tuple[int, int]]:
    """
    Tests whether an IP camera (MJPEG stream) is reachable and serving valid video frames.
    Performs a fast socket connection check first to prevent OpenCV from blocking for 30s.
    Releases all handles immediately after testing.

    Returns:
        (is_valid, error_reason, (width, height))
    """
    # 1. Fast socket reachability test
    try:
        parsed = urlparse(stream_url)
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if not host:
            return False, f"Invalid IP stream URL: {stream_url}", (0, 0)

        with socket.create_connection((host, port), timeout=timeout):
            pass  # Host and port are reachable
    except (socket.timeout, ConnectionRefusedError, OSError) as e:
        return False, f"IP Camera host unreachable ({stream_url}): {type(e).__name__}", (0, 0)

    # 2. Open VideoCapture to verify MJPEG stream
    cap = None
    try:
        cap = cv2.VideoCapture(stream_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            return False, f"IP stream cannot be opened at {stream_url}", (0, 0)

        ret, frame = cap.read()
        if not ret or frame is None:
            return False, f"IP stream opened but returned invalid frame from {stream_url}", (0, 0)

        h, w = frame.shape[:2]
        if w <= 0 or h <= 0:
            return False, f"IP stream returned invalid dimensions {w}x{h}", (0, 0)

        return True, None, (w, h)

    except Exception as e:
        return False, f"Exception probing IP stream {stream_url}: {e}", (0, 0)
    finally:
        if cap is not None:
            cap.release()


def detect_best_camera(
    local_indices: Optional[List[int]] = None,
    ip_url: Optional[str] = None,
) -> Tuple[CameraMode, Any, str, Tuple[int, int]]:
    """
    Scans for the best available camera source in priority order:
      1. Local Webcam (indices in order: 0, 1...)
      2. IP Camera (Notebook Camera Server via MJPEG)

    Returns:
        (CameraMode, target_source, description, (width, height))
    """
    if local_indices is None:
        local_indices = getattr(config, "LOCAL_CAMERA_INDICES", [0, 1])
    if ip_url is None:
        ip_url = getattr(config, "CAMERA_URL", "http://192.168.1.172:5000/video")

    print("[CameraDetection] Scanning available camera sources...")

    # Priority 1: Check Local Webcams
    for idx in local_indices:
        print(f"[CameraDetection] Probing Local Webcam Device #{idx}...")
        is_ok, err, res = probe_local_camera(idx, timeout=getattr(config, "LOCAL_CAMERA_TIMEOUT", 1.5))
        if is_ok:
            desc = f"Local Webcam (Device #{idx}, {res[0]}x{res[1]})"
            print(f"[CameraDetection] SUCCESS: Found {desc}. Selected as primary camera.")
            return CameraMode.LOCAL_WEBCAM, idx, desc, res
        else:
            print(f"[CameraDetection] Local Webcam Device #{idx} not available: {err}")

    # Priority 2: Fallback to IP Camera
    print(f"[CameraDetection] No local webcam available. Probing IP Camera at: {ip_url}...")
    is_ok, err, res = probe_ip_camera(ip_url, timeout=getattr(config, "NETWORK_STREAM_TIMEOUT", 1.0))
    if is_ok:
        desc = f"IP Camera ({ip_url}, {res[0]}x{res[1]})"
        print(f"[CameraDetection] SUCCESS: Connected to {desc}. Selected as fallback camera.")
        return CameraMode.IP_CAMERA, ip_url, desc, res
    else:
        print(f"[CameraDetection] IP Camera not available: {err}")

    # Neither available
    desc = f"No local webcam found and IP Camera ({ip_url}) is offline"
    print(f"[CameraDetection] WARNING: {desc}")
    return CameraMode.NONE, None, desc, (0, 0)


class ThreadedCameraStream:
    """
    Unified Threaded Camera Stream with Automatic Local Webcam Detection & IP Camera Fallback.
    Decouples camera frame ingestion from MediaPipe processing, completely eliminating
    buffer lag, stale frames, and blocking I/O in the main game loop.
    """

    def __init__(
        self,
        stream_url: Optional[Union[str, int]] = None,
        auto_detect: bool = True,
        local_indices: Optional[List[int]] = None,
        max_reconnect_retries: int = 5,
    ):
        self.configured_url = stream_url or getattr(config, "CAMERA_URL", "http://192.168.1.172:5000/video")
        self.auto_detect = auto_detect and getattr(config, "AUTO_CAMERA_DETECTION", True)
        self.local_indices = local_indices or getattr(config, "LOCAL_CAMERA_INDICES", [0, 1])
        self.max_reconnect_retries = max_reconnect_retries

        self.cap: Optional[cv2.VideoCapture] = None
        self.latest_frame: Optional[np.ndarray] = None
        self.frame_id: int = 0
        self.last_consumed_id: int = -1

        self.running: bool = False
        self.is_connected: bool = False
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None

        # Camera Identification & State
        self.camera_mode: CameraMode = CameraMode.NONE
        self.camera_status: CameraStatus = CameraStatus.CONNECTING
        self.active_source: Any = None
        self.source_str: str = "Initializing..."
        self.width: int = getattr(config, "FRAME_WIDTH", 640)
        self.height: int = getattr(config, "FRAME_HEIGHT", 480)

        # Fallback & Failure Tracking
        self.fallback_occurred: bool = False
        self.fallback_message: str = "None"
        self.error_message: Optional[str] = None
        self.consecutive_failures: int = 0
        self.last_switch_time: float = 0.0

        # Performance Metrics
        self.camera_fps: float = 0.0
        self.raw_frame_count: int = 0
        self.dropped_frame_count: int = 0
        self._fps_timer: float = time.time()
        self._fps_counter: int = 0
        self._read_buffer: Optional[np.ndarray] = None

        # Perform initial camera detection or connection
        self._initialize_camera()

        # Start background worker thread
        self.running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def _initialize_camera(self) -> bool:
        """Selects and opens the camera source based on detection priority."""
        if self.auto_detect:
            mode, src, desc, res = detect_best_camera(self.local_indices, self.configured_url)
            self.camera_mode = mode
            self.active_source = src
            self.source_str = desc
            if res[0] > 0 and res[1] > 0:
                self.width, self.height = res
        else:
            # Explicit source passed by user
            self.active_source = self.configured_url
            if isinstance(self.configured_url, int):
                self.camera_mode = CameraMode.LOCAL_WEBCAM
                self.source_str = f"Local Webcam (Device #{self.configured_url})"
            else:
                self.camera_mode = CameraMode.IP_CAMERA
                self.source_str = f"IP Camera ({self.configured_url})"

        if self.camera_mode == CameraMode.NONE or self.active_source is None:
            self.camera_status = CameraStatus.FAILED
            self.error_message = self.source_str
            self.is_connected = False
            return False

        return self._connect_to_source(self.active_source)

    def _connect_to_source(self, source: Any) -> bool:
        """Opens cv2.VideoCapture for the chosen source."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        try:
            if isinstance(source, int):
                # Local Webcam
                if os.name == "nt":
                    self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
                    if not self.cap.isOpened():
                        self.cap.release()
                        self.cap = cv2.VideoCapture(source, cv2.CAP_ANY)
                else:
                    self.cap = cv2.VideoCapture(source, cv2.CAP_ANY)
            else:
                # Network IP Stream (MJPEG)
                self.cap = cv2.VideoCapture(source)

            if self.cap is not None and self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    h, w = frame.shape[:2]
                    self.width, self.height = w, h
                    with self.lock:
                        self.latest_frame = frame
                        self.frame_id += 1
                    self.is_connected = True
                    self.camera_status = CameraStatus.CONNECTED
                    self.consecutive_failures = 0
                    self.error_message = None
                    print(f"[ThreadedCameraStream] Connected successfully to {self.source_str} ({w}x{h})")
                    return True

        except Exception as e:
            self.error_message = f"Error opening {source}: {e}"
            print(f"[ThreadedCameraStream] Connection failed: {self.error_message}")

        self.is_connected = False
        self.camera_status = CameraStatus.FAILED
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        return False

    def _worker_loop(self) -> None:
        """Continuously pulls latest frames in background to keep buffer latency at zero."""
        failure_threshold = getattr(config, "CAMERA_FAILURE_THRESHOLD", 15)

        while self.running:
            # Case A: Not connected -> attempt reconnection or wait
            if not self.is_connected or self.cap is None or not self.cap.isOpened():
                time.sleep(0.5)
                if not self.running:
                    break
                self._handle_reconnection_or_fallback()
                continue

            # Case B: Connected -> Read frame
            ret, frame = self.cap.read()
            curr_time = time.time()

            if ret and frame is not None:
                self.consecutive_failures = 0
                self.camera_status = CameraStatus.CONNECTED
                self.is_connected = True

                with self.lock:
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
                self.consecutive_failures += 1
                if self.consecutive_failures >= failure_threshold:
                    self.is_connected = False
                    self.camera_status = CameraStatus.RECONNECTING
                    print(f"[ThreadedCameraStream] Warning: Dropped {self.consecutive_failures} consecutive frames from {self.source_str}!")
                    self._handle_reconnection_or_fallback()
                time.sleep(0.01)

    def _handle_reconnection_or_fallback(self) -> None:
        """Handles runtime stream recovery or fallback to alternate source."""
        curr_time = time.time()
        cooldown = getattr(config, "CAMERA_RECONNECT_COOLDOWN", 3.0)

        # 1. If Local Webcam failed -> Attempt fallback to IP Camera
        if self.camera_mode == CameraMode.LOCAL_WEBCAM and self.auto_detect:
            if curr_time - self.last_switch_time > cooldown:
                print(f"[ThreadedCameraStream] Local Webcam disconnected. Attempting fallback to IP Camera...")
                is_ok, err, res = probe_ip_camera(self.configured_url, timeout=0.8)
                if is_ok:
                    self.camera_mode = CameraMode.IP_CAMERA
                    self.active_source = self.configured_url
                    self.source_str = f"IP Camera ({self.configured_url}, {res[0]}x{res[1]})"
                    self.fallback_occurred = True
                    self.fallback_message = "Local Webcam Lost -> Switched to IP Camera"
                    self.last_switch_time = curr_time
                    self._connect_to_source(self.configured_url)
                    return

        # 2. If IP Camera failed -> Check if Local Webcam became available
        elif self.camera_mode == CameraMode.IP_CAMERA and self.auto_detect:
            if curr_time - self.last_switch_time > cooldown:
                # Check if user plugged in a local webcam
                for idx in self.local_indices:
                    is_ok, err, res = probe_local_camera(idx, timeout=0.8)
                    if is_ok:
                        print(f"[ThreadedCameraStream] Local Webcam detected at Device #{idx}. Recovering to Local Webcam...")
                        self.camera_mode = CameraMode.LOCAL_WEBCAM
                        self.active_source = idx
                        self.source_str = f"Local Webcam (Device #{idx}, {res[0]}x{res[1]})"
                        self.fallback_occurred = True
                        self.fallback_message = f"Recovered to Local Webcam (Device #{idx})"
                        self.last_switch_time = curr_time
                        self._connect_to_source(idx)
                        return

        # 3. Regular reconnection attempt to current active source
        if self.active_source is not None:
            self.camera_status = CameraStatus.RECONNECTING
            self._connect_to_source(self.active_source)

        if not self.is_connected:
            self.camera_status = CameraStatus.FAILED
            self.error_message = "All camera sources unavailable"

    def read_latest(self, copy: bool = False) -> Tuple[bool, Optional[np.ndarray], bool]:
        """
        Returns:
            (success, frame, is_new_frame)
            Zero wait time; non-blocking.
            Reuses internal pre-allocated buffer when copy=False to eliminate heap allocations.
            Returns an informative standby frame if disconnected or failed.
        """
        with self.lock:
            if self.is_connected and self.latest_frame is not None:
                is_new = self.frame_id > self.last_consumed_id
                self.last_consumed_id = self.frame_id
                if copy:
                    return True, self.latest_frame.copy(), is_new
                if self._read_buffer is None or self._read_buffer.shape != self.latest_frame.shape:
                    self._read_buffer = np.empty_like(self.latest_frame)
                np.copyto(self._read_buffer, self.latest_frame)
                return True, self._read_buffer, is_new

        # Camera disconnected or offline: generate standby screen so game does not crash
        standby_frame = self._generate_standby_frame()
        return False, standby_frame, True

    def _generate_standby_frame(self) -> np.ndarray:
        """Generates a clean visual standby frame when no camera is streaming."""
        w, h = self.width, self.height
        frame = np.full((h, w, 3), 28, dtype=np.uint8)

        # Dark grid lines
        for y in range(0, h, 40):
            cv2.line(frame, (0, y), (w, y), (38, 38, 38), 1)
        for x in range(0, w, 40):
            cv2.line(frame, (x, 0), (x, h), (38, 38, 38), 1)

        # Center error card
        card_w, card_h = 480, 200
        cx, cy = w // 2, h // 2
        bx, by = cx - card_w // 2, cy - card_h // 2
        cv2.rectangle(frame, (bx, by), (bx + card_w, by + card_h), (18, 18, 18), -1)
        cv2.rectangle(frame, (bx, by), (bx + card_w, by + card_h), (0, 0, 220), 2)

        # Flashing indicator
        blink = (int(time.time() * 2) % 2) == 0
        dot_color = (0, 0, 255) if blink else (80, 80, 80)
        cv2.circle(frame, (bx + 26, by + 30), 8, dot_color, -1)

        status_title = "CAMERA OFFLINE / RECONNECTING..."
        cv2.putText(frame, status_title, (bx + 44, by + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 160, 255), 2, cv2.LINE_AA)

        # Detail lines
        mode_text = f"Mode: {self.camera_mode.value} [{self.camera_status.value}]"
        cv2.putText(frame, mode_text, (bx + 24, by + 74), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

        source_text = f"Source: {self.source_str[:48]}"
        cv2.putText(frame, source_text, (bx + 24, by + 104), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)

        if self.fallback_occurred:
            fb_text = f"Event: {self.fallback_message[:46]}"
            cv2.putText(frame, fb_text, (bx + 24, by + 134), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 220, 255), 1, cv2.LINE_AA)

        ctrl_text = "Press [R] to Retry Camera Scan | [Q] to Quit"
        cv2.putText(frame, ctrl_text, (bx + 24, by + 172), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 128), 1, cv2.LINE_AA)

        return frame

    def retry_detection(self) -> bool:
        """Explicitly re-runs camera detection scan across local webcams and IP camera."""
        print("[ThreadedCameraStream] Retrying camera detection scan...")
        self.camera_status = CameraStatus.CONNECTING
        with self.lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.is_connected = False

        success = self._initialize_camera()
        return success

    def get_metrics(self) -> Dict[str, Any]:
        """Returns camera stream health metrics and source diagnostics."""
        with self.lock:
            return {
                "camera_fps": self.camera_fps,
                "is_connected": self.is_connected,
                "raw_frames": self.raw_frame_count,
                "dropped_frames": self.dropped_frame_count,
                "camera_mode": self.camera_mode.value,
                "camera_status": self.camera_status.value,
                "camera_source": self.source_str,
                "camera_res": f"{self.width}x{self.height}",
                "fallback_occurred": self.fallback_occurred,
                "fallback_message": self.fallback_message,
                "error_message": self.error_message,
            }

    def release(self) -> None:
        """Stops background thread and releases camera handles."""
        self.running = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        with self.lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.is_connected = False
            self.camera_status = CameraStatus.DISCONNECTED
