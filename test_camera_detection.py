"""
Comprehensive Acceptance & Validation Test Suite for:
AUTOMATIC CAMERA DETECTION WITH IP CAMERA FALLBACK
Drawing Monster Defense

Validates all 8 Acceptance Criteria from user specification:
  1. Computer with a working local webcam (Primary selection)
  2. Computer without a local webcam but with a working IP Camera (Auto-fallback)
  3. Local webcam exists but cannot be opened (isOpened=False detection & fallback)
  4. Local webcam opens but returns invalid frames (Empty/broken frame validation & fallback)
  5. IP Camera server is offline (Fast non-blocking timeout & graceful handling)
  6. IP Camera disconnects during gameplay (Reconnection attempt, no stale frames, standby mode)
  7. Local webcam disconnects during gameplay (Auto-fallback to IP Camera without freeze)
  8. Both camera sources are unavailable (Standby frame, retry support, zero crash)
"""

import os
import sys
import time
import socket
import unittest
from unittest.mock import MagicMock, patch
import cv2
import numpy as np

# Ensure project directory is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import camera_config as config
from threaded_camera import (
    ThreadedCameraStream,
    CameraMode,
    CameraStatus,
    probe_local_camera,
    probe_ip_camera,
    detect_best_camera,
)


def make_valid_frame(w: int = 640, h: int = 480) -> np.ndarray:
    """Creates a synthetic BGR camera frame with valid dimensions and non-zero variance."""
    frame = np.full((h, w, 3), 120, dtype=np.uint8)
    cv2.rectangle(frame, (50, 50), (200, 200), (0, 255, 0), -1)
    return frame


class TestCameraDetectionAndFallback(unittest.TestCase):
    """Automated test cases for camera detection, selection priority, and runtime recovery."""

    def setUp(self):
        self.valid_frame = make_valid_frame(640, 480)

    # -------------------------------------------------------------------------
    # TEST 1: Working Local Webcam (Primary Selection)
    # -------------------------------------------------------------------------
    def test_case_1_working_local_webcam(self):
        print("\n--- Test 1: Computer with a working local webcam ---")

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, self.valid_frame.copy())

        with patch("cv2.VideoCapture", return_value=mock_cap):
            # Probe device 0
            is_ok, err, res = probe_local_camera(0)
            self.assertTrue(is_ok, "Working local webcam must be validated as OK")
            self.assertEqual(res, (640, 480))
            self.assertTrue(mock_cap.release.called, "Capture must be released after probing")

            # Test detect_best_camera
            mode, src, desc, res = detect_best_camera(local_indices=[0], ip_url="http://mock-ip:5000/video")
            self.assertEqual(mode, CameraMode.LOCAL_WEBCAM, "Local webcam must be primary selection")
            self.assertEqual(src, 0)
            self.assertIn("Local Webcam", desc)

            # Test ThreadedCameraStream instantiation
            cam = ThreadedCameraStream(auto_detect=True, local_indices=[0])
            try:
                metrics = cam.get_metrics()
                self.assertEqual(cam.camera_mode, CameraMode.LOCAL_WEBCAM)
                self.assertEqual(metrics["camera_mode"], "LOCAL WEBCAM")
                self.assertTrue(cam.is_connected)
                ret, frame, is_new = cam.read_latest()
                self.assertTrue(ret)
                self.assertIsNotNone(frame)
                print(f"  Selected: {cam.source_str} [{cam.camera_status.value}] (PASS)")
            finally:
                cam.release()

    # -------------------------------------------------------------------------
    # TEST 2: No Local Webcam, Working IP Camera (Auto-Fallback)
    # -------------------------------------------------------------------------
    def test_case_2_no_local_webcam_working_ip_camera(self):
        print("\n--- Test 2: Computer without local webcam, working IP Camera ---")

        def mock_video_capture(source, *args, **kwargs):
            cap = MagicMock()
            if isinstance(source, int):
                # Local webcams not present
                cap.isOpened.return_value = False
                cap.read.return_value = (False, None)
            else:
                # IP stream present
                cap.isOpened.return_value = True
                cap.read.return_value = (True, self.valid_frame.copy())
            return cap

        with patch("cv2.VideoCapture", side_effect=mock_video_capture):
            with patch("socket.create_connection", return_value=MagicMock()):
                mode, src, desc, res = detect_best_camera(
                    local_indices=[0, 1], ip_url="http://192.168.1.172:5000/video"
                )
                self.assertEqual(mode, CameraMode.IP_CAMERA, "Must automatically fallback to IP Camera")
                self.assertEqual(src, "http://192.168.1.172:5000/video")
                self.assertEqual(res, (640, 480))

                cam = ThreadedCameraStream(
                    stream_url="http://192.168.1.172:5000/video",
                    auto_detect=True,
                    local_indices=[0, 1],
                )
                try:
                    self.assertEqual(cam.camera_mode, CameraMode.IP_CAMERA)
                    self.assertTrue(cam.is_connected)
                    ret, frame, is_new = cam.read_latest()
                    self.assertTrue(ret)
                    print(f"  Auto-selected: {cam.source_str} [{cam.camera_status.value}] (PASS)")
                finally:
                    cam.release()

    # -------------------------------------------------------------------------
    # TEST 3: Local Webcam Exists but Cannot Be Opened (isOpened=False)
    # -------------------------------------------------------------------------
    def test_case_3_local_webcam_cannot_be_opened(self):
        print("\n--- Test 3: Local webcam exists but cannot be opened ---")

        def mock_capture(source, *args, **kwargs):
            cap = MagicMock()
            if isinstance(source, int):
                # Device index exists in OS but is locked / cannot be opened
                cap.isOpened.return_value = False
                cap.read.return_value = (False, None)
            else:
                cap.isOpened.return_value = True
                cap.read.return_value = (True, self.valid_frame.copy())
            return cap

        with patch("cv2.VideoCapture", side_effect=mock_capture):
            with patch("socket.create_connection", return_value=MagicMock()):
                is_ok, err, _ = probe_local_camera(0)
                self.assertFalse(is_ok, "Locked device must fail probe")
                self.assertIn("cannot be opened", err)

                # Fallback to IP Camera must occur smoothly
                mode, src, desc, _ = detect_best_camera(local_indices=[0], ip_url="http://192.168.1.172:5000/video")
                self.assertEqual(mode, CameraMode.IP_CAMERA)
                print(f"  Detected local failure ('{err}') -> Fell back to: {desc} (PASS)")

    # -------------------------------------------------------------------------
    # TEST 4: Local Webcam Opens but Returns Invalid / Dead Frames
    # -------------------------------------------------------------------------
    def test_case_4_local_webcam_returns_invalid_frames(self):
        print("\n--- Test 4: Local webcam opens but returns invalid/dead frames ---")

        def mock_capture(source, *args, **kwargs):
            cap = MagicMock()
            if isinstance(source, int):
                # Opens, but read() returns empty or pure zero black frames
                cap.isOpened.return_value = True
                cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
            else:
                cap.isOpened.return_value = True
                cap.read.return_value = (True, self.valid_frame.copy())
            return cap

        with patch("cv2.VideoCapture", side_effect=mock_capture):
            with patch("socket.create_connection", return_value=MagicMock()):
                is_ok, err, _ = probe_local_camera(0)
                self.assertFalse(is_ok, "Camera returning pure zero dead frames must be rejected")
                self.assertIn("blank/dead sensor data", err)

                # Auto fallback to IP camera
                mode, src, desc, _ = detect_best_camera(local_indices=[0], ip_url="http://192.168.1.172:5000/video")
                self.assertEqual(mode, CameraMode.IP_CAMERA)
                print(f"  Rejected dead frame source -> Selected: {desc} (PASS)")

    # -------------------------------------------------------------------------
    # TEST 5: IP Camera Server Is Offline
    # -------------------------------------------------------------------------
    def test_case_5_ip_camera_offline(self):
        print("\n--- Test 5: IP Camera server is offline ---")

        # Socket times out or connection refused
        with patch("socket.create_connection", side_effect=socket.timeout):
            t0 = time.time()
            is_ok, err, res = probe_ip_camera("http://192.168.1.172:5000/video", timeout=0.5)
            dt = time.time() - t0

            self.assertFalse(is_ok, "Offline IP camera must fail probe")
            self.assertIn("unreachable", err)
            self.assertLess(dt, 1.0, f"Socket check must be fast (< 1.0s), took {dt:.2f}s")
            print(f"  Fast offline detection in {dt:.3f}s: '{err}' (PASS)")

    # -------------------------------------------------------------------------
    # TEST 6: IP Camera Disconnects During Gameplay
    # -------------------------------------------------------------------------
    def test_case_6_ip_camera_disconnects_during_gameplay(self):
        print("\n--- Test 6: IP Camera disconnects during gameplay ---")

        state = {"connected": True}

        def mock_read():
            if state["connected"]:
                return True, self.valid_frame.copy()
            else:
                return False, None

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = mock_read

        with patch("cv2.VideoCapture", return_value=mock_cap):
            with patch("socket.create_connection", return_value=MagicMock()):
                cam = ThreadedCameraStream(
                    stream_url="http://192.168.1.172:5000/video",
                    auto_detect=False,  # Pin to IP camera
                )
                try:
                    time.sleep(0.1)
                    self.assertTrue(cam.is_connected)
                    ret, frame, _ = cam.read_latest()
                    self.assertTrue(ret)

                    # Simulate network disconnection
                    state["connected"] = False
                    # Allow worker loop to accumulate failed reads
                    time.sleep(0.25)

                    # Game must NOT freeze, stale frames must NOT be returned indefinitely
                    ret_after, standby_frame, is_new = cam.read_latest()
                    self.assertFalse(ret_after, "read_latest() ret must be False when stream is down")
                    self.assertIsNotNone(standby_frame, "Must provide standby frame during disconnect")
                    self.assertIn(cam.camera_status, [CameraStatus.RECONNECTING, CameraStatus.FAILED])
                    print(f"  Stream disconnected -> State: {cam.camera_status.value}, Standby frame active (PASS)")
                finally:
                    cam.release()

    # -------------------------------------------------------------------------
    # TEST 7: Local Webcam Disconnects During Gameplay (Auto Fallback)
    # -------------------------------------------------------------------------
    def test_case_7_local_webcam_disconnects_runtime_fallback(self):
        print("\n--- Test 7: Local webcam disconnects during gameplay (Runtime fallback) ---")

        # Start with working local webcam
        cam_state = {"local_alive": True}

        def mock_vcap(source, *args, **kwargs):
            cap = MagicMock()
            if isinstance(source, int):
                cap.isOpened.return_value = cam_state["local_alive"]
                cap.read.side_effect = lambda: (True, self.valid_frame.copy()) if cam_state["local_alive"] else (False, None)
            else:
                # IP Camera is ready as fallback
                cap.isOpened.return_value = True
                cap.read.return_value = (True, self.valid_frame.copy())
            return cap

        with patch("cv2.VideoCapture", side_effect=mock_vcap):
            with patch("socket.create_connection", return_value=MagicMock()):
                # Set cooldown to 0.1s for fast testing
                with patch.object(config, "CAMERA_RECONNECT_COOLDOWN", 0.1):
                    with patch.object(config, "CAMERA_FAILURE_THRESHOLD", 3):
                        cam = ThreadedCameraStream(
                            stream_url="http://192.168.1.172:5000/video",
                            auto_detect=True,
                            local_indices=[0],
                        )
                        try:
                            time.sleep(0.1)
                            self.assertEqual(cam.camera_mode, CameraMode.LOCAL_WEBCAM)

                            # Disconnect local webcam (e.g. unplugged USB)
                            cam_state["local_alive"] = False
                            time.sleep(0.3)  # Wait for worker thread to detect and trigger fallback

                            # Verify fallback occurred to IP Camera
                            self.assertEqual(cam.camera_mode, CameraMode.IP_CAMERA)
                            self.assertTrue(cam.fallback_occurred)
                            self.assertIn("Local Webcam", cam.fallback_message)
                            self.assertTrue(cam.is_connected)
                            print(f"  Runtime Fallback Triggered: {cam.fallback_message} (PASS)")
                        finally:
                            cam.release()

    # -------------------------------------------------------------------------
    # TEST 8: Both Camera Sources Unavailable (Standby & Retry)
    # -------------------------------------------------------------------------
    def test_case_8_both_sources_unavailable(self):
        print("\n--- Test 8: Both camera sources are unavailable ---")

        mock_failed_cap = MagicMock()
        mock_failed_cap.isOpened.return_value = False
        mock_failed_cap.read.return_value = (False, None)

        with patch("cv2.VideoCapture", return_value=mock_failed_cap):
            with patch("socket.create_connection", side_effect=socket.timeout):
                # Does NOT crash with sys.exit!
                cam = ThreadedCameraStream(auto_detect=True, local_indices=[0])
                try:
                    self.assertEqual(cam.camera_mode, CameraMode.NONE)
                    self.assertEqual(cam.camera_status, CameraStatus.FAILED)
                    self.assertFalse(cam.is_connected)

                    # read_latest returns standby error frame
                    ret, standby_frame, is_new = cam.read_latest()
                    self.assertFalse(ret)
                    self.assertIsNotNone(standby_frame)
                    self.assertEqual(standby_frame.shape, (480, 640, 3))

                    # Test user retry detection method
                    success = cam.retry_detection()
                    self.assertFalse(success, "Retry should report False while sources are down")
                    self.assertFalse(cam.is_connected)
                    print(f"  Graceful standby: Mode={cam.camera_mode.value}, Status={cam.camera_status.value}, No Crash! (PASS)")
                finally:
                    cam.release()


def run_all_tests():
    print("=" * 74)
    print(" AUTOMATIC CAMERA DETECTION & IP CAMERA FALLBACK ACCEPTANCE SUITE")
    print("=" * 74)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCameraDetectionAndFallback)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    if not success:
        sys.exit(1)
