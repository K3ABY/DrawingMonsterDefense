"""
Unit and integration test for HandTracker (headless/automated).
Validates camera connection, hand tracking data structure, landmark extraction, and frame rendering.
"""

import os
import sys
import numpy as np
import cv2

# Ensure DrawingMonsterDefense is on path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import camera_config as config
from hand_tracking import HandTracker, open_camera_stream


def test_empty_frame():
    """Verify that an empty/black frame is handled gracefully without crashing."""
    print("[TEST] Running test_empty_frame...")
    tracker = HandTracker()
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    hand_data, results = tracker.process(blank_frame)

    assert "Left" in hand_data, "Missing 'Left' in hand_data"
    assert "Right" in hand_data, "Missing 'Right' in hand_data"

    # Per specification, hand_data is None when not detected:
    assert hand_data["Left"] is None, "Expected None for undetected Left hand"
    assert hand_data["Right"] is None, "Expected None for undetected Right hand"

    # Test drawing on blank frame without error
    rendered = tracker.draw(blank_frame, hand_data, results)
    assert rendered.shape == blank_frame.shape, "Rendered frame shape mismatch"

    tracker.close()
    print("[PASS] test_empty_frame passed!")


def test_live_stream_pipeline():
    """Verify connecting to the camera stream, reading frames, and processing."""
    print(f"[TEST] Testing live stream pipeline from {config.CAMERA_URL}...")
    cap = open_camera_stream(config.CAMERA_URL, max_retries=2)
    if cap is None:
        print("[FAIL] Cannot connect to live stream!")
        return False

    tracker = HandTracker()
    frames_processed = 0

    for i in range(10):
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"[WARN] Failed reading frame {i+1}")
            continue

        hand_data, results = tracker.process(frame)
        rendered = tracker.draw(frame, hand_data, results)

        frames_processed += 1
        left_status = "Detected" if hand_data["Left"] is not None else "Not Detected"
        right_status = "Detected" if hand_data["Right"] is not None else "Not Detected"
        print(f" Frame {i+1}: Left={left_status}, Right={right_status}")

        # Save the 1st frame output as a verification image
        if i == 0:
            test_output_path = os.path.join(current_dir, "test_output.jpg")
            cv2.imwrite(test_output_path, rendered)
            print(f" Saved test snapshot to: {test_output_path}")

    cap.release()
    tracker.close()

    assert frames_processed > 0, "No frames were processed from live stream"
    print(f"[PASS] Live stream pipeline passed! ({frames_processed}/10 frames successfully processed)")
    return True


if __name__ == "__main__":
    test_empty_frame()
    success = test_live_stream_pipeline()
    if not success:
        sys.exit(1)
    print("\nAll integration tests passed successfully!")
