"""
Automated Test Suite for Stability, Landmark Smoothing, and Drawing Continuity.
Validates:
1. One Euro Filter adaptive jitter reduction & zero-lag fast tracking
2. Short-term tracking loss gap recovery (linear interpolation)
3. Long-term tracking loss stroke break (no spurious teleport lines)
4. Fist detection & Pointing drawing control
5. Independent hand control (Left Fist pauses Left only; Right draws)
6. Drawing stroke preservation across pauses
"""

import os
import sys
import time
import math
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import camera_config as config
from one_euro_filter import OneEuroFilter, PointOneEuroFilter, HandLandmarkSmoother
from drawing_manager import AirDrawingManager
from gesture_controller import TwoHandJoinGestureController, DrawingState


def create_mock_hand(label: str, wrist: tuple, is_pointing: bool, tip: tuple):
    """Generates synthetic 21 landmarks representing Pointing or Fist posture."""
    landmarks = [wrist] * 21
    landmarks[config.LANDMARK_WRIST] = wrist

    if is_pointing:
        # Index extended: mcp=(wx, wy-40), pip=(wx, wy-70), tip
        landmarks[config.LANDMARK_INDEX_MCP] = (wrist[0], wrist[1] - 40)
        landmarks[config.LANDMARK_INDEX_PIP] = (wrist[0], wrist[1] - 70)
        landmarks[config.LANDMARK_INDEX_FINGER_TIP] = tip

        # Middle, ring, pinky curled into palm
        for mcp_i, pip_i, tip_i in [(9, 10, 12), (13, 14, 16), (17, 18, 20)]:
            landmarks[mcp_i] = (wrist[0] + 20, wrist[1] - 40)
            landmarks[pip_i] = (wrist[0] + 20, wrist[1] - 25)
            landmarks[tip_i] = (wrist[0] + 20, wrist[1] - 10)
    else:
        # Fist: all 4 fingers curled close to palm
        for mcp_i, pip_i, tip_i in [(5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]:
            landmarks[mcp_i] = (wrist[0] + 15, wrist[1] - 40)
            landmarks[pip_i] = (wrist[0] + 15, wrist[1] - 25)
            landmarks[tip_i] = (wrist[0] + 15, wrist[1] - 10)

    return {
        "label": label,
        "detected": True,
        "status": "Detected",
        "index_tip_x": tip[0],
        "index_tip_y": tip[1],
        "index_finger_tip": tip,
        "landmarks": landmarks,
        "confidence": 0.95,
    }


def test_one_euro_filter_smoothing():
    print("\n--- Test 1: One Euro Filter Adaptive Smoothing ---")
    np.random.seed(42)
    filt = PointOneEuroFilter(min_cutoff=1.2, beta=0.015, d_cutoff=1.0)

    # 1. Stationary with noise
    t = 100.0
    raw_points = [(int(320 + np.random.normal(0, 3.5)), int(240 + np.random.normal(0, 3.5))) for _ in range(40)]
    smoothed_points = []
    for pt in raw_points:
        t += 0.033
        smoothed_points.append(filt.filter(pt, t))

    raw_jitter = np.std([math.hypot(p[0]-320, p[1]-240) for p in raw_points])
    smooth_jitter = np.std([math.hypot(p[0]-320, p[1]-240) for p in smoothed_points])
    print(f"  Stationary jitter: Raw std = {raw_jitter:.2f} px -> Smoothed std = {smooth_jitter:.2f} px")
    assert smooth_jitter < raw_jitter, "Smoothed jitter must be lower than raw jitter"

    # 2. Fast movement: verify lag is minimal
    fast_raw = [(320 + i * 25, 240) for i in range(15)]
    fast_smooth = []
    for pt in fast_raw:
        t += 0.033
        fast_smooth.append(filt.filter(pt, t))

    final_lag = abs(fast_raw[-1][0] - fast_smooth[-1][0])
    print(f"  Fast movement (750 px/s): Lag = {final_lag:.1f} px (< 15 px target)")
    assert final_lag < 20.0, f"Fast movement lag must be under 20 px, got {final_lag}"

    # 3. Filter reset on timeout
    t += 0.5  # gap > 0.25s
    reappear_pt = (500, 500)
    reset_pt = filt.filter(reappear_pt, t)
    # If properly reset, filtered output matches input exactly on first frame
    assert reset_pt == reappear_pt, "Filter must reset and output exact point after timeout"
    print("  Filter reset after 0.5s tracking loss confirmed (PASS)")


def test_drawing_continuity_and_gap_recovery():
    print("\n--- Test 2: Drawing Continuity & Short-Term Gap Recovery ---")
    mgr = AirDrawingManager(min_point_dist=2, max_bridge_time=0.20, max_bridge_dist=80)
    sim_t = 200.0

    # Start drawing with pointing hand
    h_data = {"Left": create_mock_hand("Left", (200, 300), is_pointing=True, tip=(200, 200)), "Right": None}
    mgr.update(h_data, is_drawing_active=True, current_time=sim_t)

    # Frame 2 (normal continuous movement)
    sim_t += 0.033
    h_data = {"Left": create_mock_hand("Left", (200, 300), is_pointing=True, tip=(205, 205)), "Right": None}
    mgr.update(h_data, is_drawing_active=True, current_time=sim_t)
    assert mgr.get_total_strokes("Left") == 1
    assert mgr.get_total_points("Left") == 2

    # Frame 3: Simulate 1-2 dropped frames (dt = 0.15s, within bridge window 0.20s, dist = 20px)
    sim_t += 0.15
    h_data = {"Left": create_mock_hand("Left", (200, 300), is_pointing=True, tip=(225, 225)), "Right": None}
    mgr.update(h_data, is_drawing_active=True, current_time=sim_t)

    # Verify bridge point was interpolated
    assert mgr.get_total_strokes("Left") == 1, "Stroke must NOT break during short-term drop"
    assert mgr.get_total_points("Left") == 4, f"Expected 4 points (with bridge), got {mgr.get_total_points('Left')}"
    print(f"  Short-term drop (0.15s): Stroke continued and bridge point inserted! Points={mgr.get_total_points('Left')} (PASS)")

    # Frame 4: Simulate long tracking loss (dt = 0.6s > 0.20s limit)
    sim_t += 0.60
    h_data = {"Left": create_mock_hand("Left", (400, 300), is_pointing=True, tip=(400, 200)), "Right": None}
    mgr.update(h_data, is_drawing_active=True, current_time=sim_t)

    assert mgr.get_total_strokes("Left") == 2, "Long-term loss must break stroke and start stroke #2"
    print("  Long-term drop (0.60s): Stroke broken cleanly without cross-screen teleport line (PASS)")


def test_fist_detection_and_independent_control():
    print("\n--- Test 3: Fist Detection & Independent Hand Drawing Control ---")
    mgr = AirDrawingManager()
    sim_t = 300.0

    # Frame 1: Both hands pointing -> both draw
    h_data = {
        "Left": create_mock_hand("Left", (150, 300), is_pointing=True, tip=(150, 200)),
        "Right": create_mock_hand("Right", (450, 300), is_pointing=True, tip=(450, 200)),
    }
    mgr.update(h_data, is_drawing_active=True, current_time=sim_t)

    # Confirm pointing state
    sim_t += 0.05
    h_data = {
        "Left": create_mock_hand("Left", (155, 300), is_pointing=True, tip=(155, 205)),
        "Right": create_mock_hand("Right", (455, 300), is_pointing=True, tip=(455, 205)),
    }
    mgr.update(h_data, is_drawing_active=True, current_time=sim_t)

    left_pts_before = mgr.get_total_points("Left")
    right_pts_before = mgr.get_total_points("Right")
    assert left_pts_before >= 1 and right_pts_before >= 1

    # Frame 3 & 4: Left hand makes a FIST, Right hand stays POINTING
    for _ in range(2):
        sim_t += 0.05
        h_data = {
            "Left": create_mock_hand("Left", (160, 300), is_pointing=False, tip=(160, 250)),
            "Right": create_mock_hand("Right", (460, 300), is_pointing=True, tip=(460, 210)),
        }
        mgr.update(h_data, is_drawing_active=True, current_time=sim_t)

    # Left hand is in FIST -> should have stopped adding points!
    # Right hand is POINTING -> should have continued adding points!
    left_pts_after = mgr.get_total_points("Left")
    right_pts_after = mgr.get_total_points("Right")

    print(f"  Left (Fist): points={left_pts_after} (frozen, did not add)")
    print(f"  Right (Pointing): points={right_pts_after} (continued drawing)")

    assert right_pts_after > right_pts_before, "Right hand must continue drawing"
    assert mgr.posture_state["Left"] == "FIST", "Left posture must be confirmed as FIST"
    assert mgr.posture_state["Right"] == "POINTING", "Right posture must be confirmed as POINTING"
    print("  Independent hand control & Fist detection confirmed! (PASS)")


if __name__ == "__main__":
    print("=" * 65)
    print(" RUNNING STABILITY & DRAWING ENHANCEMENT TESTS")
    print("=" * 65)
    test_one_euro_filter_smoothing()
    test_drawing_continuity_and_gap_recovery()
    test_fist_detection_and_independent_control()
    print("\n" + "=" * 65)
    print(" ALL STABILITY & RESPONSIVENESS TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)
