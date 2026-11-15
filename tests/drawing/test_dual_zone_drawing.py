"""
Acceptance Test Suite for DUAL-ZONE AIR DRAWING SYSTEM.
Validates all 8 Acceptance Criteria from user specification:
  1. มือซ้ายวาดในครึ่งซ้ายได้ (Left Hand in Left Zone X < 50%)
  2. มือขวาวาดในครึ่งขวาได้ (Right Hand in Right Zone X >= 50%)
  3. วาดสองมือพร้อมกันได้ (Simultaneous Dual-Zone Drawing)
  4. มือซ้ายข้ามเส้นกลางแล้วต้องไม่วาดในฝั่งขวา (Left Hand boundary violation prevented)
  5. มือขวาข้ามเส้นกลางแล้วต้องไม่วาดในฝั่งซ้าย (Right Hand boundary violation prevented)
  6. เส้นไม่เชื่อมข้ามเขต (Stroke breaks on exit, starts new stroke on re-entry, no cross-line)
  7. การกำมือยังทำงานแยกกัน (Independent Fist pause / resume per hand)
  8. การประสานมือเปิดและปิด Drawing Mode ได้ตามเดิม (Two-Hand Join Toggle preserved)
"""

import os
import sys
import math
import numpy as np

# Ensure project root directory is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
for p in [current_dir, project_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

import camera_config as config
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


def run_dual_zone_tests():
    print("=" * 70)
    print(" RUNNING ACCEPTANCE TESTS: DUAL-ZONE AIR DRAWING SYSTEM")
    print("=" * 70)

    w_frame, h_frame = 640, 480
    x_split = int(w_frame * config.ZONE_SPLIT_RATIO)  # 320 px

    mgr = AirDrawingManager(zone_split_ratio=0.5, min_point_dist=2)
    sim_t = 100.0

    # ---------------------------------------------------------
    # TEST 1: มือซ้ายวาดในครึ่งซ้ายได้ (Left Zone: X < 320)
    # ---------------------------------------------------------
    print("\n--- Test 1: มือซ้ายวาดในครึ่งซ้ายได้ (X < 50%) ---")
    h_data = {
        "Left": create_mock_hand("Left", (150, 300), is_pointing=True, tip=(150, 200)),
        "Right": None,
    }
    mgr.update(h_data, is_drawing_active=True, current_time=sim_t, frame_width=w_frame, frame_height=h_frame)

    sim_t += 0.05
    h_data = {
        "Left": create_mock_hand("Left", (150, 300), is_pointing=True, tip=(160, 210)),
        "Right": None,
    }
    mgr.update(h_data, is_drawing_active=True, current_time=sim_t, frame_width=w_frame, frame_height=h_frame)

    assert mgr.get_total_points("Left") == 2, f"Left hand must draw points in Left zone, got {mgr.get_total_points('Left')}"
    assert mgr.boundary_status["Left"] == "INSIDE_ZONE", "Left hand must be INSIDE_ZONE"
    for stroke in mgr.strokes["Left"]:
        for pt in stroke:
            assert pt["x"] < x_split, f"Left point {pt['x']} must be in Left zone (< {x_split})"
            assert pt["norm_x"] < 0.5, "norm_x must be < 0.5"
            assert 0.0 <= pt["zone_x"] <= 1.0, "zone_x must be normalized [0, 1]"
    print(f"  Left hand drew {mgr.get_total_points('Left')} points in Left Zone (X < {x_split}) (PASS)")

    # ---------------------------------------------------------
    # TEST 2: มือขวาวาดในครึ่งขวาได้ (Right Zone: X >= 320)
    # ---------------------------------------------------------
    print("\n--- Test 2: มือขวาวาดในครึ่งขวาได้ (X >= 50%) ---")
    sim_t += 0.05
    h_data = {
        "Left": None,
        "Right": create_mock_hand("Right", (450, 300), is_pointing=True, tip=(450, 200)),
    }
    mgr.update(h_data, is_drawing_active=True, current_time=sim_t, frame_width=w_frame, frame_height=h_frame)

    sim_t += 0.05
    h_data = {
        "Left": None,
        "Right": create_mock_hand("Right", (450, 300), is_pointing=True, tip=(460, 210)),
    }
    mgr.update(h_data, is_drawing_active=True, current_time=sim_t, frame_width=w_frame, frame_height=h_frame)

    assert mgr.get_total_points("Right") == 2, f"Right hand must draw points in Right zone, got {mgr.get_total_points('Right')}"
    assert mgr.boundary_status["Right"] == "INSIDE_ZONE", "Right hand must be INSIDE_ZONE"
    for stroke in mgr.strokes["Right"]:
        for pt in stroke:
            assert pt["x"] >= x_split, f"Right point {pt['x']} must be in Right zone (>= {x_split})"
            assert pt["norm_x"] >= 0.5, "norm_x must be >= 0.5"
            assert 0.0 <= pt["zone_x"] <= 1.0, "zone_x must be normalized [0, 1]"
    print(f"  Right hand drew {mgr.get_total_points('Right')} points in Right Zone (X >= {x_split}) (PASS)")

    # ---------------------------------------------------------
    # TEST 3: วาดสองมือพร้อมกันได้ (Simultaneous Dual-Zone Drawing)
    # ---------------------------------------------------------
    print("\n--- Test 3: วาดสองมือพร้อมกันได้ (Dual-Zone Simultaneous) ---")
    left_prev = mgr.get_total_points("Left")
    right_prev = mgr.get_total_points("Right")

    sim_t += 0.05
    h_data = {
        "Left": create_mock_hand("Left", (160, 300), is_pointing=True, tip=(170, 220)),
        "Right": create_mock_hand("Right", (460, 300), is_pointing=True, tip=(470, 220)),
    }
    mgr.update(h_data, is_drawing_active=True, current_time=sim_t, frame_width=w_frame, frame_height=h_frame)

    assert mgr.get_total_points("Left") > left_prev, "Left hand must record point simultaneously"
    assert mgr.get_total_points("Right") > right_prev, "Right hand must record point simultaneously"
    print(f"  Simultaneous draw: Left={mgr.get_total_points('Left')} pts, Right={mgr.get_total_points('Right')} pts (PASS)")

    # ---------------------------------------------------------
    # TEST 4: มือซ้ายข้ามเส้นกลางแล้วต้องไม่วาดในฝั่งขวา
    # ---------------------------------------------------------
    print("\n--- Test 4: มือซ้ายข้ามเส้นกลางแล้วต้องไม่วาดในฝั่งขวา ---")
    left_pts_before_cross = mgr.get_total_points("Left")

    # Move left hand across center line to X = 400 (Right Zone)
    sim_t += 0.05
    h_data = {
        "Left": create_mock_hand("Left", (400, 300), is_pointing=True, tip=(400, 220)),
        "Right": None,
    }
    mgr.update(h_data, is_drawing_active=True, current_time=sim_t, frame_width=w_frame, frame_height=h_frame)

    # Move left hand further to X = 450
    sim_t += 0.05
    h_data = {
        "Left": create_mock_hand("Left", (450, 300), is_pointing=True, tip=(450, 230)),
        "Right": None,
    }
    mgr.update(h_data, is_drawing_active=True, current_time=sim_t, frame_width=w_frame, frame_height=h_frame)

    assert mgr.boundary_status["Left"] == "OUT_OF_ZONE", "Left hand must report OUT_OF_ZONE"
    assert mgr.get_total_points("Left") == left_pts_before_cross, "Left hand must NOT add points in Right zone"
    assert mgr.active_stroke["Left"] is None, "Active stroke must be closed when crossing out of zone"
    print("  Left hand at X=400, X=450: Blocked, OUT_OF_ZONE reported, 0 points added (PASS)")

    # ---------------------------------------------------------
    # TEST 5: มือขวาข้ามเส้นกลางแล้วต้องไม่วาดในฝั่งซ้าย
    # ---------------------------------------------------------
    print("\n--- Test 5: มือขวาข้ามเส้นกลางแล้วต้องไม่วาดในฝั่งซ้าย ---")
    right_pts_before_cross = mgr.get_total_points("Right")

    # Move right hand across center line to X = 200 (Left Zone)
    sim_t += 0.05
    h_data = {
        "Left": None,
        "Right": create_mock_hand("Right", (200, 300), is_pointing=True, tip=(200, 220)),
    }
    mgr.update(h_data, is_drawing_active=True, current_time=sim_t, frame_width=w_frame, frame_height=h_frame)

    # Move right hand further to X = 150
    sim_t += 0.05
    h_data = {
        "Left": None,
        "Right": create_mock_hand("Right", (150, 300), is_pointing=True, tip=(150, 230)),
    }
    mgr.update(h_data, is_drawing_active=True, current_time=sim_t, frame_width=w_frame, frame_height=h_frame)

    assert mgr.boundary_status["Right"] == "OUT_OF_ZONE", "Right hand must report OUT_OF_ZONE"
    assert mgr.get_total_points("Right") == right_pts_before_cross, "Right hand must NOT add points in Left zone"
    assert mgr.active_stroke["Right"] is None, "Active stroke must be closed when crossing out of zone"
    print("  Right hand at X=200, X=150: Blocked, OUT_OF_ZONE reported, 0 points added (PASS)")

    # ---------------------------------------------------------
    # TEST 6: เส้นไม่เชื่อมข้ามเขต (Stroke Break & Clean Re-entry)
    # ---------------------------------------------------------
    print("\n--- Test 6: เส้นไม่เชื่อมข้ามเขต (Stroke Break & Clean Re-entry) ---")
    left_strokes_before = mgr.get_total_strokes("Left")

    # Left hand re-enters Left zone at X = 180
    sim_t += 0.05
    h_data = {
        "Left": create_mock_hand("Left", (180, 300), is_pointing=True, tip=(180, 200)),
        "Right": None,
    }
    mgr.update(h_data, is_drawing_active=True, current_time=sim_t, frame_width=w_frame, frame_height=h_frame)

    sim_t += 0.05
    h_data = {
        "Left": create_mock_hand("Left", (180, 300), is_pointing=True, tip=(190, 210)),
        "Right": None,
    }
    mgr.update(h_data, is_drawing_active=True, current_time=sim_t, frame_width=w_frame, frame_height=h_frame)

    left_strokes_after = mgr.get_total_strokes("Left")
    assert left_strokes_after == left_strokes_before + 1, "Must start a brand new stroke on re-entry!"

    # Verify no line connects old stroke to new stroke
    stroke1 = mgr.strokes["Left"][-2]
    stroke2 = mgr.strokes["Left"][-1]
    # Stroke 1 last point and Stroke 2 first point must be distinct strokes
    assert stroke1 is not stroke2, "Strokes must be separate lists"
    print(f"  Re-entry created stroke #{left_strokes_after} cleanly without bridging across border! (PASS)")

    # ---------------------------------------------------------
    # TEST 7: การกำมือยังทำงานแยกกัน (Independent Fist Control)
    # ---------------------------------------------------------
    print("\n--- Test 7: การกำมือยังทำงานแยกกัน ---")
    sim_t += 0.05
    # Left hand in FIST (inside Left zone), Right hand POINTING (inside Right zone)
    # 2 frames to confirm fist debouncing
    for _ in range(2):
        sim_t += 0.05
        h_data = {
            "Left": create_mock_hand("Left", (200, 300), is_pointing=False, tip=(200, 250)),
            "Right": create_mock_hand("Right", (480, 300), is_pointing=True, tip=(480, 210)),
        }
        mgr.update(h_data, is_drawing_active=True, current_time=sim_t, frame_width=w_frame, frame_height=h_frame)

    left_pts_fist = mgr.get_total_points("Left")
    right_pts_draw = mgr.get_total_points("Right")

    # Another frame: Left remains in fist, Right draws
    sim_t += 0.05
    h_data = {
        "Left": create_mock_hand("Left", (205, 300), is_pointing=False, tip=(205, 250)),
        "Right": create_mock_hand("Right", (480, 300), is_pointing=True, tip=(490, 220)),
    }
    mgr.update(h_data, is_drawing_active=True, current_time=sim_t, frame_width=w_frame, frame_height=h_frame)

    assert mgr.get_total_points("Left") == left_pts_fist, "Left hand in fist must not add points"
    assert mgr.get_total_points("Right") > right_pts_draw, "Right hand pointing must continue drawing"
    assert mgr.posture_state["Left"] == "FIST", "Left hand must be confirmed as FIST"
    assert mgr.posture_state["Right"] == "POINTING", "Right hand must be confirmed as POINTING"
    print(f"  Left (Fist): paused at {left_pts_fist} pts | Right (Pointing): drew to {mgr.get_total_points('Right')} pts (PASS)")

    # ---------------------------------------------------------
    # TEST 8: การประสานมือเปิดและปิด Drawing Mode ได้ตามเดิม
    # ---------------------------------------------------------
    print("\n--- Test 8: การประสานมือเปิดและปิด Drawing Mode ได้ตามเดิม ---")
    gesture = TwoHandJoinGestureController(hold_duration=0.7)
    g_time = 500.0

    # Two hands join at center (e.g. Left at 318, Right at 322)
    joined = {
        "Left": create_mock_hand("Left", (315, 300), is_pointing=True, tip=(318, 240)),
        "Right": create_mock_hand("Right", (325, 300), is_pointing=True, tip=(322, 240)),
    }

    # Frame 1: detect join
    gesture.update(joined, g_time)
    # Frame 2: hold 0.8s -> toggle IDLE -> DRAWING_ACTIVE
    g_time += 0.8
    toggled1 = gesture.update(joined, g_time)
    assert toggled1, "Must toggle on hold"
    assert gesture.drawing_state == DrawingState.DRAWING_ACTIVE, "Must enter DRAWING_ACTIVE"
    print("  Join toggle #1 confirmed: Switched to DRAWING_ACTIVE (PASS)")

    # Separate hands
    g_time += 0.2
    separated = {
        "Left": create_mock_hand("Left", (150, 300), is_pointing=True, tip=(150, 200)),
        "Right": create_mock_hand("Right", (450, 300), is_pointing=True, tip=(450, 200)),
    }
    gesture.update(separated, g_time)

    # Join a second time -> toggle DRAWING_ACTIVE -> IDLE
    g_time += 0.5  # past cooldown
    gesture.update(joined, g_time)
    g_time += 0.8
    toggled2 = gesture.update(joined, g_time)
    assert toggled2, "Must toggle on 2nd join"
    assert gesture.drawing_state == DrawingState.IDLE, "Must return to IDLE"
    print("  Join toggle #2 confirmed: Switched to IDLE (PASS)")

    # Verify strokes preserved
    assert mgr.get_total_points("Left") > 0, "Left strokes must remain in memory"
    assert mgr.get_total_points("Right") > 0, "Right strokes must remain in memory"
    print(f"  All drawn strokes preserved: Left={mgr.get_total_points('Left')} pts, Right={mgr.get_total_points('Right')} pts (PASS)")

    print("\n" + "=" * 70)
    print(" ALL 8 ACCEPTANCE TESTS FOR DUAL-ZONE AIR DRAWING PASSED 100%!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = run_dual_zone_tests()
    if not success:
        sys.exit(1)

