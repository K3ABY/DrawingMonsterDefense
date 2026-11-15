"""
Automated Acceptance Tests for Two-Hand Join Gesture (Toggle Mode) and Air Drawing.
Validates all 9 Acceptance Criteria:
  1. ประสานมือครั้งแรก -> เริ่มวาด (IDLE -> DRAWING_ACTIVE)
  2. แยกมือออก -> ยังวาดต่อได้
  3. วาดด้วยมือซ้าย -> บันทึกเส้นซ้าย
  4. วาดด้วยมือขวา -> บันทึกเส้นขวา
  5. วาดสองมือพร้อมกัน -> บันทึกทั้งสองเส้น
  6. ประสานมือครั้งที่สอง -> หยุดวาด (DRAWING_ACTIVE -> IDLE)
  7. เส้นที่วาดแล้วต้องยังอยู่ (No erase on stop)
  8. ประสานมือค้างไว้ -> ต้องไม่สลับสถานะหลายครั้ง (Debounce Lock)
  9. สามารถประสานมือครั้งที่สามเพื่อเริ่มวาดรอบใหม่ได้
"""

import os
import sys
import time
import numpy as np

# Ensure project root directory is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
for p in [current_dir, project_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

import camera_config as config
from gesture_controller import TwoHandJoinGestureController, DrawingState, GestureState
from drawing_manager import AirDrawingManager


def create_mock_hand_data(left_tip=(200, 200), right_tip=(440, 200), left_wrist=(180, 280), right_wrist=(460, 280)):
    """Creates a mock hand_data dictionary with realistic landmarks for testing."""
    def make_hand(label, tip, wrist):
        # 21 mock landmarks
        landmarks = [(wrist[0], wrist[1])] * 21
        landmarks[config.LANDMARK_WRIST] = wrist
        landmarks[config.LANDMARK_INDEX_FINGER_TIP] = tip
        # Middle MCP halfway between wrist and tip
        m_mcp = ((wrist[0] + tip[0]) // 2, (wrist[1] + tip[1]) // 2)
        landmarks[config.LANDMARK_MIDDLE_MCP] = m_mcp
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

    return {
        "Left": make_hand("Left", left_tip, left_wrist) if left_tip else None,
        "Right": make_hand("Right", right_tip, right_wrist) if right_tip else None,
    }


def run_acceptance_tests():
    print("=" * 70)
    print(" RUNNING ACCEPTANCE TESTS: TWO-HAND JOIN GESTURE (TOGGLE MODE)")
    print("=" * 70)

    gesture = TwoHandJoinGestureController(hold_duration=0.7, join_distance=65, release_distance=110, cooldown=0.3)
    drawing = AirDrawingManager(min_point_dist=2)

    sim_time = 100.0

    # Initial state verification
    assert gesture.drawing_state == DrawingState.IDLE, "Initial state must be IDLE"
    assert not gesture.is_drawing_active, "Initial drawing must be inactive"
    print("[PASS] Initial state is IDLE")

    # TEST 1: ประสานมือครั้งแรก -> เริ่มวาด
    print("\n--- Test 1: ประสานมือครั้งแรก -> เริ่มวาด ---")
    joined_hands = create_mock_hand_data(left_tip=(320, 240), right_tip=(324, 240))

    # Hold for 0.4s (under threshold) -> should NOT trigger
    for _ in range(4):
        sim_time += 0.1
        gesture.update(joined_hands, sim_time)
    assert gesture.drawing_state == DrawingState.IDLE, "Should not trigger before hold duration"
    assert gesture.gesture_state == GestureState.JOIN_HOLDING, "Should be in JOIN_HOLDING"
    print("  Hold 0.4s: Still in JOIN_HOLDING, not yet triggered (PASS)")

    # Hold for additional 0.4s (total 0.8s > 0.7s) -> MUST trigger DRAWING_ACTIVE
    sim_time += 0.4
    state_changed = gesture.update(joined_hands, sim_time)
    assert state_changed, "State change must be reported"
    assert gesture.drawing_state == DrawingState.DRAWING_ACTIVE, "State must be DRAWING_ACTIVE"
    assert gesture.last_event == "START_DRAWING", "Event must be START_DRAWING"
    assert gesture.gesture_state == GestureState.WAIT_RELEASE, "Must enter WAIT_RELEASE lock"
    print("  Hold confirmed (0.8s): Transitioned to DRAWING_ACTIVE! (PASS)")

    # TEST 8 (Part A): ประสานมือค้างไว้ -> ต้องไม่สลับสถานะหลายครั้ง
    print("\n--- Test 8 (Part A): ประสานมือค้างไว้ -> ต้องไม่สลับสถานะหลายครั้ง ---")
    for _ in range(10):
        sim_time += 0.2
        changed = gesture.update(joined_hands, sim_time)
        assert not changed, "Must NOT re-trigger while hands remain joined!"
        assert gesture.drawing_state == DrawingState.DRAWING_ACTIVE, "State must remain DRAWING_ACTIVE"
    print("  Hands kept joined for 2.0s: State remained DRAWING_ACTIVE without re-toggling (PASS)")

    # TEST 2: แยกมือออก -> ยังวาดต่อได้
    print("\n--- Test 2: แยกมือออก -> ยังวาดต่อได้ ---")
    separated_hands = create_mock_hand_data(left_tip=(200, 200), right_tip=(440, 200))
    sim_time += 0.2
    gesture.update(separated_hands, sim_time)
    assert gesture.drawing_state == DrawingState.DRAWING_ACTIVE, "Drawing must remain ACTIVE after separating hands"
    assert gesture.gesture_state == GestureState.NO_JOIN, "Lock must clear to NO_JOIN after separation"
    print("  Separated hands: Drawing still ACTIVE, Lock cleared to NO_JOIN (PASS)")

    # TEST 3: วาดด้วยมือซ้าย -> บันทึกเส้นซ้าย
    print("\n--- Test 3: วาดด้วยมือซ้าย -> บันทึกเส้นซ้าย ---")
    left_only_hands = create_mock_hand_data(left_tip=(200, 200), right_tip=None)
    drawing.update(left_only_hands, gesture.is_drawing_active, sim_time)
    # Move left hand 3 times
    for offset in [10, 20, 30]:
        sim_time += 0.05
        moving_left = create_mock_hand_data(left_tip=(200 + offset, 200 + offset), right_tip=None)
        drawing.update(moving_left, gesture.is_drawing_active, sim_time)

    assert drawing.get_total_points("Left") == 4, f"Expected 4 Left points, got {drawing.get_total_points('Left')}"
    assert drawing.get_total_points("Right") == 0, "Right points should still be 0"
    print(f"  Left hand drew: {drawing.get_total_points('Left')} points recorded (PASS)")

    # TEST 4: วาดด้วยมือขวา -> บันทึกเส้นขวา
    print("\n--- Test 4: วาดด้วยมือขวา -> บันทึกเส้นขวา ---")
    right_only_hands = create_mock_hand_data(left_tip=None, right_tip=(450, 200))
    drawing.update(right_only_hands, gesture.is_drawing_active, sim_time)
    for offset in [10, 20, 30]:
        sim_time += 0.05
        moving_right = create_mock_hand_data(left_tip=None, right_tip=(450 - offset, 200 + offset))
        drawing.update(moving_right, gesture.is_drawing_active, sim_time)

    assert drawing.get_total_points("Right") == 4, f"Expected 4 Right points, got {drawing.get_total_points('Right')}"
    print(f"  Right hand drew: {drawing.get_total_points('Right')} points recorded (PASS)")

    # TEST 5: วาดสองมือพร้อมกัน -> บันทึกทั้งสองเส้น
    print("\n--- Test 5: วาดสองมือพร้อมกัน -> บันทึกทั้งสองเส้น ---")
    prev_left_pts = drawing.get_total_points("Left")
    prev_right_pts = drawing.get_total_points("Right")

    for offset in [5, 10, 15]:
        sim_time += 0.05
        both_moving = create_mock_hand_data(left_tip=(250 + offset, 250), right_tip=(400 - offset, 250))
        drawing.update(both_moving, gesture.is_drawing_active, sim_time)

    assert drawing.get_total_points("Left") > prev_left_pts, "Left points should have increased"
    assert drawing.get_total_points("Right") > prev_right_pts, "Right points should have increased"
    print(f"  Both hands drew simultaneously: Left={drawing.get_total_points('Left')} pts, Right={drawing.get_total_points('Right')} pts (PASS)")

    # TEST 6: ประสานมือครั้งที่สอง -> หยุดวาด
    print("\n--- Test 6: ประสานมือครั้งที่สอง -> หยุดวาด ---")
    sim_time += 0.5  # past cooldown
    # Join hands again (Frame 1: detect join and start hold timer)
    joined_again = create_mock_hand_data(left_tip=(320, 240), right_tip=(322, 240))
    gesture.update(joined_again, sim_time)
    assert gesture.gesture_state == GestureState.JOIN_HOLDING, "Must enter JOIN_HOLDING on 2nd join"

    # Frame 2: Hold for 0.8s (> 0.7s) -> toggle to IDLE
    sim_time += 0.8
    changed2 = gesture.update(joined_again, sim_time)
    assert changed2, "State change must be reported on 2nd join"
    assert gesture.drawing_state == DrawingState.IDLE, "State must now be IDLE"
    assert gesture.last_event == "STOP_DRAWING", "Event must be STOP_DRAWING"
    assert not gesture.is_drawing_active, "is_drawing_active must be False"
    print("  Hold confirmed (2nd join): Transitioned from DRAWING_ACTIVE to IDLE! (PASS)")

    # TEST 7: เส้นที่วาดแล้วต้องยังอยู่
    print("\n--- Test 7: เส้นที่วาดแล้วต้องยังอยู่ ---")
    drawing.update(joined_again, gesture.is_drawing_active, sim_time)
    left_saved_pts = drawing.get_total_points("Left")
    right_saved_pts = drawing.get_total_points("Right")
    assert left_saved_pts > 0, "Left strokes must NOT be erased!"
    assert right_saved_pts > 0, "Right strokes must NOT be erased!"
    export = drawing.export_data()
    assert len(export["Left"]["strokes"]) > 0, "Left export strokes must be preserved"
    assert len(export["Right"]["strokes"]) > 0, "Right export strokes must be preserved"
    print(f"  Drawing stopped, but all data preserved: Left={left_saved_pts} pts, Right={right_saved_pts} pts (PASS)")

    # TEST 8 (Part B): ประสานมือค้างไว้ขณะ IDLE -> ต้องไม่สลับสถานะหลายครั้ง
    print("\n--- Test 8 (Part B): ประสานมือค้างไว้ขณะ IDLE -> ต้องไม่สลับสถานะหลายครั้ง ---")
    for _ in range(8):
        sim_time += 0.2
        changed = gesture.update(joined_again, sim_time)
        assert not changed, "Must NOT re-trigger while hands remain joined in IDLE!"
        assert gesture.drawing_state == DrawingState.IDLE, "Must stay in IDLE"
    print("  Hands kept joined in IDLE: Remained in IDLE without bouncing (PASS)")

    # TEST 9: สามารถประสานมือครั้งที่สามเพื่อเริ่มวาดรอบใหม่ได้
    print("\n--- Test 9: สามารถประสานมือครั้งที่สามเพื่อเริ่มวาดรอบใหม่ได้ ---")
    # First separate hands
    sim_time += 0.2
    gesture.update(separated_hands, sim_time)
    assert gesture.gesture_state == GestureState.NO_JOIN, "Lock cleared after separation"

    # Now join a third time (Frame 1: detect join and start hold timer)
    sim_time += 0.5  # past cooldown
    joined_third = create_mock_hand_data(left_tip=(320, 240), right_tip=(323, 240))
    gesture.update(joined_third, sim_time)
    assert gesture.gesture_state == GestureState.JOIN_HOLDING, "Must enter JOIN_HOLDING on 3rd join"

    # Frame 2: hold for 0.8s -> toggle back to DRAWING_ACTIVE
    sim_time += 0.8
    changed3 = gesture.update(joined_third, sim_time)
    assert changed3, "State change must be reported on 3rd join"
    assert gesture.drawing_state == DrawingState.DRAWING_ACTIVE, "State must return to DRAWING_ACTIVE"
    assert gesture.last_event == "START_DRAWING", "Event must be START_DRAWING"
    print("  3rd join confirmed: Transitioned back to DRAWING_ACTIVE for new drawing session! (PASS)")

    print("\n" + "=" * 70)
    print(" ALL 9 ACCEPTANCE CRITERIA PASSED WITHOUT ERROR!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = run_acceptance_tests()
    if not success:
        sys.exit(1)
