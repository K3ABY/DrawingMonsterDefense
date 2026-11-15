"""
Comprehensive Acceptance & Validation Test Suite for:
HAND TRACKING & INDEX FINGER RECOGNITION IMPROVEMENT
Drawing Monster Defense - Phase 1

Validates all 12 core requirements and failure modes:
  1. Index Pointing UP Recognition & Direction Classification
  2. Index Pointing FORWARD (toward camera, Z-axis) Recognition (previously failing in 2D)
  3. Index Pointing SIDEWAYS Recognition
  4. Index Pointing DOWN Recognition
  5. Index Pointing DIAGONAL Recognition
  6. Natural variation in unused fingers (relaxed curl tolerance)
  7. Accurate rejection of Open Palm and Fist postures
  8. One Euro Filter adaptive smoothing & fast movement responsiveness
  9. Tracking loss recovery (short-term smooth bridge vs prolonged break)
 10. Dual-Zone boundaries enforcement (Left X < 50%, Right X >= 50%)
 11. Fist Pause / Open Resume per hand independently
 12. Two-Hand Join Gesture Toggle Mode (debounced 0.7s hold)
"""

import os
import sys
import math
import time
import numpy as np
import cv2
from typing import Dict, Any, Tuple, List, Optional

# Ensure project root directory is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
for p in [current_dir, project_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

import camera_config as config
from drawing_manager import AirDrawingManager
from gesture_controller import TwoHandJoinGestureController, DrawingState, GestureState
from one_euro_filter import OneEuroFilter, HandLandmarkSmoother
from hand_tracking import HandTracker


def create_3d_landmarks(
    wrist: Tuple[float, float, float] = (150.0, 300.0, 0.0),
    posture: str = "UP",  # "UP", "FORWARD", "SIDEWAYS", "DOWN", "DIAGONAL", "RELAXED_POINTING", "OPEN", "FIST"
    scale: float = 60.0,
) -> Tuple[List[Tuple[int, int]], List[Tuple[float, float, float]]]:
    """
    Synthesizes mathematically accurate 3D and 2D landmark sets for MediaPipe hand tracking tests.
    Returns: (landmarks_2d_pixels, landmarks_3d_normalized)
    """
    wx, wy, wz = wrist
    lm3d: List[Tuple[float, float, float]] = [(wx, wy, wz)] * 21

    # Wrist
    lm3d[config.LANDMARK_WRIST] = (wx, wy, wz)

    # Base MCPs (Knuckles) - orient palm along general pointing direction
    if posture == "DOWN":
        idx_mcp = (wx - scale * 0.35, wy + scale * 0.95, wz)
        mid_mcp = (wx, wy + scale * 1.00, wz)
        rng_mcp = (wx + scale * 0.30, wy + scale * 0.92, wz)
        pnk_mcp = (wx + scale * 0.55, wy + scale * 0.80, wz)
        v_curl = (0.0, -scale * 0.50, 0.0)
    else:
        idx_mcp = (wx - scale * 0.35, wy - scale * 0.95, wz)
        mid_mcp = (wx, wy - scale * 1.00, wz)
        rng_mcp = (wx + scale * 0.30, wy - scale * 0.92, wz)
        pnk_mcp = (wx + scale * 0.55, wy - scale * 0.80, wz)
        v_curl = (0.0, scale * 0.50, 0.0)

    lm3d[config.LANDMARK_INDEX_MCP] = idx_mcp
    lm3d[config.LANDMARK_MIDDLE_MCP] = mid_mcp
    lm3d[config.LANDMARK_RING_MCP] = rng_mcp
    lm3d[config.LANDMARK_PINKY_MCP] = pnk_mcp

    # Helper to construct extended or curled finger
    def build_finger(mcp, v_ext, v_crl, is_ext):
        if is_ext:
            pip = (mcp[0] + v_ext[0] * 0.40, mcp[1] + v_ext[1] * 0.40, mcp[2] + v_ext[2] * 0.40)
            dip = (mcp[0] + v_ext[0] * 0.75, mcp[1] + v_ext[1] * 0.75, mcp[2] + v_ext[2] * 0.75)
            tip = (mcp[0] + v_ext[0] * 1.00, mcp[1] + v_ext[1] * 1.00, mcp[2] + v_ext[2] * 1.00)
        else:
            # Curled into palm towards wrist
            pip = (mcp[0] + v_crl[0] * 0.35, mcp[1] + v_crl[1] * 0.35, mcp[2] + v_crl[2] * 0.35)
            dip = (mcp[0] + v_crl[0] * 0.65, mcp[1] + v_crl[1] * 0.65, mcp[2] + v_crl[2] * 0.65)
            tip = (mcp[0] + v_crl[0] * 0.80, mcp[1] + v_crl[1] * 0.80, mcp[2] + v_crl[2] * 0.80)
        return pip, dip, tip

    # Index finger direction vectors
    v_up = (0.0, -scale * 1.10, 0.0)
    v_fwd = (0.0, -scale * 0.15, -scale * 1.10)
    v_side = (scale * 1.10, 0.0, 0.0)
    v_down = (0.0, scale * 1.10, 0.0)
    v_diag = (scale * 0.70, -scale * 0.70, -scale * 0.65)

    # Configure Index Finger
    if posture == "UP":
        pip, dip, tip = build_finger(idx_mcp, v_up, v_curl, is_ext=True)
    elif posture == "FORWARD":
        pip, dip, tip = build_finger(idx_mcp, v_fwd, v_curl, is_ext=True)
    elif posture == "SIDEWAYS":
        pip, dip, tip = build_finger(idx_mcp, v_side, v_curl, is_ext=True)
    elif posture == "DOWN":
        pip, dip, tip = build_finger(idx_mcp, v_down, v_curl, is_ext=True)
    elif posture == "DIAGONAL":
        pip, dip, tip = build_finger(idx_mcp, v_diag, v_curl, is_ext=True)
    elif posture == "RELAXED_POINTING":
        pip, dip, tip = build_finger(idx_mcp, v_up, v_curl, is_ext=True)
    elif posture == "OPEN":
        pip, dip, tip = build_finger(idx_mcp, v_up, v_curl, is_ext=True)
    elif posture == "FIST":
        pip, dip, tip = build_finger(idx_mcp, v_up, v_curl, is_ext=False)
    else:
        pip, dip, tip = build_finger(idx_mcp, v_up, v_curl, is_ext=True)

    lm3d[config.LANDMARK_INDEX_PIP] = pip
    lm3d[config.LANDMARK_INDEX_DIP] = dip
    lm3d[config.LANDMARK_INDEX_FINGER_TIP] = tip

    # Configure Non-Index Fingers (Middle, Ring, Pinky)
    if posture in ["UP", "FORWARD", "SIDEWAYS", "DOWN", "DIAGONAL"]:
        # Standard tight fist curl
        for mcp_i, pip_i, dip_i, tip_i, mcp in [
            (9, 10, 11, 12, mid_mcp),
            (13, 14, 15, 16, rng_mcp),
            (17, 18, 19, 20, pnk_mcp),
        ]:
            p, d, t = build_finger(mcp, v_up, v_curl, is_ext=False)
            lm3d[pip_i] = p
            lm3d[dip_i] = d
            lm3d[tip_i] = t
    elif posture == "RELAXED_POINTING":
        # Natural relaxed curl: fingers are partially curled (straightness ~0.65-0.70)
        v_relaxed_curl = (scale * 0.1, scale * 0.35, -scale * 0.2)
        for mcp_i, pip_i, dip_i, tip_i, mcp in [
            (9, 10, 11, 12, mid_mcp),
            (13, 14, 15, 16, rng_mcp),
            (17, 18, 19, 20, pnk_mcp),
        ]:
            p, d, t = build_finger(mcp, v_up, v_relaxed_curl, is_ext=False)
            lm3d[pip_i] = p
            lm3d[dip_i] = d
            lm3d[tip_i] = t
    elif posture == "OPEN":
        # All fingers extended
        for mcp_i, pip_i, dip_i, tip_i, mcp in [
            (9, 10, 11, 12, mid_mcp),
            (13, 14, 15, 16, rng_mcp),
            (17, 18, 19, 20, pnk_mcp),
        ]:
            p, d, t = build_finger(mcp, v_up, v_curl, is_ext=True)
            lm3d[pip_i] = p
            lm3d[dip_i] = d
            lm3d[tip_i] = t
    elif posture == "FIST":
        # All fingers curled tightly
        for mcp_i, pip_i, dip_i, tip_i, mcp in [
            (9, 10, 11, 12, mid_mcp),
            (13, 14, 15, 16, rng_mcp),
            (17, 18, 19, 20, pnk_mcp),
        ]:
            p, d, t = build_finger(mcp, v_up, v_curl, is_ext=False)
            lm3d[pip_i] = p
            lm3d[dip_i] = d
            lm3d[tip_i] = t

    # Convert to 2D pixel coordinates
    lm2d = [(int(p[0]), int(p[1])) for p in lm3d]

    return lm2d, lm3d


def make_mock_hand_dict(
    label: str,
    wrist: Tuple[float, float, float],
    posture: str = "UP",
    tip_override: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """Wraps synthetic landmarks in the complete hand_data dictionary structure."""
    lm2d, lm3d = create_3d_landmarks(wrist=wrist, posture=posture)
    if tip_override is not None:
        curr_tip = lm2d[config.LANDMARK_INDEX_FINGER_TIP]
        dx = tip_override[0] - curr_tip[0]
        dy = tip_override[1] - curr_tip[1]
        lm2d = [(p[0] + dx, p[1] + dy) for p in lm2d]
        lm3d = [(p[0] + dx, p[1] + dy, p[2]) for p in lm3d]

    tip = lm2d[config.LANDMARK_INDEX_FINGER_TIP]
    w = 640
    h = 480
    return {
        "label": label,
        "detected": True,
        "status": "Detected",
        "index_tip_x": tip[0],
        "index_tip_y": tip[1],
        "index_finger_tip": tip,
        "norm_tip": (float(tip[0] / w), float(tip[1] / h)),
        "landmarks": lm2d,
        "landmarks_3d": lm3d,
        "confidence": 0.94,
        "wrist_x": lm2d[config.LANDMARK_WRIST][0],
    }


def run_all_tests():
    print("=" * 72)
    print(" COMPREHENSIVE ACCEPTANCE SUITE: HAND TRACKING & INDEX RECOGNITION")
    print("=" * 72)

    passed_count = 0
    total_tests = 12

    # -------------------------------------------------------------------------
    # TEST 1: Index Pointing UP
    # -------------------------------------------------------------------------
    print("\n[TEST 1] Index Finger Pointing UP Recognition")
    _, lm3d = create_3d_landmarks(wrist=(150, 300, 0), posture="UP")
    is_pointing, is_fist, posture_str, direction = AirDrawingManager._analyze_posture_3d(lm3d)
    assert is_pointing is True, f"Failed: Expected pointing=True, got {is_pointing}"
    assert direction == "UP", f"Failed: Expected direction=UP, got {direction}"
    print(f"  Result: Posture={posture_str}, Direction={direction}, Pointing={is_pointing} (PASS)")
    passed_count += 1

    # -------------------------------------------------------------------------
    # TEST 2: Index Pointing FORWARD (toward camera, Z-axis)
    # -------------------------------------------------------------------------
    print("\n[TEST 2] Index Finger Pointing FORWARD (toward camera / Z-axis)")
    _, lm3d = create_3d_landmarks(wrist=(150, 300, 0), posture="FORWARD")
    is_pointing, is_fist, posture_str, direction = AirDrawingManager._analyze_posture_3d(lm3d)
    assert is_pointing is True, f"Failed: Expected pointing=True, got {is_pointing}"
    assert direction == "FORWARD", f"Failed: Expected direction=FORWARD, got {direction}"
    print(f"  Result: Posture={posture_str}, Direction={direction}, Pointing={is_pointing} (PASS)")
    passed_count += 1

    # -------------------------------------------------------------------------
    # TEST 3: Index Pointing SIDEWAYS
    # -------------------------------------------------------------------------
    print("\n[TEST 3] Index Finger Pointing SIDEWAYS")
    _, lm3d = create_3d_landmarks(wrist=(150, 300, 0), posture="SIDEWAYS")
    is_pointing, is_fist, posture_str, direction = AirDrawingManager._analyze_posture_3d(lm3d)
    assert is_pointing is True, f"Failed: Expected pointing=True, got {is_pointing}"
    assert direction == "SIDEWAYS", f"Failed: Expected direction=SIDEWAYS, got {direction}"
    print(f"  Result: Posture={posture_str}, Direction={direction}, Pointing={is_pointing} (PASS)")
    passed_count += 1

    # -------------------------------------------------------------------------
    # TEST 4: Index Pointing DOWN
    # -------------------------------------------------------------------------
    print("\n[TEST 4] Index Finger Pointing DOWN")
    _, lm3d = create_3d_landmarks(wrist=(150, 200, 0), posture="DOWN")
    is_pointing, is_fist, posture_str, direction = AirDrawingManager._analyze_posture_3d(lm3d)
    assert is_pointing is True, f"Failed: Expected pointing=True, got {is_pointing}"
    assert direction == "DOWN", f"Failed: Expected direction=DOWN, got {direction}"
    print(f"  Result: Posture={posture_str}, Direction={direction}, Pointing={is_pointing} (PASS)")
    passed_count += 1

    # -------------------------------------------------------------------------
    # TEST 5: Index Pointing DIAGONAL
    # -------------------------------------------------------------------------
    print("\n[TEST 5] Index Finger Pointing DIAGONAL (3D combo angles)")
    _, lm3d = create_3d_landmarks(wrist=(150, 300, 0), posture="DIAGONAL")
    is_pointing, is_fist, posture_str, direction = AirDrawingManager._analyze_posture_3d(lm3d)
    assert is_pointing is True, f"Failed: Expected pointing=True, got {is_pointing}"
    assert direction == "DIAGONAL", f"Failed: Expected direction=DIAGONAL, got {direction}"
    print(f"  Result: Posture={posture_str}, Direction={direction}, Pointing={is_pointing} (PASS)")
    passed_count += 1

    # -------------------------------------------------------------------------
    # TEST 6: Natural variation in unused fingers (Relaxed Curl tolerance)
    # -------------------------------------------------------------------------
    print("\n[TEST 6] Natural Variation in Unused Fingers (Relaxed Curl)")
    _, lm3d = create_3d_landmarks(wrist=(150, 300, 0), posture="RELAXED_POINTING")
    is_pointing, is_fist, posture_str, direction = AirDrawingManager._analyze_posture_3d(lm3d)
    assert is_pointing is True, f"Failed: Relaxed pointing must be recognized, got pointing={is_pointing}"
    assert not is_fist, "Relaxed pointing must not be classified as fist"
    print(f"  Result: Posture={posture_str}, Pointing={is_pointing} (Tolerates natural finger curvature) (PASS)")
    passed_count += 1

    # -------------------------------------------------------------------------
    # TEST 7: Accurate Rejection of Open Palm and Fist
    # -------------------------------------------------------------------------
    print("\n[TEST 7] Posture Discrimination: Open Palm and Fist Rejection")
    # 7a: Open Palm
    _, lm_open = create_3d_landmarks(wrist=(150, 300, 0), posture="OPEN")
    open_pointing, open_fist, open_posture, _ = AirDrawingManager._analyze_posture_3d(lm_open)
    assert open_pointing is False, f"Open Palm must NOT trigger pointing, got {open_pointing}"
    assert open_posture == "OPEN", f"Expected OPEN, got {open_posture}"

    # 7b: Fist
    _, lm_fist = create_3d_landmarks(wrist=(150, 300, 0), posture="FIST")
    fist_pointing, is_fist, fist_posture, _ = AirDrawingManager._analyze_posture_3d(lm_fist)
    assert fist_pointing is False, f"Fist must NOT trigger pointing, got {fist_pointing}"
    assert is_fist is True, f"Expected is_fist=True, got {is_fist}"
    assert fist_posture == "FIST", f"Expected FIST, got {fist_posture}"
    print(f"  Result: Open Palm classified as '{open_posture}' (pointing=False); Fist classified as '{fist_posture}' (fist=True) (PASS)")
    passed_count += 1

    # -------------------------------------------------------------------------
    # TEST 8: One Euro Filter Adaptive Smoothing & Fast Movement Latency
    # -------------------------------------------------------------------------
    print("\n[TEST 8] One Euro Filter Smoothing & Low-Latency Fast Tracking")
    smoother = HandLandmarkSmoother(
        min_cutoff=config.ONE_EURO_MIN_CUTOFF,
        beta=config.ONE_EURO_BETA,
        d_cutoff=config.ONE_EURO_DCUTOFF,
    )
    # Stationary jitter test
    np.random.seed(42)
    stationary_pts = [(200 + np.random.normal(0, 2.5), 200 + np.random.normal(0, 2.5)) for _ in range(60)]
    t = 100.0
    smoothed_pts = []
    for pt in stationary_pts:
        sm = smoother.smooth_index_tip((int(pt[0]), int(pt[1])), t)
        smoothed_pts.append(sm)
        t += 0.033

    raw_std = float(np.std([p[0] for p in stationary_pts]))
    sm_std = float(np.std([p[0] for p in smoothed_pts[10:]]))
    assert sm_std < raw_std * 0.65, f"Smoothing failed: raw_std={raw_std:.2f}, sm_std={sm_std:.2f}"

    # Fast movement lag test (speed ~ 750 px/s)
    fast_t = 200.0
    for step in range(30):
        fast_x = 100 + step * 25
        fast_y = 100 + step * 25
        sm_fast = smoother.smooth_index_tip((fast_x, fast_y), fast_t)
        fast_t += 0.033
    lag = math.hypot(fast_x - sm_fast[0], fast_y - sm_fast[1])
    assert lag < 12.0, f"Fast movement lag excessive: {lag:.2f}px"
    print(f"  Result: Stationary Jitter std dev reduced from {raw_std:.2f}px to {sm_std:.2f}px; Fast Tracking lag = {lag:.2f}px (< 12px) (PASS)")
    passed_count += 1

    # -------------------------------------------------------------------------
    # TEST 9: Tracking Loss Recovery (Bridge Short Drop vs Break Long Drop)
    # -------------------------------------------------------------------------
    print("\n[TEST 9] Tracking Loss Recovery: Short-term Bridge vs Long-term Break")
    mgr = AirDrawingManager()
    t_sim = 300.0

    # Draw 2 points (consecutive 30 FPS frames)
    h_data = {"Left": make_mock_hand_dict("Left", (150, 300, 0), posture="UP", tip_override=(150, 200)), "Right": None}
    mgr.update(h_data, is_drawing_active=True, current_time=t_sim)
    t_sim += 0.033
    h_data["Left"] = make_mock_hand_dict("Left", (150, 300, 0), posture="UP", tip_override=(160, 200))
    mgr.update(h_data, is_drawing_active=True, current_time=t_sim)
    assert mgr.get_total_points("Left") == 2

    # Drop hand for 0.12s (<= MAX_BRIDGE_GAP_TIME 0.22s) -> Fast recovery
    t_sim += 0.12
    h_data["Left"] = make_mock_hand_dict("Left", (150, 300, 0), posture="UP", tip_override=(175, 200))
    mgr.update(h_data, is_drawing_active=True, current_time=t_sim)
    assert mgr.get_total_strokes("Left") == 1, "Short tracking loss must bridge same stroke"
    assert mgr.get_total_points("Left") == 4, "Bridged stroke should insert intermediate point"

    # Drop hand for 0.60s (> MAX_BRIDGE_GAP_TIME) -> Stroke breaks cleanly!
    t_sim += 0.60
    h_data["Left"] = make_mock_hand_dict("Left", (150, 300, 0), posture="UP", tip_override=(250, 200))
    mgr.update(h_data, is_drawing_active=True, current_time=t_sim)
    assert mgr.get_total_strokes("Left") == 2, "Prolonged tracking loss must start new stroke segment"
    print(f"  Result: 0.12s loss seamlessly bridged; 0.60s loss cleanly broke stroke without cross-screen lines (PASS)")
    passed_count += 1

    # -------------------------------------------------------------------------
    # TEST 10: Dual-Zone Boundaries Enforcement
    # -------------------------------------------------------------------------
    print("\n[TEST 10] Dual-Zone Boundaries Enforcement (Left X < 50%, Right X >= 50%)")
    mgr_dz = AirDrawingManager()
    t_dz = 400.0

    # Left hand inside Left Zone (X=200 < 320)
    h_dz = {"Left": make_mock_hand_dict("Left", (150, 300, 0), posture="UP", tip_override=(200, 200)), "Right": None}
    mgr_dz.update(h_dz, is_drawing_active=True, current_time=t_dz)
    assert mgr_dz.boundary_status["Left"] == "INSIDE_ZONE"
    assert mgr_dz.get_total_points("Left") == 1

    # Left hand violates boundary into Right Zone (X=400 >= 320)
    t_dz += 0.05
    h_dz["Left"] = make_mock_hand_dict("Left", (400, 300, 0), posture="UP", tip_override=(400, 200))
    mgr_dz.update(h_dz, is_drawing_active=True, current_time=t_dz)
    assert mgr_dz.boundary_status["Left"] == "OUT_OF_ZONE"
    assert mgr_dz.get_total_points("Left") == 1, "Left hand must NOT create points in Right Zone"

    # Right hand inside Right Zone (X=480 >= 320)
    h_dz["Right"] = make_mock_hand_dict("Right", (480, 300, 0), posture="UP", tip_override=(480, 200))
    mgr_dz.update(h_dz, is_drawing_active=True, current_time=t_dz)
    assert mgr_dz.boundary_status["Right"] == "INSIDE_ZONE"
    assert mgr_dz.get_total_points("Right") == 1

    # Right hand violates boundary into Left Zone (X=150 < 320)
    t_dz += 0.05
    h_dz["Right"] = make_mock_hand_dict("Right", (150, 300, 0), posture="UP", tip_override=(150, 200))
    mgr_dz.update(h_dz, is_drawing_active=True, current_time=t_dz)
    assert mgr_dz.boundary_status["Right"] == "OUT_OF_ZONE"
    assert mgr_dz.get_total_points("Right") == 1, "Right hand must NOT create points in Left Zone"
    print(f"  Result: Boundary strictly enforced; cross-zone drawing prevented 100% (PASS)")
    passed_count += 1

    # -------------------------------------------------------------------------
    # TEST 11: Fist Pause / Open Resume per Hand Independently
    # -------------------------------------------------------------------------
    print("\n[TEST 11] Independent Fist Pause / Resume per Hand")
    mgr_fist = AirDrawingManager()
    t_f = 500.0

    # Both hands pointing in their zones
    h_f = {
        "Left": make_mock_hand_dict("Left", (150, 300, 0), posture="UP", tip_override=(150, 200)),
        "Right": make_mock_hand_dict("Right", (450, 300, 0), posture="UP", tip_override=(450, 200)),
    }
    mgr_fist.update(h_f, is_drawing_active=True, current_time=t_f)
    t_f += 0.05
    h_f["Left"] = make_mock_hand_dict("Left", (150, 300, 0), posture="UP", tip_override=(160, 200))
    h_f["Right"] = make_mock_hand_dict("Right", (450, 300, 0), posture="UP", tip_override=(460, 200))
    mgr_fist.update(h_f, is_drawing_active=True, current_time=t_f)
    assert mgr_fist.get_total_points("Left") == 2
    assert mgr_fist.get_total_points("Right") == 2

    # Left hand makes Fist (Pauses); Right hand continues pointing (confirmed across debounce frames)
    for _ in range(config.FIST_CONFIRM_FRAMES):
        t_f += 0.033
        h_f["Left"] = make_mock_hand_dict("Left", (150, 300, 0), posture="FIST", tip_override=(160, 200))
        h_f["Right"] = make_mock_hand_dict("Right", (450, 300, 0), posture="UP", tip_override=(470, 200))
        mgr_fist.update(h_f, is_drawing_active=True, current_time=t_f)
    assert mgr_fist.get_total_points("Left") == 2, "Left hand in Fist must pause drawing"
    assert mgr_fist.get_drawing_state_name("Left", True, True) == "FIST PAUSED"
    assert mgr_fist.get_drawing_state_name("Right", True, True) == "DRAWING"

    # Left hand uncurls and points again (Resumes drawing as clean stroke, confirmed across debounce frames)
    for _ in range(config.POINTING_CONFIRM_FRAMES):
        t_f += 0.033
        h_f["Left"] = make_mock_hand_dict("Left", (150, 300, 0), posture="UP", tip_override=(175, 200))
        h_f["Right"] = make_mock_hand_dict("Right", (450, 300, 0), posture="UP", tip_override=(480, 200))
        mgr_fist.update(h_f, is_drawing_active=True, current_time=t_f)
    assert mgr_fist.get_total_points("Left") >= 3, "Left hand opening must resume drawing"
    print(f"  Result: Left hand paused on Fist ({mgr_fist.get_total_points('Left')} pts) while Right drew ({mgr_fist.get_total_points('Right')} pts); Resumed cleanly (PASS)")
    passed_count += 1

    # -------------------------------------------------------------------------
    # TEST 12: Two-Hand Join Gesture Toggle Mode
    # -------------------------------------------------------------------------
    print("\n[TEST 12] Two-Hand Join Gesture Toggle Mode (~0.7s debounced hold)")
    ctrl = TwoHandJoinGestureController()
    t_g = 600.0

    # Initially IDLE
    assert ctrl.drawing_state == DrawingState.IDLE

    # Join hands (tips touching at center: Left=310, Right=330)
    join_data = {
        "Left": make_mock_hand_dict("Left", (280, 300, 0), posture="UP", tip_override=(310, 240)),
        "Right": make_mock_hand_dict("Right", (360, 300, 0), posture="UP", tip_override=(330, 240)),
    }
    # Hold for 0.4s (< 0.7s hold threshold) -> Must NOT toggle yet!
    ctrl.update(join_data, t_g)
    ctrl.update(join_data, t_g + 0.40)
    assert ctrl.drawing_state == DrawingState.IDLE, "Must not toggle before hold threshold"
    assert ctrl.gesture_state == GestureState.JOIN_HOLDING

    # Hold until 0.80s (>= 0.7s threshold) -> Triggers DRAWING_ACTIVE!
    ctrl.update(join_data, t_g + 0.80)
    assert ctrl.drawing_state == DrawingState.DRAWING_ACTIVE, "Must toggle to DRAWING_ACTIVE on completed hold"
    assert ctrl.last_event == "START_DRAWING"

    # Release hands
    sep_data = {
        "Left": make_mock_hand_dict("Left", (150, 300, 0), posture="UP", tip_override=(150, 200)),
        "Right": make_mock_hand_dict("Right", (450, 300, 0), posture="UP", tip_override=(450, 200)),
    }
    ctrl.update(sep_data, t_g + 1.20)
    assert ctrl.is_drawing_active is True, "Drawing must stay ACTIVE after separating hands"

    # Rejoin hands to deactivate (Hold for 0.80s)
    t_g2 = t_g + 2.0
    ctrl.update(join_data, t_g2)
    ctrl.update(join_data, t_g2 + 0.80)
    assert ctrl.drawing_state == DrawingState.IDLE, "Must toggle to IDLE on second completed hold"
    assert ctrl.last_event == "STOP_DRAWING"
    print(f"  Result: Toggle 1 (0.8s) -> DRAWING_ACTIVE; Separated -> Remains ACTIVE; Toggle 2 (0.8s) -> IDLE (PASS)")
    passed_count += 1

    print("\n" + "=" * 72)
    print(f" ALL {passed_count}/{total_tests} TESTS PASSED SUCCESSFULLY (100% GREEN)!")
    print("=" * 72)
    return True


if __name__ == "__main__":
    success = run_all_tests()
    if not success:
        sys.exit(1)
