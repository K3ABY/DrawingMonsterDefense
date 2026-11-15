"""
Comprehensive Robustness & Adaptive Image Quality Validation Suite.
Project: Drawing Monster Defense
Phase: Robust Camera Image Quality & Adaptive Hand Tracking

Tests all 10 Required Scenarios (Tests A through J):
  - Test A: Normal lighting
  - Test B: Dim / Low lighting
  - Test C: Strong backlighting
  - Test D: Low contrast / Haze
  - Test E: Fast hand movement (750 px/s)
  - Test F: Motion blur
  - Test G: Camera noise & grain
  - Test H: Partially hidden / occluded fingers
  - Test I: Sudden brightness change
  - Test J: Both hands moving simultaneously

Evaluates:
  - Image Quality Classification (ImageQualityAssessor)
  - Adaptive Enhancement latency and visual restoration (AdaptiveImageEnhancer)
  - Confidence-Aware Tracking State Transitions (TRACKING_STABLE, TRACKING_UNCERTAIN, TRACKING_LOST, TRACKING_RECOVERED)
  - Safe Air Drawing (No false strokes, clean breaks, no cross-screen lines)
  - Frame latency (< 20 ms) and throughput (>= 30 FPS)
"""

import os
import sys
import math
import time
from typing import Dict, Any, Tuple, List, Optional
import cv2
import numpy as np

# Ensure project directory is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import camera_config as config
from image_quality_assessor import ImageQualityAssessor, ImageCondition, ImageQualityMetrics
from adaptive_enhancer import AdaptiveImageEnhancer
from drawing_manager import AirDrawingManager, TrackingConfidenceState
from one_euro_filter import HandLandmarkSmoother


def create_synthetic_frame(
    condition: str = "NORMAL",
    w: int = 640,
    h: int = 480,
    hand_pos: Tuple[int, int] = (200, 240),
) -> np.ndarray:
    """
    Synthesizes test camera frames representing various real-world camera conditions.
    Draws a realistic room background and skin-tone hand silhouette with fingers.
    """
    # Realistic indoor room scene: upper wall (160), desk (80), floor (45)
    img = np.full((h, w, 3), 160, dtype=np.uint8)
    img[int(h * 0.55) : int(h * 0.75), :] = 80   # Desk surface
    img[int(h * 0.75) :, :] = 45                 # Floor

    # Add room elements (door / picture frame / window)
    cv2.rectangle(img, (40, 40), (140, 220), (110, 110, 110), -1)
    cv2.rectangle(img, (460, 60), (580, 180), (210, 210, 210), -1)

    # Draw synthetic hand silhouette (palm + index finger)
    hx, hy = hand_pos
    # Palm
    cv2.ellipse(img, (hx, hy + 40), (45, 55), 0, 0, 360, (170, 190, 220), -1)
    # Pointing index finger
    cv2.line(img, (hx, hy + 20), (hx, hy - 40), (160, 185, 215), 18)
    cv2.circle(img, (hx, hy - 40), 9, (160, 185, 215), -1)

    # Curled fingers
    for dx in [-24, 22, 42]:
        cv2.line(img, (hx + dx, hy + 30), (hx + dx, hy + 10), (150, 175, 205), 16)

    # Apply degradation based on condition
    if condition == "NORMAL":
        pass  # Standard clear indoor scene (B ~ 125, Contrast ~ 55, Sharpness ~ 120)

    elif condition == "TOO_DARK":
        # Dim ambient light: scale luminance down so mean is ~35-45
        img = (img * 0.28).astype(np.uint8)

    elif condition == "TOO_BRIGHT":
        # Overexposed scene: scale luminance up so mean is > 200
        img = np.clip(img.astype(np.float32) * 1.5 + 80, 0, 255).astype(np.uint8)

    elif condition == "BACKLIT":
        # Bright window border (luminance ~220), dark silhouette center (luminance ~45)
        img = np.full((h, w, 3), 220, dtype=np.uint8)
        # Dark silhouette in center
        margin_y, margin_x = h // 4, w // 4
        img[margin_y : h - margin_y, margin_x : w - margin_x] = 45
        # Draw dark hand silhouette in center
        cv2.ellipse(img, (hx, hy + 40), (45, 55), 0, 0, 360, (40, 40, 40), -1)
        cv2.line(img, (hx, hy + 20), (hx, hy - 40), (35, 35, 35), 18)

    elif condition == "LOW_CONTRAST":
        # Fog, haze, or dirty lens: compress dynamic range heavily around 125
        gray_center = 125.0
        img = np.clip((img.astype(np.float32) - gray_center) * 0.18 + gray_center, 0, 255).astype(np.uint8)

    elif condition == "MOTION_BLURRED":
        # Heavy camera and hand motion blur: 2D blur + directional smear
        img = cv2.GaussianBlur(img, (15, 15), 0)
        kernel_size = 15
        kernel_motion = np.zeros((kernel_size, kernel_size))
        kernel_motion[int((kernel_size - 1) / 2), :] = np.ones(kernel_size) / kernel_size
        img = cv2.filter2D(img, -1, kernel_motion)

    elif condition == "NOISY":
        # High ISO camera sensor grain: Gaussian noise with sigma = 14
        noise = np.random.normal(0, 14, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return img


def make_test_landmarks_3d(
    wrist: Tuple[float, float, float] = (180.0, 300.0, 0.0),
    posture: str = "POINTING",
    occlude_index: bool = False,
) -> List[Tuple[float, float, float]]:
    """Synthesizes normalized 3D landmarks for confidence & safe drawing testing."""
    wx, wy, wz = wrist
    lm = [(wx, wy, wz)] * 21

    # Wrist
    lm[config.LANDMARK_WRIST] = (wx, wy, wz)
    # Palm MCPs
    lm[config.LANDMARK_INDEX_MCP] = (wx, wy - 50, wz)
    lm[config.LANDMARK_MIDDLE_MCP] = (wx + 22, wy - 50, wz)
    lm[config.LANDMARK_RING_MCP] = (wx + 44, wy - 46, wz)
    lm[config.LANDMARK_PINKY_MCP] = (wx + 62, wy - 40, wz)

    if posture == "POINTING":
        if occlude_index:
            # Partially occluded index finger: joint landmark distortion & reduced extension
            lm[config.LANDMARK_INDEX_PIP] = (wx, wy - 60, wz + 5)
            lm[config.LANDMARK_INDEX_DIP] = (wx, wy - 80, wz + 10)
            lm[config.LANDMARK_INDEX_FINGER_TIP] = (wx, wy - 90, wz + 12)
        else:
            # Fully extended straight index finger pointing UP
            lm[config.LANDMARK_INDEX_PIP] = (wx, wy - 85, wz - 5)
            lm[config.LANDMARK_INDEX_DIP] = (wx, wy - 120, wz - 8)
            lm[config.LANDMARK_INDEX_FINGER_TIP] = (wx, wy - 150, wz - 10)

        # Curled middle, ring, pinky
        for mcp, pip, dip, tip in [(9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]:
            bx = lm[mcp][0]
            lm[pip] = (bx, wy - 25, wz + 15)
            lm[dip] = (bx, wy - 10, wz + 20)
            lm[tip] = (bx, wy - 5, wz + 10)

    elif posture == "FIST":
        for mcp, pip, dip, tip in [(5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]:
            bx = lm[mcp][0]
            lm[pip] = (bx, wy - 25, wz + 15)
            lm[dip] = (bx, wy - 10, wz + 20)
            lm[tip] = (bx, wy - 5, wz + 10)

    return lm


def run_robustness_test_suite():
    print("=" * 74)
    print(" ROBUST CAMERA IMAGE QUALITY & ADAPTIVE HAND TRACKING TEST SUITE")
    print("=" * 74)

    np.random.seed(42)
    assessor = ImageQualityAssessor()
    enhancer = AdaptiveImageEnhancer()
    all_passed = True
    test_results = {}

    # -------------------------------------------------------------------------
    # TEST A: Normal Lighting
    # -------------------------------------------------------------------------
    print("\n[TEST A] Normal Lighting Scenario")
    frame_a = create_synthetic_frame("NORMAL")
    metrics_a = assessor.assess(frame_a)
    enhanced_a, enh_info_a = enhancer.enhance(frame_a, metrics_a)

    assert metrics_a.condition == ImageCondition.NORMAL, f"Expected NORMAL, got {metrics_a.condition}"
    assert enh_info_a["applied"] is False, "Normal frames must bypass enhancement (zero-cost)"
    assert enh_info_a["method"] == "BYPASS"
    print(f"  Assessment: {metrics_a.description} ({metrics_a.assessment_time_ms:.2f} ms)")
    print(f"  Enhancement: {enh_info_a['method']} (Bypass time: {enh_info_a['time_ms']:.2f} ms) (PASS)")
    test_results["Test A (Normal)"] = "PASS"

    # -------------------------------------------------------------------------
    # TEST B: Dim / Low Lighting
    # -------------------------------------------------------------------------
    print("\n[TEST B] Dim / Low Lighting Scenario")
    assessor.reset()
    frame_b = create_synthetic_frame("TOO_DARK")
    metrics_b = assessor.assess(frame_b)
    enhanced_b, enh_info_b = enhancer.enhance(frame_b, metrics_b)

    assert metrics_b.condition == ImageCondition.TOO_DARK, f"Expected TOO_DARK, got {metrics_b.condition}"
    assert enh_info_b["applied"] is True
    assert enh_info_b["method"] == "GAMMA_CORRECTION"
    # Check that shadow luminance was lifted
    b_orig = float(np.mean(frame_b))
    b_enh = float(np.mean(enhanced_b))
    assert b_enh > b_orig * 1.5, f"Gamma LUT failed to lift shadows: {b_orig:.1f} -> {b_enh:.1f}"
    print(f"  Assessment: {metrics_b.description} ({metrics_b.assessment_time_ms:.2f} ms)")
    print(f"  Enhancement: {enh_info_b['method']} - Mean luminance lifted {b_orig:.1f} -> {b_enh:.1f} ({enh_info_b['time_ms']:.2f} ms) (PASS)")
    test_results["Test B (Dim Lighting)"] = "PASS"

    # -------------------------------------------------------------------------
    # TEST C: Strong Backlighting
    # -------------------------------------------------------------------------
    print("\n[TEST C] Strong Backlighting Scenario")
    assessor.reset()
    frame_c = create_synthetic_frame("BACKLIT")
    metrics_c = assessor.assess(frame_c)
    enhanced_c, enh_info_c = enhancer.enhance(frame_c, metrics_c)

    assert metrics_c.condition == ImageCondition.BACKLIT, f"Expected BACKLIT, got {metrics_c.condition}"
    assert enh_info_c["applied"] is True
    assert enh_info_c["method"] == "BACKLIGHT_RECOVERY"
    print(f"  Assessment: {metrics_c.description} ({metrics_c.assessment_time_ms:.2f} ms)")
    print(f"  Enhancement: {enh_info_c['method']} (CLAHE+Gamma time: {enh_info_c['time_ms']:.2f} ms) (PASS)")
    test_results["Test C (Backlighting)"] = "PASS"

    # -------------------------------------------------------------------------
    # TEST D: Low Contrast / Haze
    # -------------------------------------------------------------------------
    print("\n[TEST D] Low Contrast / Haze Scenario")
    assessor.reset()
    frame_d = create_synthetic_frame("LOW_CONTRAST")
    metrics_d = assessor.assess(frame_d)
    enhanced_d, enh_info_d = enhancer.enhance(frame_d, metrics_d)

    assert metrics_d.condition == ImageCondition.LOW_CONTRAST, f"Expected LOW_CONTRAST, got {metrics_d.condition}"
    assert enh_info_d["applied"] is True
    assert enh_info_d["method"] == "CLAHE_LOCAL_CONTRAST"
    std_orig = float(np.std(frame_d))
    std_enh = float(np.std(enhanced_d))
    assert std_enh > std_orig * 1.3, f"CLAHE failed to expand contrast: {std_orig:.1f} -> {std_enh:.1f}"
    print(f"  Assessment: {metrics_d.description} ({metrics_d.assessment_time_ms:.2f} ms)")
    print(f"  Enhancement: {enh_info_d['method']} - Contrast expanded {std_orig:.1f} -> {std_enh:.1f} ({enh_info_d['time_ms']:.2f} ms) (PASS)")
    test_results["Test D (Low Contrast)"] = "PASS"

    # -------------------------------------------------------------------------
    # TEST E: Fast Hand Movement (750 px/s)
    # -------------------------------------------------------------------------
    print("\n[TEST E] Fast Hand Movement Scenario (750 px/s)")
    smoother = HandLandmarkSmoother(min_cutoff=1.2, beta=0.028)
    fast_t = 100.0
    fast_pts = []
    smoothed_pts = []
    for step in range(25):
        # 25 px per frame at 30 FPS = 750 px/s
        px = 120 + step * 25
        py = 180 + step * 20
        fast_pts.append((px, py))
        sm = smoother.smooth_index_tip((px, py), fast_t)
        smoothed_pts.append(sm)
        fast_t += 0.033

    final_lag = math.hypot(fast_pts[-1][0] - smoothed_pts[-1][0], fast_pts[-1][1] - smoothed_pts[-1][1])
    assert final_lag < 12.0, f"Lag during fast hand movement exceeded limit: {final_lag:.2f}px"
    print(f"  One Euro Adaptive Tracking: Final lag at 750 px/s = {final_lag:.2f} px (< 12.0 px target) (PASS)")
    test_results["Test E (Fast Movement)"] = "PASS"

    # -------------------------------------------------------------------------
    # TEST F: Motion Blur Detection & Controlled Sharpening
    # -------------------------------------------------------------------------
    print("\n[TEST F] Motion Blur Scenario")
    assessor.reset()
    # Feed previous frame with moving hand at (160, 240) to trigger inter-frame motion delta
    frame_f_prev = create_synthetic_frame("NORMAL", hand_pos=(160, 240))
    assessor.prev_gray = cv2.cvtColor(cv2.resize(frame_f_prev, (160, 120)), cv2.COLOR_BGR2GRAY)
    frame_f = create_synthetic_frame("MOTION_BLURRED", hand_pos=(240, 240))
    metrics_f = assessor.assess(frame_f)
    enhanced_f, enh_info_f = enhancer.enhance(frame_f, metrics_f)

    assert metrics_f.condition == ImageCondition.MOTION_BLURRED, f"Expected MOTION_BLURRED, got {metrics_f.condition}"
    assert enh_info_f["applied"] is True
    assert enh_info_f["method"] == "UNSHARP_MASK_SHARPEN"
    print(f"  Assessment: {metrics_f.description} ({metrics_f.assessment_time_ms:.2f} ms)")
    print(f"  Enhancement: {enh_info_f['method']} (Unsharp mask time: {enh_info_f['time_ms']:.2f} ms) (PASS)")
    test_results["Test F (Motion Blur)"] = "PASS"

    # -------------------------------------------------------------------------
    # TEST G: Camera Noise & Grain Suppression
    # -------------------------------------------------------------------------
    print("\n[TEST G] Camera Noise & Grain Scenario")
    assessor.reset()
    frame_g = create_synthetic_frame("NOISY")
    metrics_g = assessor.assess(frame_g)
    enhanced_g, enh_info_g = enhancer.enhance(frame_g, metrics_g)

    assert metrics_g.condition == ImageCondition.NOISY, f"Expected NOISY, got {metrics_g.condition}"
    assert enh_info_g["applied"] is True
    assert enh_info_g["method"] == "BILATERAL_DENOISE"
    print(f"  Assessment: {metrics_g.description} ({metrics_g.assessment_time_ms:.2f} ms)")
    print(f"  Enhancement: {enh_info_g['method']} (Bilateral denoise time: {enh_info_g['time_ms']:.2f} ms) (PASS)")
    test_results["Test G (Camera Noise)"] = "PASS"

    # -------------------------------------------------------------------------
    # TEST H: Partially Hidden Fingers & Safe Drawing Confidence State
    # -------------------------------------------------------------------------
    print("\n[TEST H] Partially Hidden / Occluded Fingers & Confidence Gating")
    mgr_h = AirDrawingManager()
    t_h = 200.0

    # 1. Normal pointing hand (Frame 1 & 2): Stable drawing
    lm_pointing = make_test_landmarks_3d(wrist=(180, 300, 0), posture="POINTING", occlude_index=False)
    h_data_norm = {
        "Left": {
            "label": "Left",
            "detected": True,
            "confidence": 0.92,
            "landmarks_3d": lm_pointing,
            "index_finger_tip": (180, 150),
        },
        "Right": None,
    }
    mgr_h.update(h_data_norm, is_drawing_active=True, current_time=t_h)
    t_h += 0.033
    h_data_norm["Left"]["index_finger_tip"] = (185, 155)
    mgr_h.update(h_data_norm, is_drawing_active=True, current_time=t_h)
    assert mgr_h.tracking_confidence_state["Left"] == TrackingConfidenceState.TRACKING_STABLE
    assert mgr_h.get_total_points("Left") == 2

    # 2. Index finger becomes partially occluded (Frame 3):
    # Confidence drops into UNCERTAIN (0.35 - 0.60). Safe drawing pauses; no wild points added!
    t_h += 0.033
    lm_occluded = make_test_landmarks_3d(wrist=(180, 300, 0), posture="POINTING", occlude_index=True)
    h_data_occ = {
        "Left": {
            "label": "Left",
            "detected": True,
            "confidence": 0.45,  # Weakened detection confidence under occlusion
            "landmarks_3d": lm_occluded,
            "index_finger_tip": (310, 80),  # Wild jumping coordinate during occlusion
        },
        "Right": None,
    }
    mgr_h.update(h_data_occ, is_drawing_active=True, current_time=t_h)
    assert mgr_h.tracking_confidence_state["Left"] == TrackingConfidenceState.TRACKING_UNCERTAIN
    assert mgr_h.get_drawing_state_name("Left", True, True) == "TRACKING UNCERTAIN (PAUSED)"
    assert mgr_h.get_total_points("Left") == 2, "Uncertain state must NOT add wild erratic points!"

    # 3. Finger un-occluded and tracking recovers (Frame 4 & 5):
    # Transitions through RECOVERED confirmation frames to STABLE
    t_h += 0.033
    h_data_norm["Left"]["index_finger_tip"] = (190, 160)
    h_data_norm["Left"]["confidence"] = 0.94
    mgr_h.update(h_data_norm, is_drawing_active=True, current_time=t_h)
    # Confirmation frame 1: RECOVERED
    assert mgr_h.tracking_confidence_state["Left"] == TrackingConfidenceState.TRACKING_RECOVERED
    assert mgr_h.get_total_points("Left") == 2, "Must await confirmation before adding points"

    # Confirmation frame 2: STABLE -> Resumes drawing cleanly!
    t_h += 0.033
    h_data_norm["Left"]["index_finger_tip"] = (195, 165)
    mgr_h.update(h_data_norm, is_drawing_active=True, current_time=t_h)
    assert mgr_h.tracking_confidence_state["Left"] == TrackingConfidenceState.TRACKING_STABLE
    assert mgr_h.get_total_points("Left") >= 3, "Resumed drawing after confirmation!"
    print(f"  Confidence Gating: STABLE -> UNCERTAIN (Paused 0 pts) -> RECOVERED -> STABLE (Resumed clean) (PASS)")
    test_results["Test H (Occlusion & Safe Gating)"] = "PASS"

    # -------------------------------------------------------------------------
    # TEST I: Sudden Brightness Change (Sequence: Normal -> Dark -> Bright -> Normal)
    # -------------------------------------------------------------------------
    print("\n[TEST I] Sudden Brightness Change Sequence")
    sequence = [
        ("NORMAL", ImageCondition.NORMAL, "BYPASS"),
        ("TOO_DARK", ImageCondition.TOO_DARK, "GAMMA_CORRECTION"),
        ("TOO_BRIGHT", ImageCondition.TOO_BRIGHT, "EXPOSURE_ATTENUATION"),
        ("NORMAL", ImageCondition.NORMAL, "BYPASS"),
    ]

    for label, expected_cond, expected_method in sequence:
        f = create_synthetic_frame(label)
        t_start = time.perf_counter()
        m = assessor.assess(f)
        enh_f, enh_res = enhancer.enhance(f, m)
        dt = (time.perf_counter() - t_start) * 1000.0

        assert m.condition == expected_cond, f"Failed step {label}: expected {expected_cond}, got {m.condition}"
        assert enh_res["method"] == expected_method, f"Failed step {label}: expected {expected_method}, got {enh_res['method']}"
        assert dt < 12.0, f"Frame processing exceeded 12.0 ms: {dt:.2f} ms"
        print(f"  Step '{label}': Assessed {m.condition.value} -> Applied {enh_res['method']} in {dt:.2f} ms (OK)")

    print("  Sudden brightness changes handled adaptively with zero state lock or stall! (PASS)")
    test_results["Test I (Sudden Brightness)"] = "PASS"

    # -------------------------------------------------------------------------
    # TEST J: Both Hands Moving Simultaneously in Respective Dual Zones
    # -------------------------------------------------------------------------
    print("\n[TEST J] Both Hands Moving Simultaneously (Dual-Zone Parallel Operation)")
    mgr_j = AirDrawingManager()
    t_j = 300.0

    # Left hand in Left zone (X: 100 -> 240, inside X < 320)
    # Right hand in Right zone (X: 380 -> 520, inside X >= 320)
    total_frames = 20
    t_pipeline_total = 0.0

    for i in range(total_frames):
        lx = 100 + i * 7
        ly = 200 + int(20 * math.sin(i * 0.4))
        rx = 380 + i * 7
        ry = 220 + int(20 * math.cos(i * 0.4))

        lm_left = make_test_landmarks_3d(wrist=(lx, ly + 80, 0), posture="POINTING")
        lm_right = make_test_landmarks_3d(wrist=(rx, ry + 80, 0), posture="POINTING")

        h_data_both = {
            "Left": {
                "label": "Left",
                "detected": True,
                "confidence": 0.93,
                "landmarks_3d": lm_left,
                "index_finger_tip": (lx, ly),
            },
            "Right": {
                "label": "Right",
                "detected": True,
                "confidence": 0.94,
                "landmarks_3d": lm_right,
                "index_finger_tip": (rx, ry),
            },
        }

        t_sim_start = time.perf_counter()
        mgr_j.update(h_data_both, is_drawing_active=True, current_time=t_j)
        t_pipeline_total += (time.perf_counter() - t_sim_start) * 1000.0
        t_j += 0.033

    pts_left = mgr_j.get_total_points("Left")
    pts_right = mgr_j.get_total_points("Right")
    avg_mgr_ms = t_pipeline_total / total_frames

    assert pts_left == total_frames, f"Left hand points mismatch: {pts_left} != {total_frames}"
    assert pts_right == total_frames, f"Right hand points mismatch: {pts_right} != {total_frames}"
    assert mgr_j.boundary_status["Left"] == "INSIDE_ZONE"
    assert mgr_j.boundary_status["Right"] == "INSIDE_ZONE"
    assert avg_mgr_ms < 1.0, f"Dual hand update latency excessive: {avg_mgr_ms:.3f} ms"

    print(f"  Dual-Zone Simultaneous Tracking: Left={pts_left} pts | Right={pts_right} pts")
    print(f"  Dual-hand AirDrawingManager latency: {avg_mgr_ms:.3f} ms/frame (PASS)")
    test_results["Test J (Both Hands Simultaneous)"] = "PASS"

    # -------------------------------------------------------------------------
    # SUMMARY & FINAL VERIFICATION
    # -------------------------------------------------------------------------
    print("\n" + "=" * 74)
    print(" ALL 10/10 SCENARIO TESTS (TESTS A THROUGH J) PASSED WITH ZERO ERRORS!")
    print("=" * 74)
    for test_name, status in test_results.items():
        print(f"  [✓] {test_name:<36}: {status}")
    print("=" * 74)
    return True


if __name__ == "__main__":
    success = run_robustness_test_suite()
    if not success:
        sys.exit(1)
