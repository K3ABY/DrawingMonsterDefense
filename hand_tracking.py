"""
Drawing Monster Defense - Dual-Zone Hand Tracking & Responsive Air Drawing
Features:
- Dual-Zone 50/50 Canvas Division (Left Zone & Right Zone)
- MediaPipe Handedness strict zone assignment
- Boundary crossing handling: stroke cutoff on leaving, new stroke on re-entry, no cross-zone lines
- Threaded zero-buffer-lag camera ingestion (MJPEG optimization)
- One Euro Filter adaptive landmark smoothing
- Fist pause / Pointing draw control with anti-flicker debouncing
- Full diagnostic HUD with Boundary Status, Zone Coords, FPS, Latency (ms), and Point Rates
"""

import sys
import time
from typing import Optional, Tuple, Dict, Any
import cv2
import mediapipe as mp
import numpy as np

import camera_config as config
from gesture_controller import TwoHandJoinGestureController, DrawingState, GestureState
from drawing_manager import AirDrawingManager
from one_euro_filter import HandLandmarkSmoother
from threaded_camera import ThreadedCameraStream
from image_quality_assessor import ImageQualityAssessor, ImageCondition, ImageQualityMetrics
from adaptive_enhancer import AdaptiveImageEnhancer


class HandTracker:
    """
    Encapsulates MediaPipe Hands detection, adaptive One Euro Filter smoothing,
    and visual rendering.
    """

    def __init__(
        self,
        max_num_hands: int = config.MAX_NUM_HANDS,
        static_image_mode: bool = config.STATIC_IMAGE_MODE,
        model_complexity: int = config.MODEL_COMPLEXITY,
        min_detection_confidence: float = config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = config.MIN_TRACKING_CONFIDENCE,
        invert_handedness: bool = config.INVERT_HANDEDNESS,
        use_smoothing: bool = config.USE_LANDMARK_SMOOTHING,
    ):
        self.invert_handedness = invert_handedness
        self.use_smoothing = use_smoothing

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        # Separate One Euro Filter smoothers for Left and Right hands
        self.smoothers = {
            "Left": HandLandmarkSmoother(
                min_cutoff=config.ONE_EURO_MIN_CUTOFF,
                beta=config.ONE_EURO_BETA,
                d_cutoff=config.ONE_EURO_DCUTOFF,
                reset_timeout=config.FILTER_RESET_TIMEOUT,
            ),
            "Right": HandLandmarkSmoother(
                min_cutoff=config.ONE_EURO_MIN_CUTOFF,
                beta=config.ONE_EURO_BETA,
                d_cutoff=config.ONE_EURO_DCUTOFF,
                reset_timeout=config.FILTER_RESET_TIMEOUT,
            ),
        }

        # Tracking Loss Tracking & Duration
        self.tracking_loss_count = {"Left": 0, "Right": 0}
        self.prev_detected = {"Left": False, "Right": False}
        self.last_seen_time = {"Left": time.time(), "Right": time.time()}
        self.tracking_loss_duration = {"Left": 0.0, "Right": 0.0}
        self.show_hud = getattr(config, "SHOW_DEBUG_HUD", False)

        # Zero-allocation reusable buffers
        self._rgb_buffer: Optional[np.ndarray] = None
        self._hud_bg_box: Optional[np.ndarray] = None

        # Quality Assessment & Adaptive Enhancement
        self.quality_assessor = ImageQualityAssessor(
            blur_sharpness_threshold=config.QUALITY_BLUR_SHARPNESS_THRESHOLD,
            noise_threshold=config.QUALITY_NOISE_THRESHOLD,
            dark_threshold=config.QUALITY_DARK_THRESHOLD,
            bright_threshold=config.QUALITY_BRIGHT_THRESHOLD,
            contrast_threshold=config.QUALITY_LOW_CONTRAST_THRESHOLD,
        )
        self.enhancer = AdaptiveImageEnhancer(
            clahe_clip_limit=config.CLAHE_CLIP_LIMIT,
            clahe_grid_size=config.CLAHE_GRID_SIZE,
            dark_gamma=config.DARK_GAMMA,
            bright_gamma=config.BRIGHT_GAMMA,
        )
        self.last_quality_metrics: Optional[ImageQualityMetrics] = None
        self.last_enhancement_info: Optional[Dict[str, Any]] = None

    def process(self, frame: np.ndarray, current_time: Optional[float] = None) -> Tuple[Dict[str, Any], Any]:
        """
        Process a BGR frame through MediaPipe Hands with adaptive smoothing.
        Uses zero-allocation RGB conversion buffer for maximum performance.
        """
        if current_time is None:
            current_time = time.time()

        h, w, _ = frame.shape

        hand_data = {
            "Left": None,
            "Right": None,
        }

        # 1. Assess image quality and conditionally apply lightweight enhancement
        metrics = self.quality_assessor.assess(frame)
        self.last_quality_metrics = metrics
        proc_frame, enh_info = self.enhancer.enhance(frame, metrics)
        self.last_enhancement_info = enh_info

        # 2. MediaPipe requires RGB format (using reusable zero-allocation buffer)
        if self._rgb_buffer is None or self._rgb_buffer.shape != proc_frame.shape:
            self._rgb_buffer = np.empty_like(proc_frame)
        cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB, dst=self._rgb_buffer)
        self._rgb_buffer.flags.writeable = False
        results = self.hands.process(self._rgb_buffer)
        self._rgb_buffer.flags.writeable = True

        if results.multi_hand_landmarks and results.multi_handedness:
            detected_hands = []

            for hand_landmarks, handedness_info in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                raw_label = handedness_info.classification[0].label
                confidence = float(handedness_info.classification[0].score)

                # Determine effective label
                label = raw_label
                if self.invert_handedness:
                    label = "Right" if raw_label == "Left" else "Left"

                # Extract pixel coordinates and 3D normalized coordinates for all 21 landmarks
                raw_coords = []
                landmarks_3d = []
                for lm in hand_landmarks.landmark:
                    px = int(lm.x * w)
                    py = int(lm.y * h)
                    raw_coords.append((px, py))
                    landmarks_3d.append((float(lm.x), float(lm.y), float(lm.z)))

                raw_tip = raw_coords[config.LANDMARK_INDEX_FINGER_TIP]
                raw_wrist = raw_coords[config.LANDMARK_WRIST]

                # Apply One Euro Filter adaptive smoothing
                if self.use_smoothing:
                    smooth_tip = self.smoothers[label].smooth_index_tip(raw_tip, current_time)
                    smooth_wrist = self.smoothers[label].smooth_wrist(raw_wrist, current_time)
                    tip_x, tip_y = smooth_tip
                    wrist_x = smooth_wrist[0]
                    active_coords = list(raw_coords)
                    active_coords[config.LANDMARK_INDEX_FINGER_TIP] = smooth_tip
                    active_coords[config.LANDMARK_WRIST] = smooth_wrist
                else:
                    tip_x, tip_y = raw_tip
                    wrist_x = raw_wrist[0]
                    active_coords = raw_coords

                norm_x = float(round(tip_x / w, 4))
                norm_y = float(round(tip_y / h, 4))

                detected_hands.append({
                    "label": label,
                    "detected": True,
                    "status": "Detected",
                    "index_tip_x": tip_x,
                    "index_tip_y": tip_y,
                    "index_finger_tip": (tip_x, tip_y),
                    "norm_tip": (norm_x, norm_y),
                    "raw_index_tip": raw_tip,
                    "landmarks": active_coords,
                    "raw_coords": raw_coords,
                    "landmarks_3d": landmarks_3d,
                    "confidence": confidence,
                    "wrist_x": wrist_x,
                    "raw_landmarks": hand_landmarks,
                })

            # Handle assignment: if 2 hands detected but both have same label,
            # resolve by horizontal position (in mirrored view, smaller X is Left)
            if len(detected_hands) == 1:
                h_item = detected_hands[0]
                hand_data[h_item["label"]] = h_item
            elif len(detected_hands) >= 2:
                labels = [h_item["label"] for h_item in detected_hands]
                if labels[0] == labels[1]:
                    sorted_by_x = sorted(detected_hands, key=lambda x: x["wrist_x"])
                    sorted_by_x[0]["label"] = "Left"
                    sorted_by_x[1]["label"] = "Right"
                    hand_data["Left"] = sorted_by_x[0]
                    hand_data["Right"] = sorted_by_x[1]
                else:
                    for h_item in detected_hands:
                        hand_data[h_item["label"]] = h_item

        # Tracking Loss event detection, duration accumulation, and filter reset
        for label in ["Left", "Right"]:
            is_det = hand_data[label] is not None
            if is_det:
                self.last_seen_time[label] = current_time
                self.tracking_loss_duration[label] = 0.0
            else:
                self.tracking_loss_duration[label] = max(0.0, current_time - self.last_seen_time[label])

            if self.prev_detected[label] and not is_det:
                self.tracking_loss_count[label] += 1
            if not is_det:
                self.smoothers[label].check_reset(current_time)
            self.prev_detected[label] = is_det

        return hand_data, results

    def draw(
        self,
        frame: np.ndarray,
        hand_data: dict,
        results: any,
        gesture_ctrl: Optional[TwoHandJoinGestureController] = None,
        drawing_mgr: Optional[AirDrawingManager] = None,
        perf_metrics: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        Renders Dual-Zone dividing line, drawing strokes, skeletons,
        index finger tip markers, gesture visual progress, and comprehensive debug HUD.
        """
        # 1. Render Dual-Zone Canvas, center divider, and persistent strokes
        if drawing_mgr is not None:
            is_active = gesture_ctrl.is_drawing_active if gesture_ctrl else False
            display_frame = drawing_mgr.render_strokes(frame, is_active, hand_data, copy=False)
        else:
            display_frame = frame

        h, w, _ = display_frame.shape

        # 2. Draw MediaPipe hand skeletons
        for hand_label in ["Left", "Right"]:
            item = hand_data.get(hand_label)
            if item is not None and "raw_landmarks" in item:
                conn_color = (
                    config.COLOR_LEFT_HAND if hand_label == "Left" else config.COLOR_RIGHT_HAND
                )

                custom_conn_style = self.mp_drawing.DrawingSpec(
                    color=conn_color, thickness=2, circle_radius=2
                )
                custom_lm_style = self.mp_drawing.DrawingSpec(
                    color=(255, 255, 255), thickness=1, circle_radius=3
                )

                self.mp_drawing.draw_landmarks(
                    display_frame,
                    item["raw_landmarks"],
                    self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=custom_lm_style,
                    connection_drawing_spec=custom_conn_style,
                )

        # 3. Highlight index finger tips with visual indicator for recognized index pointing
        for hand_label in ["Left", "Right"]:
            item = hand_data.get(hand_label)
            if item is not None:
                tip_x = item["index_tip_x"]
                tip_y = item["index_tip_y"]
                hand_color = (
                    config.COLOR_LEFT_HAND if hand_label == "Left" else config.COLOR_RIGHT_HAND
                )

                is_pointing = False
                p_dir = "NONE"
                posture = "UNKNOWN"
                if drawing_mgr is not None:
                    is_pointing = drawing_mgr.is_pointing_recognized.get(hand_label, False)
                    p_dir = drawing_mgr.pointing_direction.get(hand_label, "NONE")
                    posture = drawing_mgr.posture_state.get(hand_label, "UNKNOWN")

                if is_pointing:
                    # Glowing outer rings for recognized index pointing
                    cv2.circle(display_frame, (tip_x, tip_y), 16, (0, 255, 128), 2, cv2.LINE_AA)
                    cv2.circle(display_frame, (tip_x, tip_y), 11, hand_color, 2, cv2.LINE_AA)
                    # Vibrant green inner center dot
                    cv2.circle(display_frame, (tip_x, tip_y), 5, (0, 255, 0), -1, cv2.LINE_AA)
                    coord_text = f"{hand_label}: ({tip_x}, {tip_y}) | POINTING: OK [{p_dir}]"
                    text_color = (0, 255, 128)
                else:
                    # Outer ring and inner dot for non-pointing states (e.g., FIST, OPEN, RELAXED)
                    cv2.circle(display_frame, (tip_x, tip_y), 12, hand_color, 2, cv2.LINE_AA)
                    dot_color = (0, 140, 255) if posture == "FIST" else (160, 160, 160)
                    cv2.circle(display_frame, (tip_x, tip_y), 5, dot_color, -1, cv2.LINE_AA)
                    coord_text = f"{hand_label}: ({tip_x}, {tip_y}) [{posture}]"
                    text_color = (220, 220, 220)

                # High-contrast coordinate and pointing badge
                cv2.putText(
                    display_frame,
                    coord_text,
                    (tip_x + 14, tip_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.46,
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    display_frame,
                    coord_text,
                    (tip_x + 14, tip_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.46,
                    text_color,
                    1,
                    cv2.LINE_AA,
                )

        # 4. Visual feedback for Two-Hand Join Gesture (Hold Progress Ring)
        if gesture_ctrl is not None:
            self._render_gesture_feedback(display_frame, gesture_ctrl)

        # 5. Render Debug Display and HUD Overlay Panel (Only when show_hud is True, toggle with 'D')
        if getattr(self, "show_hud", False):
            self._render_hud(display_frame, hand_data, gesture_ctrl, drawing_mgr, perf_metrics)

        return display_frame

    def _render_gesture_feedback(
        self, frame: np.ndarray, gesture_ctrl: TwoHandJoinGestureController
    ) -> None:
        """Draws visual circular progress at the join center during hold."""
        if gesture_ctrl.gesture_state == GestureState.JOIN_HOLDING and gesture_ctrl.join_center:
            cx, cy = gesture_ctrl.join_center
            progress = gesture_ctrl.hold_progress

            cv2.circle(frame, (cx, cy), 34, (80, 80, 80), 3, cv2.LINE_AA)
            end_angle = int(progress * 360)
            if end_angle > 0:
                cv2.ellipse(
                    frame,
                    (cx, cy),
                    (34, 34),
                    -90,
                    0,
                    end_angle,
                    config.COLOR_JOIN_ACTIVE,
                    4,
                    cv2.LINE_AA,
                )
            pulse_radius = int(6 + progress * 6)
            cv2.circle(frame, (cx, cy), pulse_radius, config.COLOR_JOIN_ACTIVE, -1, cv2.LINE_AA)

            pct_text = f"HOLD {int(progress * 100)}%"
            cv2.putText(
                frame,
                pct_text,
                (cx - 30, cy - 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                pct_text,
                (cx - 30, cy - 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                config.COLOR_JOIN_ACTIVE,
                1,
                cv2.LINE_AA,
            )

        # Confirmation Banner on state toggle
        banner_elapsed = time.time() - gesture_ctrl.last_event_time
        if banner_elapsed < 1.4 and gesture_ctrl.last_event != "NONE":
            h, w, _ = frame.shape
            is_active = gesture_ctrl.is_drawing_active
            banner_text = ">>> DRAWING ACTIVE <<<" if is_active else ">>> DRAWING STOPPED <<<"
            box_color = (0, 180, 0) if is_active else (0, 120, 255)

            text_size = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0]
            bx = (w - text_size[0]) // 2
            by = 45
            cv2.rectangle(frame, (bx - 16, by - 26), (bx + text_size[0] + 16, by + 12), box_color, -1)
            cv2.rectangle(frame, (bx - 16, by - 26), (bx + text_size[0] + 16, by + 12), (255, 255, 255), 2)
            cv2.putText(frame, banner_text, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    def _render_hud(
        self,
        frame: np.ndarray,
        hand_data: dict,
        gesture_ctrl: Optional[TwoHandJoinGestureController] = None,
        drawing_mgr: Optional[AirDrawingManager] = None,
        perf_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Draws semi-transparent HUD showing Tracking Diagnostics:
        - Hand detection status, tracking confidence & tracking loss duration
        - Index finger recognition & 3D pointing direction (FORWARD, UP, DOWN, SIDEWAYS, DIAGONAL)
        - Current drawing state (DRAWING, FIST PAUSED, OUT OF ZONE, IDLE)
        - Camera FPS, Processing FPS, frame processing time (ms)
        - Dual-Zone boundaries & coordinates
        - Two-Hand Join gesture status, hold progress & cooldown
        """
        panel_w = 320
        panel_h = 464
        panel_x = 8
        panel_y = 6

        # Fast ROI-only alpha blend (zero full-frame memory allocation)
        if self._hud_bg_box is None or self._hud_bg_box.shape != (panel_h, panel_w, 3):
            self._hud_bg_box = np.full((panel_h, panel_w, 3), 20, dtype=np.uint8)
        roi = frame[panel_y : panel_y + panel_h, panel_x : panel_x + panel_w]
        cv2.addWeighted(self._hud_bg_box, 0.82, roi, 0.18, 0, dst=roi)
        cv2.rectangle(
            frame,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            (75, 75, 75),
            1,
        )

        curr_y = panel_y + 18

        # Section 1: Performance Diagnostics, Camera Source & Image Quality
        cam_fps = perf_metrics.get("camera_fps", 0.0) if perf_metrics else 0.0
        proc_fps = perf_metrics.get("proc_fps", 0.0) if perf_metrics else 0.0
        proc_ms = perf_metrics.get("proc_ms", 0.0) if perf_metrics else 0.0
        cpu_val = perf_metrics.get("cpu_percent", 0.0) if perf_metrics else 0.0
        ram_val = perf_metrics.get("ram_mb", 0.0) if perf_metrics else 0.0
        cam_mode = perf_metrics.get("camera_mode", "UNKNOWN") if perf_metrics else "UNKNOWN"
        cam_status = perf_metrics.get("camera_status", "UNKNOWN") if perf_metrics else "UNKNOWN"
        cam_res = perf_metrics.get("camera_res", "N/A") if perf_metrics else "N/A"
        fallback_occ = perf_metrics.get("fallback_occurred", False) if perf_metrics else False

        cv2.putText(
            frame,
            "TRACKING & AIR DRAWING HUD",
            (panel_x + 12, curr_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )
        curr_y += 16
        # Camera mode and connection status
        status_color = (0, 255, 128) if cam_status == "CONNECTED" else (0, 165, 255) if cam_status == "RECONNECTING" else (0, 0, 255)
        cam_mode_text = f"Cam: {cam_mode} [{cam_status}] ({cam_res})"
        cv2.putText(
            frame,
            cam_mode_text,
            (panel_x + 12, curr_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39,
            status_color,
            1,
            cv2.LINE_AA,
        )

        curr_y += 15
        fps_text = f"Camera: {cam_fps:.1f} FPS | Proc: {proc_fps:.1f} FPS ({proc_ms:.1f} ms)"
        cv2.putText(
            frame,
            fps_text,
            (panel_x + 12, curr_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        curr_y += 15
        diag_res_text = f"CPU: {cpu_val:.1f}% | RAM: {ram_val:.1f} MB"
        cv2.putText(
            frame,
            diag_res_text,
            (panel_x + 12, curr_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )

        curr_y += 15
        loss_text = (
            f"Tracking Losses: L={self.tracking_loss_count['Left']} | "
            f"R={self.tracking_loss_count['Right']} | Smooth: {'ON' if self.use_smoothing else 'OFF'}"
        )
        cv2.putText(
            frame,
            loss_text,
            (panel_x + 12, curr_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            config.COLOR_TEXT,
            1,
            cv2.LINE_AA,
        )

        if fallback_occ:
            curr_y += 14
            fb_msg = perf_metrics.get("fallback_message", "Fallback active") if perf_metrics else ""
            cv2.putText(
                frame,
                f"Event: {fb_msg[:38]}",
                (panel_x + 12, curr_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                (0, 220, 255),
                1,
                cv2.LINE_AA,
            )

        # Image Quality Assessment & Adaptive Enhancement
        curr_y += 15
        if self.last_quality_metrics:
            m = self.last_quality_metrics
            enh = self.last_enhancement_info or {}
            enh_name = enh.get("method", "BYPASS")
            enh_ms = enh.get("time_ms", 0.0)
            q_color = (0, 255, 128) if m.condition.value == "NORMAL" else (0, 180, 255)
            q_text = f"Image: {m.condition.value} | Enh: {enh_name} ({enh_ms:.1f}ms)"
        else:
            q_color = (160, 160, 160)
            q_text = "Image Quality: N/A"

        cv2.putText(
            frame,
            q_text,
            (panel_x + 12, curr_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            q_color,
            1,
            cv2.LINE_AA,
        )

        curr_y += 8
        cv2.line(frame, (panel_x + 8, curr_y), (panel_x + panel_w - 8, curr_y), (60, 60, 60), 1)

        is_active = gesture_ctrl.is_drawing_active if gesture_ctrl else False

        # Section 2: Left Hand Tracking & Recognition Diagnostics
        curr_y += 16
        left = hand_data.get("Left")
        left_detected = left is not None
        left_conf = f"{int(left.get('confidence', 0.0) * 100)}%" if left_detected else "N/A"
        left_conf_state = drawing_mgr.tracking_confidence_state.get("Left").value if drawing_mgr else "LOST"
        left_det_text = (
            f"LEFT HAND: DETECTED ({left_conf} | {left_conf_state})"
            if left_detected
            else f"LEFT HAND: LOST (Duration: {self.tracking_loss_duration['Left']:.2f}s)"
        )
        left_det_color = config.COLOR_LEFT_HAND if left_detected else (0, 0, 255)

        cv2.putText(
            frame,
            left_det_text,
            (panel_x + 12, curr_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            left_det_color,
            2,
            cv2.LINE_AA,
        )

        curr_y += 15
        left_pointing = drawing_mgr.is_pointing_recognized.get("Left", False) if drawing_mgr else False
        left_dir = drawing_mgr.pointing_direction.get("Left", "NONE") if drawing_mgr else "NONE"
        left_posture = drawing_mgr.posture_state.get("Left", "UNKNOWN") if drawing_mgr else "N/A"
        if left_detected:
            if left_pointing:
                left_rec_text = f"Index Finger: RECOGNIZED [3D DIR: {left_dir}]"
                left_rec_color = (0, 255, 128)
            else:
                left_rec_text = f"Index Finger: NOT RECOGNIZED [{left_posture}]"
                left_rec_color = (0, 180, 255)
        else:
            left_rec_text = f"Index Finger: NO HAND (Losses: {self.tracking_loss_count['Left']})"
            left_rec_color = (130, 130, 130)

        cv2.putText(
            frame,
            left_rec_text,
            (panel_x + 12, curr_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39,
            left_rec_color,
            1,
            cv2.LINE_AA,
        )

        curr_y += 15
        left_draw_state = drawing_mgr.get_drawing_state_name("Left", is_active, left_detected) if drawing_mgr else "IDLE"
        left_b_status = drawing_mgr.boundary_status.get("Left", "NO_HAND") if drawing_mgr else "N/A"
        left_st_color = (0, 255, 0) if left_draw_state == "DRAWING" else (0, 180, 255) if left_draw_state == "FIST PAUSED" else (0, 0, 255) if left_draw_state == "OUT OF ZONE" else (160, 160, 160)
        cv2.putText(
            frame,
            f"State: {left_draw_state} | Zone: LEFT (0%-50%) [{left_b_status}]",
            (panel_x + 12, curr_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39,
            left_st_color,
            1,
            cv2.LINE_AA,
        )

        curr_y += 15
        if left_detected and drawing_mgr and drawing_mgr.last_zone_coords["Left"]:
            zx, zy = drawing_mgr.last_zone_coords["Left"]
            nx, ny = drawing_mgr.last_norm_coords["Left"]
            left_coords_text = f"Tip: ({left['index_tip_x']}, {left['index_tip_y']}) | Norm: ({nx:.2f}, {ny:.2f}) | Zone: ({zx:.2f}, {zy:.2f})"
        else:
            left_coords_text = "Tip: N/A"
        cv2.putText(
            frame,
            left_coords_text,
            (panel_x + 12, curr_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.37,
            config.COLOR_TEXT,
            1,
            cv2.LINE_AA,
        )

        curr_y += 10
        cv2.line(frame, (panel_x + 8, curr_y), (panel_x + panel_w - 8, curr_y), (60, 60, 60), 1)

        # Section 3: Right Hand Tracking & Recognition Diagnostics
        curr_y += 16
        right = hand_data.get("Right")
        right_detected = right is not None
        right_conf = f"{int(right.get('confidence', 0.0) * 100)}%" if right_detected else "N/A"
        right_conf_state = drawing_mgr.tracking_confidence_state.get("Right").value if drawing_mgr else "LOST"
        right_det_text = (
            f"RIGHT HAND: DETECTED ({right_conf} | {right_conf_state})"
            if right_detected
            else f"RIGHT HAND: LOST (Duration: {self.tracking_loss_duration['Right']:.2f}s)"
        )
        right_det_color = config.COLOR_RIGHT_HAND if right_detected else (0, 0, 255)

        cv2.putText(
            frame,
            right_det_text,
            (panel_x + 12, curr_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            right_det_color,
            2,
            cv2.LINE_AA,
        )

        curr_y += 15
        right_pointing = drawing_mgr.is_pointing_recognized.get("Right", False) if drawing_mgr else False
        right_dir = drawing_mgr.pointing_direction.get("Right", "NONE") if drawing_mgr else "NONE"
        right_posture = drawing_mgr.posture_state.get("Right", "UNKNOWN") if drawing_mgr else "N/A"
        if right_detected:
            if right_pointing:
                right_rec_text = f"Index Finger: RECOGNIZED [3D DIR: {right_dir}]"
                right_rec_color = (0, 255, 128)
            else:
                right_rec_text = f"Index Finger: NOT RECOGNIZED [{right_posture}]"
                right_rec_color = (0, 180, 255)
        else:
            right_rec_text = f"Index Finger: NO HAND (Losses: {self.tracking_loss_count['Right']})"
            right_rec_color = (130, 130, 130)

        cv2.putText(
            frame,
            right_rec_text,
            (panel_x + 12, curr_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39,
            right_rec_color,
            1,
            cv2.LINE_AA,
        )

        curr_y += 15
        right_draw_state = drawing_mgr.get_drawing_state_name("Right", is_active, right_detected) if drawing_mgr else "IDLE"
        right_b_status = drawing_mgr.boundary_status.get("Right", "NO_HAND") if drawing_mgr else "N/A"
        right_st_color = (0, 255, 0) if right_draw_state == "DRAWING" else (0, 180, 255) if right_draw_state == "FIST PAUSED" else (0, 0, 255) if right_draw_state == "OUT OF ZONE" else (160, 160, 160)
        cv2.putText(
            frame,
            f"State: {right_draw_state} | Zone: RIGHT (50%-100%) [{right_b_status}]",
            (panel_x + 12, curr_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39,
            right_st_color,
            1,
            cv2.LINE_AA,
        )

        curr_y += 15
        if right_detected and drawing_mgr and drawing_mgr.last_zone_coords["Right"]:
            zx, zy = drawing_mgr.last_zone_coords["Right"]
            nx, ny = drawing_mgr.last_norm_coords["Right"]
            right_coords_text = f"Tip: ({right['index_tip_x']}, {right['index_tip_y']}) | Norm: ({nx:.2f}, {ny:.2f}) | Zone: ({zx:.2f}, {zy:.2f})"
        else:
            right_coords_text = "Tip: N/A"
        cv2.putText(
            frame,
            right_coords_text,
            (panel_x + 12, curr_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.37,
            config.COLOR_TEXT,
            1,
            cv2.LINE_AA,
        )

        curr_y += 10
        cv2.line(frame, (panel_x + 8, curr_y), (panel_x + panel_w - 8, curr_y), (60, 60, 60), 1)

        # Section 4: Gesture & Global Air Drawing System
        if gesture_ctrl is not None and drawing_mgr is not None:
            curr_y += 18
            status_text = "ACTIVE" if is_active else "INACTIVE (IDLE)"
            status_color = (0, 255, 0) if is_active else (140, 140, 140)
            cv2.putText(
                frame,
                f"DRAWING SYSTEM: {status_text}",
                (panel_x + 12, curr_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                status_color,
                2,
                cv2.LINE_AA,
            )

            curr_y += 17
            g_status = gesture_ctrl.gesture_status_text
            g_color = (0, 255, 255) if "HOLDING" in g_status else config.COLOR_TEXT
            cv2.putText(
                frame,
                f"GESTURE STATUS: {g_status}",
                (panel_x + 12, curr_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                g_color,
                1,
                cv2.LINE_AA,
            )

            curr_y += 17
            timer_text = gesture_ctrl.hold_timer_text
            bar_w = 70
            bar_h = 8
            bx = panel_x + panel_w - bar_w - 14
            by = curr_y - 9
            cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (50, 50, 50), -1)
            fill_w = int(bar_w * gesture_ctrl.hold_progress)
            if fill_w > 0:
                cv2.rectangle(frame, (bx, by), (bx + fill_w, by + bar_h), (0, 255, 255), -1)
            cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (120, 120, 120), 1)

            cv2.putText(
                frame,
                f"HOLD TIMER: {timer_text}",
                (panel_x + 12, curr_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                config.COLOR_TEXT,
                1,
                cv2.LINE_AA,
            )

            curr_y += 17
            last_ev = gesture_ctrl.last_event
            ev_color = (0, 255, 120) if last_ev == "START_DRAWING" else (0, 140, 255) if last_ev == "STOP_DRAWING" else (140, 140, 140)
            cv2.putText(
                frame,
                f"LAST EVENT: {last_ev}",
                (panel_x + 12, curr_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                ev_color,
                1,
                cv2.LINE_AA,
            )

    def close(self):
        """Clean up MediaPipe resources."""
        self.hands.close()


def open_camera_stream(url: str, max_retries: int = 3, retry_delay: float = 1.0) -> Optional[cv2.VideoCapture]:
    """
    Attempts to connect to the video stream URL with retry and friendly error messages.
    Uses fast socket pre-check to prevent 30-second blocking timeouts when server is offline.
    Maintained for backward compatibility.
    """
    print(f"[HandTracker] Connecting to camera stream at: {url}")
    try:
        from urllib.parse import urlparse
        import socket
        parsed = urlparse(url)
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if host:
            with socket.create_connection((host, port), timeout=0.8):
                pass
    except Exception as e:
        print(f"[HandTracker] Camera stream host unreachable at {url}: {e}")
        return None

    for attempt in range(1, max_retries + 1):
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"[HandTracker] Connection successful! Frame resolution: {frame.shape[1]}x{frame.shape[0]}")
                return cap
            cap.release()

        print(f"[HandTracker] Connection attempt {attempt}/{max_retries} failed.")
        if attempt < max_retries:
            time.sleep(retry_delay)

    print(f"[ERROR] Cannot connect to Laptop Camera Server at {url}")
    return None


def main():
    """Main execution loop for Phase 1 Hand Tracking with Dual-Zone Responsive Air Drawing."""
    print("=" * 68)
    print(" Drawing Monster Defense - Dual-Zone Air Drawing System")
    print("=" * 68)
    print(f" Camera Stream: {config.CAMERA_URL}")
    print(" Dual-Zone Layout:")
    print("   - LEFT HAND  -> Left Drawing Zone ONLY (X: 0% to 50%)")
    print("   - RIGHT HAND -> Right Drawing Zone ONLY (X: 50% to 100%)")
    print(" Controls:")
    print("   - Two-Hand Join & HOLD ~0.7s: Toggle Drawing Mode (ACTIVE <-> IDLE)")
    print("   - Index Finger Pointing: Draw inside assigned zone")
    print("   - Fist (Closed Hand): Pause drawing for that specific hand")
    print("   - Crossing Zone Boundary: Stroke terminates immediately (no cross-lines)")
    print(" Keyboard Shortcuts:")
    print("   [Q] / [ESC] - Quit Application")
    print("   [R] - Retry Camera Detection / Re-scan Sources")
    print("   [C] - Clear Canvas Drawings")
    print("   [D] - Toggle Debug HUD Overlay")
    print("   [S] - Toggle One Euro Landmark Smoothing")
    print("   [H] - Toggle Handedness Inversion")
    print("   [F] - Toggle Horizontal Mirroring")
    print("=" * 68)

    # Initialize Camera Stream (Threaded with Auto-Detection & Fallback)
    if config.USE_THREADED_CAMERA:
        print("[HandTracker] Starting Unified Camera Stream (Auto-Detection & Fallback mode)...")
        cam = ThreadedCameraStream(config.CAMERA_URL, auto_detect=getattr(config, "AUTO_CAMERA_DETECTION", True))
    else:
        cam = open_camera_stream(config.CAMERA_URL)
        if cam is None:
            print(f"[ERROR] Cannot connect to camera stream at {config.CAMERA_URL}")
            cam = cv2.VideoCapture(0)

    tracker = HandTracker()
    gesture_ctrl = TwoHandJoinGestureController()
    drawing_mgr = AirDrawingManager()
    flip_horizontal = config.FLIP_HORIZONTAL

    cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    # Timing, Resource, and FPS calculations
    import psutil
    process = psutil.Process()
    cached_cpu = 0.0
    cached_ram = process.memory_info().rss / (1024 * 1024)
    last_res_query_time = time.perf_counter()

    prev_loop_time = time.perf_counter()
    proc_fps = 0.0
    avg_proc_ms = 0.0

    try:
        while True:
            t_frame_start = time.perf_counter()

            # Read latest frame
            if config.USE_THREADED_CAMERA:
                ret, frame, is_new = cam.read_latest()
                if frame is None:
                    time.sleep(0.01)
                    continue
            else:
                ret, frame = cam.read()
                if not ret or frame is None:
                    print("[HandTracker] Warning: Empty frame received.")
                    time.sleep(0.02)
                    continue

            if flip_horizontal and ret:
                frame = cv2.flip(frame, 1)

            h_frame, w_frame, _ = frame.shape
            current_time = time.time()

            if ret:
                # 1. Process hand tracking with adaptive One Euro Filter smoothing
                hand_data, results = tracker.process(frame, current_time)

                # 2. Update Two-Hand Join Gesture State Machine
                state_changed = gesture_ctrl.update(hand_data, current_time)
                if state_changed:
                    print(f"[Gesture] Event: {gesture_ctrl.last_event} -> Drawing State: {gesture_ctrl.drawing_state.value}")

                # 3. Update Dual-Zone Air Drawing (with boundary enforcement, fist pause & gap recovery)
                drawing_mgr.update(
                    hand_data,
                    gesture_ctrl.is_drawing_active,
                    current_time,
                    frame_width=w_frame,
                    frame_height=h_frame,
                )
            else:
                # Camera disconnected / standby screen: keep tracking and drawing idle
                hand_data = {"Left": None, "Right": None}
                results = None

            # Measure processing latency
            t_proc_end = time.perf_counter()
            frame_proc_ms = (t_proc_end - t_frame_start) * 1000
            avg_proc_ms = 0.9 * avg_proc_ms + 0.1 * frame_proc_ms

            # Camera FPS and Processing FPS
            curr_loop_time = time.perf_counter()
            dt_loop = curr_loop_time - prev_loop_time
            prev_loop_time = curr_loop_time
            instant_fps = 1.0 / dt_loop if dt_loop > 0 else 0.0
            proc_fps = 0.9 * proc_fps + 0.1 * instant_fps

            if curr_loop_time - last_res_query_time > 0.5:
                cached_cpu = process.cpu_percent()
                cached_ram = process.memory_info().rss / (1024 * 1024)
                last_res_query_time = curr_loop_time

            cam_metrics = cam.get_metrics() if config.USE_THREADED_CAMERA else {}
            perf_metrics = {
                "camera_fps": cam_metrics.get("camera_fps", proc_fps),
                "proc_fps": proc_fps,
                "proc_ms": avg_proc_ms,
                "cpu_percent": cached_cpu,
                "ram_mb": cached_ram,
                "camera_mode": cam_metrics.get("camera_mode", "UNKNOWN"),
                "camera_status": cam_metrics.get("camera_status", "CONNECTED" if ret else "DISCONNECTED"),
                "camera_source": cam_metrics.get("camera_source", str(config.CAMERA_URL)),
                "camera_res": cam_metrics.get("camera_res", f"{w_frame}x{h_frame}"),
                "fallback_occurred": cam_metrics.get("fallback_occurred", False),
                "fallback_message": cam_metrics.get("fallback_message", "None"),
                "error_message": cam_metrics.get("error_message", None),
            }

            # 4. Render visual annotations, Dual-Zone divider, HUD, strokes, and gesture feedback
            output_frame = tracker.draw(
                frame, hand_data, results, gesture_ctrl, drawing_mgr, perf_metrics
            )

            # Status bar at bottom
            cam_mode_badge = perf_metrics["camera_mode"]
            cam_status_badge = perf_metrics["camera_status"]
            status_text = (
                f"FPS: {proc_fps:.1f} | [{cam_mode_badge}: {cam_status_badge}] | "
                f"Draw: {'ACTIVE' if gesture_ctrl.is_drawing_active else 'IDLE'} | "
                f"HUD: {'ON' if tracker.show_hud else 'OFF'} | "
                f"[R]etry Cam [Q]uit [C]lear [D]ebug [S]mooth [H]and [F]lip"
            )
            h_out, w_out, _ = output_frame.shape
            cv2.putText(
                output_frame,
                status_text,
                (12, h_out - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Display window
            cv2.imshow(config.WINDOW_NAME, output_frame)

            # Handle Keyboard Input
            key = cv2.waitKey(1) & 0xFF
            if key in [ord("q"), ord("Q"), 27]:
                print("[HandTracker] Exit requested by user.")
                break
            elif key in [ord("r"), ord("R")]:
                print("[HandTracker] Manual camera re-scan requested by user...")
                if config.USE_THREADED_CAMERA:
                    cam.retry_detection()
            elif key in [ord("c"), ord("C")]:
                drawing_mgr.clear()
                print("[DrawingManager] Canvas cleared by user.")
            elif key in [ord("d"), ord("D")]:
                tracker.show_hud = not tracker.show_hud
                print(f"[HandTracker] Debug HUD toggled: {'ON' if tracker.show_hud else 'OFF'}")
            elif key in [ord("s"), ord("S")]:
                tracker.use_smoothing = not tracker.use_smoothing
                print(f"[HandTracker] Landmark Smoothing toggled to: {tracker.use_smoothing}")
            elif key in [ord("h"), ord("H")]:
                tracker.invert_handedness = not tracker.invert_handedness
                print(f"[HandTracker] Handedness Inversion set to: {tracker.invert_handedness}")
            elif key in [ord("f"), ord("F")]:
                flip_horizontal = not flip_horizontal
                print(f"[HandTracker] Horizontal Flip set to: {flip_horizontal}")

    except KeyboardInterrupt:
        print("\n[HandTracker] Interrupted by user.")
    except Exception as e:
        print(f"[HandTracker] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[HandTracker] Cleaning up resources...")
        if cam is not None:
            cam.release()
        tracker.close()
        cv2.destroyAllWindows()
        print("[HandTracker] Shutdown complete.")


if __name__ == "__main__":
    main()
