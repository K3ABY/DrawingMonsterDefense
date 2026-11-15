"""
Drawing Monster Defense - Phase 1: MediaPipe Hand Tracking & Two-Hand Join Toggle Drawing
Tracks up to 2 hands (Left and Right) in real-time from Laptop Camera Stream.
Supports Two-Hand Join Gesture (Toggle Mode), Debounce Lock, and Air Drawing.
"""

import sys
import time
import cv2
import mediapipe as mp
import numpy as np

import camera_config as config
from gesture_controller import TwoHandJoinGestureController, DrawingState, GestureState
from drawing_manager import AirDrawingManager


class HandTracker:
    """
    Encapsulates MediaPipe Hands detection, landmark extraction, and visualization.
    """

    def __init__(
        self,
        max_num_hands: int = config.MAX_NUM_HANDS,
        static_image_mode: bool = config.STATIC_IMAGE_MODE,
        model_complexity: int = config.MODEL_COMPLEXITY,
        min_detection_confidence: float = config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = config.MIN_TRACKING_CONFIDENCE,
        invert_handedness: bool = config.INVERT_HANDEDNESS,
    ):
        self.invert_handedness = invert_handedness
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame: np.ndarray) -> tuple[dict, any]:
        """
        Process a BGR frame through MediaPipe Hands.

        Returns:
            hand_data: dict with "Left" and "Right" keys.
                       Value is a dict when detected, or None when not detected.
            results: raw MediaPipe solution output.
        """
        h, w, _ = frame.shape

        hand_data = {
            "Left": None,
            "Right": None,
        }

        # MediaPipe requires RGB format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

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

                # Extract pixel coordinates for all 21 landmarks
                landmark_coords = []
                for lm in hand_landmarks.landmark:
                    px = int(lm.x * w)
                    py = int(lm.y * h)
                    landmark_coords.append((px, py))

                # Landmark 8: Index Finger Tip
                index_tip_lm = hand_landmarks.landmark[config.LANDMARK_INDEX_FINGER_TIP]
                tip_x = int(index_tip_lm.x * w)
                tip_y = int(index_tip_lm.y * h)

                wrist_x = landmark_coords[0][0]

                detected_hands.append({
                    "label": label,
                    "detected": True,
                    "status": "Detected",
                    "index_tip_x": tip_x,
                    "index_tip_y": tip_y,
                    "index_finger_tip": (tip_x, tip_y),
                    "landmarks": landmark_coords,
                    "confidence": confidence,
                    "wrist_x": wrist_x,
                    "raw_landmarks": hand_landmarks,
                })

            # Handle assignment: if 2 hands detected but both have same label,
            # resolve by horizontal position (in mirrored view, smaller X is Left, larger X is Right)
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

        return hand_data, results

    def draw(
        self,
        frame: np.ndarray,
        hand_data: dict,
        results: any,
        gesture_ctrl: TwoHandJoinGestureController = None,
        drawing_mgr: AirDrawingManager = None,
    ) -> np.ndarray:
        """
        Renders MediaPipe skeleton, drawing strokes, index finger tip markers,
        gesture visual progress, and debug HUD onto the frame.
        """
        # First render persistent air drawing strokes onto the frame
        if drawing_mgr is not None:
            is_active = gesture_ctrl.is_drawing_active if gesture_ctrl else False
            display_frame = drawing_mgr.render_strokes(frame, is_active, hand_data)
        else:
            display_frame = frame.copy()

        h, w, _ = display_frame.shape

        # Draw MediaPipe hand skeletons
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

        # Highlight index finger tips and draw coordinates
        for hand_label in ["Left", "Right"]:
            item = hand_data.get(hand_label)
            if item is not None:
                tip_x = item["index_tip_x"]
                tip_y = item["index_tip_y"]
                hand_color = (
                    config.COLOR_LEFT_HAND if hand_label == "Left" else config.COLOR_RIGHT_HAND
                )

                # Outer ring marker
                cv2.circle(display_frame, (tip_x, tip_y), 12, hand_color, 2, cv2.LINE_AA)
                # Inner solid circle
                cv2.circle(display_frame, (tip_x, tip_y), 5, config.COLOR_INDEX_TIP, -1, cv2.LINE_AA)

                # Coordinate badge next to index finger tip
                coord_text = f"{hand_label}: ({tip_x}, {tip_y})"
                cv2.putText(
                    display_frame,
                    coord_text,
                    (tip_x + 12, tip_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    display_frame,
                    coord_text,
                    (tip_x + 12, tip_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        # Visual feedback for Two-Hand Join Gesture (Hold Progress Ring)
        if gesture_ctrl is not None:
            self._render_gesture_feedback(display_frame, gesture_ctrl)

        # Render Debug Display and HUD Overlay Panel
        self._render_hud(display_frame, hand_data, gesture_ctrl, drawing_mgr)

        return display_frame

    def _render_gesture_feedback(
        self, frame: np.ndarray, gesture_ctrl: TwoHandJoinGestureController
    ) -> None:
        """Draws visual circular progress at the join center during hold."""
        if gesture_ctrl.gesture_state == GestureState.JOIN_HOLDING and gesture_ctrl.join_center:
            cx, cy = gesture_ctrl.join_center
            progress = gesture_ctrl.hold_progress

            # Outer guide ring
            cv2.circle(frame, (cx, cy), 34, (80, 80, 80), 3, cv2.LINE_AA)
            # Animated arc showing hold completion
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
            # Center pulse dot
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

            # Draw banner in upper center
            text_size = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0]
            bx = (w - text_size[0]) // 2
            by = 45
            cv2.rectangle(
                frame,
                (bx - 16, by - 26),
                (bx + text_size[0] + 16, by + 12),
                box_color,
                -1,
            )
            cv2.rectangle(
                frame,
                (bx - 16, by - 26),
                (bx + text_size[0] + 16, by + 12),
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                banner_text,
                (bx, by),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    def _render_hud(
        self,
        frame: np.ndarray,
        hand_data: dict,
        gesture_ctrl: TwoHandJoinGestureController = None,
        drawing_mgr: AirDrawingManager = None,
    ) -> None:
        """
        Draws semi-transparent HUD showing hand status, coordinates,
        gesture state, hold timer, and drawing status per Section 6 requirements.
        """
        panel_w = 380
        panel_h = 285
        panel_x = 10
        panel_y = 10

        # Semi-transparent overlay background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            (20, 20, 20),
            -1,
        )
        cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
        cv2.rectangle(
            frame,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            (75, 75, 75),
            1,
        )

        curr_y = panel_y + 22

        # Section 1: Hand Tracking Header
        cv2.putText(
            frame,
            "HAND TRACKING STATUS",
            (panel_x + 12, curr_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )

        # LEFT HAND
        curr_y += 22
        left = hand_data.get("Left")
        left_detected = left is not None
        left_status = "Detected" if left_detected else "Not Detected"
        left_color = (0, 255, 127) if left_detected else (140, 140, 140)

        cv2.putText(
            frame,
            "LEFT HAND",
            (panel_x + 12, curr_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            config.COLOR_LEFT_HAND,
            2,
            cv2.LINE_AA,
        )
        curr_y += 18
        cv2.putText(
            frame,
            f"Status: {left_status}",
            (panel_x + 12, curr_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            left_color,
            1,
            cv2.LINE_AA,
        )
        curr_y += 17
        left_tip_text = (
            f"Index Tip: X={left['index_tip_x']}, Y={left['index_tip_y']}"
            if left_detected
            else "Index Tip: N/A"
        )
        cv2.putText(
            frame,
            left_tip_text,
            (panel_x + 12, curr_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            config.COLOR_TEXT,
            1,
            cv2.LINE_AA,
        )

        # RIGHT HAND
        curr_y += 24
        right = hand_data.get("Right")
        right_detected = right is not None
        right_status = "Detected" if right_detected else "Not Detected"
        right_color = (0, 255, 127) if right_detected else (140, 140, 140)

        cv2.putText(
            frame,
            "RIGHT HAND",
            (panel_x + 12, curr_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            config.COLOR_RIGHT_HAND,
            2,
            cv2.LINE_AA,
        )
        curr_y += 18
        cv2.putText(
            frame,
            f"Status: {right_status}",
            (panel_x + 12, curr_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            right_color,
            1,
            cv2.LINE_AA,
        )
        curr_y += 17
        right_tip_text = (
            f"Index Tip: X={right['index_tip_x']}, Y={right['index_tip_y']}"
            if right_detected
            else "Index Tip: N/A"
        )
        cv2.putText(
            frame,
            right_tip_text,
            (panel_x + 12, curr_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            config.COLOR_TEXT,
            1,
            cv2.LINE_AA,
        )

        # Separator line between Tracking and Gesture
        curr_y += 12
        cv2.line(
            frame,
            (panel_x + 8, curr_y),
            (panel_x + panel_w - 8, curr_y),
            (60, 60, 60),
            1,
        )

        # Section 2: Debug Display for Gesture & Drawing (Section 6)
        if gesture_ctrl is not None and drawing_mgr is not None:
            curr_y += 18
            # 1. DRAWING STATUS
            is_active = gesture_ctrl.is_drawing_active
            status_text = "ACTIVE" if is_active else "INACTIVE"
            status_color = (0, 255, 0) if is_active else (140, 140, 140)
            cv2.putText(
                frame,
                f"DRAWING STATUS: {status_text}",
                (panel_x + 12, curr_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                status_color,
                2,
                cv2.LINE_AA,
            )

            # 2. GESTURE STATUS
            curr_y += 19
            g_status = gesture_ctrl.gesture_status_text
            g_color = (0, 255, 255) if "HOLDING" in g_status else config.COLOR_TEXT
            cv2.putText(
                frame,
                f"GESTURE STATUS: {g_status}",
                (panel_x + 12, curr_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                g_color,
                1,
                cv2.LINE_AA,
            )

            # 3. HOLD TIMER
            curr_y += 18
            timer_text = gesture_ctrl.hold_timer_text
            # Progress bar visualization
            bar_w = 80
            bar_h = 8
            bx = panel_x + 280
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
                0.44,
                config.COLOR_TEXT,
                1,
                cv2.LINE_AA,
            )

            # 4. LAST GESTURE EVENT
            curr_y += 18
            last_ev = gesture_ctrl.last_event
            ev_color = (0, 255, 120) if last_ev == "START_DRAWING" else (0, 140, 255) if last_ev == "STOP_DRAWING" else (140, 140, 140)
            cv2.putText(
                frame,
                f"LAST EVENT: {last_ev}",
                (panel_x + 12, curr_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                ev_color,
                1,
                cv2.LINE_AA,
            )

            # 5 & 6. LEFT & RIGHT HAND DRAWING STATUS
            curr_y += 18
            left_draw_st = drawing_mgr.get_status_text("Left", is_active, left_detected)
            cv2.putText(
                frame,
                f"LEFT DRAW: {left_draw_st}",
                (panel_x + 12, curr_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                config.COLOR_DRAW_LEFT,
                1,
                cv2.LINE_AA,
            )

            curr_y += 17
            right_draw_st = drawing_mgr.get_status_text("Right", is_active, right_detected)
            cv2.putText(
                frame,
                f"RIGHT DRAW: {right_draw_st}",
                (panel_x + 12, curr_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                config.COLOR_DRAW_RIGHT,
                1,
                cv2.LINE_AA,
            )

    def close(self):
        """Clean up MediaPipe resources."""
        self.hands.close()


def open_camera_stream(url: str, max_retries: int = 3, retry_delay: float = 1.0) -> cv2.VideoCapture:
    """
    Attempts to connect to the video stream URL with retry and friendly error messages.
    """
    print(f"[HandTracker] Connecting to camera stream at: {url}")
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

    print("\n" + "=" * 60)
    print(" [ERROR] Cannot connect to Laptop Camera Server!")
    print(f" URL: {url}")
    print(" Please check:")
    print("  1. Laptop is running camera_server.py (Flask / OpenCV stream)")
    print(f"  2. Laptop IP is correct (current: {config.CAMERA_IP})")
    print("  3. Both Desktop and Laptop are on the same Wi-Fi network")
    print("  4. Firewall on Laptop allows incoming connections on port 5000")
    print("=" * 60 + "\n")
    return None


def main():
    """Main execution loop for Phase 1 Hand Tracking with Two-Hand Join Toggle Drawing."""
    print("=" * 65)
    print(" Drawing Monster Defense - Two-Hand Join Toggle Air Drawing")
    print("=" * 65)
    print(f" Camera Stream: {config.CAMERA_URL}")
    print(" Gesture Instructions:")
    print("   - Join both hands together and HOLD for ~0.7s to START drawing.")
    print("   - Separate hands and draw freely in the air (Left, Right, or Both).")
    print("   - Join both hands together and HOLD for ~0.7s again to STOP drawing.")
    print("   - Drawings are kept on screen even after stopping.")
    print(" Controls:")
    print("   [Q] / [ESC] - Quit Application")
    print("   [C] - Clear Canvas Drawings")
    print("   [H] - Toggle Handedness Inversion (Left <-> Right)")
    print("   [F] - Toggle Horizontal Mirroring")
    print("=" * 65)

    # Initialize Camera Stream
    cap = open_camera_stream(config.CAMERA_URL)
    if cap is None:
        sys.exit(1)

    # Initialize Hand Tracker, Gesture Controller, and Drawing Manager
    tracker = HandTracker(invert_handedness=config.INVERT_HANDEDNESS)
    gesture_ctrl = TwoHandJoinGestureController()
    drawing_mgr = AirDrawingManager()
    flip_horizontal = config.FLIP_HORIZONTAL

    # Create named window
    cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    # FPS Calculation
    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[HandTracker] Warning: Failed to receive frame from stream. Reconnecting...")
                cap.release()
                time.sleep(0.5)
                cap = open_camera_stream(config.CAMERA_URL, max_retries=2)
                if cap is None:
                    print("[HandTracker] Stream ended. Exiting.")
                    break
                continue

            # Optional horizontal mirror
            if flip_horizontal:
                frame = cv2.flip(frame, 1)

            current_time = time.time()

            # 1. Process hand tracking
            hand_data, results = tracker.process(frame)

            # 2. Update Two-Hand Join Gesture State Machine
            state_changed = gesture_ctrl.update(hand_data, current_time)
            if state_changed:
                print(f"[Gesture] Event: {gesture_ctrl.last_event} -> Drawing State: {gesture_ctrl.drawing_state.value}")

            # 3. Update Air Drawing trajectories
            drawing_mgr.update(hand_data, gesture_ctrl.is_drawing_active, current_time)

            # 4. Render visual annotations, HUD, strokes, and gesture feedback
            output_frame = tracker.draw(
                frame, hand_data, results, gesture_ctrl, drawing_mgr
            )

            # 5. Calculate FPS
            fps = 0.9 * fps + 0.1 * (1.0 / (current_time - prev_time + 1e-6))
            prev_time = current_time

            # Status bar at bottom
            status_text = (
                f"FPS: {fps:.1f} | Draw: {'ACTIVE' if gesture_ctrl.is_drawing_active else 'IDLE'} | "
                f"Mirror: {'ON' if flip_horizontal else 'OFF'} | "
                f"[Q]uit [C]lear [H]andedness [F]lip"
            )
            h_out, w_out, _ = output_frame.shape
            cv2.putText(
                output_frame,
                status_text,
                (12, h_out - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
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
            elif key in [ord("c"), ord("C")]:
                drawing_mgr.clear()
                print("[DrawingManager] Canvas cleared by user.")
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
        if cap is not None:
            cap.release()
        tracker.close()
        cv2.destroyAllWindows()
        print("[HandTracker] Shutdown complete.")


if __name__ == "__main__":
    main()
