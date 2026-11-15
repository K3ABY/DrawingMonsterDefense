"""
Air Drawing Manager for Drawing Monster Defense.
Maintains separate drawing trajectories for Left and Right hands with:
- One-Hand Fist Detection (stops drawing for that hand only)
- Short-Term Tracking Loss Interpolation (prevents stroke fragmentation)
- Long-Term Tracking Loss Stroke Break (prevents spurious cross-screen lines)
- Real-time Point Update Rate Tracking
"""

import math
import time
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np

import camera_config as config


class AirDrawingManager:
    """
    Manages air drawing trajectories, stroke continuity, fist pause,
    and trajectory export for drawing recognition.
    """

    def __init__(
        self,
        min_point_dist: float = config.MIN_STROKE_POINT_DIST,
        max_bridge_time: float = config.MAX_BRIDGE_GAP_TIME,
        max_bridge_dist: float = config.MAX_BRIDGE_GAP_DIST,
        stroke_thickness: int = config.DRAW_STROKE_THICKNESS,
    ):
        self.min_point_dist = min_point_dist
        self.max_bridge_time = max_bridge_time
        self.max_bridge_dist = max_bridge_dist
        self.stroke_thickness = stroke_thickness

        # Strokes storage: hand_label -> list of strokes
        # Each stroke is a list of {"x": int, "y": int, "t": float}
        self.strokes: Dict[str, List[List[Dict[str, Any]]]] = {
            "Left": [],
            "Right": [],
        }

        # Active stroke segment currently being drawn
        self.active_stroke: Dict[str, Optional[List[Dict[str, Any]]]] = {
            "Left": None,
            "Right": None,
        }

        # Posture & Tracking continuity tracking
        self.posture_state: Dict[str, str] = {
            "Left": "UNKNOWN",
            "Right": "UNKNOWN",
        }
        self.posture_streak: Dict[str, int] = {
            "Left": 0,
            "Right": 0,
        }
        self.last_seen_point: Dict[str, Optional[Tuple[int, int]]] = {
            "Left": None,
            "Right": None,
        }
        self.last_seen_time: Dict[str, float] = {
            "Left": 0.0,
            "Right": 0.0,
        }

        # Metrics: Points update rate per second
        self.point_rate: Dict[str, float] = {"Left": 0.0, "Right": 0.0}
        self._rate_pts_counter: Dict[str, int] = {"Left": 0, "Right": 0}
        self._rate_timer: Dict[str, float] = {"Left": time.time(), "Right": time.time()}

    @staticmethod
    def _analyze_posture(landmarks: List[Tuple[int, int]]) -> Tuple[bool, bool]:
        """
        Analyzes 21 landmarks to determine if hand is POINTING or FIST.
        Returns:
            (is_pointing, is_fist)
        """
        if len(landmarks) < 21:
            return False, False

        wrist = landmarks[config.LANDMARK_WRIST]

        def dist(p1, p2):
            return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

        # Finger tip and PIP distances relative to wrist
        d_tip8 = dist(landmarks[config.LANDMARK_INDEX_FINGER_TIP], wrist)
        d_pip6 = dist(landmarks[config.LANDMARK_INDEX_PIP], wrist)
        d_mcp5 = dist(landmarks[config.LANDMARK_INDEX_MCP], wrist)

        d_tip12 = dist(landmarks[config.LANDMARK_MIDDLE_TIP], wrist)
        d_pip10 = dist(landmarks[config.LANDMARK_MIDDLE_PIP], wrist)

        d_tip16 = dist(landmarks[config.LANDMARK_RING_TIP], wrist)
        d_pip14 = dist(landmarks[config.LANDMARK_RING_PIP], wrist)

        d_tip20 = dist(landmarks[config.LANDMARK_PINKY_TIP], wrist)
        d_pip18 = dist(landmarks[config.LANDMARK_PINKY_PIP], wrist)

        # Index extended: tip is further from wrist than PIP
        index_extended = (d_tip8 > d_pip6 * 1.08) and (d_tip8 > d_mcp5 * 1.20)

        # Fist condition: all 4 non-thumb fingers are curled close to palm/wrist
        fist_curled = (
            (d_tip8 < d_pip6 * 1.05)
            and (d_tip12 < d_pip10 * 1.05)
            and (d_tip16 < d_pip14 * 1.05)
            and (d_tip20 < d_pip18 * 1.05)
        )

        is_fist = fist_curled
        is_pointing = index_extended and not is_fist

        return is_pointing, is_fist

    def _update_posture_with_hysteresis(self, hand_label: str, raw_pointing: bool, raw_fist: bool) -> str:
        """Applies 2-frame debouncing to eliminate single-frame posture noise."""
        candidate = "POINTING" if raw_pointing else "FIST" if raw_fist else "RELAXED"

        if candidate == self.posture_state[hand_label]:
            self.posture_streak[hand_label] = 0
            return self.posture_state[hand_label]

        # Count streak towards candidate
        self.posture_streak[hand_label] += 1
        confirm_needed = config.FIST_CONFIRM_FRAMES if candidate == "FIST" else config.POINTING_CONFIRM_FRAMES

        if self.posture_streak[hand_label] >= confirm_needed or self.posture_state[hand_label] == "UNKNOWN":
            self.posture_state[hand_label] = candidate
            self.posture_streak[hand_label] = 0

        return self.posture_state[hand_label]

    def update(
        self,
        hand_data: Dict[str, Any],
        is_drawing_active: bool,
        current_time: Optional[float] = None,
    ) -> None:
        """
        Updates drawing trajectories and applies short-term gap bridging.
        """
        if current_time is None:
            current_time = time.time()

        for hand_label in ["Left", "Right"]:
            # Update points/sec rate counter
            dt_rate = current_time - self._rate_timer[hand_label]
            if dt_rate >= 1.0:
                self.point_rate[hand_label] = self._rate_pts_counter[hand_label] / dt_rate
                self._rate_pts_counter[hand_label] = 0
                self._rate_timer[hand_label] = current_time

            hand_info = hand_data.get(hand_label)
            is_detected = (hand_info is not None) and hand_info.get("detected", False)

            if not is_drawing_active:
                # Mode is IDLE: Close active stroke, but PRESERVE past strokes
                self.active_stroke[hand_label] = None
                self.last_seen_point[hand_label] = None
                continue

            if is_detected:
                landmarks = hand_info.get("landmarks", [])
                raw_pointing, raw_fist = self._analyze_posture(landmarks)
                posture = self._update_posture_with_hysteresis(hand_label, raw_pointing, raw_fist)

                tip_x, tip_y = hand_info["index_finger_tip"]
                curr_pt = (tip_x, tip_y)

                if posture == "FIST":
                    # Fist stops drawing for this hand only!
                    self.active_stroke[hand_label] = None
                    self.last_seen_point[hand_label] = curr_pt
                    self.last_seen_time[hand_label] = current_time
                    continue

                if posture == "POINTING":
                    # Drawing is allowed for this hand
                    last_pt = self.last_seen_point[hand_label]
                    last_t = self.last_seen_time[hand_label]

                    if self.active_stroke[hand_label] is None or last_pt is None:
                        # Start new stroke segment
                        new_stroke = [{"x": tip_x, "y": tip_y, "t": current_time}]
                        self.strokes[hand_label].append(new_stroke)
                        self.active_stroke[hand_label] = new_stroke
                        self._rate_pts_counter[hand_label] += 1
                    else:
                        dt = current_time - last_t
                        dist = math.hypot(tip_x - last_pt[0], tip_y - last_pt[1])

                        # Case A: Short tracking loss (Gap Bridging Interpolation)
                        if (0.08 < dt <= self.max_bridge_time) and (dist <= self.max_bridge_dist):
                            # Interpolate intermediate bridge point
                            mid_x = (last_pt[0] + tip_x) // 2
                            mid_y = (last_pt[1] + tip_y) // 2
                            mid_t = (last_t + current_time) / 2.0
                            self.active_stroke[hand_label].append({"x": mid_x, "y": mid_y, "t": mid_t})
                            self.active_stroke[hand_label].append({"x": tip_x, "y": tip_y, "t": current_time})
                            self._rate_pts_counter[hand_label] += 2

                        # Case B: Continuous normal movement
                        elif dt <= self.max_bridge_time and dist >= self.min_point_dist:
                            self.active_stroke[hand_label].append({"x": tip_x, "y": tip_y, "t": current_time})
                            self._rate_pts_counter[hand_label] += 1

                        # Case C: Long tracking loss or teleport distance -> Break stroke!
                        elif dt > self.max_bridge_time or dist > self.max_bridge_dist:
                            new_stroke = [{"x": tip_x, "y": tip_y, "t": current_time}]
                            self.strokes[hand_label].append(new_stroke)
                            self.active_stroke[hand_label] = new_stroke
                            self._rate_pts_counter[hand_label] += 1

                    self.last_seen_point[hand_label] = curr_pt
                    self.last_seen_time[hand_label] = current_time
            else:
                # Hand lost in current frame: check timeout to break stroke
                if current_time - self.last_seen_time[hand_label] > self.max_bridge_time:
                    self.active_stroke[hand_label] = None
                    self.last_seen_point[hand_label] = None

    def clear(self) -> None:
        """Clears all drawing strokes when explicitly requested by user."""
        self.strokes["Left"].clear()
        self.strokes["Right"].clear()
        self.active_stroke["Left"] = None
        self.active_stroke["Right"] = None
        self.last_seen_point["Left"] = None
        self.last_seen_point["Right"] = None

    def get_total_points(self, hand_label: str) -> int:
        return sum(len(stroke) for stroke in self.strokes.get(hand_label, []))

    def get_total_strokes(self, hand_label: str) -> int:
        return len(self.strokes.get(hand_label, []))

    def get_hand_state_desc(self, hand_label: str, is_detected: bool) -> str:
        """Returns posture description for HUD metrics."""
        if not is_detected:
            return "Lost"
        posture = self.posture_state[hand_label]
        if posture == "POINTING":
            return "Pointing (Draw)"
        elif posture == "FIST":
            return "Fist (Paused)"
        else:
            return "Open (Idle)"

    def get_status_text(self, hand_label: str, is_drawing_active: bool, is_detected: bool) -> str:
        pts = self.get_total_points(hand_label)
        n_strokes = self.get_total_strokes(hand_label)
        rate = self.point_rate[hand_label]

        if not is_drawing_active:
            return f"IDLE ({pts} pts, {n_strokes} strk)"

        if is_detected:
            posture = self.posture_state[hand_label]
            if posture == "FIST":
                return f"FIST PAUSED ({pts} pts)"
            elif posture == "POINTING":
                return f"DRAWING ({pts} pts, {rate:.1f} pts/s)"
            else:
                return f"READY ({pts} pts)"
        else:
            return f"NO HAND ({pts} pts)"

    def export_data(self) -> Dict[str, Any]:
        """Prepares clean trajectory data for drawing recognition."""
        return {
            "Left": {
                "strokes": [
                    [(pt["x"], pt["y"], pt["t"]) for pt in stroke]
                    for stroke in self.strokes["Left"]
                ],
                "total_points": self.get_total_points("Left"),
                "total_strokes": self.get_total_strokes("Left"),
            },
            "Right": {
                "strokes": [
                    [(pt["x"], pt["y"], pt["t"]) for pt in stroke]
                    for stroke in self.strokes["Right"]
                ],
                "total_points": self.get_total_points("Right"),
                "total_strokes": self.get_total_strokes("Right"),
            },
        }

    def render_strokes(
        self,
        frame: np.ndarray,
        is_drawing_active: bool,
        hand_data: Dict[str, Any],
    ) -> np.ndarray:
        """Renders persistent strokes and drawing nib / fist indicator."""
        output = frame.copy()

        # Render Left hand strokes (Cyan)
        for stroke in self.strokes["Left"]:
            for i in range(1, len(stroke)):
                p1 = (stroke[i - 1]["x"], stroke[i - 1]["y"])
                p2 = (stroke[i]["x"], stroke[i]["y"])
                cv2.line(output, p1, p2, config.COLOR_DRAW_LEFT, self.stroke_thickness, cv2.LINE_AA)

        # Render Right hand strokes (Orange)
        for stroke in self.strokes["Right"]:
            for i in range(1, len(stroke)):
                p1 = (stroke[i - 1]["x"], stroke[i - 1]["y"])
                p2 = (stroke[i]["x"], stroke[i]["y"])
                cv2.line(output, p1, p2, config.COLOR_DRAW_RIGHT, self.stroke_thickness, cv2.LINE_AA)

        # Draw active pen or fist indicator
        if is_drawing_active:
            for hand_label in ["Left", "Right"]:
                hand_info = hand_data.get(hand_label)
                if hand_info is not None and hand_info.get("detected", False):
                    tip = hand_info["index_finger_tip"]
                    posture = self.posture_state[hand_label]

                    if posture == "POINTING":
                        pen_color = (
                            config.COLOR_DRAW_LEFT if hand_label == "Left" else config.COLOR_DRAW_RIGHT
                        )
                        cv2.circle(output, tip, 13, pen_color, 2, cv2.LINE_AA)
                        cv2.circle(output, tip, 4, (255, 255, 255), -1, cv2.LINE_AA)
                    elif posture == "FIST":
                        cv2.circle(output, tip, 10, config.COLOR_FIST_PAUSED, 2, cv2.LINE_AA)
                        cv2.circle(output, tip, 4, config.COLOR_FIST_PAUSED, -1, cv2.LINE_AA)

        return output
