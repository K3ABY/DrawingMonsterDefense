"""
Air Drawing Manager for Drawing Monster Defense.
Maintains separate drawing trajectories for Left and Right hands,
renders smooth anti-aliased strokes on frame, and prepares trajectory data
for future Drawing Recognition without losing data when paused.
"""

import math
import time
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np

import camera_config as config


class AirDrawingManager:
    """
    Records and visualizes hand trajectories for Left and Right hands.
    Preserves all drawn strokes across IDLE and DRAWING_ACTIVE states.
    """

    def __init__(
        self,
        min_point_dist: float = config.MIN_STROKE_POINT_DIST,
        stroke_thickness: int = config.DRAW_STROKE_THICKNESS,
    ):
        self.min_point_dist = min_point_dist
        self.stroke_thickness = stroke_thickness

        # Strokes storage: hand_label -> list of strokes
        # Each stroke is a list of {"x": int, "y": int, "t": float}
        self.strokes: Dict[str, List[List[Dict[str, Any]]]] = {
            "Left": [],
            "Right": [],
        }

        # Current active stroke for each hand
        self.active_stroke: Dict[str, Optional[List[Dict[str, Any]]]] = {
            "Left": None,
            "Right": None,
        }

    def update(
        self,
        hand_data: Dict[str, Any],
        is_drawing_active: bool,
        current_time: Optional[float] = None,
    ) -> None:
        """
        Updates drawing trajectories for detected hands when drawing is active.
        """
        if current_time is None:
            current_time = time.time()

        if is_drawing_active:
            for hand_label in ["Left", "Right"]:
                hand_info = hand_data.get(hand_label)
                if hand_info is not None and hand_info.get("detected", False):
                    tip_x, tip_y = hand_info["index_finger_tip"]

                    if self.active_stroke[hand_label] is None:
                        # Start a new stroke segment
                        new_stroke = [{"x": tip_x, "y": tip_y, "t": current_time}]
                        self.strokes[hand_label].append(new_stroke)
                        self.active_stroke[hand_label] = new_stroke
                    else:
                        # Append point if distance exceeds noise threshold
                        last_pt = self.active_stroke[hand_label][-1]
                        dist = math.hypot(tip_x - last_pt["x"], tip_y - last_pt["y"])
                        if dist >= self.min_point_dist:
                            self.active_stroke[hand_label].append(
                                {"x": tip_x, "y": tip_y, "t": current_time}
                            )
                else:
                    # Hand temporarily out of frame: close active stroke
                    self.active_stroke[hand_label] = None
        else:
            # Drawing is paused/idle: close active strokes but DO NOT ERASE past strokes
            self.active_stroke["Left"] = None
            self.active_stroke["Right"] = None

    def clear(self) -> None:
        """Clears all drawing strokes when explicitly requested by user."""
        self.strokes["Left"].clear()
        self.strokes["Right"].clear()
        self.active_stroke["Left"] = None
        self.active_stroke["Right"] = None

    def get_total_points(self, hand_label: str) -> int:
        """Calculates total points recorded for the given hand."""
        return sum(len(stroke) for stroke in self.strokes.get(hand_label, []))

    def get_total_strokes(self, hand_label: str) -> int:
        """Calculates total number of stroke segments for the given hand."""
        return len(self.strokes.get(hand_label, []))

    def get_status_text(self, hand_label: str, is_drawing_active: bool, hand_detected: bool) -> str:
        """Returns human-readable drawing status for HUD display."""
        pts = self.get_total_points(hand_label)
        n_strokes = self.get_total_strokes(hand_label)
        if is_drawing_active and hand_detected and self.active_stroke[hand_label] is not None:
            return f"DRAWING ({pts} pts, {n_strokes} strk)"
        elif pts > 0:
            return f"IDLE ({pts} pts, {n_strokes} strk)"
        else:
            return "IDLE (0 pts)"

    def export_data(self) -> Dict[str, Any]:
        """
        Exports clean trajectory data formatted for future Drawing Recognition.
        """
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
        """
        Draws all recorded strokes onto the frame.
        Active index fingertips receive an animated glowing pen marker.
        """
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

        # Draw drawing nib / pen glow on active fingertips
        if is_drawing_active:
            for hand_label in ["Left", "Right"]:
                hand_info = hand_data.get(hand_label)
                if hand_info is not None and hand_info.get("detected", False):
                    tip = hand_info["index_finger_tip"]
                    pen_color = (
                        config.COLOR_DRAW_LEFT if hand_label == "Left" else config.COLOR_DRAW_RIGHT
                    )
                    # Outer glow ring
                    cv2.circle(output, tip, 14, pen_color, 2, cv2.LINE_AA)
                    # Core drawing point
                    cv2.circle(output, tip, 5, (255, 255, 255), -1, cv2.LINE_AA)

        return output

