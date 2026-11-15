"""
Two-Hand Join Gesture Controller for Drawing Monster Defense.
Implements Two-Hand Join detection in Toggle Mode with State Machine,
Hold Timer (~0.7s), Hysteresis, and Debounce Lock (Wait for Release).
"""

import math
import time
from enum import Enum
from typing import Optional, Tuple, Dict, Any

try:
    import src.config.camera_config as config
except ImportError:
    import camera_config as config


class DrawingState(Enum):
    IDLE = "IDLE"
    DRAWING_ACTIVE = "DRAWING_ACTIVE"


class GestureState(Enum):
    NO_JOIN = "NO_JOIN"
    JOIN_HOLDING = "JOIN_HOLDING"
    WAIT_RELEASE = "WAIT_RELEASE"


class TwoHandJoinGestureController:
    """
    Manages detection of Two-Hand Join Gesture, Hold Confirmation,
    and State Machine toggling between IDLE and DRAWING_ACTIVE.
    """

    def __init__(
        self,
        hold_duration: float = config.JOIN_HOLD_DURATION,
        join_distance: float = config.JOIN_DISTANCE_THRESHOLD,
        release_distance: float = config.RELEASE_DISTANCE_THRESHOLD,
        cooldown: float = config.GESTURE_COOLDOWN,
    ):
        self.hold_duration = hold_duration
        self.join_distance = join_distance
        self.release_distance = release_distance
        self.cooldown = cooldown

        # State Machine
        self.drawing_state: DrawingState = DrawingState.IDLE
        self.gesture_state: GestureState = GestureState.NO_JOIN

        # Timers and Debounce Tracking
        self.hold_start_time: Optional[float] = None
        self.hold_progress: float = 0.0
        self.cooldown_until: float = 0.0

        # Event tracking
        self.last_event: str = "NONE"
        self.last_event_time: float = 0.0

        # Geometric tracking for visual feedback
        self.join_center: Optional[Tuple[int, int]] = None
        self.current_distance: float = 999.0
        self.is_hands_joined: bool = False
        self.is_hands_released: bool = True

    @staticmethod
    def _calc_dist(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """Euclidean distance between two pixel coordinates."""
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def evaluate_hands(self, hand_data: Dict[str, Any]) -> Tuple[bool, bool, Optional[Tuple[int, int]], float]:
        """
        Calculates geometric metrics between Left and Right hands.
        Returns:
            (is_joined, is_released, join_center, min_distance)
        """
        left = hand_data.get("Left")
        right = hand_data.get("Right")

        if left is None or right is None:
            # When one or both hands are absent, hands cannot be joined, and are considered released
            return False, True, None, 999.0

        # Extract landmarks
        l_tip = left["index_finger_tip"]
        r_tip = right["index_finger_tip"]

        l_wrist = left["landmarks"][config.LANDMARK_WRIST]
        r_wrist = right["landmarks"][config.LANDMARK_WRIST]

        l_mcp = left["landmarks"][config.LANDMARK_MIDDLE_MCP]
        r_mcp = right["landmarks"][config.LANDMARK_MIDDLE_MCP]

        # Distances
        d_tips = self._calc_dist(l_tip, r_tip)
        d_palms = self._calc_dist(l_mcp, r_mcp)
        min_dist = min(d_tips, d_palms)

        # Scale estimation (wrist to middle MCP)
        s_left = max(20.0, self._calc_dist(l_wrist, l_mcp))
        s_right = max(20.0, self._calc_dist(r_wrist, r_mcp))
        avg_scale = (s_left + s_right) / 2.0

        # Hysteresis & Scale-Normalized checks
        norm_dist_tips = d_tips / avg_scale
        norm_dist_palms = d_palms / avg_scale

        is_joined = (
            (d_tips <= self.join_distance or norm_dist_tips <= 0.85)
            or (d_palms <= (self.join_distance + 15) or norm_dist_palms <= 0.95)
        )

        is_released = (
            (d_tips >= self.release_distance and norm_dist_tips >= 1.25)
            and (d_palms >= self.release_distance and norm_dist_palms >= 1.25)
        )

        # Compute midpoint for visual feedback
        if d_tips <= d_palms:
            center = ((l_tip[0] + r_tip[0]) // 2, (l_tip[1] + r_tip[1]) // 2)
        else:
            center = ((l_mcp[0] + r_mcp[0]) // 2, (l_mcp[1] + r_mcp[1]) // 2)

        return is_joined, is_released, center, min_dist

    def update(self, hand_data: Dict[str, Any], current_time: Optional[float] = None) -> bool:
        """
        Updates the gesture state machine.
        Returns:
            bool: True if a state transition occurred in this frame, False otherwise.
        """
        if current_time is None:
            current_time = time.time()

        is_joined, is_released, center, min_dist = self.evaluate_hands(hand_data)
        self.is_hands_joined = is_joined
        self.is_hands_released = is_released
        self.join_center = center
        self.current_distance = min_dist

        state_changed = False

        # State 1: WAIT_RELEASE (Debounce Lock)
        # Prevents re-triggering while hands remain joined after a toggle.
        if self.gesture_state == GestureState.WAIT_RELEASE:
            self.hold_progress = 0.0
            self.hold_start_time = None
            if is_released:
                # Hands have separated -> Lock released!
                self.gesture_state = GestureState.NO_JOIN

        # State 2: NO_JOIN
        elif self.gesture_state == GestureState.NO_JOIN:
            self.hold_progress = 0.0
            if is_joined and (current_time >= self.cooldown_until):
                self.gesture_state = GestureState.JOIN_HOLDING
                self.hold_start_time = current_time

        # State 3: JOIN_HOLDING
        elif self.gesture_state == GestureState.JOIN_HOLDING:
            if is_joined:
                elapsed = current_time - (self.hold_start_time or current_time)
                self.hold_progress = min(1.0, elapsed / self.hold_duration)

                if elapsed >= (self.hold_duration - 1e-3):
                    # TOGGLE TRIGGERED!
                    if self.drawing_state == DrawingState.IDLE:
                        self.drawing_state = DrawingState.DRAWING_ACTIVE
                        self.last_event = "START_DRAWING"
                    else:
                        self.drawing_state = DrawingState.IDLE
                        self.last_event = "STOP_DRAWING"

                    self.last_event_time = current_time
                    self.cooldown_until = current_time + self.cooldown
                    self.gesture_state = GestureState.WAIT_RELEASE
                    self.hold_start_time = None
                    self.hold_progress = 1.0
                    state_changed = True
            else:
                # Join gesture was released before hold time was satisfied
                self.gesture_state = GestureState.NO_JOIN
                self.hold_start_time = None
                self.hold_progress = 0.0

        return state_changed

    @property
    def is_drawing_active(self) -> bool:
        """Returns True if the drawing state is currently active."""
        return self.drawing_state == DrawingState.DRAWING_ACTIVE

    @property
    def gesture_status_text(self) -> str:
        """Friendly text representation of current gesture status for HUD."""
        if self.gesture_state == GestureState.WAIT_RELEASE:
            return "LOCKED (WAIT RELEASE)"
        elif self.gesture_state == GestureState.JOIN_HOLDING:
            pct = int(self.hold_progress * 100)
            return f"HOLDING ({pct}%)"
        elif self.is_hands_joined:
            return "JOIN DETECTED"
        else:
            return "NO JOIN"

    @property
    def hold_timer_text(self) -> str:
        """Displays current elapsed hold time vs target."""
        if self.gesture_state == GestureState.JOIN_HOLDING and self.hold_start_time is not None:
            elapsed = time.time() - self.hold_start_time
            return f"{min(elapsed, self.hold_duration):.2f}s / {self.hold_duration:.2f}s"
        elif self.gesture_state == GestureState.WAIT_RELEASE:
            return f"{self.hold_duration:.2f}s (CONFIRMED)"
        else:
            return f"0.00s / {self.hold_duration:.2f}s"
