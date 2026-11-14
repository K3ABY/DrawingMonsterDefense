"""
Air Drawing Manager for Drawing Monster Defense with 3D Index Finger Recognition.
Features:
- Robust 3D Index Finger Pose Recognition across all directions (Up, Down, Forward, Sideways, Diagonal)
- Rotation and palm-angle invariant (uses 3D joint angles and chord-to-segment ratios)
- Natural finger variation tolerance (does not require tight clenching of unused fingers)
- Dual-Zone 50/50 Canvas Division (Left Zone & Right Zone)
- Boundary crossing handling: stroke cutoff on leaving, new stroke on re-entry, no cross-zone lines
- Fist Detection pause per hand & short-term gap recovery
"""

import math
import time
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np

import camera_config as config


class TrackingConfidenceState(Enum):
    TRACKING_STABLE = "TRACKING_STABLE"
    TRACKING_UNCERTAIN = "TRACKING_UNCERTAIN"
    TRACKING_LOST = "TRACKING_LOST"
    TRACKING_RECOVERED = "TRACKING_RECOVERED"


class AirDrawingManager:
    """
    Manages dual-zone air drawing trajectories, 3D index-pointing recognition,
    confidence-aware safe drawing states, zone boundary enforcement, and gap recovery.
    """

    def __init__(
        self,
        zone_split_ratio: float = config.ZONE_SPLIT_RATIO,
        min_point_dist: float = config.MIN_STROKE_POINT_DIST,
        max_bridge_time: float = config.MAX_BRIDGE_GAP_TIME,
        max_bridge_dist: float = config.MAX_BRIDGE_GAP_DIST,
        stroke_thickness: int = config.DRAW_STROKE_THICKNESS,
    ):
        self.zone_split_ratio = zone_split_ratio
        self.min_point_dist = min_point_dist
        self.max_bridge_time = max_bridge_time
        self.max_bridge_dist = max_bridge_dist
        self.stroke_thickness = stroke_thickness

        # Strokes storage: hand_label -> list of strokes
        self.strokes: Dict[str, List[List[Dict[str, Any]]]] = {
            "Left": [],
            "Right": [],
        }

        # Active stroke segment currently being drawn
        self.active_stroke: Dict[str, Optional[List[Dict[str, Any]]]] = {
            "Left": None,
            "Right": None,
        }

        # Posture & 3D Recognition status
        self.posture_state: Dict[str, str] = {
            "Left": "UNKNOWN",
            "Right": "UNKNOWN",
        }
        self.posture_streak: Dict[str, int] = {
            "Left": 0,
            "Right": 0,
        }
        self.is_pointing_recognized: Dict[str, bool] = {
            "Left": False,
            "Right": False,
        }
        self.pointing_direction: Dict[str, str] = {
            "Left": "NONE",
            "Right": "NONE",
        }

        # Confidence-Aware Tracking States (Section 6)
        self.tracking_confidence_state: Dict[str, TrackingConfidenceState] = {
            "Left": TrackingConfidenceState.TRACKING_LOST,
            "Right": TrackingConfidenceState.TRACKING_LOST,
        }
        self.tracking_confidence_score: Dict[str, float] = {
            "Left": 0.0,
            "Right": 0.0,
        }
        self.hand_detection_confidence: Dict[str, float] = {
            "Left": 0.0,
            "Right": 0.0,
        }
        self.index_pose_confidence: Dict[str, float] = {
            "Left": 0.0,
            "Right": 0.0,
        }
        self.recovery_streak: Dict[str, int] = {
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

        # Boundary status tracking per hand
        self.boundary_status: Dict[str, str] = {
            "Left": "NO_HAND",
            "Right": "NO_HAND",
        }
        self.last_norm_coords: Dict[str, Optional[Tuple[float, float]]] = {
            "Left": None,
            "Right": None,
        }
        self.last_zone_coords: Dict[str, Optional[Tuple[float, float]]] = {
            "Left": None,
            "Right": None,
        }

        # Metrics: Points update rate per second
        self.point_rate: Dict[str, float] = {"Left": 0.0, "Right": 0.0}
        self._rate_pts_counter: Dict[str, int] = {"Left": 0, "Right": 0}
        self._rate_timer: Dict[str, float] = {"Left": time.time(), "Right": time.time()}

    @staticmethod
    def _norm3d(p1: Tuple[float, ...], p2: Tuple[float, ...]) -> float:
        """Euclidean distance in 2D or 3D space."""
        dim = min(len(p1), len(p2))
        return math.sqrt(sum((p1[i] - p2[i]) ** 2 for i in range(dim)))

    @classmethod
    def _finger_extension_3d(
        cls,
        mcp: Tuple[float, ...],
        pip: Tuple[float, ...],
        dip: Tuple[float, ...],
        tip: Tuple[float, ...],
        wrist: Optional[Tuple[float, ...]] = None,
    ) -> Tuple[bool, bool, float, float]:
        """
        Computes 3D straightness ratio, PIP angle cosine, and curl status.
        Gracefully handles both full 21-landmark MediaPipe models and degenerate/missing DIP joints.
        Returns:
            (is_extended, is_curled, straightness_ratio, cos_pip)
        """
        d_pip_tip = cls._norm3d(pip, tip)
        d_pip_dip = cls._norm3d(pip, dip)
        d_mcp_pip = cls._norm3d(mcp, pip)

        # Handle synthetic/unpopulated mock landmarks where MCP and PIP are collocated at wrist
        if d_mcp_pip < 1e-4:
            d_tip_mcp = cls._norm3d(tip, mcp)
            is_ext = d_tip_mcp > 15.0
            return is_ext, not is_ext, 1.0, 1.0

        # Anatomically DIP must lie between PIP and Tip: dist(pip, dip) <= dist(pip, tip)
        dip_valid = (
            (d_pip_dip <= d_pip_tip * 1.08)
            and (dip != wrist)
            and (d_pip_dip > 1e-4)
        )

        if dip_valid:
            seg_sum = cls._norm3d(mcp, pip) + d_pip_dip + cls._norm3d(dip, tip)
            dim = min(len(mcp), len(pip), len(dip))
            v1 = [pip[i] - mcp[i] for i in range(dim)]
            v2 = [dip[i] - pip[i] for i in range(dim)]
        else:
            # Fallback to MCP -> PIP -> Tip if DIP is degenerate or unpopulated
            seg_sum = cls._norm3d(mcp, pip) + d_pip_tip
            dim = min(len(mcp), len(pip), len(tip))
            v1 = [pip[i] - mcp[i] for i in range(dim)]
            v2 = [tip[i] - pip[i] for i in range(dim)]

        chord = cls._norm3d(mcp, tip)
        ratio = chord / seg_sum if seg_sum > 0 else 0.0

        dot = sum(v1[i] * v2[i] for i in range(dim))
        l1 = math.sqrt(sum(v1[i] ** 2 for i in range(dim)))
        l2 = math.sqrt(sum(v2[i] ** 2 for i in range(dim)))
        cos_pip = dot / (l1 * l2) if (l1 * l2) > 0 else 0.0

        # Curl detection:
        # A finger is curled if its 3D straightness is low OR if its tip is curled back toward the wrist/palm
        curled_towards_wrist = False
        if wrist is not None:
            d_tip_wrist = cls._norm3d(tip, wrist)
            d_pip_wrist = cls._norm3d(pip, wrist)
            curled_towards_wrist = (d_tip_wrist < d_pip_wrist * 1.05)

        is_curled = (ratio <= config.FIST_MAX_STRAIGHTNESS) or (cos_pip <= 0.35) or curled_towards_wrist
        is_extended = (
            (ratio >= config.INDEX_STRAIGHTNESS_MIN)
            and (cos_pip >= config.INDEX_PIP_ANGLE_MIN_COS)
            and not curled_towards_wrist
        )

        return is_extended, is_curled, ratio, cos_pip

    @classmethod
    def _analyze_posture_3d(
        cls, landmarks: List[Tuple[float, ...]], return_confidence: bool = False
    ) -> Any:
        """
        Robust 3D pose classification for Index-Pointing vs Fist vs Open Palm.
        Evaluates 3D landmark geometry, joint collinearity, and palm extension ratio.
        Invariant to pointing direction (Forward, Up, Down, Sideways, Diagonal).
        Returns:
            (is_pointing, is_fist, posture_label, pointing_direction) or
            (is_pointing, is_fist, posture_label, pointing_direction, pose_confidence)
        """
        if len(landmarks) < 21:
            if return_confidence:
                return False, False, "UNKNOWN", "NONE", 0.0
            return False, False, "UNKNOWN", "NONE"

        wrist = landmarks[config.LANDMARK_WRIST]
        palm_center = landmarks[config.LANDMARK_MIDDLE_MCP]

        # 1. 3D Extension of all 4 fingers
        idx_ext, idx_crl, idx_r, _ = cls._finger_extension_3d(
            landmarks[config.LANDMARK_INDEX_MCP],
            landmarks[config.LANDMARK_INDEX_PIP],
            landmarks[config.LANDMARK_INDEX_DIP],
            landmarks[config.LANDMARK_INDEX_FINGER_TIP],
            wrist=wrist,
        )

        mid_ext, mid_crl, mid_r, _ = cls._finger_extension_3d(
            landmarks[config.LANDMARK_MIDDLE_MCP],
            landmarks[config.LANDMARK_MIDDLE_PIP],
            landmarks[config.LANDMARK_MIDDLE_DIP],
            landmarks[config.LANDMARK_MIDDLE_TIP],
            wrist=wrist,
        )

        rng_ext, rng_crl, rng_r, _ = cls._finger_extension_3d(
            landmarks[config.LANDMARK_RING_MCP],
            landmarks[config.LANDMARK_RING_PIP],
            landmarks[config.LANDMARK_RING_DIP],
            landmarks[config.LANDMARK_RING_TIP],
            wrist=wrist,
        )

        pnk_ext, pnk_crl, pnk_r, _ = cls._finger_extension_3d(
            landmarks[config.LANDMARK_PINKY_MCP],
            landmarks[config.LANDMARK_PINKY_PIP],
            landmarks[config.LANDMARK_PINKY_DIP],
            landmarks[config.LANDMARK_PINKY_TIP],
            wrist=wrist,
        )

        # 2. Relative Palm Extension: Index Tip vs Middle/Ring Tip distance from palm center
        d_idx_palm = cls._norm3d(palm_center, landmarks[config.LANDMARK_INDEX_FINGER_TIP])
        d_mid_palm = cls._norm3d(palm_center, landmarks[config.LANDMARK_MIDDLE_TIP])
        palm_ratio = d_idx_palm / max(d_mid_palm, 1e-4)

        # 3. Pointing Direction Calculation (Vector from Index MCP to Tip)
        idx_mcp = landmarks[config.LANDMARK_INDEX_MCP]
        idx_tip = landmarks[config.LANDMARK_INDEX_FINGER_TIP]
        dx = idx_tip[0] - idx_mcp[0]
        dy = idx_tip[1] - idx_mcp[1]
        dz = idx_tip[2] - idx_mcp[2] if len(idx_tip) >= 3 and len(idx_mcp) >= 3 else 0.0

        # Classify dominant 3D pointing direction
        abs_x, abs_y, abs_z = abs(dx), abs(dy), abs(dz)
        if abs_z > abs_x * 1.2 and abs_z > abs_y * 1.2:
            direction = "FORWARD" if dz < 0 else "BACKWARD"
        elif abs_y > abs_x * 1.2 and abs_y > abs_z * 1.2:
            direction = "UP" if dy < 0 else "DOWN"
        elif abs_x > abs_y * 1.2 and abs_x > abs_z * 1.2:
            direction = "SIDEWAYS"
        else:
            direction = "DIAGONAL"

        # 4. Posture Classification
        # Fist: Index, Middle, and Ring are all curled
        is_fist = idx_crl and mid_crl and rng_crl

        # Open Hand: Index, Middle, and Ring are all extended and not curled into palm
        is_open = (
            idx_ext
            and mid_ext
            and rng_ext
            and not mid_crl
            and not rng_crl
            and (palm_ratio < config.INDEX_PALM_EXTENSION_RATIO)
        )

        # Index Pointing:
        # Index is straight/extended, not open hand, not fist, and index extends past middle finger
        is_pointing = (
            idx_ext
            and not is_open
            and not is_fist
            and (palm_ratio >= config.INDEX_PALM_EXTENSION_RATIO or mid_crl or not mid_ext)
        )

        posture = "POINTING" if is_pointing else "FIST" if is_fist else "OPEN" if is_open else "RELAXED"

        # Pose confidence evaluation
        straightness_conf = min(1.0, max(0.0, (idx_r - 0.50) / 0.50))
        palm_conf = min(1.0, max(0.0, (palm_ratio - 1.0) / 0.30))
        pose_conf = float(round(0.6 * straightness_conf + 0.4 * palm_conf, 3))
        if is_fist:
            pose_conf = 0.95
        elif is_open:
            pose_conf = 0.90
        elif not is_pointing:
            pose_conf = 0.50

        if return_confidence:
            return is_pointing, is_fist, posture, direction, pose_conf

        return is_pointing, is_fist, posture, direction

    def _update_posture_with_hysteresis(self, hand_label: str, raw_pointing: bool, raw_fist: bool, raw_posture: str) -> str:
        """Applies responsive debouncing to eliminate single-frame posture noise."""
        candidate = "POINTING" if raw_pointing else "FIST" if raw_fist else raw_posture

        if candidate == self.posture_state[hand_label]:
            self.posture_streak[hand_label] = 0
            return self.posture_state[hand_label]

        self.posture_streak[hand_label] += 1
        confirm_needed = config.FIST_CONFIRM_FRAMES if candidate == "FIST" else config.POINTING_CONFIRM_FRAMES

        if self.posture_streak[hand_label] >= confirm_needed or self.posture_state[hand_label] == "UNKNOWN":
            self.posture_state[hand_label] = candidate
            self.posture_streak[hand_label] = 0

        return self.posture_state[hand_label]

    def check_zone_boundary(
        self, hand_label: str, tip_x: int, frame_width: int
    ) -> Tuple[bool, float, float]:
        """
        Enforces Dual-Zone hand-to-zone assignment:
        - LEFT HAND: Allowed only in X < 50% (Left Zone)
        - RIGHT HAND: Allowed only in X >= 50% (Right Zone)
        """
        x_split = frame_width * self.zone_split_ratio
        norm_x = max(0.0, min(1.0, tip_x / frame_width))

        if hand_label == "Left":
            is_in_zone = (tip_x < x_split)
            zone_x = max(0.0, min(1.0, tip_x / x_split)) if x_split > 0 else 0.0
        else:
            is_in_zone = (tip_x >= x_split)
            right_width = frame_width - x_split
            zone_x = max(0.0, min(1.0, (tip_x - x_split) / right_width)) if right_width > 0 else 0.0

        return is_in_zone, norm_x, zone_x

    def update(
        self,
        hand_data: Dict[str, Any],
        is_drawing_active: bool,
        current_time: Optional[float] = None,
        frame_width: int = 640,
        frame_height: int = 480,
    ) -> None:
        """
        Updates dual-zone drawing trajectories with 3D index-pointing recognition,
        strict zone boundary enforcement, and fast gap recovery.
        """
        if current_time is None:
            current_time = time.time()

        for hand_label in ["Left", "Right"]:
            dt_rate = current_time - self._rate_timer[hand_label]
            if dt_rate >= 1.0:
                self.point_rate[hand_label] = self._rate_pts_counter[hand_label] / dt_rate
                self._rate_pts_counter[hand_label] = 0
                self._rate_timer[hand_label] = current_time

            hand_info = hand_data.get(hand_label)
            is_detected = (hand_info is not None) and hand_info.get("detected", False)

            if not is_drawing_active:
                self.active_stroke[hand_label] = None
                self.last_seen_point[hand_label] = None
                self.is_pointing_recognized[hand_label] = False
                if not is_detected:
                    self.boundary_status[hand_label] = "NO_HAND"
                continue

            if is_detected:
                tip_x, tip_y = hand_info["index_finger_tip"]
                curr_pt = (tip_x, tip_y)

                # 1. Coordinate Mapping & Dual-Zone Boundary Check
                is_in_zone, norm_x, zone_x = self.check_zone_boundary(hand_label, tip_x, frame_width)
                norm_y = max(0.0, min(1.0, tip_y / frame_height))
                zone_y = norm_y

                self.boundary_status[hand_label] = "INSIDE_ZONE" if is_in_zone else "OUT_OF_ZONE"
                self.last_norm_coords[hand_label] = (norm_x, norm_y)
                self.last_zone_coords[hand_label] = (zone_x, zone_y)

                # 2. 3D Index Finger Pose Recognition
                # Prefer 3D landmarks (x, y, z) when available, falling back to 2D
                landmarks_3d = hand_info.get("landmarks_3d") or hand_info.get("landmarks", [])
                raw_pointing, raw_fist, raw_posture, direction, pose_conf = self._analyze_posture_3d(
                    landmarks_3d, return_confidence=True
                )

                self.is_pointing_recognized[hand_label] = raw_pointing
                self.pointing_direction[hand_label] = direction if raw_pointing else "NONE"
                posture = self._update_posture_with_hysteresis(hand_label, raw_pointing, raw_fist, raw_posture)

                # 3. Confidence Evaluation & State Machine Transitions (Section 4 & 6)
                raw_det_conf = hand_info.get("confidence", 0.90)
                det_conf = min(1.0, max(0.0, float(raw_det_conf)))
                self.hand_detection_confidence[hand_label] = det_conf
                self.index_pose_confidence[hand_label] = pose_conf

                if raw_pointing or raw_fist:
                    combined_conf = 0.5 * det_conf + 0.5 * pose_conf
                else:
                    combined_conf = det_conf
                self.tracking_confidence_score[hand_label] = round(combined_conf, 3)

                prev_state = self.tracking_confidence_state[hand_label]
                if combined_conf >= config.CONFIDENCE_STABLE_MIN:
                    if prev_state == TrackingConfidenceState.TRACKING_UNCERTAIN:
                        self.recovery_streak[hand_label] += 1
                        if self.recovery_streak[hand_label] >= config.TRACKING_RECOVERY_CONFIRM_FRAMES:
                            self.tracking_confidence_state[hand_label] = TrackingConfidenceState.TRACKING_STABLE
                            self.recovery_streak[hand_label] = 0
                        else:
                            self.tracking_confidence_state[hand_label] = TrackingConfidenceState.TRACKING_RECOVERED
                    else:
                        self.tracking_confidence_state[hand_label] = TrackingConfidenceState.TRACKING_STABLE
                        self.recovery_streak[hand_label] = 0
                elif combined_conf >= config.CONFIDENCE_UNCERTAIN_MIN:
                    self.tracking_confidence_state[hand_label] = TrackingConfidenceState.TRACKING_UNCERTAIN
                    self.recovery_streak[hand_label] = 0
                else:
                    self.tracking_confidence_state[hand_label] = TrackingConfidenceState.TRACKING_LOST
                    self.recovery_streak[hand_label] = 0

                # 4. Boundary Crossing Enforcement:
                # When leaving permitted zone, immediately break active stroke!
                if not is_in_zone:
                    self.active_stroke[hand_label] = None
                    self.last_seen_point[hand_label] = None
                    continue

                curr_conf_state = self.tracking_confidence_state[hand_label]

                # 5. Tracking Lost under severe conditions -> Cleanly break stroke!
                if curr_conf_state == TrackingConfidenceState.TRACKING_LOST:
                    self.active_stroke[hand_label] = None
                    self.last_seen_point[hand_label] = None
                    continue

                # 6. Tracking Uncertain or Recovering -> Temporarily pause drawing safely!
                # Does NOT add wild erratic points; keeps active_stroke segment alive for potential recovery
                if curr_conf_state in (TrackingConfidenceState.TRACKING_UNCERTAIN, TrackingConfidenceState.TRACKING_RECOVERED):
                    continue

                # 7. Fist Pose: Pause drawing for this hand
                if posture == "FIST":
                    self.active_stroke[hand_label] = None
                    self.last_seen_point[hand_label] = curr_pt
                    self.last_seen_time[hand_label] = current_time
                    continue

                # 8. Pointing Pose with STABLE Confidence: Draw inside assigned zone
                if posture == "POINTING" and curr_conf_state == TrackingConfidenceState.TRACKING_STABLE:
                    last_pt = self.last_seen_point[hand_label]
                    last_t = self.last_seen_time[hand_label]

                    point_data = {
                        "x": tip_x,
                        "y": tip_y,
                        "norm_x": float(round(norm_x, 4)),
                        "norm_y": float(round(norm_y, 4)),
                        "zone_x": float(round(zone_x, 4)),
                        "zone_y": float(round(zone_y, 4)),
                        "t": float(current_time),
                    }

                    if self.active_stroke[hand_label] is None or last_pt is None:
                        # Start a brand new stroke segment inside permitted zone
                        new_stroke = [point_data]
                        self.strokes[hand_label].append(new_stroke)
                        self.active_stroke[hand_label] = new_stroke
                        self._rate_pts_counter[hand_label] += 1
                    else:
                        dt = current_time - last_t
                        dist = math.hypot(tip_x - last_pt[0], tip_y - last_pt[1])
                        last_in_zone, _, _ = self.check_zone_boundary(hand_label, last_pt[0], frame_width)

                        # Case A: Short tracking loss (Gap Bridging Interpolation within same zone, dropped >= 2 frames)
                        if (0.07 < dt <= self.max_bridge_time) and (dist <= self.max_bridge_dist) and last_in_zone:
                            mid_x = (last_pt[0] + tip_x) // 2
                            mid_y = (last_pt[1] + tip_y) // 2
                            mid_in_zone, mid_norm_x, mid_zone_x = self.check_zone_boundary(hand_label, mid_x, frame_width)

                            if mid_in_zone:
                                mid_norm_y = max(0.0, min(1.0, mid_y / frame_height))
                                mid_point = {
                                    "x": mid_x,
                                    "y": mid_y,
                                    "norm_x": float(round(mid_norm_x, 4)),
                                    "norm_y": float(round(mid_norm_y, 4)),
                                    "zone_x": float(round(mid_zone_x, 4)),
                                    "zone_y": float(round(mid_norm_y, 4)),
                                    "t": (last_t + current_time) / 2.0,
                                }
                                self.active_stroke[hand_label].append(mid_point)
                                self._rate_pts_counter[hand_label] += 1

                            self.active_stroke[hand_label].append(point_data)
                            self._rate_pts_counter[hand_label] += 1

                        # Case B: Continuous normal movement inside zone
                        elif dt <= self.max_bridge_time and dist >= self.min_point_dist:
                            self.active_stroke[hand_label].append(point_data)
                            self._rate_pts_counter[hand_label] += 1

                        # Case C: Long tracking loss or teleport jump -> Break stroke!
                        elif dt > self.max_bridge_time or dist > self.max_bridge_dist or not last_in_zone:
                            new_stroke = [point_data]
                            self.strokes[hand_label].append(new_stroke)
                            self.active_stroke[hand_label] = new_stroke
                            self._rate_pts_counter[hand_label] += 1

                    self.last_seen_point[hand_label] = curr_pt
                    self.last_seen_time[hand_label] = current_time
            else:
                self.boundary_status[hand_label] = "NO_HAND"
                self.is_pointing_recognized[hand_label] = False
                self.pointing_direction[hand_label] = "NONE"
                self.tracking_confidence_state[hand_label] = TrackingConfidenceState.TRACKING_LOST
                self.tracking_confidence_score[hand_label] = 0.0
                self.recovery_streak[hand_label] = 0
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
        """Returns posture and zone status description for HUD metrics."""
        if not is_detected:
            return "Lost"
        b_status = self.boundary_status.get(hand_label, "NO_HAND")
        if b_status == "OUT_OF_ZONE":
            return "Out of Zone"
        conf_state = self.tracking_confidence_state.get(hand_label, TrackingConfidenceState.TRACKING_LOST)
        if conf_state == TrackingConfidenceState.TRACKING_UNCERTAIN:
            return "Uncertain (Paused)"
        if conf_state == TrackingConfidenceState.TRACKING_RECOVERED:
            return "Recovering..."
        posture = self.posture_state[hand_label]
        d = self.pointing_direction.get(hand_label, "NONE")
        if posture == "POINTING":
            return f"Pointing [{d}]"
        elif posture == "FIST":
            return "Fist (Paused)"
        elif posture == "OPEN":
            return "Open Palm"
        else:
            return "Relaxed (Idle)"

    def get_drawing_state_name(self, hand_label: str, is_drawing_active: bool, is_detected: bool) -> str:
        """Returns exact drawing state string for HUD and testing: DRAWING, FIST PAUSED, OUT OF ZONE, TRACKING UNCERTAIN (PAUSED), RECOVERING, or IDLE."""
        if not is_drawing_active:
            return "IDLE"
        if not is_detected:
            return "LOST"
        b_status = self.boundary_status.get(hand_label, "NO_HAND")
        if b_status == "OUT_OF_ZONE":
            return "OUT OF ZONE"
        conf_state = self.tracking_confidence_state.get(hand_label, TrackingConfidenceState.TRACKING_LOST)
        if conf_state == TrackingConfidenceState.TRACKING_UNCERTAIN:
            return "TRACKING UNCERTAIN (PAUSED)"
        elif conf_state == TrackingConfidenceState.TRACKING_RECOVERED:
            return "RECOVERING"
        posture = self.posture_state.get(hand_label, "UNKNOWN")
        if posture == "FIST":
            return "FIST PAUSED"
        elif posture == "POINTING":
            return "DRAWING"
        else:
            return f"IDLE ({posture})"

    def get_status_text(self, hand_label: str, is_drawing_active: bool, is_detected: bool) -> str:
        pts = self.get_total_points(hand_label)
        n_strokes = self.get_total_strokes(hand_label)
        rate = self.point_rate[hand_label]
        b_status = self.boundary_status.get(hand_label, "NO_HAND")

        if not is_drawing_active:
            return f"IDLE ({pts} pts, {n_strokes} strk)"

        if is_detected:
            if b_status == "OUT_OF_ZONE":
                return f"OUT OF ZONE ({pts} pts)"
            posture = self.posture_state[hand_label]
            d = self.pointing_direction.get(hand_label, "NONE")
            if posture == "FIST":
                return f"FIST PAUSED ({pts} pts)"
            elif posture == "POINTING":
                return f"DRAWING [{d}] ({pts} pts, {rate:.1f} pts/s)"
            else:
                return f"READY [{posture}] ({pts} pts)"
        else:
            return f"NO HAND ({pts} pts)"

    def export_data(self) -> Dict[str, Any]:
        """Prepares clean trajectory data for drawing recognition with Dual-Zone metadata."""
        return {
            "Left": {
                "zone": "LEFT_DRAWING_ZONE",
                "x_range": [0.0, self.zone_split_ratio],
                "strokes": self.strokes["Left"],
                "total_points": self.get_total_points("Left"),
                "total_strokes": self.get_total_strokes("Left"),
            },
            "Right": {
                "zone": "RIGHT_DRAWING_ZONE",
                "x_range": [self.zone_split_ratio, 1.0],
                "strokes": self.strokes["Right"],
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
        Renders Dual-Zone Canvas Division, dividing center line,
        persistent strokes, and dynamic fingertip status markers.
        """
        output = frame.copy()
        h, w, _ = output.shape
        x_split = int(w * self.zone_split_ratio)

        # 1. Render Dual-Zone Canvas Division and Vertical Dividing Line
        self._render_dual_zone_overlay(output, w, h, x_split)

        # 2. Render Left hand strokes (Cyan)
        for stroke in self.strokes["Left"]:
            for i in range(1, len(stroke)):
                p1 = (stroke[i - 1]["x"], stroke[i - 1]["y"])
                p2 = (stroke[i]["x"], stroke[i]["y"])
                cv2.line(output, p1, p2, config.COLOR_DRAW_LEFT, self.stroke_thickness, cv2.LINE_AA)

        # 3. Render Right hand strokes (Orange)
        for stroke in self.strokes["Right"]:
            for i in range(1, len(stroke)):
                p1 = (stroke[i - 1]["x"], stroke[i - 1]["y"])
                p2 = (stroke[i]["x"], stroke[i]["y"])
                cv2.line(output, p1, p2, config.COLOR_DRAW_RIGHT, self.stroke_thickness, cv2.LINE_AA)

        # 4. Draw active pen, pointing pose recognition badge, out-of-zone warning, or fist indicator
        if is_drawing_active:
            for hand_label in ["Left", "Right"]:
                hand_info = hand_data.get(hand_label)
                if hand_info is not None and hand_info.get("detected", False):
                    tip = hand_info["index_finger_tip"]
                    b_status = self.boundary_status.get(hand_label, "INSIDE_ZONE")
                    posture = self.posture_state[hand_label]
                    direction = self.pointing_direction.get(hand_label, "NONE")

                    # Condition A: Out of assigned zone -> Show Red Warning Marker
                    if b_status == "OUT_OF_ZONE":
                        cv2.circle(output, tip, 14, config.COLOR_ZONE_OUT, 2, cv2.LINE_AA)
                        cv2.line(output, (tip[0] - 6, tip[1] - 6), (tip[0] + 6, tip[1] + 6), config.COLOR_ZONE_OUT, 2, cv2.LINE_AA)
                        cv2.line(output, (tip[0] - 6, tip[1] + 6), (tip[0] + 6, tip[1] - 6), config.COLOR_ZONE_OUT, 2, cv2.LINE_AA)
                        cv2.putText(
                            output,
                            "OUT OF ZONE",
                            (tip[0] + 12, tip[1] + 18),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            config.COLOR_ZONE_OUT,
                            1,
                            cv2.LINE_AA,
                        )

                    # Condition B: Tracking Uncertain -> Amber warning marker
                    elif self.tracking_confidence_state.get(hand_label) == TrackingConfidenceState.TRACKING_UNCERTAIN:
                        cv2.circle(output, tip, 14, (0, 165, 255), 2, cv2.LINE_AA)
                        cv2.circle(output, tip, 4, (0, 165, 255), -1, cv2.LINE_AA)
                        cv2.putText(
                            output,
                            "UNCERTAIN (PAUSED)",
                            (tip[0] + 12, tip[1] + 18),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.40,
                            (0, 165, 255),
                            1,
                            cv2.LINE_AA,
                        )

                    # Condition C: Tracking Recovering -> Cyan confirmation marker
                    elif self.tracking_confidence_state.get(hand_label) == TrackingConfidenceState.TRACKING_RECOVERED:
                        streak = self.recovery_streak.get(hand_label, 1)
                        cv2.circle(output, tip, 14, (255, 255, 0), 2, cv2.LINE_AA)
                        cv2.circle(output, tip, 4, (255, 255, 0), -1, cv2.LINE_AA)
                        cv2.putText(
                            output,
                            f"RECOVERING ({streak}/{config.TRACKING_RECOVERY_CONFIRM_FRAMES})",
                            (tip[0] + 12, tip[1] + 18),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.40,
                            (255, 255, 0),
                            1,
                            cv2.LINE_AA,
                        )

                    # Condition D: Inside zone and Pointing Recognized -> Prominent Drawing Indicator
                    elif posture == "POINTING":
                        pen_color = (
                            config.COLOR_DRAW_LEFT if hand_label == "Left" else config.COLOR_DRAW_RIGHT
                        )
                        # Outer glowing ring
                        cv2.circle(output, tip, 15, pen_color, 2, cv2.LINE_AA)
                        # Middle ring
                        cv2.circle(output, tip, 9, (0, 255, 0), 1, cv2.LINE_AA)
                        # Core drawing point
                        cv2.circle(output, tip, 4, (255, 255, 255), -1, cv2.LINE_AA)
                        # Recognition label with 3D direction
                        rec_text = f"POINTING: OK [{direction}]"
                        cv2.putText(
                            output,
                            rec_text,
                            (tip[0] + 14, tip[1] + 18),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.42,
                            (0, 255, 127),
                            1,
                            cv2.LINE_AA,
                        )

                    # Condition E: Inside zone and Fist -> Show Paused Marker
                    elif posture == "FIST":
                        cv2.circle(output, tip, 10, config.COLOR_FIST_PAUSED, 2, cv2.LINE_AA)
                        cv2.circle(output, tip, 4, config.COLOR_FIST_PAUSED, -1, cv2.LINE_AA)
                        cv2.putText(
                            output,
                            "PAUSED (FIST)",
                            (tip[0] + 12, tip[1] + 18),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.40,
                            config.COLOR_FIST_PAUSED,
                            1,
                            cv2.LINE_AA,
                        )

                    # Condition F: Inside zone and Open/Relaxed -> Show Idle Pose Badge
                    else:
                        cv2.circle(output, tip, 8, (180, 180, 180), 1, cv2.LINE_AA)
                        cv2.putText(
                            output,
                            f"NOT POINTING ({posture})",
                            (tip[0] + 12, tip[1] + 18),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.38,
                            (180, 180, 180),
                            1,
                            cv2.LINE_AA,
                        )

        return output

    def _render_dual_zone_overlay(self, frame: np.ndarray, w: int, h: int, x_split: int) -> None:
        """Renders the vertical dividing line and zone watermark tags."""
        dash_len = 12
        gap_len = 8
        y_pos = 0
        while y_pos < h:
            y_end = min(y_pos + dash_len, h)
            cv2.line(frame, (x_split, y_pos), (x_split, y_end), config.COLOR_ZONE_DIVIDER, 1, cv2.LINE_AA)
            y_pos += dash_len + gap_len

        # Center boundary badge at the top
        badge_w = 120
        badge_h = 22
        bx = x_split - badge_w // 2
        by = 8
        cv2.rectangle(frame, (bx, by), (bx + badge_w, by + badge_h), (25, 25, 25), -1)
        cv2.rectangle(frame, (bx, by), (bx + badge_w, by + badge_h), config.COLOR_ZONE_DIVIDER, 1)
        cv2.putText(
            frame,
            "DIVIDER 50%",
            (bx + 14, by + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            config.COLOR_ZONE_DIVIDER,
            1,
            cv2.LINE_AA,
        )

        # Left Zone Tag (Top Left area)
        cv2.putText(
            frame,
            "LEFT ZONE (0% - 50%)",
            (x_split - 190, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            config.COLOR_ZONE_LEFT_TAG,
            1,
            cv2.LINE_AA,
        )

        # Right Zone Tag (Top Right area)
        cv2.putText(
            frame,
            "RIGHT ZONE (50% - 100%)",
            (x_split + 20, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            config.COLOR_ZONE_RIGHT_TAG,
            1,
            cv2.LINE_AA,
        )
