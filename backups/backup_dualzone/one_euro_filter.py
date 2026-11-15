"""
One Euro Filter (1€ Filter) for adaptive landmark smoothing in Drawing Monster Defense.
Provides strong jitter reduction at low speeds while eliminating lag during fast motion.
Reference: Casiez et al., CHI 2012 (https://cristal.univ-lille.fr/~casiez/1euro/)
"""

import math
import time
from typing import Tuple, List, Optional


class LowPassFilter:
    """First-order low-pass filter."""

    def __init__(self, alpha: float = 0.5):
        self.y: Optional[float] = None
        self.s: Optional[float] = None
        self.alpha: float = float(alpha)

    def set_alpha(self, alpha: float) -> None:
        self.alpha = float(alpha)

    def filter(self, value: float) -> float:
        if self.y is None:
            s = float(value)
        else:
            s = self.alpha * float(value) + (1.0 - self.alpha) * float(self.s)
        self.y = float(value)
        self.s = s
        return s

    def reset(self) -> None:
        self.y = None
        self.s = None


class OneEuroFilter:
    """
    1D One Euro Filter with dynamic cutoff frequency based on signal velocity.
    """

    def __init__(
        self,
        t0: float,
        x0: float,
        min_cutoff: float = 1.2,
        beta: float = 0.015,
        d_cutoff: float = 1.0,
    ):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)

        self.x_filt = LowPassFilter()
        self.dx_filt = LowPassFilter()

        self.t_prev = float(t0)
        self.x_filt.filter(x0)
        self.dx_filt.filter(0.0)

    @staticmethod
    def _alpha(rate: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / rate if rate > 0 else 0.016
        return 1.0 / (1.0 + tau / te)

    def filter(self, x: float, t: float) -> float:
        dt = t - self.t_prev
        self.t_prev = t

        if dt <= 0.0:
            dt = 1e-4

        rate = 1.0 / dt
        dx = (x - (self.x_filt.s if self.x_filt.s is not None else x)) * rate
        edx = self.dx_filt.filter(dx)

        # Dynamic cutoff increases with speed to eliminate lag
        cutoff = self.min_cutoff + self.beta * abs(edx)
        alpha = self._alpha(rate, cutoff)
        self.x_filt.set_alpha(alpha)

        return self.x_filt.filter(x)

    def reset(self, t0: float = 0.0, x0: Optional[float] = None) -> None:
        self.t_prev = float(t0)
        self.x_filt.reset()
        self.dx_filt.reset()
        if x0 is not None:
            self.x_filt.filter(x0)
            self.dx_filt.filter(0.0)


class PointOneEuroFilter:
    """
    2D (X, Y) Point filter using two OneEuroFilter instances.
    Includes auto-reset when tracking is lost past a specified timeout.
    """

    def __init__(
        self,
        min_cutoff: float = 1.2,
        beta: float = 0.015,
        d_cutoff: float = 1.0,
        reset_timeout: float = 0.25,
    ):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset_timeout = reset_timeout

        self.filter_x: Optional[OneEuroFilter] = None
        self.filter_y: Optional[OneEuroFilter] = None
        self.last_update_time: float = 0.0

    def filter(self, pt: Tuple[int, int], t: float) -> Tuple[int, int]:
        x, y = pt

        # Reset if tracking was lost for too long
        if self.filter_x is None or (t - self.last_update_time) > self.reset_timeout:
            self.filter_x = OneEuroFilter(t, x, self.min_cutoff, self.beta, self.d_cutoff)
            self.filter_y = OneEuroFilter(t, y, self.min_cutoff, self.beta, self.d_cutoff)
            self.last_update_time = t
            return (int(round(x)), int(round(y)))

        self.last_update_time = t
        smooth_x = self.filter_x.filter(x, t)
        smooth_y = self.filter_y.filter(y, t)

        return (int(round(smooth_x)), int(round(smooth_y)))

    def reset(self) -> None:
        self.filter_x = None
        self.filter_y = None
        self.last_update_time = 0.0


class HandLandmarkSmoother:
    """
    Manages 2D OneEuroFilters for hand landmarks (specifically index tip and wrist).
    Operates independently per hand to avoid cross-hand interference.
    """

    def __init__(
        self,
        min_cutoff: float = 1.2,
        beta: float = 0.015,
        d_cutoff: float = 1.0,
        reset_timeout: float = 0.25,
    ):
        self.index_tip_filter = PointOneEuroFilter(min_cutoff, beta, d_cutoff, reset_timeout)
        self.wrist_filter = PointOneEuroFilter(min_cutoff, beta, d_cutoff, reset_timeout)
        self.reset_timeout = reset_timeout
        self.last_seen_time = 0.0

    def smooth_index_tip(self, raw_tip: Tuple[int, int], current_time: float) -> Tuple[int, int]:
        self.last_seen_time = current_time
        return self.index_tip_filter.filter(raw_tip, current_time)

    def smooth_wrist(self, raw_wrist: Tuple[int, int], current_time: float) -> Tuple[int, int]:
        return self.wrist_filter.filter(raw_wrist, current_time)

    def check_reset(self, current_time: float) -> None:
        """Call when hand is not detected to check if timeout exceeded."""
        if current_time - self.last_seen_time > self.reset_timeout:
            self.reset()

    def reset(self) -> None:
        self.index_tip_filter.reset()
        self.wrist_filter.reset()
        self.last_seen_time = 0.0

