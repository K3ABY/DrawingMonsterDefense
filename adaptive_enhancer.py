"""
Adaptive Image Enhancement for Robust Hand Tracking.
Project: Drawing Monster Defense

Applies targeted, lightweight image enhancement ONLY when necessary:
- NORMAL: Bypasses processing completely (0.0 ms, zero artifact risk)
- TOO_DARK: Fast Gamma Correction (gamma = 0.65) via precomputed 256-entry LUT (~0.15 ms)
- TOO_BRIGHT: Fast Exposure attenuation via LUT (~0.15 ms)
- LOW_CONTRAST / HAZY: CLAHE on L channel of LAB space (~1.2 ms)
- BACKLIT: Combined shadow-lift and CLAHE on LAB L channel (~1.4 ms)
- NOISY: Fast edge-preserving spatial filter (~1.5 ms)
- MOTION_BLURRED: Controlled unsharp mask with conservative weights to avoid ringing (~0.8 ms)
"""

import time
import math
import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional

import camera_config as config
from image_quality_assessor import ImageCondition, ImageQualityMetrics


class AdaptiveImageEnhancer:
    """
    Conditionally applies targeted image enhancements based on ImageQualityMetrics.
    Prioritizes real-time performance and avoids over-processing or artifact generation.
    """

    def __init__(
        self,
        clahe_clip_limit: float = 2.0,
        clahe_grid_size: Tuple[int, int] = (8, 8),
        dark_gamma: float = 0.65,
        bright_gamma: float = 1.35,
    ):
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_grid_size)

        # Precompute 256-element Look-Up Tables (LUT) for sub-millisecond gamma correction
        # dark_gamma < 1.0 lifts shadows; bright_gamma > 1.0 compresses highlights
        self.lut_dark = np.array(
            [min(255, int(((i / 255.0) ** dark_gamma) * 255.0 + 0.5)) for i in range(256)], dtype=np.uint8
        )

        self.lut_bright = np.array(
            [min(255, int(((i / 255.0) ** bright_gamma) * 255.0 + 0.5)) for i in range(256)], dtype=np.uint8
        )

        # Cache for bilateral filter kernels and reusable buffers
        self.bilateral_d = 5
        self.bilateral_sigma_color = 30
        self.bilateral_sigma_space = 30
        self._lut_buffer: Optional[np.ndarray] = None

    def enhance(
        self, frame: np.ndarray, metrics: ImageQualityMetrics
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Enhances the frame conditionally based on assessed metrics.
        Returns:
            (processed_frame, enhancement_info)
        """
        t_start = time.perf_counter()
        condition = metrics.condition

        # 1. Condition NORMAL: Zero-cost bypass
        if condition == ImageCondition.NORMAL:
            return frame, {
                "applied": False,
                "method": "BYPASS",
                "time_ms": 0.0,
                "description": "Original image used (Normal quality)",
            }

        # Ensure reusable buffer is sized to frame
        if self._lut_buffer is None or self._lut_buffer.shape != frame.shape:
            self._lut_buffer = np.empty_like(frame)

        # 2. Condition TOO_DARK: Fast Gamma Correction via LUT
        if condition == ImageCondition.TOO_DARK:
            cv2.LUT(frame, self.lut_dark, dst=self._lut_buffer)
            enhanced = self._lut_buffer
            method = "GAMMA_CORRECTION"
            desc = f"Shadow lifted with Gamma={1.0 / (1.0 / 0.65):.2f}"

        # 3. Condition TOO_BRIGHT: Attenuation via LUT
        elif condition == ImageCondition.TOO_BRIGHT:
            cv2.LUT(frame, self.lut_bright, dst=self._lut_buffer)
            enhanced = self._lut_buffer
            method = "EXPOSURE_ATTENUATION"
            desc = "Highlights compressed with Gamma attenuation"

        # 4. Condition LOW_CONTRAST / HAZY: Contrast stretch + CLAHE on L channel
        elif condition == ImageCondition.LOW_CONTRAST:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            norm_l = cv2.normalize(l_channel, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            enhanced_l = self.clahe.apply(norm_l)
            enhanced_lab = cv2.merge([enhanced_l, a_channel, b_channel])
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            method = "CLAHE_LOCAL_CONTRAST"
            desc = "Local contrast restored with CLAHE on L-channel"

        # 5. Condition BACKLIT: Shadow-lift + CLAHE
        elif condition == ImageCondition.BACKLIT:
            # Lift shadows slightly first
            gamma_lifted = cv2.LUT(frame, self.lut_dark)
            lab = cv2.cvtColor(gamma_lifted, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            enhanced_l = self.clahe.apply(l_channel)
            enhanced_lab = cv2.merge([enhanced_l, a_channel, b_channel])
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            method = "BACKLIGHT_RECOVERY"
            desc = "Dual-stage shadow lift + CLAHE for backlit silhouette recovery"

        # 6. Condition NOISY: Fast Edge-Preserving Denoising
        elif condition == ImageCondition.NOISY:
            enhanced = cv2.bilateralFilter(
                frame,
                d=self.bilateral_d,
                sigmaColor=self.bilateral_sigma_color,
                sigmaSpace=self.bilateral_sigma_space,
            )
            method = "BILATERAL_DENOISE"
            desc = "Edge-preserving spatial smoothing to suppress sensor grain"

        # 7. Condition MOTION_BLURRED: Controlled Unsharp Mask
        elif condition == ImageCondition.MOTION_BLURRED:
            # Gaussian blur for low-frequency reference
            gaussian = cv2.GaussianBlur(frame, (0, 0), sigmaX=2.0)
            # Unsharp mask formula: enhanced = original * (1 + amount) - blurred * amount
            # Conservative weight 0.25 prevents high-frequency ringing and noise explosion
            enhanced = cv2.addWeighted(frame, 1.25, gaussian, -0.25, 0)
            method = "UNSHARP_MASK_SHARPEN"
            desc = "Controlled edge gradient restoration for motion-blurred silhouettes"

        else:
            enhanced = frame
            method = "BYPASS"
            desc = "Bypassed"

        t_end = time.perf_counter()
        enhancement_time_ms = (t_end - t_start) * 1000.0

        return enhanced, {
            "applied": True,
            "method": method,
            "time_ms": enhancement_time_ms,
            "description": desc,
        }

