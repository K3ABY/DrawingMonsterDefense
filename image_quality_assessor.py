"""
Lightweight Image Quality Assessment (IQA) for Hand Tracking.
Project: Drawing Monster Defense

Evaluates incoming camera frames in real-time (< 1.0 ms) across:
- Mean Brightness & Luminance Distribution
- Contrast & Dynamic Range
- Image Sharpness (Laplacian Variance)
- Motion Blur Indicators (Temporal gradient & frequency drop)
- High-Frequency Sensor Noise Estimation
- Overexposed (> 240) and Underexposed (< 15) Pixel Ratios
- Classification into: NORMAL, TOO_DARK, TOO_BRIGHT, BACKLIT, LOW_CONTRAST, MOTION_BLURRED, NOISY
"""

import time
import math
import cv2
import numpy as np
from enum import Enum
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass

import camera_config as config


class ImageCondition(Enum):
    NORMAL = "NORMAL"
    TOO_DARK = "TOO_DARK"
    TOO_BRIGHT = "TOO_BRIGHT"
    BACKLIT = "BACKLIT"
    LOW_CONTRAST = "LOW_CONTRAST"
    MOTION_BLURRED = "MOTION_BLURRED"
    NOISY = "NOISY"


@dataclass
class ImageQualityMetrics:
    brightness: float             # Mean luminance (0 - 255)
    contrast: float               # Standard deviation of luminance
    sharpness: float              # Variance of Laplacian (higher = sharper)
    motion_blur_metric: float     # Temporal gradient / blur ratio
    noise_level: float            # Estimated high-frequency noise standard deviation
    underexposed_ratio: float     # Fraction of pixels with luminance < 15
    overexposed_ratio: float      # Fraction of pixels with luminance > 240
    condition: ImageCondition     # Classified image quality state
    assessment_time_ms: float     # Execution time in milliseconds
    description: str              # Human-readable summary for diagnostics


class ImageQualityAssessor:
    """
    Sub-millisecond Image Quality Assessor using downscaled luminance analysis.
    Designed to add negligible overhead to the real-time game loop while accurately
    identifying degraded camera conditions that jeopardize MediaPipe tracking.
    """

    def __init__(
        self,
        assessment_width: int = 160,
        assessment_height: int = 120,
        blur_sharpness_threshold: float = 50.0,
        noise_threshold: float = 8.0,
        dark_threshold: float = 60.0,
        bright_threshold: float = 195.0,
        contrast_threshold: float = 32.0,
    ):
        self.target_size = (assessment_width, assessment_height)
        self.blur_sharpness_threshold = blur_sharpness_threshold
        self.noise_threshold = noise_threshold
        self.dark_threshold = dark_threshold
        self.bright_threshold = bright_threshold
        self.contrast_threshold = contrast_threshold

        self.prev_gray: Optional[np.ndarray] = None
        self.prev_sharpness: float = 100.0

    def assess(self, frame: np.ndarray) -> ImageQualityMetrics:
        """
        Assesses image quality of the incoming BGR camera frame.
        Takes < 1.0 ms via subsampled spatial processing.
        """
        t_start = time.perf_counter()

        # 1. Downscale frame to assessment size for ultra-fast statistics
        small_frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_NEAREST)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        # 2. Luminance & Contrast
        mean_b, std_c = cv2.meanStdDev(gray)
        brightness = float(mean_b[0][0])
        contrast = float(std_c[0][0])

        # 3. Overexposed & Underexposed pixel ratios
        total_pixels = self.target_size[0] * self.target_size[1]
        underexposed_count = int(np.count_nonzero(gray < 15))
        overexposed_count = int(np.count_nonzero(gray > 240))
        underexposed_ratio = float(underexposed_count / total_pixels)
        overexposed_ratio = float(overexposed_count / total_pixels)

        # 4. Sharpness (Variance of the Laplacian)
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        sharpness = float(laplacian.var())

        # 5. Motion Blur Indicator (Inter-frame difference + sharpness drop)
        motion_metric = 0.0
        if self.prev_gray is not None and self.prev_gray.shape == gray.shape:
            # Temporal difference between consecutive frames
            diff = cv2.absdiff(gray, self.prev_gray)
            mean_motion = float(np.mean(diff))

            # Motion blur occurs when motion is substantial AND sharpness drops
            if mean_motion > 10.0 and sharpness < self.blur_sharpness_threshold:
                motion_metric = mean_motion * (self.blur_sharpness_threshold / max(sharpness, 1.0))
            else:
                motion_metric = mean_motion
        else:
            motion_metric = 0.0

        self.prev_gray = gray.copy()
        self.prev_sharpness = sharpness

        # 6. High-Frequency Noise Estimation (Median Absolute Deviation of High-Pass Residual)
        # Fast 3x3 box blur residual
        blurred = cv2.blur(gray, (3, 3))
        residual = cv2.absdiff(gray, blurred)
        noise_level = float(np.mean(residual) * 1.2533)  # Scale mean absolute error to std dev for Gaussian noise

        # 7. Spatial Distribution for Backlighting Detection
        # Compare center region vs peripheral border
        h, w = gray.shape
        margin_y, margin_x = h // 4, w // 4
        center_roi = gray[margin_y : h - margin_y, margin_x : w - margin_x]
        center_mean = float(np.mean(center_roi))

        # Peripheral borders
        border_pixels = np.concatenate([
            gray[0:margin_y, :].flatten(),
            gray[h - margin_y : h, :].flatten(),
            gray[margin_y : h - margin_y, 0:margin_x].flatten(),
            gray[margin_y : h - margin_y, w - margin_x : w].flatten(),
        ])
        border_mean = float(np.mean(border_pixels)) if len(border_pixels) > 0 else center_mean

        # 8. Multi-Factor Condition Classification
        # Do not rely on a single metric
        is_backlit = (
            (border_mean > 160.0 and center_mean < 75.0)
            or (overexposed_ratio > 0.12 and underexposed_ratio > 0.18)
        )
        is_too_dark = (brightness < self.dark_threshold) or (underexposed_ratio > 0.30 and brightness < 75.0)
        is_too_bright = (brightness > self.bright_threshold) or (overexposed_ratio > 0.35)
        is_motion_blurred = (motion_metric > 18.0) and (sharpness < self.blur_sharpness_threshold)
        is_noisy = (noise_level > self.noise_threshold) and (contrast > 15.0)
        is_low_contrast = (contrast < self.contrast_threshold) and not is_too_dark and not is_too_bright

        if is_backlit:
            condition = ImageCondition.BACKLIT
            desc = f"Backlit (Border: {border_mean:.0f}, Center: {center_mean:.0f})"
        elif is_too_dark:
            condition = ImageCondition.TOO_DARK
            desc = f"Too Dark (Luminance: {brightness:.1f})"
        elif is_too_bright:
            condition = ImageCondition.TOO_BRIGHT
            desc = f"Too Bright (Luminance: {brightness:.1f})"
        elif is_motion_blurred:
            condition = ImageCondition.MOTION_BLURRED
            desc = f"Motion Blurred (Motion: {motion_metric:.1f}, Sharp: {sharpness:.1f})"
        elif is_low_contrast:
            condition = ImageCondition.LOW_CONTRAST
            desc = f"Low Contrast / Hazy (Std: {contrast:.1f})"
        elif is_noisy:
            condition = ImageCondition.NOISY
            desc = f"Camera Noise (Level: {noise_level:.1f})"
        else:
            condition = ImageCondition.NORMAL
            desc = f"Normal (B: {brightness:.0f}, C: {contrast:.0f}, S: {sharpness:.0f})"

        t_end = time.perf_counter()
        assessment_time_ms = (t_end - t_start) * 1000.0

        return ImageQualityMetrics(
            brightness=brightness,
            contrast=contrast,
            sharpness=sharpness,
            motion_blur_metric=motion_metric,
            noise_level=noise_level,
            underexposed_ratio=underexposed_ratio,
            overexposed_ratio=overexposed_ratio,
            condition=condition,
            assessment_time_ms=assessment_time_ms,
            description=desc,
        )

