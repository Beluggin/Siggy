"""
SignalBot Tank - Camera Obstacle Detector
Detects obstacles by finding what ISN'T floor in the camera image.

Approach:
  1. Take bottom portion of camera frame (where floor should be)
  2. Build a floor color model (HSV range)
  3. Anything in the floor zone that doesn't match = obstacle
  4. Convert obstacle positions to angular bins (ray-cast style)
  5. Return which directions are blocked vs free

This is deliberately simple. It works on hard floors with consistent
color. For outdoor/carpet/varied surfaces, you'd want a learned model
or depth sensor. But this ships today on a Pi 5.

Usage:
    detector = ObstacleDetector()
    result = detector.detect(frame)
    # result.bins = list of (angle_rad, distance_estimate, is_blocked)
"""

import cv2
import numpy as np
import math
from dataclasses import dataclass, field

from mapping_config import (
    CAMERA_HFOV_DEG, OBSTACLE_RANGE_M, FREE_RANGE_M,
    FLOOR_DETECT_ROWS, FLOOR_HSV_LOWER, FLOOR_HSV_UPPER,
    OBSTACLE_MIN_AREA, OBSTACLE_ANGULAR_BINS
)


@dataclass
class DetectionBin:
    angle_rad: float      # angle relative to tank forward (0 = center)
    distance_m: float     # estimated distance to obstacle (or FREE_RANGE if clear)
    is_blocked: bool      # True if obstacle detected in this angular bin


@dataclass
class DetectionResult:
    bins: list = field(default_factory=list)
    obstacle_mask: np.ndarray = None   # binary mask for debug visualization
    floor_mask: np.ndarray = None      # what we think is floor


class ObstacleDetector:
    def __init__(self):
        self.hfov = math.radians(CAMERA_HFOV_DEG)
        self.n_bins = OBSTACLE_ANGULAR_BINS
        self.floor_hsv_lower = np.array(FLOOR_HSV_LOWER)
        self.floor_hsv_upper = np.array(FLOOR_HSV_UPPER)

        # Adaptive floor model — starts with config values,
        # refines from a "known floor" patch in the first few frames
        self.floor_model_ready = False
        self.floor_samples = []

    def calibrate_floor(self, frame):
        """
        Sample the very bottom-center of the frame as 'definitely floor'.
        Call this a few times when the tank starts on a clear floor.
        """
        h, w = frame.shape[:2]
        # Bottom center patch: 20% width, bottom 10% height
        patch = frame[int(h * 0.88):int(h * 0.98),
                      int(w * 0.4):int(w * 0.6)]
        hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        self.floor_samples.append(hsv_patch.reshape(-1, 3))

        if len(self.floor_samples) >= 5:
            all_samples = np.vstack(self.floor_samples)
            mean = all_samples.mean(axis=0)
            std = all_samples.std(axis=0)
            # Set floor range to mean +/- 2*std
            self.floor_hsv_lower = np.clip(mean - 2.5 * std, 0, 255).astype(np.uint8)
            self.floor_hsv_upper = np.clip(mean + 2.5 * std, 0, 255).astype(np.uint8)
            self.floor_model_ready = True

    def detect(self, frame) -> DetectionResult:
        """
        Given a BGR camera frame, detect obstacles.
        Returns DetectionResult with angular bins.
        """
        h, w = frame.shape[:2]

        # 1. Extract floor zone (bottom portion of image)
        row_start = int(h * FLOOR_DETECT_ROWS[0])
        row_end = int(h * FLOOR_DETECT_ROWS[1])
        floor_zone = frame[row_start:row_end, :]

        # 2. HSV floor segmentation
        hsv = cv2.cvtColor(floor_zone, cv2.COLOR_BGR2HSV)
        floor_mask = cv2.inRange(hsv, self.floor_hsv_lower, self.floor_hsv_upper)

        # 3. Obstacle = NOT floor
        obstacle_mask = cv2.bitwise_not(floor_mask)

        # Clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_OPEN, kernel)
        obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_CLOSE, kernel)

        # 4. Divide into angular bins
        bin_width = w // self.n_bins
        half_hfov = self.hfov / 2.0
        bins = []

        for i in range(self.n_bins):
            col_start = i * bin_width
            col_end = (i + 1) * bin_width if i < self.n_bins - 1 else w
            bin_slice = obstacle_mask[:, col_start:col_end]

            # Angle of this bin center (left of image = positive angle)
            bin_center_x = (col_start + col_end) / 2.0
            angle = half_hfov - (bin_center_x / w) * self.hfov

            # Check for obstacle
            obstacle_pixels = cv2.countNonZero(bin_slice)
            total_pixels = bin_slice.shape[0] * bin_slice.shape[1]
            obstacle_ratio = obstacle_pixels / max(total_pixels, 1)

            is_blocked = obstacle_ratio > 0.15  # >15% obstacle pixels = blocked

            if is_blocked:
                # Estimate distance: higher in the floor zone = farther away
                # Find topmost obstacle row (closest to horizon = farthest)
                rows_with_obstacle = np.any(bin_slice > 0, axis=1)
                if np.any(rows_with_obstacle):
                    topmost = np.argmax(rows_with_obstacle)
                    zone_height = row_end - row_start
                    # Linear mapping: top of zone = max range, bottom = 0
                    dist_ratio = 1.0 - (topmost / zone_height)
                    distance = dist_ratio * OBSTACLE_RANGE_M
                else:
                    distance = FREE_RANGE_M
            else:
                distance = FREE_RANGE_M

            bins.append(DetectionBin(
                angle_rad=angle,
                distance_m=distance,
                is_blocked=is_blocked
            ))

        # Full-frame masks for debug viz
        full_obstacle = np.zeros((h, w), dtype=np.uint8)
        full_obstacle[row_start:row_end, :] = obstacle_mask
        full_floor = np.zeros((h, w), dtype=np.uint8)
        full_floor[row_start:row_end, :] = floor_mask

        return DetectionResult(
            bins=bins,
            obstacle_mask=full_obstacle,
            floor_mask=full_floor
        )
