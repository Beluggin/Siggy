"""
SignalBot Tank - Dead Reckoning Odometry
Tracks tank position (x, y, theta) from motor commands.

This is command-based dead reckoning — it WILL drift over time.
That's fine for a first pass. Later you can add:
  - Visual odometry (OpenCV feature tracking)
  - IMU integration (if XiaoR GEEK has one)
  - Loop closure when tank recognizes a place it's been

Usage:
    odom = TankOdometry()
    # Every time you send a motor command:
    odom.update(left_power, right_power)
    x, y, theta = odom.get_pose()
"""

import time
import math
import threading

from mapping_config import (
    TRACK_WIDTH, SPEED_SCALE, TURN_SCALE
)


class TankOdometry:
    def __init__(self):
        self.x = 0.0          # meters from start
        self.y = 0.0          # meters from start
        self.theta = 0.0      # radians, 0 = forward at start
        self.last_update = time.time()
        self._lock = threading.Lock()

    def update(self, left_power: float, right_power: float):
        """
        Call this every time you send motor commands.
        left_power, right_power: -100 to 100 (% motor power)
        Positive = forward.
        """
        now = time.time()
        with self._lock:
            dt = now - self.last_update
            self.last_update = now

            if dt <= 0 or dt > 1.0:
                # Skip if clock is weird or too long between updates
                return

            # Convert motor power to velocity
            v_left = left_power * SPEED_SCALE    # m/s
            v_right = right_power * SPEED_SCALE  # m/s

            # Differential drive kinematics
            v_linear = (v_left + v_right) / 2.0
            v_angular = (v_right - v_left) / TRACK_WIDTH

            # Update pose
            if abs(v_angular) < 1e-6:
                # Straight line
                self.x += v_linear * math.cos(self.theta) * dt
                self.y += v_linear * math.sin(self.theta) * dt
            else:
                # Arc
                radius = v_linear / v_angular
                dtheta = v_angular * dt
                self.x += radius * (math.sin(self.theta + dtheta) - math.sin(self.theta))
                self.y -= radius * (math.cos(self.theta + dtheta) - math.cos(self.theta))
                self.theta += dtheta

            # Normalize theta to [-pi, pi]
            self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))

    def get_pose(self):
        """Returns (x, y, theta) in meters and radians."""
        with self._lock:
            return self.x, self.y, self.theta

    def get_grid_pose(self, resolution, origin_x, origin_y):
        """Returns (grid_x, grid_y, theta) in grid cell coordinates."""
        with self._lock:
            gx = int(self.x / resolution) + origin_x
            gy = int(self.y / resolution) + origin_y
            return gx, gy, self.theta

    def reset(self):
        """Reset to origin."""
        with self._lock:
            self.x = 0.0
            self.y = 0.0
            self.theta = 0.0
            self.last_update = time.time()
