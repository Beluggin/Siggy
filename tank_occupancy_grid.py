"""
SignalBot Tank - Occupancy Grid
2D grid map that fills in as the tank drives around.

Each cell is:
  -1  = unknown (never seen)
   0  = free (confirmed driveable)
  100 = occupied (obstacle detected)

Uses log-odds updating so repeated observations strengthen confidence.
A single false positive won't permanently mark a cell.

Usage:
    grid = OccupancyGrid()
    grid.update(tank_x, tank_y, tank_theta, detection_bins)
    data = grid.serialize()  # for network streaming
"""

import numpy as np
import math
import struct
import zlib

from mapping_config import (
    GRID_RESOLUTION, GRID_WIDTH, GRID_HEIGHT,
    GRID_ORIGIN_X, GRID_ORIGIN_Y,
    CELL_UNKNOWN, CELL_FREE, CELL_OCCUPIED
)


class OccupancyGrid:
    def __init__(self):
        self.width = GRID_WIDTH
        self.height = GRID_HEIGHT
        self.resolution = GRID_RESOLUTION
        self.origin_x = GRID_ORIGIN_X
        self.origin_y = GRID_ORIGIN_Y

        # Log-odds grid (0 = unknown, positive = occupied, negative = free)
        self.log_odds = np.zeros((self.height, self.width), dtype=np.float32)

        # Update parameters
        self.l_occ = 0.85    # log-odds increment for occupied
        self.l_free = -0.40  # log-odds increment for free
        self.l_max = 3.5     # clamp to prevent over-confidence
        self.l_min = -2.0

        # Track which cells have ever been observed
        self.observed = np.zeros((self.height, self.width), dtype=bool)

        # Version counter for delta streaming
        self.version = 0

    def world_to_grid(self, wx, wy):
        """Convert world coords (meters) to grid cell indices."""
        gx = int(wx / self.resolution) + self.origin_x
        gy = int(wy / self.resolution) + self.origin_y
        return gx, gy

    def grid_to_world(self, gx, gy):
        """Convert grid cell indices to world coords (meters)."""
        wx = (gx - self.origin_x) * self.resolution
        wy = (gy - self.origin_y) * self.resolution
        return wx, wy

    def _in_bounds(self, gx, gy):
        return 0 <= gx < self.width and 0 <= gy < self.height

    def update(self, tank_x, tank_y, tank_theta, detection_bins):
        """
        Update the grid with new observations.

        tank_x, tank_y: world position in meters
        tank_theta: heading in radians
        detection_bins: list of DetectionBin from obstacle detector
        """
        gx_tank, gy_tank = self.world_to_grid(tank_x, tank_y)

        for detection_bin in detection_bins:
            # Absolute angle of this bin in world frame
            ray_angle = tank_theta + detection_bin.angle_rad
            distance = detection_bin.distance_m

            # Ray-cast: mark cells along the ray
            step = self.resolution * 0.5  # half-cell steps for accuracy
            num_steps = int(distance / step)

            for s in range(num_steps):
                d = s * step
                wx = tank_x + d * math.cos(ray_angle)
                wy = tank_y + d * math.sin(ray_angle)
                gx, gy = self.world_to_grid(wx, wy)

                if not self._in_bounds(gx, gy):
                    break

                # Everything before the endpoint is free
                self.log_odds[gy, gx] += self.l_free
                self.observed[gy, gx] = True

            # If blocked, mark the endpoint as occupied
            if detection_bin.is_blocked:
                wx_end = tank_x + distance * math.cos(ray_angle)
                wy_end = tank_y + distance * math.sin(ray_angle)
                gx_end, gy_end = self.world_to_grid(wx_end, wy_end)

                if self._in_bounds(gx_end, gy_end):
                    self.log_odds[gy_end, gx_end] += self.l_occ
                    self.observed[gy_end, gx_end] = True

        # Clamp
        np.clip(self.log_odds, self.l_min, self.l_max, out=self.log_odds)
        self.version += 1

    def get_display_grid(self):
        """
        Convert log-odds to display values:
          -1 = unknown, 0 = free, 100 = occupied
        Returns int8 numpy array.
        """
        display = np.full((self.height, self.width), CELL_UNKNOWN, dtype=np.int8)
        display[self.observed & (self.log_odds < -0.1)] = CELL_FREE
        display[self.observed & (self.log_odds > 0.5)] = CELL_OCCUPIED
        return display

    def serialize(self, tank_x, tank_y, tank_theta):
        """
        Serialize grid + pose for network transmission.
        Returns compressed bytes.
        Format:
          [version:u32][tank_x:f32][tank_y:f32][tank_theta:f32]
          [width:u16][height:u16][resolution:f32]
          [zlib compressed grid data]
        """
        display = self.get_display_grid()
        header = struct.pack(
            '<I fff HH f',
            self.version,
            tank_x, tank_y, tank_theta,
            self.width, self.height,
            self.resolution
        )
        grid_bytes = display.tobytes()
        compressed = zlib.compress(grid_bytes, level=1)  # fast compression
        payload = header + compressed
        # Prefix with total length for framing
        return struct.pack('<I', len(payload)) + payload

    @staticmethod
    def deserialize(data):
        """
        Deserialize grid + pose from network data.
        Returns dict with grid, pose, and metadata.
        """
        header_size = struct.calcsize('<I fff HH f')
        header = struct.unpack('<I fff HH f', data[:header_size])

        version = header[0]
        tank_x, tank_y, tank_theta = header[1], header[2], header[3]
        width, height = header[4], header[5]
        resolution = header[6]

        compressed = data[header_size:]
        grid_bytes = zlib.decompress(compressed)
        grid = np.frombuffer(grid_bytes, dtype=np.int8).reshape(height, width)

        return {
            'version': version,
            'tank_x': tank_x,
            'tank_y': tank_y,
            'tank_theta': tank_theta,
            'width': width,
            'height': height,
            'resolution': resolution,
            'grid': grid.copy()  # copy so buffer can be freed
        }
