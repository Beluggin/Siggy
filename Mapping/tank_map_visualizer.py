"""
SignalBot Tank - Map Visualizer (runs on Desktop / Nitro N50)
Connects to the Pi's MapServer and renders the occupancy grid
in a real-time 3D view using slamd.

Requirements (desktop only):
    pip install slamd numpy

Usage:
    python tank_map_visualizer.py                     # default: connect to signalbot-jail
    python tank_map_visualizer.py 192.168.1.100       # specify Pi IP
    python tank_map_visualizer.py 192.168.1.100 5555  # specify IP + port

What you'll see:
    - Gray plane = unknown area
    - Green cells = confirmed free space
    - Red cells = obstacles
    - Blue triad = tank's current position + heading
    - Yellow trail = path the tank has driven
"""

import sys
import socket
import struct
import threading
import time
import numpy as np
import math

try:
    import slamd
except ImportError:
    print("ERROR: slamd not installed. Run: pip install slamd")
    print("Requires Linux or macOS, Python >= 3.11, GPU with OpenGL")
    sys.exit(1)

from tank_occupancy_grid import OccupancyGrid
from mapping_config import (
    STREAM_PORT, CELL_UNKNOWN, CELL_FREE, CELL_OCCUPIED,
    GRID_RESOLUTION
)


class MapVisualizer:
    def __init__(self, host, port=STREAM_PORT):
        self.host = host
        self.port = port
        self._running = False
        self._latest_map = None
        self._map_lock = threading.Lock()

        # Trail of tank positions
        self._trail = []
        self._max_trail = 2000

    def start(self):
        self._running = True

        # Start network receiver
        threading.Thread(target=self._receive_loop, daemon=True).start()

        # Run slamd visualizer (blocks on main thread)
        self._run_visualizer()

    def _receive_loop(self):
        """Connect to Pi and receive map updates."""
        while self._running:
            try:
                print(f"[Viz] Connecting to {self.host}:{self.port}...")
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5.0)
                sock.connect((self.host, self.port))
                print(f"[Viz] Connected!")
                sock.settimeout(2.0)

                buf = b""
                while self._running:
                    try:
                        chunk = sock.recv(65536)
                        if not chunk:
                            break
                        buf += chunk

                        # Parse framed messages: [length:u32][payload]
                        while len(buf) >= 4:
                            msg_len = struct.unpack('<I', buf[:4])[0]
                            total = 4 + msg_len
                            if len(buf) < total:
                                break
                            payload = buf[4:total]
                            buf = buf[total:]

                            try:
                                map_data = OccupancyGrid.deserialize(payload)
                                with self._map_lock:
                                    self._latest_map = map_data
                            except Exception as e:
                                print(f"[Viz] Deserialize error: {e}")

                    except socket.timeout:
                        continue

            except (ConnectionRefusedError, OSError) as e:
                print(f"[Viz] Connection failed: {e}, retrying in 3s...")
                time.sleep(3)
            except Exception as e:
                print(f"[Viz] Error: {e}, retrying in 3s...")
                time.sleep(3)

    def _run_visualizer(self):
        """Main slamd visualization loop."""
        vis = slamd.Visualizer("SignalBot Tank Map")
        scene = vis.scene("map")

        # Ground reference
        scene.set_object("/ground", slamd.geom.Plane(
            normal=np.array([0.0, 0.0, 1.0]),
            point=np.array([0.0, 0.0, -0.01]),
            color=np.array([0.2, 0.2, 0.2]),
            scale=5.0,
            opacity=0.3
        ))

        # Origin marker
        scene.set_object("/origin", slamd.geom.Triad(scale=0.3))

        last_version = -1
        print("[Viz] Visualizer running. Waiting for map data...")

        try:
            while True:
                map_data = None
                with self._map_lock:
                    if self._latest_map is not None:
                        map_data = self._latest_map

                if map_data and map_data['version'] != last_version:
                    last_version = map_data['version']
                    self._update_scene(scene, map_data)

                time.sleep(0.05)  # 20fps viz update

        except KeyboardInterrupt:
            print("\n[Viz] Shutting down...")
            self._running = False

    def _update_scene(self, scene, map_data):
        """Update the slamd scene with new map data."""
        grid = map_data['grid']
        res = map_data['resolution']
        tank_x = map_data['tank_x']
        tank_y = map_data['tank_y']
        tank_theta = map_data['tank_theta']
        w = map_data['width']
        h = map_data['height']
        ox = w // 2
        oy = h // 2

        # --- Build point cloud from grid ---
        # Collect free and occupied cells
        free_ys, free_xs = np.where(grid == CELL_FREE)
        occ_ys, occ_xs = np.where(grid == CELL_OCCUPIED)

        points = []
        colors = []

        if len(free_xs) > 0:
            free_world_x = (free_xs - ox).astype(np.float32) * res
            free_world_y = (free_ys - oy).astype(np.float32) * res
            free_z = np.zeros_like(free_world_x)
            free_pts = np.stack([free_world_x, free_world_y, free_z], axis=-1)
            free_cols = np.full((len(free_xs), 3), [0.2, 0.8, 0.3], dtype=np.float32)  # green
            points.append(free_pts)
            colors.append(free_cols)

        if len(occ_xs) > 0:
            occ_world_x = (occ_xs - ox).astype(np.float32) * res
            occ_world_y = (occ_ys - oy).astype(np.float32) * res
            occ_z = np.full(len(occ_xs), 0.05, dtype=np.float32)  # slightly elevated
            occ_pts = np.stack([occ_world_x, occ_world_y, occ_z], axis=-1)
            occ_cols = np.full((len(occ_xs), 3), [0.9, 0.15, 0.15], dtype=np.float32)  # red
            points.append(occ_pts)
            colors.append(occ_cols)

        if points:
            all_points = np.concatenate(points, axis=0)
            all_colors = np.concatenate(colors, axis=0)
            scene.set_object("/map/grid", slamd.geom.PointCloud(
                points=all_points,
                colors=all_colors,
                point_size=max(res * 80, 3.0),  # scale point size to resolution
                opacity=0.9
            ))

        # --- Tank position (triad) ---
        # Build 4x4 pose matrix
        pose = np.eye(4, dtype=np.float32)
        pose[0, 0] = math.cos(tank_theta)
        pose[0, 1] = -math.sin(tank_theta)
        pose[1, 0] = math.sin(tank_theta)
        pose[1, 1] = math.cos(tank_theta)
        pose[0, 3] = tank_x
        pose[1, 3] = tank_y
        pose[2, 3] = 0.02  # slightly above ground

        scene.set_object("/tank", slamd.geom.Triad(scale=0.15))
        scene.set_transform("/tank", pose)

        # --- Trail ---
        self._trail.append([tank_x, tank_y, 0.01])
        if len(self._trail) > self._max_trail:
            self._trail = self._trail[-self._max_trail:]

        if len(self._trail) >= 2:
            trail_pts = np.array(self._trail, dtype=np.float32)
            scene.set_object("/tank/trail", slamd.geom.PolyLine(
                points=trail_pts,
                color=np.array([0.3, 0.5, 1.0]),  # blue trail
                line_width=2.0
            ))


def main():
    # Default to signalbot-jail hostname (your Pi on the LAN)
    host = "signalbot-jail"
    port = STREAM_PORT

    if len(sys.argv) >= 2:
        host = sys.argv[1]
    if len(sys.argv) >= 3:
        port = int(sys.argv[2])

    print(f"=== SignalBot Map Visualizer ===")
    print(f"Connecting to {host}:{port}")
    print(f"Controls: left-click drag to orbit, scroll to zoom, right-click to pan")
    print()

    viz = MapVisualizer(host, port)
    viz.start()


if __name__ == "__main__":
    main()
