"""
SignalBot Tank - Map Server (runs on Pi 5)
Integrates odometry + obstacle detection + occupancy grid,
streams map data to the desktop visualizer over TCP.

Integration point: this hooks into your existing tank_server.py.
You need to call map_server.on_motor_command(left, right) every
time a motor command is sent, and feed camera frames.

Usage (standalone test):
    python tank_map_server.py

Usage (integrated with tank_server.py):
    from tank_map_server import MapServer
    map_server = MapServer()
    map_server.start()

    # In your motor command handler:
    map_server.on_motor_command(left_power, right_power)

    # In your camera loop:
    map_server.on_camera_frame(frame)
"""

import socket
import threading
import time
import cv2
import sys

from mapping_config import (
    STREAM_HOST, STREAM_PORT, STREAM_INTERVAL,
    GRID_RESOLUTION, GRID_ORIGIN_X, GRID_ORIGIN_Y
)
from tank_odometry import TankOdometry
from tank_obstacle_detector import ObstacleDetector
from tank_occupancy_grid import OccupancyGrid


class MapServer:
    def __init__(self, camera_index=0, use_internal_camera=False):
        """
        Args:
            camera_index: OpenCV camera index (only used if use_internal_camera=True)
            use_internal_camera: If True, captures frames internally.
                                 If False, you must call on_camera_frame() externally.
        """
        self.odom = TankOdometry()
        self.detector = ObstacleDetector()
        self.grid = OccupancyGrid()

        self.use_internal_camera = use_internal_camera
        self.camera_index = camera_index
        self.cap = None

        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._clients = []
        self._clients_lock = threading.Lock()
        self._running = False

        # Calibration state
        self._calibration_frames = 0
        self._calibrated = False

    def start(self):
        """Start all mapping threads."""
        self._running = True

        if self.use_internal_camera:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                print(f"[MapServer] ERROR: Cannot open camera {self.camera_index}")
                return False
            threading.Thread(target=self._camera_loop, daemon=True).start()

        threading.Thread(target=self._accept_loop, daemon=True).start()
        threading.Thread(target=self._mapping_loop, daemon=True).start()
        print(f"[MapServer] Started on port {STREAM_PORT}")
        return True

    def stop(self):
        """Stop everything."""
        self._running = False
        if self.cap:
            self.cap.release()
        with self._clients_lock:
            for c in self._clients:
                try:
                    c.close()
                except Exception:
                    pass
            self._clients.clear()

    def on_motor_command(self, left_power: float, right_power: float):
        """
        Hook this into your motor command handler.
        Call EVERY TIME you send a command to the motors.
        """
        self.odom.update(left_power, right_power)

    def on_camera_frame(self, frame):
        """
        Feed a camera frame (BGR numpy array).
        Call this from your existing camera capture loop.
        """
        with self._frame_lock:
            self._latest_frame = frame.copy()

    # --- Internal threads ---

    def _camera_loop(self):
        """Internal camera capture (only if use_internal_camera=True)."""
        while self._running:
            ret, frame = self.cap.read()
            if ret:
                with self._frame_lock:
                    self._latest_frame = frame
            time.sleep(0.033)  # ~30fps

    def _mapping_loop(self):
        """Main mapping update loop."""
        while self._running:
            frame = None
            with self._frame_lock:
                if self._latest_frame is not None:
                    frame = self._latest_frame.copy()

            if frame is not None:
                # Auto-calibrate floor model on first frames
                if not self._calibrated:
                    self.detector.calibrate_floor(frame)
                    self._calibration_frames += 1
                    if self._calibration_frames >= 5:
                        self._calibrated = True
                        print("[MapServer] Floor model calibrated")

                # Detect obstacles
                result = self.detector.detect(frame)

                # Get current pose
                x, y, theta = self.odom.get_pose()

                # Update occupancy grid
                self.grid.update(x, y, theta, result.bins)

                # Stream to connected clients
                try:
                    data = self.grid.serialize(x, y, theta)
                    self._broadcast(data)
                except Exception as e:
                    print(f"[MapServer] Serialize error: {e}")

            time.sleep(STREAM_INTERVAL)

    def _accept_loop(self):
        """Accept TCP connections from desktop visualizer."""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.settimeout(1.0)
        server.bind((STREAM_HOST, STREAM_PORT))
        server.listen(3)

        while self._running:
            try:
                client, addr = server.accept()
                print(f"[MapServer] Visualizer connected from {addr}")
                with self._clients_lock:
                    self._clients.append(client)
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[MapServer] Accept error: {e}")

        server.close()

    def _broadcast(self, data):
        """Send data to all connected clients."""
        dead = []
        with self._clients_lock:
            for c in self._clients:
                try:
                    c.sendall(data)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    dead.append(c)
            for c in dead:
                self._clients.remove(c)
                try:
                    c.close()
                except Exception:
                    pass


# --- Standalone test mode ---
if __name__ == "__main__":
    print("=== SignalBot Map Server (standalone test) ===")
    print("Using internal camera capture.")
    print(f"Streaming on port {STREAM_PORT}")
    print("Drive commands simulated — connect visualizer to see the grid.")
    print()

    server = MapServer(camera_index=0, use_internal_camera=True)
    if not server.start():
        sys.exit(1)

    try:
        # Simulate some movement for testing
        import math
        t = 0
        while True:
            # Gentle forward + slight turning to test grid building
            left = 30 + 10 * math.sin(t * 0.3)
            right = 30 - 10 * math.sin(t * 0.3)
            server.on_motor_command(left, right)
            t += STREAM_INTERVAL
            time.sleep(STREAM_INTERVAL)
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.stop()
