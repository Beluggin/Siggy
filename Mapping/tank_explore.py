"""
SignalBot Tank — Explore Behavior
Autonomous mapping: drive toward the unknown, avoid obstacles, fill the grid.

This is ONE behavior that plugs into the initiative daemon as a mode.
It doesn't need a route. It doesn't need waypoints. It just chases
the fog of war.

Algorithm:
    1. Cast rays in 8 directions from the tank's current grid position
    2. Score each direction by how many UNKNOWN cells it hits
    3. Penalize directions that have OCCUPIED cells (obstacles) nearby
    4. Pick the best direction
    5. Turn toward it, drive until blocked or until the score shifts
    6. Repeat

Usage (standalone test):
    python tank_explore.py --simulate

Usage (integrated with initiative daemon):
    from tank_explore import ExploreController
    explorer = ExploreController(map_server, motor_controller)
    explorer.start()   # begins autonomous exploration
    explorer.stop()    # halts, returns to idle
    explorer.is_exploring  # bool

Integration with tank_server.py:
    Add as a SignalBot action:
        elif action == "explore":
            explorer.start()
            return {"ok": True, "mode": "EXPLORE"}
        elif action == "stop_explore":
            explorer.stop()
            return {"ok": True, "mode": "IDLE"}
"""

import math
import time
import threading
import logging
import numpy as np

from mapping_config import (
    GRID_RESOLUTION, GRID_WIDTH, GRID_HEIGHT,
    GRID_ORIGIN_X, GRID_ORIGIN_Y,
    CELL_UNKNOWN, CELL_FREE, CELL_OCCUPIED
)

log = logging.getLogger("explore")


class ExploreController:
    """
    Autonomous exploration: chase the unknown, avoid the known-bad.
    """

    def __init__(self, map_server, motor):
        """
        Args:
            map_server: MapServer instance (has .grid and .odom)
            motor: MotorController instance (has .drive() and .set_speed())
        """
        self.map_server = map_server
        self.motor = motor

        # Explore parameters — tune these
        self.explore_speed = 28          # conservative speed (0-100)
        self.turn_speed = 25             # speed during turns
        self.obstacle_stop_dist = 0.30   # meters — stop if obstacle this close
        self.ray_length = 40             # grid cells to cast per ray (~2m at 5cm/cell)
        self.turn_duration = 0.6         # seconds per ~45° turn (tune on real tank)
        self.drive_chunk = 1.5           # seconds to drive before re-evaluating
        self.stuck_threshold = 0.02      # meters — if we moved less than this, we're stuck
        self.stuck_turn_count = 0        # how many times we've been stuck in a row

        # State
        self.is_exploring = False
        self._thread = None
        self._stop_event = threading.Event()

        # 8 directions to evaluate (every 45°)
        self._directions = [
            ("N",  0,             0.0),
            ("NE", math.pi / 4,   0.0),
            ("E",  math.pi / 2,   0.0),
            ("SE", 3 * math.pi / 4, 0.0),
            ("S",  math.pi,       0.0),
            ("SW", -3 * math.pi / 4, 0.0),
            ("W",  -math.pi / 2,  0.0),
            ("NW", -math.pi / 4,  0.0),
        ]

    def start(self):
        """Begin autonomous exploration."""
        if self.is_exploring:
            log.info("Already exploring")
            return

        self._stop_event.clear()
        self.is_exploring = True
        self.stuck_turn_count = 0
        self._thread = threading.Thread(target=self._explore_loop, daemon=True)
        self._thread.start()
        log.info("EXPLORE: started")

    def stop(self):
        """Halt exploration, stop motors."""
        self._stop_event.set()
        self.is_exploring = False
        self._drive_stop()
        log.info("EXPLORE: stopped")

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def _explore_loop(self):
        """Main exploration loop."""
        # Set conservative speed
        self.motor.set_speed(self.explore_speed, self.explore_speed)
        time.sleep(0.3)  # let speed settle

        while not self._stop_event.is_set():
            try:
                # 1. Where are we? What does the grid look like?
                x, y, theta = self.map_server.odom.get_pose()
                grid = self.map_server.grid

                # 2. Score each direction
                scores = self._score_directions(grid, x, y, theta)

                if not scores:
                    # Grid not ready yet, just wait
                    time.sleep(0.5)
                    continue

                # 3. Pick the best direction
                best_name, best_angle, best_score = scores[0]
                log.info(f"EXPLORE: best={best_name} score={best_score:.1f} "
                         f"pos=({x:.2f},{y:.2f}) heading={math.degrees(theta):.0f}°")

                # 4. If best score is 0, we've explored everything reachable
                if best_score <= 0:
                    log.info("EXPLORE: area fully mapped (no unknown cells reachable)")
                    time.sleep(2.0)
                    continue

                # 5. Turn toward the best direction
                angle_diff = self._normalize_angle(best_angle - theta)
                self._turn_toward(angle_diff)

                if self._stop_event.is_set():
                    break

                # 6. Drive forward for a chunk
                prev_x, prev_y = x, y
                self._drive_forward(self.drive_chunk)

                if self._stop_event.is_set():
                    break

                # 7. Check if we actually moved (stuck detection)
                new_x, new_y, _ = self.map_server.odom.get_pose()
                dist_moved = math.sqrt((new_x - prev_x)**2 + (new_y - prev_y)**2)

                if dist_moved < self.stuck_threshold:
                    self.stuck_turn_count += 1
                    log.warning(f"EXPLORE: stuck (moved {dist_moved:.3f}m), "
                                f"count={self.stuck_turn_count}")
                    self._unstick()
                else:
                    self.stuck_turn_count = 0

            except Exception as e:
                log.error(f"EXPLORE error: {e}")
                self._drive_stop()
                time.sleep(1.0)

        self._drive_stop()

    # ------------------------------------------------------------------
    # Direction scoring
    # ------------------------------------------------------------------

    def _score_directions(self, grid, tank_x, tank_y, tank_theta):
        """
        Cast rays in 8 directions from the tank's position.
        Score = unknown cells hit — penalty for obstacles.

        Returns sorted list of (name, world_angle, score), best first.
        """
        display = grid.get_display_grid()
        gx, gy = grid.world_to_grid(tank_x, tank_y)

        results = []

        for name, relative_angle, _ in self._directions:
            world_angle = tank_theta + relative_angle
            unknown_count = 0
            obstacle_penalty = 0
            hit_wall = False

            for step in range(1, self.ray_length + 1):
                # Grid cell along this ray
                cx = gx + int(step * math.cos(world_angle))
                cy = gy + int(step * math.sin(world_angle))

                # Out of bounds
                if cx < 0 or cx >= GRID_WIDTH or cy < 0 or cy >= GRID_HEIGHT:
                    break

                cell = display[cy, cx]

                if cell == CELL_UNKNOWN:
                    unknown_count += 1
                elif cell == CELL_OCCUPIED:
                    # Closer obstacles are worse
                    distance_penalty = max(0, self.ray_length - step)
                    obstacle_penalty += distance_penalty * 2
                    if step < 5:  # obstacle within ~25cm
                        hit_wall = True
                        break
                # CELL_FREE: keep going, nothing to count

            # Final score
            score = unknown_count - obstacle_penalty
            if hit_wall:
                score = -100  # never pick a direction with an immediate wall

            results.append((name, world_angle, score))

        # Sort by score descending
        results.sort(key=lambda x: x[2], reverse=True)
        return results

    # ------------------------------------------------------------------
    # Motor commands
    # ------------------------------------------------------------------

    def _turn_toward(self, angle_diff):
        """
        Turn the tank by angle_diff radians.
        Positive = turn left, negative = turn right.
        """
        if abs(angle_diff) < 0.15:  # ~8.5° — close enough, don't bother
            return

        # Estimate turn time from angle
        # Full 180° should take roughly turn_duration * 4
        turn_time = abs(angle_diff) / (math.pi / 4) * self.turn_duration
        turn_time = max(0.15, min(turn_time, 3.0))  # clamp

        self.motor.set_speed(self.turn_speed, self.turn_speed)

        if angle_diff > 0:
            self._drive_left(turn_time)
        else:
            self._drive_right(turn_time)

        self.motor.set_speed(self.explore_speed, self.explore_speed)

    def _drive_forward(self, duration):
        """Drive forward, but check for obstacles periodically."""
        from enum import IntEnum
        # Import Direction from wherever it's defined in the tank code
        try:
            from tank_server import Direction
        except ImportError:
            # Fallback for standalone testing
            class Direction(IntEnum):
                STOP = 0
                FORWARD = 1
                BACKWARD = 2
                LEFT = 3
                RIGHT = 4

        self.motor.drive(Direction.FORWARD)

        # Drive in small increments, checking for obstacles
        elapsed = 0
        check_interval = 0.2
        while elapsed < duration and not self._stop_event.is_set():
            time.sleep(check_interval)
            elapsed += check_interval

            # Check if obstacle detector sees something close
            if self._obstacle_ahead():
                log.info("EXPLORE: obstacle ahead, stopping early")
                break

        self.motor.drive(Direction.STOP)

    def _drive_left(self, duration):
        try:
            from tank_server import Direction
        except ImportError:
            class Direction(IntEnum):
                STOP = 0; FORWARD = 1; BACKWARD = 2; LEFT = 3; RIGHT = 4

        self.motor.drive(Direction.LEFT)
        self._wait(duration)
        self.motor.drive(Direction.STOP)

    def _drive_right(self, duration):
        try:
            from tank_server import Direction
        except ImportError:
            class Direction(IntEnum):
                STOP = 0; FORWARD = 1; BACKWARD = 2; LEFT = 3; RIGHT = 4

        self.motor.drive(Direction.RIGHT)
        self._wait(duration)
        self.motor.drive(Direction.STOP)

    def _drive_stop(self):
        try:
            from tank_server import Direction
        except ImportError:
            class Direction(IntEnum):
                STOP = 0; FORWARD = 1; BACKWARD = 2; LEFT = 3; RIGHT = 4

        self.motor.drive(Direction.STOP)

    def _wait(self, duration):
        """Sleeps in small chunks so _stop_event can interrupt."""
        elapsed = 0
        while elapsed < duration and not self._stop_event.is_set():
            time.sleep(0.05)
            elapsed += 0.05

    # ------------------------------------------------------------------
    # Obstacle checking
    # ------------------------------------------------------------------

    def _obstacle_ahead(self):
        """
        Quick check: are any of the center detection bins blocked
        at close range?
        """
        grid = self.map_server.grid
        x, y, theta = self.map_server.odom.get_pose()
        gx, gy = grid.world_to_grid(x, y)

        # Check 3 cells directly ahead
        for step in range(1, 6):  # ~25cm at 5cm resolution
            cx = gx + int(step * math.cos(theta))
            cy = gy + int(step * math.sin(theta))
            if 0 <= cx < GRID_WIDTH and 0 <= cy < GRID_HEIGHT:
                display = grid.get_display_grid()
                if display[cy, cx] == CELL_OCCUPIED:
                    return True
        return False

    # ------------------------------------------------------------------
    # Stuck recovery
    # ------------------------------------------------------------------

    def _unstick(self):
        """
        We haven't moved. Likely pressed against something.
        Back up, turn a random-ish amount, try again.
        """
        try:
            from tank_server import Direction
        except ImportError:
            class Direction(IntEnum):
                STOP = 0; FORWARD = 1; BACKWARD = 2; LEFT = 3; RIGHT = 4

        log.info("EXPLORE: unsticking...")

        # Back up
        self.motor.drive(Direction.BACKWARD)
        self._wait(0.8)
        self.motor.drive(Direction.STOP)
        self._wait(0.2)

        # Turn — alternate left/right, increasing duration with stuck count
        turn_time = 0.5 + (self.stuck_turn_count * 0.3)
        turn_time = min(turn_time, 2.5)

        if self.stuck_turn_count % 2 == 0:
            self._drive_right(turn_time)
        else:
            self._drive_left(turn_time)

        # If stuck too many times, do a big 180
        if self.stuck_turn_count >= 4:
            log.warning("EXPLORE: very stuck, doing 180°")
            self._drive_right(self.turn_duration * 4)
            self.stuck_turn_count = 0

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_angle(angle):
        """Normalize to [-pi, pi]."""
        return math.atan2(math.sin(angle), math.cos(angle))
