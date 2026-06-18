"""
SignalBot Tank - Mapping Configuration
Shared constants for Pi-side mapping and desktop-side visualization.
"""

# --- Grid ---
GRID_RESOLUTION = 0.05       # meters per cell (5cm)
GRID_WIDTH = 200             # cells (10m total)
GRID_HEIGHT = 200            # cells (10m total)
GRID_ORIGIN_X = GRID_WIDTH // 2   # tank starts at center
GRID_ORIGIN_Y = GRID_HEIGHT // 2

# Cell states
CELL_UNKNOWN = -1
CELL_FREE = 0
CELL_OCCUPIED = 100

# --- Tank physical params ---
TRACK_WIDTH = 0.18           # meters between tracks (measure your XiaoR GEEK)
# Speed calibration: map motor command % to m/s
# You WILL need to tune this by driving a known distance
SPEED_SCALE = 0.002          # m/s per 1% motor power (rough starting guess)
TURN_SCALE = 0.015           # rad/s per 1% differential (rough starting guess)

# --- Camera ---
CAMERA_HFOV_DEG = 62.0       # horizontal FOV (typical Pi camera v2)
CAMERA_VFOV_DEG = 48.8
OBSTACLE_RANGE_M = 2.0       # max detection range for camera-based obstacles
FREE_RANGE_M = 1.5           # range we confidently mark as free
FLOOR_DETECT_ROWS = (0.6, 0.95)  # bottom 35% of image is "floor zone"

# --- Network ---
STREAM_HOST = "0.0.0.0"
STREAM_PORT = 5555
STREAM_INTERVAL = 0.1        # seconds between map updates to desktop

# --- Obstacle detection ---
FLOOR_HSV_LOWER = (0, 0, 40)     # tune to your floor color
FLOOR_HSV_UPPER = (180, 80, 200)
OBSTACLE_MIN_AREA = 500           # min contour area (pixels) to count as obstacle
OBSTACLE_ANGULAR_BINS = 7         # divide camera FOV into N bins for ray-casting
