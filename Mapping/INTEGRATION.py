"""
SignalBot Tank - Integration Guide
How to wire the mapping system into your existing tank_server.py

This is NOT a standalone file — it shows the edits you need to make
to your existing tank code on the Pi.
"""

# ============================================================
# STEP 1: Add to your imports in tank_server.py
# ============================================================

# At the top of tank_server.py, add:
from tank_map_server import MapServer

# ============================================================
# STEP 2: Create the MapServer alongside your existing server
# ============================================================

# Wherever you initialize your tank (near the top of main or __init__):
map_server = MapServer(use_internal_camera=False)  # we feed frames ourselves
map_server.start()

# ============================================================
# STEP 3: Hook motor commands
# ============================================================

# Find wherever you actually send power to the motors.
# It probably looks something like:
#
#   def set_motors(left, right):
#       # ... your existing motor control code ...
#       send_to_motor_board(left, right)
#
# Add ONE LINE after the motor send:

def set_motors(left, right):
    # ... your existing motor control code ...
    send_to_motor_board(left, right)
    map_server.on_motor_command(left, right)    # <-- ADD THIS

# ============================================================
# STEP 4: Hook camera frames
# ============================================================

# Find your existing camera capture loop.
# It probably looks something like:
#
#   while True:
#       ret, frame = cap.read()
#       # ... encode and stream to GUI ...
#
# Add ONE LINE inside the loop:

# In your camera loop:
while True:
    ret, frame = cap.read()
    if ret:
        # ... your existing frame processing ...
        map_server.on_camera_frame(frame)    # <-- ADD THIS

# ============================================================
# STEP 5: Copy mapping files to the Pi
# ============================================================

# From your desktop, SCP the mapping files to the Pi:
#
#   scp mapping_config.py tank_odometry.py tank_obstacle_detector.py \
#       tank_occupancy_grid.py tank_map_server.py \
#       luggin@signalbot-jail:~/tank_mapping/
#
# Make sure the tank_mapping directory is on the Python path:
#
#   export PYTHONPATH=$PYTHONPATH:~/tank_mapping
#
# Or add to your tank's startup script:
#
#   sys.path.insert(0, os.path.expanduser('~/tank_mapping'))

# ============================================================
# STEP 6: Run the visualizer on the desktop
# ============================================================

# On the Nitro N50:
#
#   pip install slamd numpy
#   cd tank_mapping
#   python tank_map_visualizer.py signalbot-jail
#
# Or if you want to test without the real tank:
#
#   python tank_map_visualizer.py localhost

# ============================================================
# STEP 7: Calibration
# ============================================================

# CRITICAL: You need to tune these values in mapping_config.py:
#
# 1. SPEED_SCALE — drive the tank forward for exactly 1 meter
#    at 50% power, time how long it takes. Then:
#    SPEED_SCALE = 1.0 / (time_seconds * 50)
#
# 2. TRACK_WIDTH — measure center-to-center of the tank treads
#    in meters. Default 0.18m is a guess.
#
# 3. FLOOR_HSV_LOWER / FLOOR_HSV_UPPER — point the camera at
#    your floor and run:
#
#    python -c "
#    import cv2, numpy as np
#    cap = cv2.VideoCapture(0)
#    _, frame = cap.read()
#    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#    h, w = frame.shape[:2]
#    patch = hsv[int(h*0.85):, int(w*0.3):int(w*0.7)]
#    print('Floor HSV mean:', patch.reshape(-1,3).mean(axis=0).astype(int))
#    print('Floor HSV std:', patch.reshape(-1,3).std(axis=0).astype(int))
#    cap.release()
#    "
#
#    Set LOWER = mean - 2.5*std, UPPER = mean + 2.5*std
#    (The auto-calibrator does this too, but manual is more reliable)
