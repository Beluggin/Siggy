#!/usr/bin/env python3
"""
SignalBot Tank HUD — Flask Backend
Serves the web dashboard on :5000 and bridges API calls to tank hardware.

Drop this + tank_hud.html into /home/pi/work/python_src/

Usage:
    python3 tank_hud_server.py [--simulate]
"""

import json
import time
import os
import threading
import argparse
from datetime import datetime
from flask import Flask, send_file, jsonify, request
from flask_cors import CORS

parser = argparse.ArgumentParser()
parser.add_argument('--simulate', action='store_true', help='Run without hardware')
args, _ = parser.parse_known_args()
SIMULATE = args.simulate

# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------
if not SIMULATE:
    try:
        from gpiozero import DigitalInputDevice, DistanceSensor, Motor, PWMOutputDevice
        import smbus2
        print("[HUD] Loading hardware...")

        PCA9685_ADDR = 0x40
        bus = smbus2.SMBus(1)

        def pca9685_init():
            bus.write_byte_data(PCA9685_ADDR, 0x00, 0x10)
            bus.write_byte_data(PCA9685_ADDR, 0xFE, 121)
            bus.write_byte_data(PCA9685_ADDR, 0x00, 0x00)
            time.sleep(0.005)
            bus.write_byte_data(PCA9685_ADDR, 0x00, 0x20)

        def set_servo_angle(channel, angle):
            pulse = int(150 + (angle / 180.0) * 450)
            reg = 0x06 + 4 * channel
            bus.write_byte_data(PCA9685_ADDR, reg, 0)
            bus.write_byte_data(PCA9685_ADDR, reg + 1, 0)
            bus.write_byte_data(PCA9685_ADDR, reg + 2, pulse & 0xFF)
            bus.write_byte_data(PCA9685_ADDR, reg + 3, pulse >> 8)

        pca9685_init()

        # L298 motor driver — adjust GPIO pins to your wiring
        motor_left = Motor(forward=17, backward=18)
        motor_right = Motor(forward=22, backward=23)
        pwm_left = PWMOutputDevice(12)
        pwm_right = PWMOutputDevice(13)

        def drive_proportional(speed_l, speed_r):
            """Proportional drive. speed_l/speed_r are -100 to 100."""
            al = abs(speed_l) / 100.0
            ar = abs(speed_r) / 100.0
            pwm_left.value = min(al, 1.0)
            pwm_right.value = min(ar, 1.0)
            if speed_l > 0:
                motor_left.forward()
            elif speed_l < 0:
                motor_left.backward()
            else:
                motor_left.stop()
            if speed_r > 0:
                motor_right.forward()
            elif speed_r < 0:
                motor_right.backward()
            else:
                motor_right.stop()

        def drive_stop():
            motor_left.stop()
            motor_right.stop()
            pwm_left.value = 0
            pwm_right.value = 0

        # IR sensors — adjust GPIO pins to your wiring
        IR_PINS = [5, 6, 16, 20, 21, 26]
        ir_sensors = [DigitalInputDevice(pin, pull_up=True) for pin in IR_PINS]

        # Ultrasonic — adjust GPIO pins
        ultrasonic = DistanceSensor(echo=24, trigger=25, max_distance=3)

        print("[HUD] Hardware ready")

    except Exception as e:
        print(f"[HUD] Hardware init error: {e}")
        print("[HUD] Falling back to simulate mode")
        SIMULATE = True

if SIMULATE:
    print("[HUD] SIMULATE mode — no hardware")
    import random

    def set_servo_angle(channel, angle):
        pass

    def drive_proportional(speed_l, speed_r):
        pass

    def drive_stop():
        pass

    ir_sensors = None
    ultrasonic = None

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
state = {
    'mode': 'IDLE',
    'detections': 0,
    'hazard_stops': 0,
    'headlight': False,
    'buzzer': False,
    'recording': False,
    'servo_positions': [115, 180, 0, 50, 95, 79],
    'dwm': False,
    'daemon_status': '—',
}

# ---------------------------------------------------------------------------
# Flask
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

HUD_DIR = os.path.dirname(os.path.abspath(__file__))


@app.route('/')
def index():
    return send_file(os.path.join(HUD_DIR, 'tank_hud.html'))


@app.route('/api/status')
def api_status():
    ir_data = [0] * 6
    us_dist = 0
    if SIMULATE:
        ir_data = [random.choice([0, 0, 0, 0, 1]) for _ in range(6)]
        us_dist = random.randint(5, 200)
    else:
        try:
            ir_data = [1 if not s.value else 0 for s in ir_sensors]
        except:
            pass
        try:
            us_dist = round(ultrasonic.distance * 100, 1)
        except:
            us_dist = -1

    return jsonify({
        'connected': True,
        'mode': state['mode'],
        'ir': ir_data,
        'ultrasonic': us_dist,
        'detections': state['detections'],
        'hazard_stops': state['hazard_stops'],
        'daemon_status': state['daemon_status'],
        'timestamp': datetime.now().isoformat(),
    })


@app.route('/api/drive', methods=['POST'])
def api_drive():
    data = request.json or {}
    direction = data.get('direction', 'stop')

    if direction == 'stop':
        drive_stop()
        return jsonify({'ok': True, 'direction': 'stop'})

    if direction == 'proportional':
        # Joystick sends signed speed values: -100 to 100
        sl = data.get('speed_l', 0)
        sr = data.get('speed_r', 0)
        sl = max(-100, min(100, int(sl)))
        sr = max(-100, min(100, int(sr)))
        drive_proportional(sl, sr)
        return jsonify({'ok': True, 'direction': 'proportional', 'l': sl, 'r': sr})

    # Legacy simple directions
    speed_l = data.get('speed_l', 30)
    speed_r = data.get('speed_r', 26)
    mapping = {
        'fwd': (speed_l, speed_r),
        'rev': (-speed_l, -speed_r),
        'left': (-speed_l, speed_r),
        'right': (speed_l, -speed_r),
    }
    sl, sr = mapping.get(direction, (0, 0))
    drive_proportional(sl, sr)
    return jsonify({'ok': True, 'direction': direction})


@app.route('/api/servo', methods=['POST'])
def api_servo():
    data = request.json or {}
    sid = data.get('id', 1)
    angle = max(0, min(180, int(data.get('angle', 90))))
    set_servo_angle(sid - 1, angle)
    if 1 <= sid <= 6:
        state['servo_positions'][sid - 1] = angle
    return jsonify({'ok': True, 'servo': sid, 'angle': angle})


@app.route('/api/arm_preset', methods=['POST'])
def api_arm_preset():
    data = request.json or {}
    positions = data.get('positions', [])
    for i, angle in enumerate(positions):
        set_servo_angle(i, int(angle))
        if i < 6:
            state['servo_positions'][i] = int(angle)
        time.sleep(0.05)
    return jsonify({'ok': True, 'preset': data.get('preset', 'custom')})


@app.route('/api/system', methods=['POST'])
def api_system():
    data = request.json or {}
    name = data.get('name', '')
    val = data.get('state', False)
    if name in state:
        state[name] = val
    # TODO: Wire headlight/buzzer to GPIO
    return jsonify({'ok': True, 'system': name, 'state': val})


@app.route('/api/sb_action', methods=['POST'])
def api_sb_action():
    data = request.json or {}
    action = data.get('action', '')

    if action == 'estop':
        drive_stop()
        state['mode'] = 'ESTOP'
        return jsonify({'ok': True, 'action': 'estop'})

    action_modes = {
        'patrol': 'PATROL',
        'socialize': 'SOCIAL',
        'play_cats': 'PLAY',
        'find': 'SEARCH',
    }
    if action in action_modes:
        state['mode'] = action_modes[action]
        # TODO: Hook into signalbot_tank_main.py
        return jsonify({'ok': True, 'action': action})

    return jsonify({'ok': False, 'error': f'Unknown: {action}'})


@app.route('/api/tts', methods=['POST'])
def api_tts():
    data = request.json or {}
    text = data.get('text', '')
    volume = data.get('volume', 50)
    if text and not SIMULATE:
        vol_pct = max(0, min(200, volume * 2))
        os.system(f'espeak -a {vol_pct} "{text}" &')
    return jsonify({'ok': True})


@app.route('/api/screenshot', methods=['POST'])
def api_screenshot():
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = f'/home/pi/screenshots/tank_{ts}.jpg'
    os.makedirs('/home/pi/screenshots', exist_ok=True)
    if not SIMULATE:
        os.system(f'ffmpeg -y -i http://localhost:8080/video_feed -frames:v 1 -q:v 2 {path} 2>/dev/null &')
    return jsonify({'ok': True, 'path': path})


@app.route('/api/record', methods=['POST'])
def api_record():
    data = request.json or {}
    state['recording'] = data.get('state', False)
    # TODO: ffmpeg record start/stop
    return jsonify({'ok': True, 'recording': state['recording']})


if __name__ == '__main__':
    print(f"""
╔══════════════════════════════════════════╗
║  SignalBot Tank HUD                      ║
║  http://192.168.0.11:5000                ║
║  Simulate: {SIMULATE}                           ║
╚══════════════════════════════════════════╝
    """)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
