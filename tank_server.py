#!/usr/bin/env python3
"""
XiaoR Geek Tank — Pi 5 Control Server
Dual-mode: Manual Web UI + SignalBot REST/WebSocket API

Hardware: XiaoR Geek GFS tank chassis, Raspberry Pi 5
GPIO: lgpio (Pi 5 compatible, replaces RPi.GPIO)
Protocol: Preserves original XiaoRGEEK 0xFF-delimited command structure
"""

import os
import sys
import json
import time
import signal
import logging
import asyncio
import argparse
import cv2
import numpy as np
import urllib.request
from enum import IntEnum
from dataclasses import dataclass, field, asdict
from typing import Optional, Callable
from pathlib import Path
from tank_map_server import MapServer


# ---------------------------------------------------------------------------
# Conditional hardware imports — graceful fallback for dev/test
# ---------------------------------------------------------------------------
try:
    import lgpio
    HW_AVAILABLE = True
except ImportError:
    lgpio = None
    HW_AVAILABLE = False
    print("[WARN] lgpio not available — running in SIMULATION mode")

from flask import Flask, request, jsonify, render_template_string, Response
from flask_socketio import SocketIO, emit
import threading

from servo_controller import XR_Servo

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("tank")

# ---------------------------------------------------------------------------
# GPIO Pin Map (from original _XiaoRGEEK_GPIO_.py, BCM numbering)
# ---------------------------------------------------------------------------
class Pins:
    # LEDs
    LED0 = 10
    LED1 = 9
    LED2 = 25
    # Motor driver (L298)
    ENA = 13    # Left enable (PWM)
    ENB = 20    # Right enable (PWM)
    IN1 = 19    # Motor 1
    IN2 = 16    # Motor 2
    IN3 = 21    # Motor 3
    IN4 = 26    # Motor 4
    # Ultrasonic
    ECHO = 4
    TRIG = 17
    # IR sensors
    IR_R = 18
    IR_L = 27
    IR_M = 22
    IRF_R = 23
    IRF_L = 24

# ---------------------------------------------------------------------------
# Hardware Abstraction Layer
# ---------------------------------------------------------------------------
class HardwareLayer:
    """Wraps lgpio calls. In simulation mode, logs instead of driving pins."""

    def __init__(self, simulate: bool = False):
        self.simulate = simulate or not HW_AVAILABLE
        self.chip = None
        self._pwm_a_duty = 100
        self._pwm_b_duty = 100

        if not self.simulate:
            self.chip = lgpio.gpiochip_open(0)
            self._setup_outputs()
            self._setup_inputs()
            self._setup_pwm()
            log.info("Hardware initialized (lgpio, Pi 5)")
        else:
            log.info("Hardware in SIMULATION mode")

    def _setup_outputs(self):
        for pin in [Pins.LED0, Pins.LED1, Pins.LED2,
                     Pins.IN1, Pins.IN2, Pins.IN3, Pins.IN4,
                     Pins.TRIG]:
            lgpio.gpio_claim_output(self.chip, pin, 0)

    def _setup_inputs(self):
        for pin in [Pins.ECHO, Pins.IR_R, Pins.IR_L,
                     Pins.IR_M, Pins.IRF_R, Pins.IRF_L]:
            lgpio.gpio_claim_input(self.chip, pin, lgpio.SET_PULL_UP)

    def _setup_pwm(self):
        # Pi 5 hardware PWM via lgpio — 1kHz like original
        lgpio.gpio_claim_output(self.chip, Pins.ENA, 0)
        lgpio.gpio_claim_output(self.chip, Pins.ENB, 0)
        # We'll use software PWM through lgpio tx_pwm
        lgpio.tx_pwm(self.chip, Pins.ENA, 1000, self._pwm_a_duty)
        lgpio.tx_pwm(self.chip, Pins.ENB, 1000, self._pwm_b_duty)

    def set_pin(self, pin: int, high: bool):
        if self.simulate:
            return
        lgpio.gpio_write(self.chip, pin, 1 if high else 0)

    def read_pin(self, pin: int) -> int:
        if self.simulate:
            return 0
        return lgpio.gpio_read(self.chip, pin)

    def set_speed_a(self, duty: int):
        """Set left motor speed (0-100)."""
        self._pwm_a_duty = max(0, min(100, duty))
        if not self.simulate:
            lgpio.tx_pwm(self.chip, Pins.ENA, 1000, self._pwm_a_duty)

    def set_speed_b(self, duty: int):
        """Set right motor speed (0-100)."""
        self._pwm_b_duty = max(0, min(100, duty))
        if not self.simulate:
            lgpio.tx_pwm(self.chip, Pins.ENB, 1000, self._pwm_b_duty)

    def cleanup(self):
        if self.chip is not None:
            lgpio.gpiochip_close(self.chip)
            log.info("GPIO cleaned up")

# ---------------------------------------------------------------------------
# Motor Controller
# ---------------------------------------------------------------------------
class Direction(IntEnum):
    STOP = 0
    FORWARD = 1
    BACKWARD = 2
    LEFT = 3
    RIGHT = 4

class MotorController:
    """
    Drives the L298N H-bridge. Preserves the original motor_flag remapping
    system (8 wiring permutations) from the XiaoR firmware.
    """

    def __init__(self, hw: HardwareLayer):
        self.hw = hw
        self.motor_flag = 1  # default wiring config
        self.current_direction = Direction.STOP
        self.speed_left = 100
        self.speed_right = 100
        self.map_server = None

    def _fwd(self):
        self.hw.set_pin(Pins.IN1, True)
        self.hw.set_pin(Pins.IN2, False)
        self.hw.set_pin(Pins.IN3, True)
        self.hw.set_pin(Pins.IN4, False)

    def _back(self):
        self.hw.set_pin(Pins.IN1, False)
        self.hw.set_pin(Pins.IN2, True)
        self.hw.set_pin(Pins.IN3, False)
        self.hw.set_pin(Pins.IN4, True)

    def _left(self):
        self.hw.set_pin(Pins.IN1, True)
        self.hw.set_pin(Pins.IN2, False)
        self.hw.set_pin(Pins.IN3, False)
        self.hw.set_pin(Pins.IN4, True)

    def _right(self):
        self.hw.set_pin(Pins.IN1, False)
        self.hw.set_pin(Pins.IN2, True)
        self.hw.set_pin(Pins.IN3, True)
        self.hw.set_pin(Pins.IN4, False)

    def _stop(self):
        for pin in [Pins.IN1, Pins.IN2, Pins.IN3, Pins.IN4]:
            self.hw.set_pin(pin, False)

    def _notify_odometry(self):
        """
        Derive effective left/right power from current direction + speed,
        and push to the mapping system's odometry.
        """
        if self.map_server is None:
            return

        direction = self.current_direction
        sl = self.speed_left
        sr = self.speed_right

        if direction == Direction.STOP:
            left_power, right_power = 0.0, 0.0
        elif direction == Direction.FORWARD:
            left_power, right_power = float(sl), float(sr)
        elif direction == Direction.BACKWARD:
            left_power, right_power = -float(sl), -float(sr)
        elif direction == Direction.LEFT:
            left_power, right_power = -float(sl), float(sr)
        elif direction == Direction.RIGHT:
            left_power, right_power = float(sl), -float(sr)
        else:
            left_power, right_power = 0.0, 0.0

        self.map_server.on_motor_command(left_power, right_power)

    # Motor flag remapping (preserves original firmware's 8-way wiring support)
    _REMAP = {
        Direction.FORWARD:  {1: '_fwd', 2: '_fwd', 3: '_back', 4: '_back',
                             5: '_left', 6: '_left', 7: '_right', 8: '_right'},
        Direction.BACKWARD: {1: '_back', 2: '_back', 3: '_fwd', 4: '_fwd',
                             5: '_right', 6: '_right', 7: '_left', 8: '_left'},
        Direction.LEFT:     {1: '_left', 2: '_right', 3: '_left', 4: '_right',
                             5: '_fwd', 6: '_back', 7: '_fwd', 8: '_back'},
        Direction.RIGHT:    {1: '_right', 2: '_left', 3: '_right', 4: '_left',
                             5: '_back', 6: '_fwd', 7: '_back', 8: '_fwd'},
    }

    def drive(self, direction: Direction):
        self.current_direction = direction
        if direction == Direction.STOP:
            self._stop()
            log.info("MOTOR: stop")
            self._notify_odometry()
            return
        method_name = self._REMAP[direction].get(self.motor_flag, '_stop')
        getattr(self, method_name)()
        log.info(f"MOTOR: {direction.name} (flag={self.motor_flag})")
        self._notify_odometry()              # <-- THIS WAS MISSING

    def set_speed(self, left: int, right: int):
        self.speed_left = left
        self.speed_right = right
        self.hw.set_speed_a(left)
        self.hw.set_speed_b(right)
        log.info(f"SPEED: L={left} R={right}")
        self._notify_odometry()

# ---------------------------------------------------------------------------
# Sensor Reader
# ---------------------------------------------------------------------------
class SensorReader:
    def __init__(self, hw: HardwareLayer):
        self.hw = hw

    def ultrasonic_distance_cm(self) -> float:
        """Trigger ultrasonic sensor and return distance in cm."""
        if hw.simulate:
            return 99.0
        hw = self.hw
        hw.set_pin(Pins.TRIG, True)
        time.sleep(0.000015)
        hw.set_pin(Pins.TRIG, False)

        timeout = time.time() + 0.04  # 40ms max
        while not hw.read_pin(Pins.ECHO):
            if time.time() > timeout:
                return -1.0
        t1 = time.time()

        while hw.read_pin(Pins.ECHO):
            if time.time() > timeout:
                return -1.0
        t2 = time.time()

        distance = (t2 - t1) * 34000 / 2  # cm
        return round(distance, 1) if distance < 500 else -1.0

    def ir_sensors(self) -> dict:
        return {
            "ir_left":     bool(self.hw.read_pin(Pins.IR_L)),
            "ir_right":    bool(self.hw.read_pin(Pins.IR_R)),
            "ir_middle":   bool(self.hw.read_pin(Pins.IR_M)),
            "ir_follow_l": bool(self.hw.read_pin(Pins.IRF_L)),
            "ir_follow_r": bool(self.hw.read_pin(Pins.IRF_R)),
        }

# ---------------------------------------------------------------------------
# LED Controller
# ---------------------------------------------------------------------------
class LEDController:
    def __init__(self, hw: HardwareLayer):
        self.hw = hw

    def headlight(self, on: bool):
        # LED0 is active-low (positive to 5V, GPIO sinks)
        self.hw.set_pin(Pins.LED0, not on)

    def status_leds(self, led1: bool, led2: bool):
        self.hw.set_pin(Pins.LED1, led1)
        self.hw.set_pin(Pins.LED2, led2)

    def flow_sequence(self):
        """Startup flow LED pattern (non-blocking version)."""
        def _flow():
            for _ in range(5):
                for pin in [Pins.LED0, Pins.LED1, Pins.LED2]:
                    self.hw.set_pin(Pins.LED0, pin == Pins.LED0)
                    self.hw.set_pin(Pins.LED1, pin == Pins.LED1)
                    self.hw.set_pin(Pins.LED2, pin == Pins.LED2)
                    time.sleep(0.15)
                for pin in [Pins.LED0, Pins.LED1, Pins.LED2]:
                    self.hw.set_pin(pin, False)
                time.sleep(0.1)
        threading.Thread(target=_flow, daemon=True).start()

# ---------------------------------------------------------------------------
# Tank State (shared across manual + SignalBot control)
# ---------------------------------------------------------------------------
@dataclass
class TankState:
    direction: str = "stop"
    speed_left: int = 100
    speed_right: int = 100
    headlight: bool = False
    mode: str = "manual"          # "manual" | "signalbot"
    ultrasonic_cm: float = -1.0
    ir_sensors: dict = field(default_factory=dict)
    servo_angles: dict = field(default_factory=dict)
    signalbot_connected: bool = False
    last_command_source: str = "none"   # "web" | "signalbot" | "none"
    last_command_time: float = 0.0

    def to_dict(self):
        return asdict(self)

# ---------------------------------------------------------------------------
# SignalBot Integration Layer
# ---------------------------------------------------------------------------
class SignalBotBridge:
    """
    API bridge for SignalBot daemon integration.

    SignalBot can:
    - Send movement commands via REST or WebSocket
    - Read sensor data
    - Claim/release control authority
    - Receive state change events via WebSocket

    Auth uses a shared secret token (set via env or config).
    """

    def __init__(self, state: TankState, motor: MotorController,
                 sensors: SensorReader, leds: LEDController,
                 servo: XR_Servo):
        self.state = state
        self.motor = motor
        self.sensors = sensors
        self.leds = leds
        self.servo = servo
        self.token = os.environ.get("SIGNALBOT_TOKEN", "signalbot_dev_key")
        self._command_queue = []

    def authenticate(self, token: str) -> bool:
        return token == self.token

    def claim_control(self):
        self.state.mode = "signalbot"
        self.state.signalbot_connected = True
        log.info("SignalBot: CLAIMED control")

    def release_control(self):
        self.state.mode = "manual"
        self.state.signalbot_connected = False
        self.motor.drive(Direction.STOP)
        log.info("SignalBot: RELEASED control")

    def execute_command(self, cmd: dict) -> dict:
        """
        Execute a SignalBot command.

        Supported commands:
            {"action": "move", "direction": "forward|backward|left|right|stop"}
            {"action": "speed", "left": 0-100, "right": 0-100}
            {"action": "sensor_read"}
            {"action": "headlight", "on": true|false}
            {"action": "servo", "num": 1-8, "angle": 0-180}
            {"action": "servo_sweep", "num": 1-8, "start": 0-180, "end": 0-180}
            {"action": "servo_center"}
            {"action": "servo_save"}
            {"action": "servo_reset"}
            {"action": "claim"}
            {"action": "release"}
            {"action": "status"}
            {"action": "emergency_stop"}
        """
        action = cmd.get("action", "")
        self.state.last_command_source = "signalbot"
        self.state.last_command_time = time.time()

        if action == "move":
            direction = cmd.get("direction", "stop")
            dir_map = {
                "forward": Direction.FORWARD, "backward": Direction.BACKWARD,
                "left": Direction.LEFT, "right": Direction.RIGHT,
                "stop": Direction.STOP,
            }
            d = dir_map.get(direction, Direction.STOP)
            self.motor.drive(d)
            self.state.direction = direction
            return {"ok": True, "direction": direction}

        elif action == "speed":
            left = cmd.get("left", 100)
            right = cmd.get("right", 100)
            self.motor.set_speed(left, right)
            self.state.speed_left = left
            self.state.speed_right = right
            return {"ok": True, "speed": {"left": left, "right": right}}

        elif action == "sensor_read":
            dist = self.sensors.ultrasonic_distance_cm()
            ir = self.sensors.ir_sensors()
            self.state.ultrasonic_cm = dist
            self.state.ir_sensors = ir
            return {"ok": True, "ultrasonic_cm": dist, "ir": ir}

        elif action == "headlight":
            on = cmd.get("on", False)
            self.leds.headlight(on)
            self.state.headlight = on
            return {"ok": True, "headlight": on}

        elif action == "servo":
            num = cmd.get("num", 1)
            angle = cmd.get("angle", 90)
            self.servo.set_angle(num, angle)
            self.state.servo_angles = self.servo.get_all_angles()
            return {"ok": True, "servo": num, "angle": angle}

        elif action == "servo_sweep":
            num = cmd.get("num", 1)
            start = cmd.get("start", 30)
            end = cmd.get("end", 150)
            step = cmd.get("step", 2)
            delay = cmd.get("delay", 0.02)
            self.servo.sweep(num, start, end, step, delay)
            self.state.servo_angles = self.servo.get_all_angles()
            return {"ok": True, "servo": num, "swept_to": end}

        elif action == "servo_center":
            self.servo.center_all()
            self.state.servo_angles = self.servo.get_all_angles()
            return {"ok": True, "servos": "centered"}

        elif action == "servo_save":
            self.servo.XiaoRGEEK_SaveServo()
            return {"ok": True, "servos": "saved"}

        elif action == "servo_reset":
            self.servo.XiaoRGEEK_ReSetServo()
            self.state.servo_angles = self.servo.get_all_angles()
            return {"ok": True, "servos": "reset"}

        elif action == "claim":
            self.claim_control()
            return {"ok": True, "mode": "signalbot"}

        elif action == "release":
            self.release_control()
            return {"ok": True, "mode": "manual"}

        elif action == "status":
            self.state.servo_angles = self.servo.get_all_angles()
            return {"ok": True, "state": self.state.to_dict()}

        elif action == "emergency_stop":
            self.motor.drive(Direction.STOP)
            self.state.direction = "stop"
            log.warning("SignalBot: EMERGENCY STOP")
            return {"ok": True, "emergency_stop": True}

        else:
            return {"ok": False, "error": f"Unknown action: {action}"}

# ---------------------------------------------------------------------------
# Flask App + SocketIO
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('TANK_SECRET', 'xr-tank-dev-secret')
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Globals (initialized in main)
hw: Optional[HardwareLayer] = None
motor: Optional[MotorController] = None
sensors: Optional[SensorReader] = None
leds: Optional[LEDController] = None
servo: Optional[XR_Servo] = None
state: Optional[TankState] = None
bridge: Optional[SignalBotBridge] = None

# ---------------------------------------------------------------------------
# Web UI (manual control)
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template_string(WEB_UI_HTML)

# ---------------------------------------------------------------------------
# REST API — SignalBot
# ---------------------------------------------------------------------------
@app.route("/api/v1/command", methods=["POST"])
def api_command():
    """SignalBot command endpoint."""
    token = request.headers.get("X-SignalBot-Token", "")
    if not bridge.authenticate(token):
        return jsonify({"ok": False, "error": "Unauthorized"}), 401

    cmd = request.get_json(silent=True) or {}
    result = bridge.execute_command(cmd)

    # Broadcast state update to all WebSocket clients
    socketio.emit("state_update", state.to_dict())
    return jsonify(result)

@app.route("/api/v1/status", methods=["GET"])
def api_status():
    """Public status endpoint (no auth required)."""
    return jsonify(state.to_dict())

@app.route("/api/v1/sensors", methods=["GET"])
def api_sensors():
    """Quick sensor read."""
    dist = sensors.ultrasonic_distance_cm()
    ir = sensors.ir_sensors()
    state.ultrasonic_cm = dist
    state.ir_sensors = ir
    return jsonify({"ultrasonic_cm": dist, "ir": ir})

# ---------------------------------------------------------------------------
# WebSocket — bidirectional for both Web UI and SignalBot
# ---------------------------------------------------------------------------
@socketio.on("connect")
def ws_connect():
    log.info(f"WebSocket client connected")
    emit("state_update", state.to_dict())

@socketio.on("manual_command")
def ws_manual_command(data):
    """Handle manual control from web UI."""
    if state.mode == "signalbot":
        emit("error", {"msg": "SignalBot has control. Release first."})
        return

    action = data.get("action", "")
    state.last_command_source = "web"
    state.last_command_time = time.time()

    if action == "move":
        direction = data.get("direction", "stop")
        dir_map = {
            "forward": Direction.FORWARD, "backward": Direction.BACKWARD,
            "left": Direction.LEFT, "right": Direction.RIGHT,
            "stop": Direction.STOP,
        }
        motor.drive(dir_map.get(direction, Direction.STOP))
        state.direction = direction

    elif action == "speed":
        left = data.get("left", 100)
        right = data.get("right", 100)
        motor.set_speed(left, right)
        state.speed_left = left
        state.speed_right = right

    elif action == "headlight":
        on = data.get("on", False)
        leds.headlight(on)
        state.headlight = on

    elif action == "servo":
        num = data.get("num", 1)
        angle = data.get("angle", 90)
        servo.set_angle(num, angle)
        state.servo_angles = servo.get_all_angles()

    elif action == "servo_save":
        servo.XiaoRGEEK_SaveServo()

    elif action == "servo_reset":
        servo.XiaoRGEEK_ReSetServo()
        state.servo_angles = servo.get_all_angles()

    elif action == "emergency_stop":
        motor.drive(Direction.STOP)
        state.direction = "stop"

    socketio.emit("state_update", state.to_dict())

@socketio.on("signalbot_command")
def ws_signalbot_command(data):
    """Handle SignalBot commands over WebSocket (alternative to REST)."""
    token = data.get("token", "")
    if not bridge.authenticate(token):
        emit("error", {"msg": "Unauthorized"})
        return

    cmd = data.get("command", {})
    result = bridge.execute_command(cmd)
    emit("command_result", result)
    socketio.emit("state_update", state.to_dict())

# ---------------------------------------------------------------------------
# Background sensor polling (pushes to all WS clients)
# ---------------------------------------------------------------------------
def sensor_polling_loop():
    """Poll sensors every 500ms, broadcast to connected clients."""
    while True:
        try:
            state.ultrasonic_cm = sensors.ultrasonic_distance_cm()
            state.ir_sensors = sensors.ir_sensors()
            socketio.emit("sensor_data", {
                "ultrasonic_cm": state.ultrasonic_cm,
                "ir": state.ir_sensors,
                "timestamp": time.time()
            })
        except Exception as e:
            log.error(f"Sensor poll error: {e}")
        time.sleep(0.5)

# ---------------------------------------------------------------------------
# Web UI HTML (embedded — single file deployment)
# ---------------------------------------------------------------------------
WEB_UI_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no"/>
<title>XR Tank — Command Interface</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;800&family=Orbitron:wght@400;700;900&display=swap');

  :root {
    --bg: #0a0c10;
    --surface: #12151c;
    --surface-2: #1a1e28;
    --border: #2a2f3d;
    --accent: #00ff88;
    --accent-dim: #00ff8833;
    --danger: #ff3355;
    --danger-dim: #ff335533;
    --warn: #ffaa00;
    --text: #e0e4ec;
    --text-dim: #6a7088;
    --signalbot: #7b61ff;
    --signalbot-dim: #7b61ff33;
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    font-family: 'JetBrains Mono', monospace;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* Scanline overlay */
  body::after {
    content: '';
    position: fixed; inset: 0;
    background: repeating-linear-gradient(
      0deg, transparent, transparent 2px,
      rgba(0,0,0,0.03) 2px, rgba(0,0,0,0.03) 4px
    );
    pointer-events: none;
    z-index: 9999;
  }

  .header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 24px;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
  }

  .header h1 {
    font-family: 'Orbitron', sans-serif;
    font-size: 18px;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    background: linear-gradient(135deg, var(--accent), #00ccff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }

  .mode-badge {
    font-size: 11px;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 20px;
    letter-spacing: 1px;
    text-transform: uppercase;
  }
  .mode-badge.manual {
    background: var(--accent-dim);
    color: var(--accent);
    border: 1px solid var(--accent);
  }
  .mode-badge.signalbot {
    background: var(--signalbot-dim);
    color: var(--signalbot);
    border: 1px solid var(--signalbot);
    animation: pulse-glow 2s ease-in-out infinite;
  }

  @keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 4px var(--signalbot-dim); }
    50% { box-shadow: 0 0 16px var(--signalbot-dim); }
  }

  .main {
    display: grid;
    grid-template-columns: 1fr 280px;
    gap: 16px;
    padding: 16px 24px;
    max-width: 960px;
    margin: 0 auto;
  }

  @media (max-width: 700px) {
    .main { grid-template-columns: 1fr; }
  }

  .panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
  }

  .panel-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 12px;
  }

  /* D-pad */
  .dpad {
    display: grid;
    grid-template-areas:
      ". up ."
      "left stop right"
      ". down .";
    grid-template-columns: repeat(3, 72px);
    grid-template-rows: repeat(3, 72px);
    gap: 6px;
    justify-content: center;
    margin: 8px 0;
  }

  .dpad button {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text);
    font-size: 22px;
    cursor: pointer;
    transition: all 0.1s;
    display: flex;
    align-items: center;
    justify-content: center;
    -webkit-user-select: none;
    user-select: none;
    touch-action: manipulation;
  }

  .dpad button:active, .dpad button.active {
    background: var(--accent-dim);
    border-color: var(--accent);
    color: var(--accent);
    box-shadow: 0 0 20px var(--accent-dim);
    transform: scale(0.96);
  }

  .dpad .up    { grid-area: up; }
  .dpad .down  { grid-area: down; }
  .dpad .left  { grid-area: left; }
  .dpad .right { grid-area: right; }
  .dpad .stop  { grid-area: stop; font-size: 13px; font-weight: 600; }

  /* Emergency stop */
  .estop {
    width: 100%;
    padding: 14px;
    margin-top: 12px;
    background: var(--danger-dim);
    border: 2px solid var(--danger);
    border-radius: 8px;
    color: var(--danger);
    font-family: 'Orbitron', sans-serif;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    cursor: pointer;
    transition: all 0.15s;
  }
  .estop:active {
    background: var(--danger);
    color: #fff;
    box-shadow: 0 0 30px var(--danger-dim);
  }

  /* Speed slider */
  .speed-control { margin-top: 16px; }
  .speed-control label {
    font-size: 11px;
    color: var(--text-dim);
    display: block;
    margin-bottom: 6px;
  }
  .speed-control input[type=range] {
    width: 100%;
    accent-color: var(--accent);
  }
  .speed-val {
    font-size: 13px;
    color: var(--accent);
    font-weight: 600;
    text-align: center;
    margin-top: 4px;
  }

  /* Headlight toggle */
  .toggle-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 12px;
    padding: 10px 12px;
    background: var(--surface-2);
    border-radius: 6px;
  }
  .toggle-row span { font-size: 12px; }
  .toggle-switch {
    width: 44px; height: 24px;
    background: var(--border);
    border-radius: 12px;
    cursor: pointer;
    position: relative;
    transition: background 0.2s;
  }
  .toggle-switch.on { background: var(--accent); }
  .toggle-switch::after {
    content: '';
    position: absolute;
    top: 3px; left: 3px;
    width: 18px; height: 18px;
    background: #fff;
    border-radius: 50%;
    transition: transform 0.2s;
  }
  .toggle-switch.on::after { transform: translateX(20px); }

  /* Sensor display */
  .sensor-grid {
    display: grid;
    gap: 8px;
  }
  .sensor-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 10px;
    background: var(--surface-2);
    border-radius: 4px;
    font-size: 12px;
  }
  .sensor-label { color: var(--text-dim); }
  .sensor-value { color: var(--accent); font-weight: 600; }
  .sensor-value.triggered { color: var(--warn); }

  /* Status bar */
  .status-bar {
    display: flex;
    gap: 16px;
    align-items: center;
    padding: 10px 24px;
    border-top: 1px solid var(--border);
    background: var(--surface);
    font-size: 11px;
    color: var(--text-dim);
    position: fixed;
    bottom: 0;
    left: 0; right: 0;
  }
  .status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
  }
  .status-dot.connected { background: var(--accent); box-shadow: 0 0 6px var(--accent); }
  .status-dot.disconnected { background: var(--danger); }

  /* Camera placeholder */
  .camera-feed {
    background: var(--surface-2);
    border: 1px dashed var(--border);
    border-radius: 6px;
    height: 180px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--text-dim);
    font-size: 11px;
    margin-bottom: 12px;
    overflow: hidden;
  }
  .camera-feed img {
    width: 100%;
    height: 100%;
    object-fit: cover;
  }

  /* SignalBot indicator */
  .sb-status {
    margin-top: 12px;
    padding: 10px;
    border-radius: 6px;
    font-size: 11px;
    text-align: center;
  }
  .sb-status.idle {
    background: var(--surface-2);
    color: var(--text-dim);
    border: 1px solid var(--border);
  }
  .sb-status.active {
    background: var(--signalbot-dim);
    color: var(--signalbot);
    border: 1px solid var(--signalbot);
  }

  /* Servo controls */
  .servo-grid { display: flex; flex-direction: column; gap: 6px; }
  .servo-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 4px 0;
  }
  .servo-label {
    font-size: 11px;
    color: var(--text-dim);
    min-width: 64px;
    white-space: nowrap;
  }
  .servo-val { color: var(--accent); font-weight: 600; }
  .servo-slider {
    flex: 1;
    accent-color: var(--accent);
    height: 4px;
  }
  .servo-btn {
    flex: 1;
    padding: 6px 8px;
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 4px;
    color: var(--text);
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    cursor: pointer;
    transition: all 0.15s;
  }
  .servo-btn:active {
    background: var(--accent-dim);
    border-color: var(--accent);
    color: var(--accent);
  }
</style>
</head>
<body>
<div class="header">
  <h1>XR Tank</h1>
  <div class="mode-badge manual" id="modeBadge">MANUAL</div>
</div>

<div class="main">
  <div style="display:flex;flex-direction:column;gap:16px;">
    <!-- Camera -->
    <div class="panel">
      <div class="panel-title">Camera Feed</div>
      <div class="camera-feed" id="cameraFeed">
        <span>mjpg-streamer @ :8080 — connect after boot</span>
      </div>
    </div>

    <!-- Controls -->
    <div class="panel">
      <div class="panel-title">Drive Control</div>
      <div class="dpad" id="dpad">
        <button class="up" data-dir="forward">▲</button>
        <button class="left" data-dir="left">◄</button>
        <button class="stop" data-dir="stop">■</button>
        <button class="right" data-dir="right">►</button>
        <button class="down" data-dir="backward">▼</button>
      </div>
      <button class="estop" id="estop">⚡ Emergency Stop</button>

      <div class="speed-control">
        <label>Speed</label>
        <input type="range" id="speedSlider" min="0" max="100" value="100"/>
        <div class="speed-val"><span id="speedVal">100</span>%</div>
      </div>

      <div class="toggle-row">
        <span>💡 Headlight</span>
        <div class="toggle-switch" id="headlightToggle"></div>
      </div>
    </div>

    <!-- Servo Control -->
    <div class="panel">
      <div class="panel-title">Servos (PCA9685)</div>
      <div class="servo-grid" id="servoGrid">
        <div class="servo-row">
          <span class="servo-label">S1 <span class="servo-val" id="sv1">90</span>°</span>
          <input type="range" class="servo-slider" data-servo="1" min="15" max="160" value="90"/>
        </div>
        <div class="servo-row">
          <span class="servo-label">S2 <span class="servo-val" id="sv2">90</span>°</span>
          <input type="range" class="servo-slider" data-servo="2" min="15" max="160" value="90"/>
        </div>
        <div class="servo-row">
          <span class="servo-label">S3 <span class="servo-val" id="sv3">90</span>°</span>
          <input type="range" class="servo-slider" data-servo="3" min="15" max="160" value="90"/>
        </div>
        <div class="servo-row">
          <span class="servo-label">S4 <span class="servo-val" id="sv4">90</span>°</span>
          <input type="range" class="servo-slider" data-servo="4" min="15" max="160" value="90"/>
        </div>
        <div class="servo-row">
          <span class="servo-label">S5 <span class="servo-val" id="sv5">90</span>°</span>
          <input type="range" class="servo-slider" data-servo="5" min="15" max="160" value="90"/>
        </div>
        <div class="servo-row">
          <span class="servo-label">S6 <span class="servo-val" id="sv6">90</span>°</span>
          <input type="range" class="servo-slider" data-servo="6" min="15" max="160" value="90"/>
        </div>
        <div class="servo-row">
          <span class="servo-label">S7 <span class="servo-val" id="sv7">90</span>°</span>
          <input type="range" class="servo-slider" data-servo="7" min="15" max="160" value="90"/>
        </div>
        <div class="servo-row">
          <span class="servo-label">S8 <span class="servo-val" id="sv8">90</span>°</span>
          <input type="range" class="servo-slider" data-servo="8" min="15" max="160" value="90"/>
        </div>
      </div>
      <div style="display:flex;gap:6px;margin-top:10px;">
        <button class="servo-btn" id="servoCenterBtn">Center All</button>
        <button class="servo-btn" id="servoSaveBtn">Save</button>
        <button class="servo-btn" id="servoResetBtn">Restore</button>
      </div>
    </div>
  </div>

  <!-- Sidebar -->
  <div style="display:flex;flex-direction:column;gap:16px;">
    <div class="panel">
      <div class="panel-title">Sensors</div>
      <div class="sensor-grid" id="sensorGrid">
        <div class="sensor-item">
          <span class="sensor-label">Ultrasonic</span>
          <span class="sensor-value" id="sUltra">—</span>
        </div>
        <div class="sensor-item">
          <span class="sensor-label">IR Left</span>
          <span class="sensor-value" id="sIrL">—</span>
        </div>
        <div class="sensor-item">
          <span class="sensor-label">IR Right</span>
          <span class="sensor-value" id="sIrR">—</span>
        </div>
        <div class="sensor-item">
          <span class="sensor-label">IR Middle</span>
          <span class="sensor-value" id="sIrM">—</span>
        </div>
        <div class="sensor-item">
          <span class="sensor-label">Follow L</span>
          <span class="sensor-value" id="sFlL">—</span>
        </div>
        <div class="sensor-item">
          <span class="sensor-label">Follow R</span>
          <span class="sensor-value" id="sFlR">—</span>
        </div>
      </div>
    </div>

    <div class="panel">
      <div class="panel-title">SignalBot</div>
      <div class="sb-status idle" id="sbStatus">
        NOT CONNECTED
      </div>
      <div style="margin-top:8px;font-size:10px;color:var(--text-dim);line-height:1.5;">
        REST: POST /api/v1/command<br>
        WS: emit("signalbot_command")<br>
        Auth: X-SignalBot-Token header
      </div>
    </div>

    <div class="panel">
      <div class="panel-title">System</div>
      <div class="sensor-grid">
        <div class="sensor-item">
          <span class="sensor-label">Direction</span>
          <span class="sensor-value" id="sDir">stop</span>
        </div>
        <div class="sensor-item">
          <span class="sensor-label">Last Cmd</span>
          <span class="sensor-value" id="sLastCmd">—</span>
        </div>
      </div>
    </div>
  </div>
</div>

<div class="status-bar">
  <span><span class="status-dot disconnected" id="wsDot"></span>WebSocket</span>
  <span id="wsStatus">Connecting...</span>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.4/socket.io.min.js"></script>
<script>
(function() {
  const socket = io();
  const dpad = document.getElementById('dpad');
  const estop = document.getElementById('estop');
  const speedSlider = document.getElementById('speedSlider');
  const speedVal = document.getElementById('speedVal');
  const headlightToggle = document.getElementById('headlightToggle');
  const wsDot = document.getElementById('wsDot');
  const wsStatus = document.getElementById('wsStatus');
  let headlightOn = false;

  // Connection status
  socket.on('connect', () => {
    wsDot.className = 'status-dot connected';
    wsStatus.textContent = 'Connected';
  });
  socket.on('disconnect', () => {
    wsDot.className = 'status-dot disconnected';
    wsStatus.textContent = 'Disconnected';
  });

  // D-pad controls (touch + mouse)
  dpad.querySelectorAll('button').forEach(btn => {
    const dir = btn.dataset.dir;

    function sendMove() {
      btn.classList.add('active');
      socket.emit('manual_command', { action: 'move', direction: dir });
    }
    function sendStop() {
      btn.classList.remove('active');
      if (dir !== 'stop') {
        socket.emit('manual_command', { action: 'move', direction: 'stop' });
      }
    }

    // For stop button, just send once
    if (dir === 'stop') {
      btn.addEventListener('mousedown', sendMove);
      btn.addEventListener('touchstart', (e) => { e.preventDefault(); sendMove(); });
      btn.addEventListener('mouseup', () => btn.classList.remove('active'));
      btn.addEventListener('touchend', () => btn.classList.remove('active'));
    } else {
      btn.addEventListener('mousedown', sendMove);
      btn.addEventListener('mouseup', sendStop);
      btn.addEventListener('mouseleave', sendStop);
      btn.addEventListener('touchstart', (e) => { e.preventDefault(); sendMove(); });
      btn.addEventListener('touchend', (e) => { e.preventDefault(); sendStop(); });
    }
  });

  // Keyboard controls
  const keyMap = {
    'ArrowUp': 'forward', 'w': 'forward', 'W': 'forward',
    'ArrowDown': 'backward', 's': 'backward', 'S': 'backward',
    'ArrowLeft': 'left', 'a': 'left', 'A': 'left',
    'ArrowRight': 'right', 'd': 'right', 'D': 'right',
    ' ': 'stop'
  };
  const keysHeld = new Set();

  document.addEventListener('keydown', (e) => {
    if (keyMap[e.key] && !keysHeld.has(e.key)) {
      keysHeld.add(e.key);
      socket.emit('manual_command', { action: 'move', direction: keyMap[e.key] });
      // Highlight matching button
      const btn = dpad.querySelector(`[data-dir="${keyMap[e.key]}"]`);
      if (btn) btn.classList.add('active');
    }
  });

  document.addEventListener('keyup', (e) => {
    if (keyMap[e.key]) {
      keysHeld.delete(e.key);
      if (keyMap[e.key] !== 'stop' && keysHeld.size === 0) {
        socket.emit('manual_command', { action: 'move', direction: 'stop' });
      }
      const btn = dpad.querySelector(`[data-dir="${keyMap[e.key]}"]`);
      if (btn) btn.classList.remove('active');
    }
  });

  // Emergency stop
  estop.addEventListener('click', () => {
    socket.emit('manual_command', { action: 'emergency_stop' });
  });

  // Speed
  speedSlider.addEventListener('input', () => {
    const v = parseInt(speedSlider.value);
    speedVal.textContent = v;
    socket.emit('manual_command', { action: 'speed', left: v, right: v });
  });

  // Headlight
  headlightToggle.addEventListener('click', () => {
    headlightOn = !headlightOn;
    headlightToggle.classList.toggle('on', headlightOn);
    socket.emit('manual_command', { action: 'headlight', on: headlightOn });
  });

  // Servo sliders
  document.querySelectorAll('.servo-slider').forEach(slider => {
    slider.addEventListener('input', () => {
      const num = parseInt(slider.dataset.servo);
      const angle = parseInt(slider.value);
      document.getElementById('sv' + num).textContent = angle;
      socket.emit('manual_command', { action: 'servo', num: num, angle: angle });
    });
  });
  document.getElementById('servoCenterBtn').addEventListener('click', () => {
    document.querySelectorAll('.servo-slider').forEach(s => { s.value = 90; });
    for (let i = 1; i <= 8; i++) document.getElementById('sv' + i).textContent = '90';
    socket.emit('manual_command', { action: 'servo', num: 0, angle: 90 });
    // Send center for each servo
    for (let i = 1; i <= 8; i++) {
      socket.emit('manual_command', { action: 'servo', num: i, angle: 90 });
    }
  });
  document.getElementById('servoSaveBtn').addEventListener('click', () => {
    socket.emit('manual_command', { action: 'servo_save' });
  });
  document.getElementById('servoResetBtn').addEventListener('click', () => {
    socket.emit('manual_command', { action: 'servo_reset' });
  });

  // State updates
  socket.on('state_update', (s) => {
    document.getElementById('sDir').textContent = s.direction;
    document.getElementById('sLastCmd').textContent = s.last_command_source;
    const badge = document.getElementById('modeBadge');
    if (s.mode === 'signalbot') {
      badge.className = 'mode-badge signalbot';
      badge.textContent = 'SIGNALBOT';
    } else {
      badge.className = 'mode-badge manual';
      badge.textContent = 'MANUAL';
    }
    const sb = document.getElementById('sbStatus');
    if (s.signalbot_connected) {
      sb.className = 'sb-status active';
      sb.textContent = 'CONNECTED — IN CONTROL';
    } else {
      sb.className = 'sb-status idle';
      sb.textContent = 'NOT CONNECTED';
    }
  });

  // Sensor data
  socket.on('sensor_data', (d) => {
    document.getElementById('sUltra').textContent =
      d.ultrasonic_cm >= 0 ? d.ultrasonic_cm + ' cm' : '—';
    if (d.ir) {
      const setIr = (id, val) => {
        const el = document.getElementById(id);
        el.textContent = val ? 'CLEAR' : 'TRIG';
        el.className = val ? 'sensor-value' : 'sensor-value triggered';
      };
      setIr('sIrL', d.ir.ir_left);
      setIr('sIrR', d.ir.ir_right);
      setIr('sIrM', d.ir.ir_middle);
      setIr('sFlL', d.ir.ir_follow_l);
      setIr('sFlR', d.ir.ir_follow_r);
    }
  });

  // Error handling
  socket.on('error', (e) => {
    console.warn('Server error:', e.msg);
  });

  // Camera feed — try to load mjpg-streamer
  setTimeout(() => {
    const cam = document.getElementById('cameraFeed');
    const img = document.createElement('img');
    img.src = 'http://' + window.location.hostname + ':8080/?action=stream';
    img.onerror = () => { /* leave placeholder */ };
    img.onload = () => { cam.innerHTML = ''; cam.appendChild(img); };
  }, 2000);

})();
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
def camera_feed_loop(map_server, stream_url="http://localhost:8080/?action=snapshot"):
    """
    Pulls JPEG snapshots from mjpg-streamer and feeds them to the mapper.

    Why snapshot instead of stream:
      - The MJPEG stream is already being consumed by the web UI
      - Snapshots are simpler, no MJPEG boundary parsing needed
      - 5-10 fps is plenty for floor mapping
    """
    while True:
        try:
            resp = urllib.request.urlopen(stream_url, timeout=2)
            jpg_data = resp.read()
            frame = cv2.imdecode(
                np.frombuffer(jpg_data, dtype=np.uint8),
                cv2.IMREAD_COLOR
            )
            if frame is not None:
                map_server.on_camera_frame(frame)
        except Exception as e:
            # mjpg-streamer might not be running yet — just retry
            pass
        time.sleep(0.15)  # ~6-7 fps is fine for mapping

def main():
    global hw, motor, sensors, leds, servo, state, bridge

    parser = argparse.ArgumentParser(description="XR Tank Control Server")
    parser.add_argument("--simulate", action="store_true",
                        help="Run without GPIO hardware (dev mode)")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=5000, help="HTTP port")
    args = parser.parse_args()

    # Initialize hardware FIRST
    hw = HardwareLayer(simulate=args.simulate)
    motor = MotorController(hw)
    sensors = SensorReader(hw)
    leds = LEDController(hw)
    servo = XR_Servo(simulate=args.simulate or not HW_AVAILABLE)
    state = TankState(servo_angles=servo.get_all_angles())
    bridge = SignalBotBridge(state, motor, sensors, leds, servo)

    # Initialize mapping AFTER motor exists
    map_server = MapServer(use_internal_camera=False)
    motor.map_server = map_server
    map_server.start()

    # Camera frame grabber — pulls from mjpg-streamer
    cam_thread = threading.Thread(
        target=camera_feed_loop,
        args=(map_server,),
        daemon=True
    )
    cam_thread.start()

    # Startup LED sequence
    leds.flow_sequence()

    # Background sensor polling
    sensor_thread = threading.Thread(target=sensor_polling_loop, daemon=True)
    sensor_thread.start()

    # Graceful shutdown
    def shutdown(sig, frame):
        log.info("Shutting down...")
        motor.drive(Direction.STOP)
        map_server.stop()
        servo.cleanup()
        hw.cleanup()
        sys.exit(0)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    log.info(f"Tank server starting on {args.host}:{args.port}")
    log.info(f"Hardware: {'LIVE' if not hw.simulate else 'SIMULATED'}")
    log.info(f"Servos: {servo.get_all_angles()}")
    log.info(f"SignalBot API: POST /api/v1/command")
    log.info(f"Map server: TCP port 5555 (connect slamd visualizer)")
    log.info(f"Web UI: http://{args.host}:{args.port}/")

    socketio.run(app, host=args.host, port=args.port,
                 allow_unsafe_werkzeug=True)

if __name__ == "__main__":
    main()
