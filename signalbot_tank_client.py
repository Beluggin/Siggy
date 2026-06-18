#!/usr/bin/env python3
"""
SignalBot Tank Client — Drop-in integration for SignalBot v7

This module lets any SignalBot daemon or the autonomous loop
issue commands to the XR Tank via REST or WebSocket.

Usage from SignalBot:
    from signalbot_tank_client import TankClient

    tank = TankClient("http://192.168.x.x:5000", token="your_token")

    # Claim control
    tank.claim()

    # Drive
    tank.forward()
    tank.set_speed(left=60, right=60)
    tank.stop()

    # Read sensors
    data = tank.sensors()
    print(data["ultrasonic_cm"])

    # Release back to manual
    tank.release()
"""

import json
import time
import logging
import requests
from typing import Optional, Callable

log = logging.getLogger("signalbot.tank")


class TankClient:
    """
    Synchronous REST client for the XR Tank server.
    Designed to be imported directly into SignalBot daemons.
    """

    def __init__(self, base_url: str, token: str = "signalbot_dev_key",
                 timeout: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({
            "X-SignalBot-Token": self.token,
            "Content-Type": "application/json"
        })

    def _cmd(self, command: dict) -> dict:
        """Send a command to the tank server."""
        try:
            resp = self._session.post(
                f"{self.base_url}/api/v1/command",
                json=command,
                timeout=self.timeout
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            log.error(f"Tank command failed: {e}")
            return {"ok": False, "error": str(e)}

    # -- Control authority --

    def claim(self) -> dict:
        """Claim control from manual mode."""
        return self._cmd({"action": "claim"})

    def release(self) -> dict:
        """Release control back to manual mode."""
        return self._cmd({"action": "release"})

    # -- Movement --

    def forward(self) -> dict:
        return self._cmd({"action": "move", "direction": "forward"})

    def backward(self) -> dict:
        return self._cmd({"action": "move", "direction": "backward"})

    def left(self) -> dict:
        return self._cmd({"action": "move", "direction": "left"})

    def right(self) -> dict:
        return self._cmd({"action": "move", "direction": "right"})

    def stop(self) -> dict:
        return self._cmd({"action": "move", "direction": "stop"})

    def emergency_stop(self) -> dict:
        return self._cmd({"action": "emergency_stop"})

    # -- Speed --

    def set_speed(self, left: int = 100, right: int = 100) -> dict:
        return self._cmd({"action": "speed", "left": left, "right": right})

    # -- Sensors --

    def sensors(self) -> dict:
        """Read all sensors (ultrasonic + IR)."""
        return self._cmd({"action": "sensor_read"})

    def distance(self) -> float:
        """Quick ultrasonic distance read (cm). Returns -1 on error."""
        result = self.sensors()
        return result.get("ultrasonic_cm", -1.0)

    # -- Accessories --

    def headlight(self, on: bool = True) -> dict:
        return self._cmd({"action": "headlight", "on": on})

    # -- Servos (PCA9685, 8 channels) --

    def servo(self, num: int, angle: int) -> dict:
        """Set servo angle. num=1-8, angle=0-180."""
        return self._cmd({"action": "servo", "num": num, "angle": angle})

    def servo_sweep(self, num: int, start: int, end: int,
                    step: int = 2, delay: float = 0.02) -> dict:
        """Sweep a servo smoothly between two angles."""
        return self._cmd({"action": "servo_sweep", "num": num,
                         "start": start, "end": end,
                         "step": step, "delay": delay})

    def servo_center(self) -> dict:
        """Center all servos to 90 degrees."""
        return self._cmd({"action": "servo_center"})

    def servo_save(self) -> dict:
        """Save current servo positions to EEPROM."""
        return self._cmd({"action": "servo_save"})

    def servo_reset(self) -> dict:
        """Restore saved servo positions from EEPROM."""
        return self._cmd({"action": "servo_reset"})

    # -- Status --

    def status(self) -> dict:
        return self._cmd({"action": "status"})

    def is_connected(self) -> bool:
        """Quick health check."""
        try:
            resp = self._session.get(
                f"{self.base_url}/api/v1/status",
                timeout=2.0
            )
            return resp.status_code == 200
        except:
            return False


class TankWebSocketClient:
    """
    WebSocket client for real-time bidirectional communication.
    Use this when SignalBot needs streaming sensor data or
    low-latency command execution.

    Requires: python-socketio[client]
        pip install python-socketio[client]
    """

    def __init__(self, base_url: str, token: str = "signalbot_dev_key"):
        self.base_url = base_url
        self.token = token
        self._sio = None
        self._sensor_callback: Optional[Callable] = None
        self._state_callback: Optional[Callable] = None

    def connect(self, on_sensor_data: Optional[Callable] = None,
                on_state_update: Optional[Callable] = None):
        """
        Connect to the tank server via WebSocket.

        Args:
            on_sensor_data: callback(data_dict) for sensor updates (~2Hz)
            on_state_update: callback(state_dict) for state changes
        """
        import socketio
        self._sio = socketio.Client()
        self._sensor_callback = on_sensor_data
        self._state_callback = on_state_update

        @self._sio.on("sensor_data")
        def _on_sensor(data):
            if self._sensor_callback:
                self._sensor_callback(data)

        @self._sio.on("state_update")
        def _on_state(data):
            if self._state_callback:
                self._state_callback(data)

        @self._sio.on("command_result")
        def _on_result(data):
            log.debug(f"Command result: {data}")

        @self._sio.on("error")
        def _on_error(data):
            log.warning(f"Tank error: {data}")

        self._sio.connect(self.base_url)
        log.info(f"WebSocket connected to {self.base_url}")

    def send_command(self, command: dict):
        """Send a command over WebSocket."""
        if self._sio and self._sio.connected:
            self._sio.emit("signalbot_command", {
                "token": self.token,
                "command": command
            })

    def disconnect(self):
        if self._sio:
            self._sio.disconnect()


# ---------------------------------------------------------------------------
# Example: SignalBot daemon integration sketch
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Quick test — run this to verify connectivity.
    Set TANK_URL environment variable or pass as argument.
    """
    import sys

    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5000"
    token = sys.argv[2] if len(sys.argv) > 2 else "signalbot_dev_key"

    print(f"Connecting to tank at {url}...")
    tank = TankClient(url, token=token)

    if not tank.is_connected():
        print("FAIL: Cannot reach tank server")
        sys.exit(1)

    print("OK: Tank server reachable")
    print(f"Status: {json.dumps(tank.status(), indent=2)}")

    # Quick movement test
    print("\nClaiming control...")
    print(tank.claim())

    print("Forward for 1s...")
    tank.set_speed(50, 50)
    tank.forward()
    time.sleep(1)

    print("Stop")
    tank.stop()

    print("Sensor read:")
    print(json.dumps(tank.sensors(), indent=2))

    print("\nReleasing control...")
    print(tank.release())
    print("Done.")
