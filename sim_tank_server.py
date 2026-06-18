#!/usr/bin/env python3
"""
sim_tank_server.py — protocol-faithful stand-in for tank_server.py.

A tiny plain-Flask server that mirrors the EXACT wire contract of the real
`tank_server.py` SignalBot API (endpoint, auth header, payload shape, responses),
so the bridge loop can be tested offline WITHOUT the Pi, without flask_socketio,
and without the SLAM/explore stack. It logs "would drive <dir>" instead of
moving motors — same as the real server's simulation mode.

This is a TEST FIXTURE, not the robot server. The real tank_server.py drops in
unchanged once flask_socketio is installed and its map/explore modules are
consolidated (it speaks the identical contract).

    python3 sim_tank_server.py          # then: python3 test_tank_loop.py
"""

import os
from flask import Flask, request, jsonify

TOKEN = os.environ.get("SIGNALBOT_TOKEN", "signalbot_dev_key")  # matches client default
app = Flask(__name__)
state = {"mode": "manual", "direction": "stop"}


def _auth():
    return request.headers.get("X-SignalBot-Token", "") == TOKEN


@app.route("/api/v1/command", methods=["POST"])
def command():
    if not _auth():
        return jsonify({"ok": False, "error": "Unauthorized"}), 401
    cmd = request.get_json(silent=True) or {}
    action = cmd.get("action", "")

    if action == "move":
        direction = cmd.get("direction", "stop")
        state["direction"] = direction
        print(f"[SIM] would drive: {direction}", flush=True)   # the motor line
        return jsonify({"ok": True, "direction": direction})
    if action == "claim":
        state["mode"] = "signalbot"
        print("[SIM] control CLAIMED", flush=True)
        return jsonify({"ok": True, "mode": "signalbot"})
    if action == "release":
        state["mode"] = "manual"
        print("[SIM] control RELEASED", flush=True)
        return jsonify({"ok": True, "mode": "manual"})
    if action == "emergency_stop":
        state["direction"] = "stop"
        print("[SIM] EMERGENCY STOP", flush=True)
        return jsonify({"ok": True, "emergency_stop": True})
    if action == "status":
        return jsonify({"ok": True, "state": state})
    return jsonify({"ok": False, "error": f"Unknown action: {action}"})


@app.route("/api/v1/status", methods=["GET"])
def status():
    return jsonify({"ok": True, **state})


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    print(f"[SIM] protocol-faithful tank stub on {host}:{port} "
          f"(token={'set' if TOKEN else 'none'})", flush=True)
    app.run(host=host, port=port, threaded=True)
