#!/usr/bin/env python3
"""
tank_omni_explorer.py — Autonomous explorer with OmniAddress telemetry
XiaoR GEEK GFS tank / Raspberry Pi 5

STANDALONE — copy just this file to the tank. No other SignalBot files needed.
    scp tank_omni_explorer.py pi@192.168.0.22:~/

Run on the tank:
    python3 tank_omni_explorer.py

What it does:
  - Drives the tank autonomously using ultrasonic obstacle avoidance
  - Every state change is parsed into OmniAddress format: subject.verb.object.tense.negator
  - All events logged to telemetry.jsonl (one JSON line per event)
  - Rule-based OmniParser: fast, no model, no internet needed

Optional T5 parser:
  If you copy the omniaddress_model/ folder alongside this file, it will use
  the fine-tuned T5 model instead of the rule-based parser (needs transformers + torch).

Hardware: XiaoR GEEK GFS, Pi 5, L298 motors, HC-SR04 ultrasonic
Pin map from xr_gpio.py (BCM numbering):
  ENA=13  ENB=20  IN1=19  IN2=16  IN3=21  IN4=26
  TRIG=17  ECHO=5
  LED0=10  LED1=9  LED2=25
"""

import json
import math
import time
import random
import signal
import sys
from datetime import datetime
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════════
# OMNIADDRESS  —  address format: subject.verb.object.tense.negator
# ═══════════════════════════════════════════════════════════════════════════════

def omni(subject: str, verb: str, obj: str,
         tense: str = "now", negator: str = "true") -> str:
    """Build an OmniAddress string."""
    return f"{subject}.{verb}.{obj}.{tense}.{negator}"


# ── Rule-based parser ─────────────────────────────────────────────────────────
# Maps robot event phrases → OmniAddress. Priority order: first match wins.
# Fast enough for real-time telemetry (no model required).

_RULES = [
    # First-match-wins — longer/more-specific phrases must come first.
    # Rule of thumb: if phrase A contains phrase B, A must appear before B.
    # phrase fragment          subject    verb         object         tense    neg
    ("loop trap",             ("robot",  "detected",  "loop_trap",  "now",  "true")),
    ("turn left",             ("robot",  "turned",    "left",       "now",  "true")),
    ("turn right",            ("robot",  "turned",    "right",      "now",  "true")),
    ("backed",                ("robot",  "backed",    "away",       "now",  "true")),
    ("back",                  ("robot",  "backed",    "away",       "now",  "true")),
    ("reverse",               ("robot",  "backed",    "away",       "now",  "true")),
    ("summary",               ("robot",  "reported",  "summary",    "now",  "true")),
    ("clear",                 ("robot",  "observed",  "clear_path", "now",  "true")),
    ("obstacle",              ("robot",  "detected",  "obstacle",   "now",  "true")),
    ("blocked",               ("robot",  "detected",  "obstacle",   "now",  "false")),
    ("forward",               ("robot",  "navigated", "forward",    "now",  "true")),
    ("left",                  ("robot",  "turned",    "left",       "now",  "true")),
    ("right",                 ("robot",  "turned",    "right",      "now",  "true")),
    ("shutdown",              ("robot",  "stopped",   "explore",    "past", "true")),
    ("exit",                  ("robot",  "stopped",   "explore",    "past", "true")),
    ("stopped",               ("robot",  "stopped",   "motors",     "now",  "true")),
    ("stop",                  ("robot",  "stopped",   "motors",     "now",  "true")),
    ("start",                 ("robot",  "started",   "explore",    "now",  "true")),
    ("battery",               ("robot",  "reported",  "battery",    "now",  "true")),
    ("charging",              ("robot",  "is",        "charging",   "now",  "true")),
    ("loop",                  ("robot",  "detected",  "loop_trap",  "now",  "true")),
    ("ir",                    ("robot",  "detected",  "ir_trigger", "now",  "true")),
    ("edge",                  ("robot",  "detected",  "edge",       "now",  "true")),
]

def parse_omni(text: str) -> str:
    """
    Rule-based OmniAddress parser.
    Tries each rule in priority order, returns first match.
    Falls back to a generic 'observed.event' address if nothing matches.
    """
    t = text.lower()
    for fragment, (subj, verb, obj, tense, neg) in _RULES:
        if fragment in t:
            return omni(subj, verb, obj, tense, neg)
    # Fallback — still a valid address, just generic
    return omni("robot", "observed", "event", "now", "true")


# ── Optional T5 parser ────────────────────────────────────────────────────────
# Only loads if omniaddress_model/ folder is present and transformers is installed.
# Falls back to rule-based if anything fails.

_t5_model     = None
_t5_tokenizer = None
_T5_MODEL_DIR = Path(__file__).parent / "omniaddress_model"

def _try_load_t5():
    global _t5_model, _t5_tokenizer
    if not _T5_MODEL_DIR.exists():
        return False
    try:
        from transformers import AutoTokenizer, T5ForConditionalGeneration
        import torch
        print("[parser] Loading fine-tuned T5 model...")
        _t5_tokenizer = AutoTokenizer.from_pretrained(str(_T5_MODEL_DIR))
        _t5_model     = T5ForConditionalGeneration.from_pretrained(str(_T5_MODEL_DIR))
        _t5_model.eval()
        print("[parser] T5 parser ready (fine-tuned OmniAddress model)")
        return True
    except Exception as e:
        print(f"[parser] T5 load failed ({e}) — falling back to rule-based parser")
        return False

def auto_parse(text: str) -> str:
    """
    Parse natural language → OmniAddress.
    Uses T5 if model folder is present, otherwise rule-based.
    """
    if _t5_model is not None:
        try:
            import torch
            inp = _t5_tokenizer(
                "omniaddress: " + text,
                return_tensors="pt", max_length=96, truncation=True,
            )
            with torch.no_grad():
                out = _t5_model.generate(**inp, max_new_tokens=24, num_beams=4)
            return _t5_tokenizer.decode(out[0], skip_special_tokens=True).strip()
        except Exception:
            pass
    return parse_omni(text)  # rule-based fallback


# ═══════════════════════════════════════════════════════════════════════════════
# TELEMETRY  —  JSONL log, one event per line
# ═══════════════════════════════════════════════════════════════════════════════

TELEM_PATH = Path("telemetry.jsonl")

def log_event(description: str, distance_cm: float = None, extra: dict = None):
    """Parse description → OmniAddress, write to log, print to terminal."""
    address = auto_parse(description)
    entry = {
        "ts":      datetime.now().isoformat(),
        "address": address,
        "desc":    description,
    }
    if distance_cm is not None:
        entry["dist_cm"] = round(distance_cm, 1)
    if extra:
        entry.update(extra)

    with open(TELEM_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")

    ts_short = entry["ts"][11:19]
    dist_str = f"  [{distance_cm:.0f}cm]" if distance_cm is not None else ""
    print(f"[{ts_short}] {address:<46} {description}{dist_str}")


# ═══════════════════════════════════════════════════════════════════════════════
# HARDWARE  —  gpiozero wrappers, simulates if not on Pi
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from gpiozero import LED as GPIOPin, PWMLED, DistanceSensor
    HW_AVAILABLE = True
except ImportError:
    HW_AVAILABLE = False
    print("[hw] gpiozero not found — SIMULATION MODE (no motors will move)")


# BCM pin map — from xr_gpio.py (XiaoR GEEK GFS tank)
class Pins:
    ENA  = 13    # Left  track PWM enable  (L298)
    ENB  = 20    # Right track PWM enable  (L298)
    IN1  = 19    # Left  forward
    IN2  = 16    # Left  reverse
    IN3  = 21    # Right forward
    IN4  = 26    # Right reverse
    TRIG = 17    # HC-SR04 trigger
    ECHO = 5     # HC-SR04 echo
    LED0 = 10
    LED1 = 9
    LED2 = 25

CRUISE_SPEED   = 0.70   # 0.0–1.0 maps to PWM duty cycle
TURN_SPEED     = 0.85
OBSTACLE_CM    = 30.0   # Stop and turn if closer than this
BACK_SECS      = 0.30   # Reverse this long before turning
TURN_SECS      = 0.50   # Turn this long (tuned for carpet; adjust for hard floor)


class TankHardware:
    """
    Motors + ultrasonic, with full simulation fallback.
    When HW_AVAILABLE=False, all pin calls are no-ops and distance returns
    a random stream that occasionally simulates an obstacle.
    """

    def __init__(self):
        self.sim = not HW_AVAILABLE
        if self.sim:
            print("[hw] Simulation mode — distance will occasionally return obstacles")
            self._sim_tick = 0
            return

        # Direction pins — start all off
        self._in1 = GPIOPin(Pins.IN1); self._in1.off()
        self._in2 = GPIOPin(Pins.IN2); self._in2.off()
        self._in3 = GPIOPin(Pins.IN3); self._in3.off()
        self._in4 = GPIOPin(Pins.IN4); self._in4.off()

        # PWM enable (speed control)
        # frequency=100 matches xr_gpio.py
        self._ena = PWMLED(Pins.ENA, active_high=True, initial_value=0, frequency=100)
        self._enb = PWMLED(Pins.ENB, active_high=True, initial_value=0, frequency=100)

        # LEDs — all on at start (high = off on this board)
        self._led0 = GPIOPin(Pins.LED0); self._led0.on()
        self._led1 = GPIOPin(Pins.LED1); self._led1.on()
        self._led2 = GPIOPin(Pins.LED2); self._led2.on()

        # HC-SR04 via gpiozero — handles the timing loop internally
        self._sonar = DistanceSensor(
            echo=Pins.ECHO, trigger=Pins.TRIG,
            max_distance=2.0,   # 2m max range
            threshold_distance=OBSTACLE_CM / 100,
        )

        print("[hw] Motors, LEDs, ultrasonic initialised (Pi 5 / gpiozero)")

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _speed(self, s: float):
        s = max(0.0, min(1.0, s))
        self._ena.value = s
        self._enb.value = s

    # ── Motion ──────────────────────────────────────────────────────────────

    def forward(self, speed: float = CRUISE_SPEED):
        if self.sim: return
        self._speed(speed)
        self._in1.on();  self._in2.off()   # Left  forward
        self._in3.on();  self._in4.off()   # Right forward
        self._led0.off(); self._led1.on(); self._led2.on()

    def back(self, speed: float = CRUISE_SPEED):
        if self.sim: return
        self._speed(speed)
        self._in1.off(); self._in2.on()    # Left  reverse
        self._in3.off(); self._in4.on()    # Right reverse

    def turn_left(self, speed: float = TURN_SPEED):
        """Tank turn: left track reverse, right track forward."""
        if self.sim: return
        self._speed(speed)
        self._in1.off(); self._in2.on()    # Left  reverse
        self._in3.on();  self._in4.off()   # Right forward
        self._led2.off()

    def turn_right(self, speed: float = TURN_SPEED):
        """Tank turn: left track forward, right track reverse."""
        if self.sim: return
        self._speed(speed)
        self._in1.on();  self._in2.off()   # Left  forward
        self._in3.off(); self._in4.on()    # Right reverse
        self._led2.on(); self._led1.off()

    def stop(self):
        if self.sim: return
        self._speed(0)
        self._in1.off(); self._in2.off()
        self._in3.off(); self._in4.off()
        self._led0.on(); self._led1.on(); self._led2.on()

    # ── Sensors ─────────────────────────────────────────────────────────────

    def distance_cm(self) -> float:
        """
        Read ultrasonic distance in cm.
        Returns 999.0 on read error (treat as clear path).
        """
        if self.sim:
            # Simulate mostly-clear environment with occasional obstacles
            self._sim_tick += 1
            if self._sim_tick % 40 in (0, 1, 2):
                return random.uniform(10.0, 25.0)   # Simulated obstacle
            return random.uniform(60.0, 150.0)      # Clear
        try:
            d = self._sonar.distance * 100.0
            return d if d > 2.0 else 999.0          # <2cm = bad read
        except Exception:
            return 999.0

    # ── Cleanup ──────────────────────────────────────────────────────────────

    def cleanup(self):
        """Stop all motors. gpiozero releases GPIO on garbage collection."""
        self.stop()
        print("[hw] Motors stopped")


# ═══════════════════════════════════════════════════════════════════════════════
# EXPLORER  —  autonomous navigation state machine
# ═══════════════════════════════════════════════════════════════════════════════
#
# States:  FORWARD → (obstacle) → STOP → BACK → TURN → FORWARD
#
# Loop trap detection: if we've hit N obstacles in a short window,
# we try a longer turn to break out of a corner.

LOOP_DETECT_WINDOW   = 8    # Look at last N obstacles
LOOP_DETECT_THRESH   = 6    # If >= this many in window, declare loop trap
LONG_TURN_SECS       = 1.4  # How long to turn when loop-trapped
LOG_CLEAR_EVERY      = 20   # Only log "path clear" every N forward-ticks (reduces noise)
SUMMARY_EVERY_SECS   = 60   # Print and log a summary every N seconds


class Explorer:

    def __init__(self):
        self.hw              = TankHardware()
        self.running         = False
        self.cycle           = 0
        self.obstacles_hit   = 0
        self.obstacle_recent = []   # Ring buffer for loop detection
        self.next_turn       = "right"  # Alternates each obstacle
        self.start_time      = None
        self.last_summary_ts = 0.0

    # ── Main loop ────────────────────────────────────────────────────────────

    def run(self):
        self.running   = True
        self.start_time = time.time()
        log_event("robot start autonomous explore",
                  extra={"mode": "obstacle_avoidance"})

        print("\n[explorer] Running. Ctrl+C to stop.\n")
        try:
            while self.running:
                self._tick()
                time.sleep(0.1)   # 10 Hz main loop
        except KeyboardInterrupt:
            pass
        finally:
            self._shutdown()

    def _tick(self):
        self.cycle += 1
        dist = self.hw.distance_cm()
        now  = time.time()

        # ── Periodic summary ─────────────────────────────────────────────────
        if now - self.last_summary_ts >= SUMMARY_EVERY_SECS:
            elapsed = int(now - self.start_time)
            log_event(
                f"robot summary: {self.obstacles_hit} obstacles in {elapsed}s",
                extra={"cycle": self.cycle, "elapsed_s": elapsed,
                       "obstacles": self.obstacles_hit},
            )
            self.last_summary_ts = now

        # ── Obstacle handling ─────────────────────────────────────────────────
        if dist < OBSTACLE_CM:
            self.hw.stop()
            self.obstacles_hit += 1
            self.obstacle_recent.append(now)
            # Trim ring buffer to window
            self.obstacle_recent = [t for t in self.obstacle_recent
                                    if now - t < LOOP_DETECT_WINDOW * 2]

            log_event(f"obstacle detected at {dist:.0f}cm",
                      distance_cm=dist, extra={"cycle": self.cycle,
                                               "obstacle_n": self.obstacles_hit})

            # Check for loop trap (stuck in a corner)
            loop_trapped = len(self.obstacle_recent) >= LOOP_DETECT_THRESH
            if loop_trapped:
                log_event("robot detected loop trap — using long turn",
                          extra={"cycle": self.cycle})
                self.obstacle_recent.clear()

            # Back up
            self.hw.back()
            time.sleep(BACK_SECS)
            self.hw.stop()
            log_event("robot backed away from obstacle",
                      extra={"cycle": self.cycle})

            # Turn — longer if loop-trapped
            turn_secs = LONG_TURN_SECS if loop_trapped else TURN_SECS
            if self.next_turn == "right":
                self.hw.turn_right()
                log_event("robot turn right to avoid obstacle",
                          extra={"cycle": self.cycle})
            else:
                self.hw.turn_left()
                log_event("robot turn left to avoid obstacle",
                          extra={"cycle": self.cycle})

            time.sleep(turn_secs)
            self.hw.stop()
            # Flip direction for next obstacle
            self.next_turn = "left" if self.next_turn == "right" else "right"

        else:
            # ── Clear path ────────────────────────────────────────────────────
            if self.cycle % LOG_CLEAR_EVERY == 1:
                log_event(f"path clear forward",
                          distance_cm=dist, extra={"cycle": self.cycle})
            self.hw.forward()

    # ── Shutdown ─────────────────────────────────────────────────────────────

    def _shutdown(self):
        self.running = False
        self.hw.cleanup()

        elapsed = int(time.time() - self.start_time) if self.start_time else 0
        log_event(
            "robot shutdown — explore complete",
            extra={
                "total_cycles":    self.cycle,
                "total_obstacles": self.obstacles_hit,
                "elapsed_s":       elapsed,
            },
        )
        print(f"\n[explorer] Done. {self.cycle} cycles, "
              f"{self.obstacles_hit} obstacles, {elapsed}s")
        print(f"[telem]    Log: {TELEM_PATH.resolve()}")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 62)
    print("  OmniAddress Explorer — XiaoR GEEK Tank")
    print(f"  Telemetry → {TELEM_PATH}")
    print(f"  Obstacle threshold: {OBSTACLE_CM}cm")
    print(f"  Hardware mode: {'REAL (gpiozero)' if HW_AVAILABLE else 'SIMULATION'}")
    print("  Ctrl+C to stop")
    print("=" * 62)

    # Try loading the fine-tuned T5 parser if the model folder is present
    t5_loaded = _try_load_t5()
    if not t5_loaded:
        print("[parser] Using rule-based OmniParser (fast, no model needed)")

    print()

    explorer = Explorer()

    # SIGTERM handler for clean shutdown (e.g. `kill` or systemd stop)
    def _sigterm(sig, frame):
        print("\n[signal] SIGTERM received — shutting down...")
        explorer.running = False

    signal.signal(signal.SIGTERM, _sigterm)

    explorer.run()


if __name__ == "__main__":
    main()
