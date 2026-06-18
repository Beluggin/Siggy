#!/usr/bin/env python3
"""
liveness.py — the negator-as-liveness register + the daemon-agnostic settle bus.

THE IDEA (Adam, 2026-06-03 — "the 7th field is the negator"):
The 6th address field, `negator`, was never really "did it happen." It's
"is this thread still alive." So it's a LIVENESS register, not a boolean:

    true   → ALIVE   succeeded, the thread continues   (the default)
    done   → DONE    completed, this CLOSES the goal    (terminus folded in here)
    false  → FAIL    failed / blocked, closes by failure

The address stays 6 fields. We did NOT add a 7th. negator is the one slot that
already scopes over the WHOLE proposition (you can't negate without reading every
other slot), so terminus — also a whole-address operator — reuses it instead of
bloating the schema. Constituents (subject/verb/object/destination) describe the
action; operators (tense/negator) comment on the whole. Terminus is an operator.

WHAT THIS BUYS — the closed loop (the actual goal: the safety gate):
A motor command goes out as INTENT (negator=true). The world executes it. Then
the WORLD writes back the settled liveness (done/fail) plus the measured outcome.
"go forward" → "I went forward an inch." The command and its receipt are the SAME
address at two liveness states:
    out:  robot.move.forward.none.now.true     (intent / [DREAM])
    back: robot.move.forward.none.past.done     (receipt / [GROUND])
The loop is closed exactly when the address comes back SETTLED.

THE OPEN BUS (so BOTH daemons can subscribe, plus any future one):
The settle is PUBLISHED, never written into a daemon's struct. Each daemon
registers a thin on_settle(goal_id, status, telemetry) adapter:
  - goal_engine_DAEMON.py: flips Goal.unresolved (the same liveness bit, one
    altitude up — `unresolved` is the goal-level negator).
  - temporal_daemon.py: reflects into its cognitive cycle (STUB for now — seated
    in the bus from day one so wiring its real reaction later is purely additive).
This module knows about NEITHER. It only knows the bus. That's what "open enough
for both daemons" means structurally: the import arrow never points from here to a
daemon.

SCOPE TODAY: telic + success-only. The single dispatch point on `status` in
close_loop() is the SEAM where the deferred work attaches with no retrofit —
chains as (done → next address), exceptions as (fail → recovery). They only READ
status; they don't restructure the loop.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable

import omni_gate as og   # cheap: og loads the flan-t5 model lazily, not on import
from verbalizer import gloss   # address→English (no model); intent + receipt-fallback

# ── liveness states = the three negator values ──
ALIVE = "true"    # thread continues (default; not a settle)
DONE  = "done"    # completed, closes the goal
FAIL  = "false"   # failed/blocked, closes by failure


# ═══════════════════════════════════════════════════════════════════
# THE SETTLE BUS  — daemon-agnostic pub/sub. Knows about no daemon.
# ═══════════════════════════════════════════════════════════════════

_settle_handlers: List[Callable] = []   # each: (goal_id, status, telemetry) -> None

def on_settle(handler: Callable) -> Callable:
    """Register a daemon adapter. Every settle fans out to every handler. Adding a
    third daemon someday = one more on_settle() call; the bus doesn't change."""
    _settle_handlers.append(handler)
    return handler

def clear_handlers() -> None:
    """Reset the bus (tests)."""
    _settle_handlers.clear()

def publish_settle(goal_id: Optional[str], status: str, telemetry: Dict) -> None:
    """Fan a settle out to every registered adapter. One adapter throwing must not
    starve the others (fail gracefully in autonomous mode — CLAUDE.md)."""
    for h in list(_settle_handlers):
        try:
            h(goal_id, status, telemetry)
        except Exception as e:
            print(f"[settle] handler {getattr(h, '__name__', h)} errored: {e}",
                  flush=True)


# ═══════════════════════════════════════════════════════════════════
# SETTLING AN ADDRESS  — the command address, returned settled.
# ═══════════════════════════════════════════════════════════════════

def settle_address(fields_: Dict[str, str], status: str) -> Dict[str, str]:
    """Same address, two liveness states. The receipt is the command with
    tense→past and negator→the world's verdict. Nothing else moves."""
    settled = dict(fields_)
    settled["tense"] = "past"
    settled["negator"] = status
    return settled


# ── faked outcome telemetry. On the Pi this reads tank_odometry.py instead; the
#    loop structure is identical, only the numbers stop being made up. ──
ASSUMED_SPEED_IN_PER_S = 4.0   # FAKED stand-in for real wheel odometry
DEFAULT_TAP_S = 0.25           # a bare "move forward" with no duration = one tap

def _measure(fields_: Dict[str, str], raw_text: str, resp: Dict) -> Dict:
    """What actually happened, as best the world can report it. Faked in sim."""
    if not resp.get("ok"):
        return {}
    verb, obj = fields_["verb"], fields_["object"]
    if verb == "move":
        dur = og.resolve_duration(raw_text) or DEFAULT_TAP_S
        return {"moved_in": round(ASSUMED_SPEED_IN_PER_S * dur),
                "direction": obj, "faked": True}
    if verb == "turn":
        return {"turned": obj, "faked": True}
    if verb in ("stop", "hold", "wait"):
        return {"stopped": True}
    return {"faked": True}


def receipt(fields_: Dict[str, str], status: str, tele: Dict) -> str:
    """The [GROUND] receipt spoken back — measured, real. (The command was [DREAM].)"""
    if status == FAIL:
        return f"[GROUND] I couldn't {fields_['verb']} — it didn't take."
    if "moved_in" in tele:
        n = tele["moved_in"]
        dist = "an inch" if n == 1 else f"{n} inches"
        return f"[GROUND] I moved {tele['direction']} {dist}."
    if "turned" in tele:
        return f"[GROUND] I turned {tele['turned']}."
    if tele.get("stopped"):
        return "[GROUND] I stopped."
    # No telemetry phrasing for this verb (yet) — fall back to the schema gloss of
    # the SETTLED address so the talk lane still speaks sensibly ("I observed …"),
    # not a bare "done." Motor verbs above keep their telemetry-rich receipts.
    return f"[GROUND] {gloss(settle_address(fields_, status))}."


# ═══════════════════════════════════════════════════════════════════
# THE LOOP CLOSER  — dispatch → settle → receipt → publish.
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SettleResult:
    status:   str                       # DONE | FAIL | "NO_FIRE"
    settled:  Optional[Dict[str, str]]  # the address, settled (past + liveness)
    receipt:  str                       # the [GROUND] line to speak (settled outcome)
    intent:   str = ""                   # the [DREAM] line — gloss of the command (pre-settle)
    telemetry: Dict = field(default_factory=dict)
    goal_id:  Optional[str] = None
    fired:    bool = False               # did a motor command actually execute?

    def __str__(self):
        a = ".".join(self.settled.values()) if self.settled else "—"
        g = f" goal={self.goal_id}" if self.goal_id else ""
        return f"[{self.status}] {a}{g}  «{self.receipt}»"


def close_loop(gate_result, tank, talk: Optional[Callable] = None,
               goal_id: Optional[str] = None, raw_text: str = "") -> SettleResult:
    """One closed loop = TWO lane firings on a single motor utterance:

        FIRING #1 (tank lane): execute the motor command — do the thing.
        FIRING #2 (talk lane): speak the [GROUND] receipt — say what happened.

    The two are a BOUND co-fire. Talk fires *because* tank settled, and reports
    *what* the settle measured ("an inch"). You cannot speak the outcome at command
    time — it doesn't exist until the action runs — so talk is strictly downstream
    of the tank settle. That binding (talk ← settle ← tank) is the smallest real
    goal: the co-fire 'goal' from the bridge, in its degenerate 1-step form.

    `tank` = motor sink (TankClient-like, forward/left/...). `talk` = response
    sink: any callable talk(line) that emits the receipt (response_engine / TTS /
    print). goal_id ties the action to a daemon goal (None = bare user command);
    carried HERE at the routing layer, never inside the address — same as the
    odometry magnitude.
    """
    # The [DREAM] intent: the command gloss, spoken/grounded BEFORE the world acts.
    # Same address the receipt settles — command and receipt are one address at two
    # liveness states, both rendered to English by the codec. ("" if no parsed addr.)
    intent_line = f"[DREAM] {gloss(gate_result.address)}." if gate_result.address else ""

    # ── FIRING #1 — TANK LANE: do the thing ──
    resp = og.dispatch(gate_result, tank)

    # The gate didn't route this to the tank (SAFE_STOP / CLARIFY / talk-verb).
    # Nothing executed, so there's nothing to settle, no receipt to co-fire, and
    # nothing to tell the daemons. A non-event is not a settle; the caller owns
    # non-tank routing.
    if resp.get("skipped"):
        return SettleResult(status="NO_FIRE", settled=gate_result.address,
                            receipt=resp.get("why", "no motor action"),
                            intent=intent_line, goal_id=goal_id, fired=False)

    fields_ = gate_result.address

    # ── the world's verdict becomes the liveness value ──
    status = DONE if resp.get("ok") else FAIL
    tele = _measure(fields_, raw_text, resp)
    settled = settle_address(fields_, status)
    line = receipt(fields_, status, tele)

    # ── THE SEAM ─────────────────────────────────────────────────────────────
    # One dispatch on the settled liveness value. TODAY both branches just speak
    # the receipt, so the happy path falls through. The deferred work attaches
    # HERE with no retrofit, because it only reads `status`:
    #   chains:     if status == DONE  → planner picks the next address for goal_id
    #   exceptions: if status == FAIL  → recovery loop for goal_id
    # ──────────────────────────────────────────────────────────────────────────

    # Tell every daemon, through the open bus. The bus, not us, owns who reacts.
    publish_settle(goal_id, status, tele)

    # ── FIRING #2 — TALK LANE: say what happened ──
    # The receipt is not a return value to be maybe-logged later; it's an actual
    # talk-lane emission that co-fires with the tank action. Speaking it IS the
    # second half of closing the loop. (Both DONE and FAIL speak — "an inch" or
    # "it didn't take".)
    if talk is not None:
        talk(line)

    return SettleResult(status=status, settled=settled, receipt=line,
                        intent=intent_line, telemetry=tele, goal_id=goal_id, fired=True)


# ═══════════════════════════════════════════════════════════════════
# DAEMON ADAPTERS  — thin translators from the shared settle to each
# daemon's private worldview. The ONLY place a daemon's internals appear.
# ═══════════════════════════════════════════════════════════════════

def make_goal_engine_adapter(engine) -> Callable:
    """goal_engine_DAEMON adapter: reflect a settle into Goal.unresolved.

    `unresolved` IS the negator one altitude up. done/fail both CLOSE the goal as
    far as 'is the world still working on it' — but we only clear unresolved on
    DONE. A FAIL leaves unresolved=True on purpose, so the daemon re-surfaces it;
    that re-surfacing is exactly the (future) exception loop's trigger.

    `engine` is anything with a `.goals` dict of {id: Goal}. We deliberately do
    NOT depend on GoalEngine's constructor — only on the field the map needs."""
    def goal_engine_adapter(goal_id, status, telemetry):
        if goal_id is None:
            return                      # bare user command, no goal behind it
        g = engine.goals.get(goal_id)
        if g is None:
            return                      # not our goal
        if status == DONE:
            g.unresolved = False        # world satisfied it → stop nagging
            g.last_active = __import__("time").time()
        # FAIL: leave unresolved=True (the daemon will revisit → exception loop)
    return goal_engine_adapter


def temporal_adapter(goal_id, status, telemetry):
    """temporal_daemon adapter — STUB. It's seated in the bus from day one so its
    real cycle-side reaction is additive later; for now it just observes."""
    print(f"[temporal] saw settle: goal={goal_id} status={status} tele={telemetry}",
          flush=True)
