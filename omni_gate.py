#!/usr/bin/env python3
"""
omni_gate.py — Safety gate + 2-lane router for the SignalBot↔OmniAddress bridge.

This is STAGE 0 of the router: nothing reaches the tank lane until an address
clears the gate. Tank-lane misfires move a physical robot, so the gate sits in
front of motion, not after it.

Flow:
    text  ──parse──▶  "subj.verb.obj.dest.tense.neg" + confidence
                          │
                          ▼
                   ┌──────────────┐
                   │  GATE (here) │  malformed? low-conf? unroutable?
                   └──────────────┘
                     │pass        │fail
                     ▼            ▼
              route to lane(s)   SAFE_STOP / CLARIFY
              talk / tank        (clarify IS the retry, cap 2)

The gate logic (validate + gate) is a PURE function on the address string — no
model needed, so it's testable on its own (`python3 omni_gate.py`). Model
loading for parse() is lazy and separate.

PoC scope (low vocab on purpose): the tank only does raw directions
(forward/back/left/right/stop), so the only DIRECTLY routable commands today are
turn-left / turn-right / stop. Named-place navigation ("go to the kitchen") is
well-formed but UNROUTABLE until a path-planner exists — the gate says so
instead of pretending. Widen CAPABILITIES as the tank gains skills.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "omniaddress_model_v3_2"   # current deployed 6-field model

# ═══════════════════════════════════════════════════════════════════
# CLOSED-SLOT VOCAB  (derived from omni_phase1_corpus.v3_2.jsonl, 16003 rows)
# These slots are genuinely closed — the model can only emit these. object and
# destination are OPEN (363 / 124 values), so we don't enforce membership there;
# we only check they're well-formed tokens. Refresh with derive_vocab() if the
# corpus grows.
# ═══════════════════════════════════════════════════════════════════

SUBJECTS = {"adam", "battery", "camera", "charger", "griffin", "mason",
            "robot", "sophie", "system", "user"}

VERBS = {"ask", "avoid", "command", "detect", "greet", "hold", "map", "move",
         "navigate", "observe", "report", "respond", "return", "scan", "speak",
         "stop", "track", "turn", "wait"}

TENSES   = {"future", "now", "past"}
NEGATORS = {"true", "false"}   # "true" = it holds / do it; "false" = negated

# ═══════════════════════════════════════════════════════════════════
# LANE CLASSIFICATION  (which verbs belong to which lane)
# talk = produce a response (text/voice).  tank = physical/sensor action.
# A single utterance is usually one verb → one lane. Both lanes firing on one
# bound utterance ("go to X and tell me") is a GOAL — that's the planner, parked.
# ═══════════════════════════════════════════════════════════════════

TALK_VERBS = {"ask", "greet", "report", "respond", "speak", "command"}
TANK_VERBS = {"avoid", "detect", "hold", "map", "move", "navigate", "observe",
              "return", "scan", "stop", "track", "turn", "wait"}

# ═══════════════════════════════════════════════════════════════════
# CAPABILITIES  (verb, key) → tank primitive method on TankClient.
# key = the value in the address that picks the motion. "*" = any/none.
# This is the HONEST routable set: if (verb,key) isn't here, the command is
# UNROUTABLE and the gate refuses to move. Grow this as the tank gains skills
# (a path-planner would add ("navigate", <place>) → goto(place)).
# ═══════════════════════════════════════════════════════════════════

CAPABILITIES = {
    ("turn", "left"):      "left",
    ("turn", "right"):     "right",
    ("move", "forward"):   "forward",
    ("move", "backward"):  "backward",
    ("stop", "*"):         "stop",
    ("hold", "*"):         "stop",
    ("wait", "*"):         "stop",
}

# The model collapses directional movement to `move.none.none` (trained on
# named places, not nudges — "move forward" and "back up" parse identically).
# So for raw direction we read the word straight from the user's text: the model
# decides it's a motion command, this picks which way. Closed unambiguous set.
DIRECTION_WORDS = {
    "forward": "forward", "ahead": "forward", "straight": "forward",
    "back": "backward", "backward": "backward", "backwards": "backward",
    "reverse": "backward",
    "left": "left", "right": "right",
}

def resolve_direction(raw_text: str) -> Optional[str]:
    """First explicit direction word in the raw input, normalized. None if absent."""
    for tok in re.findall(r"[a-z]+", raw_text.lower()):
        if tok in DIRECTION_WORDS:
            return DIRECTION_WORDS[tok]
    return None

def resolve_duration(raw_text: str) -> Optional[float]:
    """Seconds from 'for N seconds' / 'N sec' / 'Ns' in raw text. None if absent.
    The model doesn't carry duration — like direction, we read it from the words.
    Used to drive-then-auto-stop ('drive forward for 3 seconds')."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:s\b|sec|secs|second|seconds)", raw_text.lower())
    return float(m.group(1)) if m else None

def enrich(fields_: Dict[str, str], raw_text: str) -> Dict[str, str]:
    """Fill a motor verb's blank direction from raw text. No word → stays 'none'
    → gate SAFE_STOPs (we never guess a direction)."""
    if fields_["verb"] in ("move", "turn") and fields_["object"] == "none":
        d = resolve_direction(raw_text)
        if d:
            fields_["object"] = d
    return fields_

# Confidence floor for the parse. flan-t5 sequences_scores → exp() pseudo-prob.
# TUNE on Adam's real dry-test input — synthetic held-out runs hot.
CONF_THRESHOLD = 0.45

FIELDS = ("subject", "verb", "object", "destination", "tense", "negator")


# ═══════════════════════════════════════════════════════════════════
# GATE RESULT
# ═══════════════════════════════════════════════════════════════════

@dataclass
class GateResult:
    status:  str                      # "ROUTE" | "CLARIFY" | "SAFE_STOP"
    reason:  str                      # human-readable why
    address: Optional[Dict[str, str]] = None   # parsed fields, if well-formed
    lanes:   List[str]   = field(default_factory=list)   # subset of talk/tank
    tank_cmd: Optional[str] = None    # TankClient method name, if routable
    confidence: Optional[float] = None

    def __str__(self):
        a = ".".join(self.address.values()) if self.address else "—"
        extra = f" tank={self.tank_cmd}" if self.tank_cmd else ""
        lanes = "+".join(self.lanes) if self.lanes else "—"
        return f"[{self.status}] {a}  lanes={lanes}{extra}  ({self.reason})"


# ═══════════════════════════════════════════════════════════════════
# STAGE 0a — STRUCTURAL VALIDATION (malformed check). Pure, no model.
# ═══════════════════════════════════════════════════════════════════

def validate(address_str: str) -> Tuple[bool, str, Optional[Dict[str, str]]]:
    """Check the address is structurally sound and closed slots are in-vocab.
    Returns (ok, reason, fields_dict_or_None)."""
    parts = address_str.strip().split(".")
    if len(parts) != 6:
        return False, f"expected 6 fields, got {len(parts)}", None

    fields_ = dict(zip(FIELDS, parts))

    # closed slots must be in-vocab; out-of-vocab here means the model garbled it
    if fields_["verb"] not in VERBS:
        return False, f"unknown verb '{fields_['verb']}'", fields_
    if fields_["tense"] not in TENSES:
        return False, f"unknown tense '{fields_['tense']}'", fields_
    if fields_["negator"] not in NEGATORS:
        return False, f"bad negator '{fields_['negator']}'", fields_
    if fields_["subject"] not in SUBJECTS:
        return False, f"unknown subject '{fields_['subject']}'", fields_

    # open slots (object/destination): just demand a non-empty clean token
    for slot in ("object", "destination"):
        v = fields_[slot]
        if not v or " " in v:
            return False, f"malformed {slot} '{v}'", fields_

    return True, "well-formed", fields_


# ═══════════════════════════════════════════════════════════════════
# STAGE 0 — THE GATE.  malformed / low-conf / unroutable → don't move.
# ═══════════════════════════════════════════════════════════════════

def gate(address_str: str, confidence: Optional[float] = None) -> GateResult:
    """Decide whether an address may be routed, and to which lane(s)."""

    # 1. MALFORMED → can't trust it, ask again
    ok, reason, fields_ = validate(address_str)
    if not ok:
        return GateResult("CLARIFY", f"malformed: {reason}",
                          address=fields_, confidence=confidence)

    # 2. LOW CONFIDENCE → clarify (the clarify IS the retry)
    if confidence is not None and confidence < CONF_THRESHOLD:
        return GateResult("CLARIFY",
                          f"low confidence {confidence:.2f} < {CONF_THRESHOLD}",
                          address=fields_, confidence=confidence)

    verb = fields_["verb"]

    # talk lane: communicative verbs route straight to the response engine.
    if verb in TALK_VERBS:
        return GateResult("ROUTE", "talk-lane verb", address=fields_,
                          lanes=["talk"], confidence=confidence)

    # ── from here, it's a tank-lane verb ──

    # 3a. NEGATED command ("don't go to the kitchen") → never auto-execute motion
    if fields_["negator"] == "false":
        return GateResult("SAFE_STOP", "negated command — not auto-executing",
                          address=fields_, confidence=confidence)

    # 3b. PAST TENSE → this is a memory/log, not a live command. Don't move;
    #     it's really a statement, so hand it to the talk lane.
    if fields_["tense"] == "past":
        return GateResult("ROUTE", "past-tense — statement, not a command",
                          address=fields_, lanes=["talk"], confidence=confidence)

    # 3c. UNROUTABLE → well-formed + confident but no capability binds. The
    #     motion key comes from the object (e.g. left/right/forward) for turn/
    #     move; for stop/hold/wait it's "*".
    key = fields_["object"]
    tank_cmd = CAPABILITIES.get((verb, key)) or CAPABILITIES.get((verb, "*"))
    if tank_cmd is None:
        dest = fields_["destination"]
        hint = (f" — '{verb} to {dest}' needs a path-planner (not built)"
                if dest not in ("none", "unknown") else "")
        return GateResult("SAFE_STOP",
                          f"unroutable: no capability for ({verb},{key}){hint}",
                          address=fields_, confidence=confidence)

    # PASS — tank may move
    return GateResult("ROUTE", "routable motor command", address=fields_,
                      lanes=["tank"], tank_cmd=tank_cmd, confidence=confidence)


# ═══════════════════════════════════════════════════════════════════
# PARSE  (lazy-loaded v3.2 model → address + confidence)
# ═══════════════════════════════════════════════════════════════════

_model = None
_tok = None

def _load():
    """Load the deployed 6-field flan-t5. Returns True on success."""
    global _model, _tok
    if _model is not None:
        return True
    if not MODEL_DIR.exists():
        return False
    from transformers import AutoTokenizer, T5ForConditionalGeneration
    _tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    _model = T5ForConditionalGeneration.from_pretrained(str(MODEL_DIR))
    _model.eval()
    return True


def parse(text: str) -> Optional[Tuple[str, float]]:
    """text → ('subj.verb.obj.dest.tense.neg', confidence in 0..1).
    confidence = exp(length-normalized beam log-prob). None if model missing."""
    if not _load():
        return None
    import torch, math
    inp = _tok("omniaddress: " + text, return_tensors="pt",
               max_length=96, truncation=True)
    with torch.no_grad():
        out = _model.generate(**inp, max_new_tokens=24, num_beams=4,
                              output_scores=True, return_dict_in_generate=True)
    addr = _tok.decode(out.sequences[0], skip_special_tokens=True).strip()
    # sequences_scores is the length-normalized log prob of the chosen beam
    conf = math.exp(float(out.sequences_scores[0])) if out.sequences_scores is not None else 1.0
    return addr, conf


def route(text: str) -> GateResult:
    """Full path: parse text, then gate it. The one call app.py will use."""
    parsed = parse(text)
    if parsed is None:
        return GateResult("SAFE_STOP", "parser model not loaded")
    addr, conf = parsed
    # enrich blank motor direction from raw text before gating (model drops it)
    ok, _, fields_ = validate(addr)
    if ok:
        addr = ".".join(enrich(fields_, text)[f] for f in FIELDS)
    return gate(addr, conf)


def dispatch(result: GateResult, tank) -> Dict:
    """Call the tank for a routed motor result and return the server's RAW
    response dict (TankClient methods return the server JSON). Non-routes/no-ops
    come back as {"ok": False, "skipped": True, "why": ...}. Kept separate from
    execute() so the liveness layer (liveness.py) can read the real response to
    settle the loop — the gate already decided it's safe; this just calls."""
    if result.status != "ROUTE" or "tank" not in result.lanes or not result.tank_cmd:
        return {"ok": False, "skipped": True, "why": f"no tank action ({result.status})"}
    method = getattr(tank, result.tank_cmd, None)
    if method is None:
        return {"ok": False, "why": f"tank has no '{result.tank_cmd}' method"}
    resp = method()
    return resp if isinstance(resp, dict) else {"ok": True, "result": resp}


def execute(result: GateResult, tank) -> str:
    """Human-string wrapper over dispatch() — used by the simple sim test."""
    resp = dispatch(result, tank)
    if "why" in resp:
        return resp["why"]
    return f"{result.tank_cmd} → {resp}"


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

def derive_vocab(corpus="omni_phase1_corpus.v3_2.jsonl"):
    """Reprint the closed-slot vocab from the corpus — run if the data grows."""
    import json
    from collections import Counter
    cnt = {f: Counter() for f in FIELDS}
    for line in open(BASE_DIR / corpus):
        if line.strip():
            for f, v in zip(FIELDS, json.loads(line)["omni"].split(".")):
                cnt[f][v] += 1
    for f in ("subject", "verb", "tense", "negator"):
        print(f"{f} ({len(cnt[f])}): {sorted(cnt[f])}")


# ═══════════════════════════════════════════════════════════════════
# SMOKE TEST — gate logic only, no model needed.
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # (address, confidence) → expected behaviour
    cases = [
        ("robot.turn.left.none.now.true",        0.91),  # ROUTE tank → left
        ("robot.turn.right.none.now.true",       0.88),  # ROUTE tank → right
        ("robot.stop.none.none.now.true",        0.95),  # ROUTE tank → stop
        ("robot.navigate.none.kitchen.now.true", 0.90),  # SAFE_STOP (no planner)
        ("robot.move.none.kitchen.future.true",  0.85),  # SAFE_STOP (no planner)
        ("robot.move.none.kitchen.now.false",    0.85),  # SAFE_STOP (negated)
        ("robot.navigate.none.kitchen.past.true",0.85),  # ROUTE talk (statement)
        ("robot.report.status.none.now.true",    0.80),  # ROUTE talk
        ("adam.ask.robot.none.now.true",         0.80),  # ROUTE talk
        ("robot.turn.left.none.now.true",        0.30),  # CLARIFY (low conf)
        ("robot.frobnicate.left.none.now.true",  0.90),  # CLARIFY (bad verb)
        ("robot.turn.left.now.true",             0.90),  # CLARIFY (5 fields)
    ]
    print("── GATE SMOKE TEST ──")
    for addr, conf in cases:
        print(f"  {gate(addr, conf)}")

    # Direction enrichment: model gives bare move.none.none; raw text fills it.
    # (raw_text, conf) — the address is the identical blank-move every time.
    print("\n── DIRECTION ENRICHMENT (no model) ──")
    blank_move = "robot.move.none.none.now.true"
    for raw, conf in [("move forward", 0.99), ("back up", 0.99),
                      ("go straight ahead", 0.99), ("just move", 0.99)]:
        ok, _, f = validate(blank_move)
        addr2 = ".".join(enrich(f, raw)[x] for x in FIELDS)
        print(f"  {raw!r:22} → {addr2}\n      {gate(addr2, conf)}")
