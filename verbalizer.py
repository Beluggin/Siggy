"""
verbalizer.py — the address→English decoder (the cheap other half of OmniAddress).

OmniAddress is the ENCODER: English → 6-field address. This is the DECODER:
address → short English. Together they're a codec around the address bus.

WHY this exists — it's the sensory/narration membrane for a TEXT-ONLY brain:
  • perception-in : YOLO emits an address ("griffin.detect.none.left.now.true")
                    → gloss → "I see Griffin on the left" → a blind model READS it.
  • narration-out : the liveness settle writes a receipt address
                    ("robot.move.forward.none.past.done") → gloss → "I moved forward".
  • grounding     : feed the gloss alongside the raw address to the talk-lane LLM so
                    its prose is anchored to what actually happened, not hallucinated.

This is SCHEMA-LEVEL and deterministic (no model, no hardware). It is the superset
of liveness.receipt(), which adds measured magnitude ("...an inch") from telemetry.
gloss() never invents magnitude — that rides telemetry, not the schema.

PROVISIONAL — the perception grammar (for detect/observe/scan/track the SUBJECT is
the thing perceived, not the agent) is an assumption: the YOLO encoder isn't built
yet. When it is, fix the mapping HERE and update test_verbalizer.py — the tests are
the spec, so adjust the strings to taste.
"""

from typing import Dict, Optional, Union
from omni_gate import FIELDS  # ("subject","verb","object","destination","tense","negator")

# Subjects that ARE the agent (render as "I"). Everything else is a named thing.
SELF = {"robot", "system"}

# Perception verbs: the SUBJECT field holds the thing perceived, not the doer.
# (For all other verbs the subject is the agent doing the action.)
PERCEPTION = {"detect", "observe", "scan", "track"}

# Direction/relative-place words — rendered as a location phrase, not "the X".
DIRECTIONS = {"forward", "backward", "left", "right", "ahead", "front", "behind"}

# Hand-tuned English per verb for the closed set. Columns: (now_form, past_form, base).
# 'now_form' is whatever reads best live — progressive for actions ("moving"),
# simple present for perception ("see") — so we don't fight English morphology.
# 'base' is the UNINFLECTED form. It's not a "future tense" — English builds future
# and negation periphrastically ("will move", "don't move", "couldn't move"), where
# the main verb stays bare and the auxiliary carries tense. So past/present inflect
# the verb; future + every negation reuse the bare base. That asymmetry is the grammar.
VERB_RENDER = {
    # motor / action
    "move":     ("moving",     "moved",     "move"),
    "turn":     ("turning",    "turned",    "turn"),
    "stop":     ("stopping",   "stopped",   "stop"),
    "hold":     ("holding",    "held",      "hold"),
    "wait":     ("waiting",    "waited",    "wait"),
    "navigate": ("heading",    "went",      "head"),
    "return":   ("returning",  "returned",  "return"),
    "map":      ("mapping",    "mapped",    "map"),
    "avoid":    ("avoiding",   "avoided",   "avoid"),
    # perception (now-form is simple present so "I see X" reads right)
    "detect":   ("see",        "saw",       "see"),
    "observe":  ("watching",   "watched",   "watch"),
    "scan":     ("scanning",   "scanned",   "scan"),
    "track":    ("tracking",   "tracked",   "track"),
    # talk
    "ask":      ("asking",     "asked",     "ask"),
    "greet":    ("greeting",   "greeted",   "greet"),
    "report":   ("reporting",  "reported",  "report"),
    "respond":  ("responding", "responded", "respond"),
    "speak":    ("speaking",   "spoke",     "speak"),
    "command":  ("telling",    "told",      "tell"),
}

_NULL = ("none", "unknown", "")  # empty-slot fillers we drop from the gloss


def _is_set(slot: str) -> bool:
    return slot not in _NULL


# Proper-name subjects render bare ("Griffin"); common nouns get an article
# ("a person"). YOLO emits common COCO nouns; a future face-recognizer emits the
# kid names. Both flow through this same perception gloss.
PROPER = {"griffin", "mason", "sophie", "adam", "robot", "system"}


def _article(noun: str) -> str:
    return f"an {noun}" if noun[:1] in "aeiou" else f"a {noun}"


def _name(subject: str) -> str:
    """Subject → thing-phrase: 'user'→'someone'; proper names bare; common nouns get 'a/an'."""
    if subject == "user":
        return "someone"
    if subject in PROPER:
        return subject.capitalize()
    return _article(subject.replace("_", " "))  # COCO labels use '_' for spaces


def _aux_now(agent: str) -> str:
    return "I'm" if agent == "I" else f"{agent} is"


def _aux_fut(agent: str) -> str:
    return "I'll" if agent == "I" else f"{agent} will"


def _location(obj: str, dest: str) -> str:
    """A place phrase for perception, from whichever slot carries it (dest first)."""
    for slot in (dest, obj):
        if slot in ("left", "right"):
            return f"on the {slot}"
        if slot in DIRECTIONS:
            return "ahead" if slot in ("forward", "ahead", "front") else "behind"
        if _is_set(slot):
            return f"near the {slot}"
    return ""


def _complement(obj: str, dest: str) -> str:
    """The trailing 'forward' / 'the charger' / 'to the kitchen' for action verbs."""
    parts = []
    if _is_set(obj):
        parts.append(obj if obj in DIRECTIONS else f"the {obj}")
    if _is_set(dest):
        parts.append(f"to the {dest}")
    return " ".join(parts)


def _as_fields(address: Union[str, Dict[str, str]]) -> Optional[Dict[str, str]]:
    """Accept a dotted string OR an already-parsed fields dict. None if malformed."""
    if isinstance(address, dict):
        return address if all(k in address for k in FIELDS) else None
    parts = address.strip().split(".")
    if len(parts) != len(FIELDS):
        return None
    return dict(zip(FIELDS, parts))


def gloss(address: Union[str, Dict[str, str]]) -> str:
    """Render a 6-field address as a short English clause. Deterministic, schema-only."""
    f = _as_fields(address)
    if f is None:
        return "(malformed address)"

    verb = f["verb"]
    tense = f["tense"]
    neg = f["negator"]
    now_form, past_form, base = VERB_RENDER.get(verb, (verb, verb + "ed", verb))
    failed = neg == "false"  # liveness: false = negated (now/future) or FAIL (past)

    # ── PERCEPTION: subject is the thing perceived → "I see Griffin on the left" ──
    if verb in PERCEPTION:
        what = _name(f["subject"])
        loc = _location(f["object"], f["destination"])
        if failed:
            vp = {"now": f"don't {base}", "past": f"couldn't {base}",
                  "future": f"won't {base}"}[tense]
        else:
            vp = {"now": now_form, "past": past_form, "future": f"will {base}"}[tense]
        out = f"I {vp} {what}"
        return f"{out} {loc}".strip() if loc else out

    # ── ACTION / TALK: subject is the agent ──
    agent = "I" if f["subject"] in SELF else _name(f["subject"])
    comp = _complement(f["object"], f["destination"])

    if failed:
        if tense == "past":          # FAIL — it didn't take
            core = f"{agent} couldn't {base}"
        elif tense == "future":
            core = f"{_aux_fut(agent)} not {base}"
        else:                        # negated live command
            core = f"{_aux_now(agent)} not {now_form}"
    else:                            # affirmative (true / done)
        if tense == "past":
            core = f"{agent} {past_form}"
        elif tense == "future":
            core = f"{_aux_fut(agent)} {base}"
        else:
            core = f"{_aux_now(agent)} {now_form}"

    return f"{core} {comp}".strip() if comp else core


if __name__ == "__main__":
    # quick eyeball — `python3 verbalizer.py`
    for a in ("robot.move.forward.none.now.true",
              "robot.move.forward.none.past.done",
              "robot.move.forward.none.past.false",
              "griffin.detect.none.left.now.true",
              "sophie.detect.none.none.now.false",
              "robot.navigate.none.kitchen.now.true"):
        print(f"{a:42s} → {gloss(a)}")
