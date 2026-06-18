"""
preference_engine.py — Signal's OWN tastes (the "wanting" leg of agency).

This is the TRAIT layer that sits under the curiosity engine. The distinction
that makes it work:

    curiosity = STATE  (fast, per-goal, decays toward CURIO_FLOOR every cycle —
                        "what am I drawn to right NOW")
    preference = TRAIT (the slow integral of curiosity, clustered by topic ×
                        action — "what I keep coming back to", survives restart)

It is EMERGENT + SEEDED: ships with a few starter leanings, then accretes real
tastes from engagement history (initiative_log.jsonl). A thread Signal pinged
that Adam *answered* reinforces that topic+action; one he *ignored* decays it.
Same "the misses are the gold" signal as the gate logs and the thought ladder.

Model-agnostic by construction: pure Python, no model/network/hardware. The
tastes live in the overlay, so they survive a model swap — same standard as the
model-agnostic identity milestone.

Offline-testable: test_preference_engine.py is the spec.
"""

import json
import time
from pathlib import Path

# ─── Tunables (TUNE on real engagement data) ──────────────────────────────
AFF_MIN, AFF_MAX = -1.0, 1.0      # affinity range
TOPIC_LR = 0.15                   # learning rate, topic affinity
ACTION_LR = 0.10                  # learning rate, action affinity (habit < taste)
TRAIT_DECAY = 0.01                # per-tick pull toward 0 — DELIBERATELY tiny.
                                  # This is what makes it a trait: curiosity
                                  # decays fast each cycle, a taste fades slowly.
BIAS_W_TOPIC = 0.20               # weight of topic affinity in the score bias
BIAS_W_ACTION = 0.10              # weight of action affinity in the score bias
BIAS_CLAMP = 0.30                 # hard cap on total bias — biases, never
                                  # overrides (good-sense gate stays in charge)

# Seed leanings — Signal isn't a blank slate, but these are a thumb on the
# scale, not law. Real engagement quickly outweighs them.
DEFAULT_SEED_TOPIC = {
    "memory-arch": 0.50,
    "smalltalk": -0.30,
}
DEFAULT_SEED_ACTION = {}

# Topic buckets keyed by the words that route a goal into them. Coarse on
# purpose — a real clusterer is overkill; these are SignalBot's actual domains.
TOPIC_KEYWORDS = {
    "memory-arch": {"memory", "twdc", "archive", "decay", "recall", "gradient",
                    "forget", "consolidat"},
    "omniaddress": {"address", "omni", "verb", "gate", "encode", "decode",
                    "verbalizer", "schema", "semantic"},
    "robot": {"tank", "robot", "motor", "navigate", "slam", "yolo", "vision",
              "camera", "occupancy", "explore behavior"},
    "identity": {"identity", "agency", "thesis", "persona", "self", "conscious",
                 "preference", "values", "personality", "agent"},
    "gamedev": {"game", "dungeon", "roblox", "builder", "crawler", "carl"},
    "evolve": {"evolve", "patch", "snapshot", "capability", "problem queue",
               "ladder", "daemon"},
    "smalltalk": {"hi", "hey", "hello", "thanks", "thank you", "lol", "haha",
                  "how are you", "good morning", "good night", "sup"},
}

KNOWN_ACTIONS = {"explore", "think", "ask_user", "revisit", "resolve"}


def classify_topic(text):
    """Route free text into a topic bucket by keyword hit, else 'other'.
    Pluggable: swap this for an embedding clusterer later without touching
    the rest of the engine."""
    if not text:
        return "other"
    low = text.lower()
    best, best_hits = "other", 0
    for topic, kws in TOPIC_KEYWORDS.items():
        hits = sum(1 for kw in kws if kw in low)
        if hits > best_hits:
            best, best_hits = topic, hits
    return best


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


class PreferenceLayer:
    """Signal's emergent tastes. One instance per user."""

    def __init__(self, path=None, seed=True):
        # path: where affinity.json lives (per-user). None => memory-only.
        self.path = Path(path) if path else None
        self.affinity_topic = {}
        self.affinity_action = {}
        self.last_update = time.time()
        # Incremental-learning cursor: how many log lines we've consumed, and
        # the fired-but-not-yet-settled pings waiting for their answered/ignored
        # receipt. Persisted so the daemon can re-read the growing
        # initiative_log every cycle WITHOUT double-counting old events.
        self.log_offset = 0
        self.pending = {}

        if self.path and self.path.exists():
            self._load()
        elif seed:
            # Fresh layer: plant the seed leanings.
            self.affinity_topic = dict(DEFAULT_SEED_TOPIC)
            self.affinity_action = dict(DEFAULT_SEED_ACTION)

    # ─── learning ─────────────────────────────────────────────────────────
    def reinforce(self, topic, action, engaged, strength=1.0):
        """One engagement event. engaged=True (answered) pulls affinity toward
        +1; engaged=False (ignored) decays it toward 0. Saturating, so a topic
        can't run away — repeated hits give diminishing returns."""
        s = _clamp(strength, 0.0, 1.0)

        if topic:
            a = self.affinity_topic.get(topic, 0.0)
            if engaged:
                a += TOPIC_LR * (AFF_MAX - a) * s      # saturate toward +1
            else:
                a -= TOPIC_LR * (a - 0.0) * s          # decay toward 0
            self.affinity_topic[topic] = _clamp(a, AFF_MIN, AFF_MAX)

        if action:
            a = self.affinity_action.get(action, 0.0)
            if engaged:
                a += ACTION_LR * (AFF_MAX - a) * s
            else:
                a -= ACTION_LR * (a - 0.0) * s
            self.affinity_action[action] = _clamp(a, AFF_MIN, AFF_MAX)

        self.last_update = time.time()

    def learn_event(self, description, action, engaged, strength=1.0):
        """Convenience: classify the goal's text to a topic, then reinforce."""
        self.reinforce(classify_topic(description), action, engaged, strength)

    def decay_tick(self, n=1):
        """Slow trait-decay toward 0. Call once per daemon cycle. Tiny by
        design — without this a one-time spike would be permanent; with a big
        value it'd be mood, not trait."""
        for _ in range(max(1, n)):
            for m in (self.affinity_topic, self.affinity_action):
                for k in list(m.keys()):
                    v = m[k]
                    m[k] = v - TRAIT_DECAY * v   # exponential pull to 0
        self.last_update = time.time()

    # ─── output ───────────────────────────────────────────────────────────
    def bias(self, topic, action=None):
        """The number the curiosity engine actually consumes: a small bounded
        additive term for a (topic, action) pair. Sits next to
        identity_relevance in the composite score."""
        b = BIAS_W_TOPIC * self.affinity_topic.get(topic, 0.0)
        if action:
            b += BIAS_W_ACTION * self.affinity_action.get(action, 0.0)
        return _clamp(b, -BIAS_CLAMP, BIAS_CLAMP)

    def bias_for_text(self, description, action=None):
        return self.bias(classify_topic(description), action)

    # ─── engagement backfill ──────────────────────────────────────────────
    def learn_from_initiative_log(self, log_path, incremental=True):
        """Replay an initiative_log.jsonl to grow tastes from engagement.
        Pairs each 'fired' (carries action_type + message) with the later
        'answered' (engaged) or 'ignored' (not) for the same goal_id.

        incremental=True (the daemon path): resume from the saved line cursor
        and the saved pending map, so calling this every cycle on the growing
        log learns each event exactly once — including a fired/answered pair
        that straddles two calls. Returns count of events learned this call."""
        log_path = Path(log_path)
        if not log_path.exists():
            return 0

        lines = log_path.read_text().splitlines()
        start = self.log_offset if incremental else 0
        # tuple/list both unpack fine — reload turns the saved tuples to lists
        pending = dict(self.pending) if incremental else {}
        learned = 0
        for line in lines[start:]:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue   # fail-safe: a junk line punishes nobody
            ev, gid = e.get("event"), e.get("goal_id")
            if ev == "fired":
                pending[gid] = (e.get("action_type", ""), e.get("message", ""))
            elif ev in ("answered", "ignored") and gid in pending:
                action, msg = pending.pop(gid)
                engaged = (ev == "answered")
                # Real conversation (long answer latency) is a stronger taste
                # signal than a reflexive reply. Cap so one chat doesn't swamp.
                strength = 1.0
                if engaged:
                    lat = e.get("latency", 0) or 0
                    strength = _clamp(0.5 + lat / 600.0, 0.5, 1.0)
                self.learn_event(msg, action, engaged, strength)
                learned += 1
        if incremental:
            self.log_offset = len(lines)
            self.pending = pending
        return learned

    # ─── persistence ──────────────────────────────────────────────────────
    def to_dict(self):
        return {
            "affinity_topic": self.affinity_topic,
            "affinity_action": self.affinity_action,
            "last_update": self.last_update,
            "log_offset": self.log_offset,
            "pending": self.pending,
        }

    def save(self):
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.to_dict(), indent=1))

    def _load(self):
        d = json.loads(self.path.read_text())
        self.affinity_topic = d.get("affinity_topic", {})
        self.affinity_action = d.get("affinity_action", {})
        self.last_update = d.get("last_update", time.time())
        self.log_offset = d.get("log_offset", 0)
        self.pending = d.get("pending", {})

    # ─── readout (for daemon diagnostic / debugging) ──────────────────────
    def top_tastes(self, n=5):
        """Human-readable 'what Signal is into right now', strongest first."""
        items = sorted(self.affinity_topic.items(),
                       key=lambda kv: kv[1], reverse=True)
        return items[:n]
