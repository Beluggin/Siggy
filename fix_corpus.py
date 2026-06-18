#!/usr/bin/env python3
"""
fix_corpus.py — Clean omni_phase1_corpus.jsonl before retraining.

Fixes applied:
  1. tense slot: "present" → "now"  (24 cases — removes split vocabulary)
  2. verb slot: gerunds (-ing) → base form  (10 cases)
  3. verb slot: 3rd-person singular (-s/-es) → base form  (9 cases)

Run: python3 fix_corpus.py
Overwrites omni_phase1_corpus.jsonl in place.
"""

import json
from pathlib import Path
from collections import Counter

CORPUS = Path("omni_phase1_corpus.jsonl")

# ── Gerund → base form lookup ─────────────────────────────────────────────────
# Strip -ing is not always right (running→run, hiding→hide, calling→call)
# Handle the specific ones that appear in the corpus explicitly.
GERUND_MAP = {
    "searching":   "search",
    "calling":     "call",
    "scanning":    "scan",
    "hiding":      "hide",
    "playing":     "play",
    "performing":  "perform",
    "checking":    "check",
    "responding":  "respond",
    "locating":    "locate",
}

# ── 3rd-person singular → base form ──────────────────────────────────────────
# Covers the specific ones that appear in the corpus.
THIRD_PERSON_MAP = {
    "approaches":  "approach",
    "navigates":   "navigate",
    "finds":       "find",
    "scans":       "scan",
    "avoids":      "avoid",
    "searches":    "search",
    "calibrates":  "calibrate",
    "continues":   "continue",
    "applies":     "apply",
    "sees":        "see",
    "contains":    "contain",
}

# ── Past tense → base form ───────────────────────────────────────────────────
# Schema requires base verb form. Past tense in the verb slot caused ~30% of
# the corpus to teach the model contradictory verb/tense pairings.
# Includes irregular forms (drove→drive, found→find, told→tell).
PAST_TENSE_MAP = {
    # Regular -ed (just strip -ed or -d)
    "moved":         "move",
    "detected":      "detect",
    "blocked":       "block",
    "commanded":     "command",
    "visualized":    "visualize",
    "advanced":      "advance",
    "asked":         "ask",
    "confirmed":     "confirm",
    "discussed":     "discuss",
    "searched":      "search",
    "scanned":       "scan",
    "identified":    "identify",
    "responded":     "respond",
    "explained":     "explain",
    "provided":      "provide",
    "received":      "receive",
    "reached":       "reach",
    "reported":      "report",
    "planned":       "plan",
    "collided":      "collide",
    "steered":       "steer",
    "progressed":    "progress",
    "proceeded":     "proceed",
    "pushed":        "push",
    "requested":     "request",
    "opened":        "open",
    "reviewed":      "review",
    "clarified":     "clarify",
    "accessed":      "access",
    "located":       "locate",
    "initiated":     "initiate",
    "disconnected":  "disconnect",
    "displayed":     "display",
    "avoided":       "avoid",
    "recognized":    "recognize",
    "propelled":     "propel",
    "cleared":       "clear",
    "achieved":      "achieve",
    "traversed":     "traverse",
    "impeded":       "impede",
    "ascended":      "ascend",
    "gained":        "gain",
    "engaged":       "engage",
    "denied":        "deny",
    "edited":        "edit",
    "shared":        "share",
    "interrupted":   "interrupt",
    "suggested":     "suggest",
    "informed":      "inform",
    "talked":        "talk",
    "agreed":        "agree",
    "communicated":  "communicate",
    "updated":       "update",
    "generated":     "generate",
    "mentioned":     "mention",
    "checked":       "check",
    "observed":      "observe",
    "exited":        "exit",
    "entered":       "enter",
    "navigated":     "navigate",
    "connected":     "connect",
    "predicted":     "predict",
    "notified":      "notify",
    "approached":    "approach",
    "required":      "require",
    "indicated":     "indicate",
    "established":   "establish",
    "continued":     "continue",
    "triggered":     "trigger",
    "attempted":     "attempt",
    "docked":        "dock",
    "managed":       "manage",
    "secured":       "secure",
    "waited":        "wait",
    "prioritized":   "prioritize",
    "registered":    "register",
    "rerouted":      "reroute",
    "expected":      "expect",
    # Irregular past forms
    "drove":         "drive",
    "found":         "find",
    "wrote":         "write",
    "told":          "tell",
    "got":           "get",
    "held":          "hold",
    "had":           "have",
    "heard":         "hear",
    "understood":    "understand",
    "seen":          "see",
    "overcame":      "overcome",
    # Compound verbs — flatten to a base form (schema disallows underscores)
    "moved_to":      "move",
    "looking_for":   "search",
    "re_plan":       "plan",
    "is_in":         "be",
    "is":            "be",
}

# ── Verb synonym canonicalisation ─────────────────────────────────────────────
# After the past-tense fix, eval showed the model picking semantically valid
# synonyms that disagreed with the gold label (evade vs avoid, find vs detect,
# show vs report, turn vs steer …). These are corpus-side noise — Gemini wasn't
# constrained to a fixed verb vocabulary, so it invented synonyms across runs.
# Map rare synonyms into the most-common canonical verb in each cluster.
VERB_CANONICAL = {
    # Perception cluster → detect (37)
    "visualize":  "detect",
    "see":        "detect",
    "observe":    "detect",
    "perceive":   "detect",
    "sense":      "detect",
    "hear":       "detect",
    "find":       "detect",
    "locate":     "detect",
    "identify":   "detect",
    "recognize":  "detect",
    # Movement cluster → move (56)
    "gain":       "move",
    "advance":    "move",
    "progress":   "move",
    "propel":     "move",
    "push":       "move",
    "proceed":    "move",
    "traverse":   "move",
    "ascend":     "move",
    # Reporting cluster → report (34)
    "show":       "report",
    "display":    "report",
    "mention":    "report",
    "inform":     "report",
    "notify":     "report",
    "suggest":    "report",
    "present":    "report",
    "indicate":   "report",
    "share":      "report",
    "communicate":"report",
    # Verification cluster → confirm (7)
    "verify":     "confirm",
    "check":      "confirm",
    "review":     "confirm",
    # Command cluster → command (8)
    "ask":        "command",
    "request":    "command",
    "instruct":   "command",
    # Steering cluster → steer (2)
    "turn":       "steer",
    "pan":        "steer",
    "tilt":       "steer",
    # Avoid cluster → avoid (6)
    "evade":      "avoid",
    "dodge":      "avoid",
    # Hold cluster → hold (1)
    "maintain":   "hold",
    "preserve":   "hold",
    # Completion cluster → complete (2)
    "accomplish": "complete",
    "achieve":    "complete",
    "secure":     "complete",
    # Writing cluster → write (3)
    "edit":       "write",
    "generate":   "write",
    "modify":     "write",
    # Explanation cluster → explain (3)
    "clarify":    "explain",
    "discuss":    "explain",
    "talk":       "explain",
    "speak":      "explain",
    # Search cluster → search (23)
    "investigate":"search",
    # Adjust cluster → adjust (6)
    "calibrate":  "adjust",
    # Misc one-off mappings
    "reroute":    "navigate",
    "register":   "log",
    "delete":     "remove",
    "rescan":     "scan",
    "warn":       "report",
    "perform":    "complete",
}

# ── Object normalisation ──────────────────────────────────────────────────────
# Plurals, gerund-forms, and underscored compounds in the object slot caused
# eval mismatches (obstacles vs obstacle, tracking vs track) and outright
# violated the schema's "single simple word" rule.
OBJECT_NORMALIZE = {
    # Plural → singular
    "obstacles":     "obstacle",
    "branches":      "branch",
    "tracks":        "track",
    "wires":         "wire",
    "keys":          "key",
    "calls":         "call",
    "changes":       "change",
    "concerns":      "concern",
    "coordinates":   "coordinate",
    "dependencies":  "dependency",
    "instructions":  "instruction",
    "names":         "name",
    "parameters":    "parameter",
    "sensors":       "sensor",
    "tasks":         "task",
    "threats":       "threat",
    # Gerund → base
    "tracking":      "track",
    "docking":       "dock",
    "charging":      "charge",
    "saving":        "save",
    "responding":    "respond",
    "warning":       "warn",
    # Compound (underscored) → most meaningful single word
    "basement_door":          "door",
    "bathrooms_first_floor":  "bathroom",
    "childrens_rooms":        "room",
    "dining_room":            "room",
    "drop_off":               "dropoff",
    "entry_exit":             "entry",
    "faint_sound_basement":   "sound",
    "front_door_ajar":        "door",
    "griffin_in_house":       "griffin",
    "house_areas":            "area",
    "human_presence_upstairs":"human",
    "living_room":            "room",
    "livingroom":             "room",
    "mason_inside":           "mason",
    "mason_neighbors_yard":   "yard",
    "mason_toy_family_room":  "toy",
    "master_bedroom":         "bedroom",
    "multiple_targets":       "target",
    "object_alpha":           "object",
    "play_areas":             "area",
    "recent_movement":        "movement",
    "secondary_target":       "target",
    "sophie_signs":           "sophie",
    "sophie_voice":           "sophie",
    "sweep_ground_floor":     "floor",
    "target_b":               "target",
    "target_gamma":           "target",
    "tracking_data":          "data",
    "visual_pantry":          "pantry",
    # Fix the French token leakage we saw in eval predictions —
    # these never appeared as gold but the model produced them, so making sure
    # gold uses unambiguous English helps the model lock onto the right token.
}


def fix_address(addr: str) -> tuple[str, list[str]]:
    """
    Apply all fixes to an omni address string.
    Returns (fixed_addr, list_of_changes).
    """
    parts = addr.split(".")
    if len(parts) != 5:
        return addr, []

    subj, verb, obj, tense, neg = parts
    changes = []

    # Fix 1: present → now
    if tense == "present":
        tense = "now"
        changes.append("tense:present→now")

    # Fix 2: gerund verb → base form
    if verb in GERUND_MAP:
        new_verb = GERUND_MAP[verb]
        changes.append(f"verb:{verb}→{new_verb}")
        verb = new_verb

    # Fix 3: 3rd-person singular → base form
    if verb in THIRD_PERSON_MAP:
        new_verb = THIRD_PERSON_MAP[verb]
        changes.append(f"verb:{verb}→{new_verb}")
        verb = new_verb

    # Fix 4: past tense → base form
    # This is the big one — ~30% of corpus had past-tense verbs in the slot,
    # often paired with a non-past tense, training the model on contradictions.
    if verb in PAST_TENSE_MAP:
        new_verb = PAST_TENSE_MAP[verb]
        changes.append(f"verb:{verb}→{new_verb}")
        verb = new_verb

    # Fix 5: verb synonym → canonical form
    # Collapses corpus-side synonym noise (evade↔avoid, find↔detect, …) that
    # was the dominant cause of "model is right, gold disagrees" eval misses.
    if verb in VERB_CANONICAL:
        new_verb = VERB_CANONICAL[verb]
        changes.append(f"verb:{verb}→{new_verb}")
        verb = new_verb

    # Fix 6: object normalisation
    # Singularises plurals, strips gerund forms, and flattens underscored
    # compound objects (which violated the schema's "single simple word" rule).
    if obj in OBJECT_NORMALIZE:
        new_obj = OBJECT_NORMALIZE[obj]
        changes.append(f"object:{obj}→{new_obj}")
        obj = new_obj

    fixed = f"{subj}.{verb}.{obj}.{tense}.{neg}"
    return fixed, changes


def main():
    pairs = [json.loads(l) for l in CORPUS.read_text().splitlines() if l.strip()]
    print(f"Loaded {len(pairs)} pairs")

    total_changes = Counter()
    changed_count = 0
    fixed_pairs = []

    for p in pairs:
        orig = p["omni"]
        fixed, changes = fix_address(orig)
        for c in changes:
            total_changes[c.split(":")[0]] += 1  # count by category
        if orig != fixed:
            changed_count += 1
            print(f"  {orig:<55} → {fixed}")
        fixed_pairs.append({"english": p["english"], "omni": fixed})

    print(f"\n── Summary ───────────────────────────────────────────")
    for cat, n in total_changes.most_common():
        print(f"  {cat}: {n} fixed")
    print(f"  Total entries changed: {changed_count}/{len(pairs)}")

    # Write back
    out = "\n".join(json.dumps(p) for p in fixed_pairs) + "\n"
    CORPUS.write_text(out)
    print(f"\n[done] {CORPUS} updated in place")


if __name__ == "__main__":
    main()
