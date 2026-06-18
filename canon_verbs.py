#!/usr/bin/env python3
"""
canon_verbs.py — collapse the OmniAddress corpus's 100 loose verbs down to a
tight "caveman" canonical set so the model stops losing exact-match on synonyms.

Two lanes:
  TANK  (live, roam vocabulary)  — tight, primal action verbs
  TALK  (placeholder until speech API) — folded down to 4 comms verbs

Writes a NEW file (omni_phase1_corpus.canon.jsonl); never overwrites the original.
Any verb not in the map is printed as UNMAPPED and left unchanged so we catch gaps.
"""
import json
from collections import Counter
from pathlib import Path

SRC = Path("omni_phase1_corpus.jsonl")
DST = Path("omni_phase1_corpus.canon.jsonl")

# canonical -> list of synonyms that collapse into it
GROUPS = {
    # ── TANK: locomotion ──────────────────────────────────────────
    "move":    ["move", "drive", "go", "navigate", "steer", "enter", "exit",
                "pass", "continue", "advance", "proceed", "begin", "start",
                "guide", "manage", "follow", "return",   # follow-a-path = move
                # audit additions (2026-05-28): un-canonicalized locomotion verbs
                "traverse", "cross", "walk", "maneuver", "propel", "reverse",
                "ascend", "descend", "leave", "resume", "progress", "push", "pull"],
    "turn":    ["turn", "adjust", "change", "zoom",
                "pan", "tilt", "orient", "align", "unzoom", "spin"],
    "stop":    ["stop", "hold", "wait", "stay", "halt",
                "maintain", "keep", "occupy", "pause", "await", "sleep"],
    "avoid":   ["avoid", "block", "collide", "bump", "impede", "slip",
                "overcome", "deny", "interrupt", "contain", "engage",
                # audit additions: skip/blocked/stuck-state verbs
                "ignore", "bypass", "prevent", "reject", "hit", "stick", "jam",
                "bind", "immobilize", "impair", "dislodge", "disengage",
                "disable", "trip", "fall", "miss", "close", "compromise"],
    "reach":   ["reach", "approach", "arrive", "complete", "get", "regain",
                # audit additions: completion + object-handling (no manipulator lane yet)
                "achieve", "finish", "resolve", "free", "retrieve", "fetch",
                "pick", "pickup", "grip", "draw", "drop"],
    # ── TANK: perception ─────────────────────────────────────────
    "detect":  ["detect", "see", "acquire", "notice", "monitor", "expect",
                "predict",
                # audit additions
                "identify", "observe", "check", "hear", "sense", "inspect",
                "register", "capture", "anticipate", "listen", "feel",
                "experience", "appear"],
    "scan":    ["scan", "map", "survey", "measure"],
    "search":  ["search", "find", "seek", "explore", "look", "want", "lose",
                "locate", "need"],
    "track":   ["track", "focus", "fixate", "pursue", "chase", "target"],
    # ── TANK: power ──────────────────────────────────────────────
    "charge":  ["charge", "dock", "connect", "disconnect", "unplug",
                "activate", "lock", "empty", "decrease",
                # audit additions
                "recharge", "deplete", "drain", "consume", "power", "supply",
                "increase", "secure", "attach", "enable", "shut"],
    # ── TALK: thin placeholder lane (until speech API) ───────────
    "report":  ["report", "explain", "write", "log", "send", "provide",
                "describe", "conclude", "analyze", "read", "debug", "process",
                "update", "test", "commit", "merge", "fix", "create", "remove",
                "apply", "clear", "have", "be", "open", "access", "hide",
                "map_data",
                # audit additions: comms + dev/system actions fold to report
                "review", "discuss", "warn", "share", "show", "alert",
                "indicate", "suggest", "note", "speak", "talk", "signal",
                "clarify", "present", "flag", "record", "compile", "edit",
                "deploy", "generate", "modify", "save", "store", "delete",
                "execute", "run", "compare", "verify", "reset", "recalibrate",
                "initialize", "make", "perform", "operate", "give"],
    "command": ["command", "tell", "request", "require", "trigger", "initiate",
                "prioritize", "plan", "call", "establish", "attempt", "play",
                # audit additions: directives
                "ask", "order", "instruct", "direct", "query", "decide",
                "prepare", "allow", "obey", "help"],
    "confirm": ["confirm", "agree", "understand", "receive",
                "know", "accept", "approve", "acknowledge"],
    "respond": ["respond", "reply", "answer", "interact", "bark"],
}

# invert -> synonym:canonical
CANON = {}
for canon, syns in GROUPS.items():
    for s in syns:
        CANON[s] = canon

def main():
    out = []
    before = Counter()
    after = Counter()
    unmapped = Counter()
    changed = 0

    for line in open(SRC):
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        parts = o["omni"].split(".")
        if len(parts) != 5:
            out.append(o)
            continue
        verb = parts[1]
        before[verb] += 1
        canon = CANON.get(verb)
        if canon is None:
            unmapped[verb] += 1
            after[verb] += 1          # leave unchanged
        else:
            if canon != verb:
                changed += 1
            parts[1] = canon
            after[canon] += 1
            o["omni"] = ".".join(parts)
        out.append(o)

    with open(DST, "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")

    print(f"[canon] {len(out)} pairs  |  verbs rewritten: {changed}")
    print(f"[canon] unique verbs: {len(before)} -> {len(after)}")
    if unmapped:
        print(f"\n[!] UNMAPPED verbs (left unchanged — add to GROUPS):")
        for v, c in unmapped.most_common():
            print(f"      {c:3}  {v}")
    print(f"\n[canon] new verb distribution:")
    for v, c in after.most_common():
        print(f"      {c:4}  {v}")
    print(f"\n[canon] wrote {DST}")

if __name__ == "__main__":
    main()
