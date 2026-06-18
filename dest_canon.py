#!/usr/bin/env python3
# Collapse TRUE-SYNONYM destinations to one canonical place (e.g. dock/outlet/socket -> charger).
# ONLY genuine same-place synonyms — lossless. NOT distinct things (a stool != an armchair),
# so those are deliberately left alone. Applies IN PLACE to OMNI_GEN_OUT. Run after dedup,
# before training. Same role as canon_verbs.py, but for the destination slot.
import json, os
from pathlib import Path

# canonical : [synonyms that are literally the same destination for the tank]
GROUPS = {
    "charger":     ["charger", "dock", "outlet", "socket", "station", "base", "plug"],
    "couch":       ["couch", "sofa"],
    "hallway":     ["hallway", "hall", "corridor", "passage"],
    "bathroom":    ["bathroom", "restroom", "washroom", "lavatory"],
    "fridge":      ["fridge", "refrigerator"],
    "lab":         ["lab", "laboratory"],
    "elevator":    ["elevator", "lift"],
    "diningroom":  ["diningroom", "dining"],
    "laundryroom": ["laundryroom", "laundry"],
    "driveway":    ["driveway", "drive"],
    "tv":          ["tv", "television"],
}
CANON = {syn: c for c, syns in GROUPS.items() for syn in syns}

P = Path(os.environ.get("OMNI_GEN_OUT", "omni_phase1_corpus.v3.jsonl"))
rows = [json.loads(l) for l in open(P) if l.strip()]
changed = 0
for r in rows:
    p = r["omni"].split(".")
    if len(p) == 6 and p[3] in CANON and p[3] != CANON[p[3]]:
        p[3] = CANON[p[3]]
        r["omni"] = ".".join(p)
        changed += 1
with open(P, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"[dest_canon] collapsed {changed} synonym destinations to canonical in {P.name}")
