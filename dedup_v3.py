#!/usr/bin/env python3
# Dedup the v3 corpus in place: keep first row per unique english, drop malformed.
# v3 requires a clean 6-field address (subject.verb.object.destination.tense.negator).
import json
import os
from pathlib import Path

P = Path(os.environ.get("OMNI_GEN_OUT", "omni_phase1_corpus.v3.jsonl"))  # match corpus1.py's OUT
if not P.exists():
    print("[dedup] no v3 file yet"); raise SystemExit

seen, out = set(), []
for line in open(P):
    line = line.strip()
    if not line:
        continue
    try:
        o = json.loads(line)
    except json.JSONDecodeError:
        continue
    e = o.get("english", "").strip()
    a = o.get("omni", "").strip()
    if not e or len(a.split(".")) != 6:   # require a clean 6-field address
        continue
    if e in seen:
        continue
    seen.add(e)
    out.append(o)

with open(P, "w") as f:
    for o in out:
        f.write(json.dumps(o) + "\n")
print(f"[dedup] {len(out)} unique 6-field pairs in {P.name}")
