#!/usr/bin/env python3
# Dedup the v2 corpus in place: keep first row per unique english, drop malformed.
# (Many englishes -> same omni is fine/good; we only kill duplicate sentences.)
import json
from pathlib import Path

P = Path("omni_phase1_corpus.v2.jsonl")
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
    if not e or len(a.split(".")) != 5:   # require a clean 5-field address
        continue
    if e in seen:
        continue
    seen.add(e)
    out.append(o)

with open(P, "w") as f:
    for o in out:
        f.write(json.dumps(o) + "\n")
print(f"[dedup] {len(out)} unique pairs in {P.name}")
