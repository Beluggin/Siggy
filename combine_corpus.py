#!/usr/bin/env python3
# Combine frozen v1 + pro v2 into one corpus, dedup on unique english (keep first).
# Writes omni_phase1_corpus.combined.jsonl. v1 and v2 are left untouched (frozen).
import json
seen, out = set(), []
for fn in ("omni_phase1_corpus.jsonl", "omni_phase1_corpus.v2.jsonl"):
    for line in open(fn):
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        e = o.get("english", "").strip()
        a = o.get("omni", "").strip()
        if not e or len(a.split(".")) != 5:
            continue
        if e in seen:
            continue
        seen.add(e)
        out.append(o)
with open("omni_phase1_corpus.combined.jsonl", "w") as f:
    for o in out:
        f.write(json.dumps(o) + "\n")
print(f"[combine] {len(out)} unique pairs -> omni_phase1_corpus.combined.jsonl")
