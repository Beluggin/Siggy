#!/usr/bin/env python3
"""
object_synonymizer.py — propose object-canon groups by embedding proximity.

First working brick of the embedding-addressing layer: encode every distinct
object with all-MiniLM-L6-v2, cluster by cosine distance, and surface tight
synonym groups for Adam to APPROVE. The approved groups freeze into a
deterministic map (like canon_verbs.py) — the model never runs at apply time.

GUARDRAIL: proximity proposes, action-equivalence disposes. Embeddings will
cluster kitchen/bedroom (both rooms) — but the tank navigates to those, so they
stay distinct. Human gates every group by "does the tank care?"

Usage: python3 object_synonymizer.py [corpus.jsonl] [threshold]
"""
import sys, json, collections
from sentence_transformers import SentenceTransformer
import torch

corpus = sys.argv[1] if len(sys.argv) > 1 else "omni_phase1_corpus.v2.jsonl"
THRESH = float(sys.argv[2]) if len(sys.argv) > 2 else 0.72

freq = collections.Counter()
for line in open(corpus):
    line = line.strip()
    if not line: continue
    p = json.loads(line)["omni"].split(".")
    if len(p) == 5: freq[p[2]] += 1
objects = [o for o, _ in freq.most_common()]
print(f"[synonymizer] {len(objects)} distinct objects from {corpus}  (cosine >= {THRESH})")

model = SentenceTransformer("all-MiniLM-L6-v2")
emb = model.encode(objects, normalize_embeddings=True, convert_to_tensor=True)
sim = (emb @ emb.T).cpu()

# union-find over pairs above threshold
parent = list(range(len(objects)))
def find(i):
    while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
    return i
def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb: parent[ra] = rb

pairs = []
for i in range(len(objects)):
    for j in range(i+1, len(objects)):
        s = sim[i, j].item()
        if s >= THRESH:
            union(i, j); pairs.append((s, objects[i], objects[j]))

groups = collections.defaultdict(list)
for i in range(len(objects)): groups[find(i)].append(i)
clusters = [sorted([objects[i] for i in idx], key=lambda o: -freq[o])
            for idx in groups.values() if len(idx) > 1]
clusters.sort(key=lambda c: -sum(freq[o] for o in c))

print(f"\n=== {len(clusters)} proposed groups (review & gate by tank-action-equivalence) ===")
for c in clusters:
    head = c[0]                                   # most frequent = canonical
    members = "  ".join(f"{o}({freq[o]})" for o in c)
    print(f"  [{head}]  {members}")

print(f"\n=== tightest individual pairs (sanity check) ===")
for s, a, b in sorted(pairs, reverse=True)[:20]:
    print(f"  {s:.3f}  {a} ~ {b}")
