#!/usr/bin/env python3
# typo_augment.py — make the parser robust to REAL messy input.
# Takes clean English->omni pairs and emits MESSY variants. The omni LABEL never changes:
# "navigaet to teh kitchen" still maps to robot.navigate.none.kitchen — that's the whole point.
# No dependency (skipped the lib on purpose — a 40-line injector tuned to YOUR typo style beats a
# heavyweight dep). Keyboard-neighbor swaps, transposition, drop, double, lowercasing, lost
# punctuation/apostrophes — the exact stuff you've been doing to me all night.
import json, os, random

ADJ = {  # qwerty neighbors for fat-finger swaps
  'q':'wa','w':'qes','e':'wrd','r':'etf','t':'rgy','y':'tuh','u':'yij','i':'uok','o':'ipl','p':'ol',
  'a':'qsz','s':'awdz','d':'sefx','f':'drgc','g':'fthv','h':'gyjb','j':'hukn','k':'jilm','l':'kop',
  'z':'asx','x':'zsdc','c':'xdfv','v':'cfgb','b':'vghn','n':'bhjm','m':'njk'}

def _typo_word(w):
    if len(w) < 3 or random.random() > 0.5:        # leave ~half the words clean
        return w
    i = random.randrange(len(w)); c = w[i].lower(); op = random.random()
    if op < 0.30 and c in ADJ:  return w[:i] + random.choice(ADJ[c]) + w[i+1:]  # adjacent key
    if op < 0.55 and i < len(w)-1: return w[:i] + w[i+1] + w[i] + w[i+2:]        # transpose
    if op < 0.75:               return w[:i] + w[i+1:]                           # drop
    return w[:i] + w[i] + w[i:]                                                  # double

def messify(s):
    s = " ".join(_typo_word(w) for w in s.split(" "))
    if random.random() < 0.6: s = s.lower()
    if random.random() < 0.5: s = s.replace("'", "")
    if random.random() < 0.4: s = s.rstrip(".!?")
    if random.random() < 0.2: s = s.replace(",", "")
    return s

if __name__ == "__main__":
    SRC = os.environ.get("AUG_IN",  "omni_phase1_corpus.v3_2.jsonl")
    DST = os.environ.get("AUG_OUT", "omni_phase1_corpus.v3_2.aug.jsonl")
    K   = int(os.environ.get("AUG_K", "2"))   # messy variants per clean pair
    rows = [json.loads(l) for l in open(SRC) if l.strip()]
    out = []
    for r in rows:
        out.append(r)                          # keep the clean original
        seen = {r["english"]}
        for _ in range(K):
            m = messify(r["english"])
            if m and m not in seen:
                seen.add(m); out.append({"english": m, "omni": r["omni"]})
    with open(DST, "w") as f:
        for o in out: f.write(json.dumps(o) + "\n")
    print(f"[aug] {len(rows)} clean -> {len(out)} total ({K} messy each) -> {DST}")
