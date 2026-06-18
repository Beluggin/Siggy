#!/usr/bin/env python3
"""
head_to_head.py — fair comparison of the v1 and v2 models on the SAME exam.

Both models scored on v1's seeded held-out set (SEED=42, 20%), minus any example
the v2 model trained on (leakage guard). Same questions, both models — the only
honest way to ask "is v2 actually better, or did it just sit a harder test?"
"""
import json, random
from pathlib import Path
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
from canon_verbs import CANON

SEED, FRAC = 42, 0.20

def split(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rng = random.Random(SEED); sh = rows[:]; rng.shuffle(sh)
    ne = int(len(sh) * FRAC)
    return sh[ne:], sh[:ne]            # train, eval

v1_train, v1_eval = split("omni_phase1_corpus.jsonl")
v2_train, v2_eval = split("omni_phase1_corpus.v2.jsonl")
v2_train_eng = {r["english"] for r in v2_train}

leak  = [r for r in v1_eval if r["english"] in v2_train_eng]
clean = [r for r in v1_eval if r["english"] not in v2_train_eng]
print(f"v1 held-out: {len(v1_eval)}  |  in v2-train (excluded as leak): {len(leak)}  |  clean exam: {len(clean)}")

dev = "cuda" if torch.cuda.is_available() else "cpu"
labels = ["subject", "verb", "object", "tense", "negator"]

def evaluate(model_path, data):
    tok = AutoTokenizer.from_pretrained(model_path)
    m = T5ForConditionalGeneration.from_pretrained(model_path).to(dev).eval()
    ok = cok = 0; fields = [0]*5
    for it in data:
        inp = tok("omniaddress: " + it["english"], return_tensors="pt",
                  max_length=96, truncation=True).to(dev)
        with torch.no_grad():
            out = m.generate(**inp, max_new_tokens=24, num_beams=4)
        pred = tok.decode(out[0], skip_special_tokens=True).strip()
        gold = it["omni"].strip()
        if pred == gold: ok += 1
        pf, gf = pred.split("."), gold.split(".")
        for i in range(min(5, len(gf))):
            if i < len(pf) and pf[i] == gf[i]: fields[i] += 1
        if len(pf) == 5 and len(gf) == 5:        # execution-canon: collapse verb both sides
            p, g = pf[:], gf[:]
            p[1] = CANON.get(pf[1], pf[1]); g[1] = CANON.get(gf[1], gf[1])
            if p == g: cok += 1
    del m
    if dev == "cuda": torch.cuda.empty_cache()
    return ok, cok, fields, len(data)

print(f"\n=== HEAD-TO-HEAD on {len(clean)} clean v1 held-out examples ===")
for name, path in [("v1 model (live)", "omniaddress_model"),
                   ("v2 model       ", "omniaddress_model_v2")]:
    ok, cok, f, n = evaluate(path, clean)
    print(f"\n{name}  [{path}]")
    print(f"  rich exact:  {ok}/{n} = {100.0*ok/n:.1f}%")
    print(f"  canon exact: {cok}/{n} = {100.0*cok/n:.1f}%")
    print("  per-field: " + "  ".join(f"{l} {100.0*fc/n:.1f}" for l, fc in zip(labels, f)))
