#!/usr/bin/env python3
"""
eval_functional.py — score the ORIGINAL-trained model (checkpoint-120) on the
ORIGINAL held-out set, but two ways:
  1. strict exact match (current 40% number)
  2. FUNCTIONAL match: canonicalize the verb on BOTH pred and gold before comparing
     (does the tank pick the right ACTION, ignoring synonym choice?)

This tells us whether to canonicalize at INFERENCE time (rich training, blunt
execution) instead of retraining on collapsed labels.
"""
import json, random
from pathlib import Path
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
from canon_verbs import CANON   # synonym -> canonical

MODEL = "omniaddress_model"
CORPUS = Path("omni_phase1_corpus.jsonl")   # ORIGINAL labels
SEED, EVAL_SPLIT, MAXIN, MAXOUT = 42, 0.20, 96, 24

def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]
def split(pairs):
    s = pairs[:]; random.Random(SEED).shuffle(s)
    n = int(len(s)*EVAL_SPLIT); return s[n:], s[:n]
def canon_addr(a):
    p = a.split(".")
    if len(p) == 5: p[1] = CANON.get(p[1], p[1])
    return ".".join(p)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_, ev = split(load(CORPUS))
tok = AutoTokenizer.from_pretrained(MODEL)
m = T5ForConditionalGeneration.from_pretrained(MODEL).to(dev).eval()

strict = func = 0
for it in ev:
    inp = tok("omniaddress: "+it["english"], return_tensors="pt",
              max_length=MAXIN, truncation=True).to(dev)
    with torch.no_grad():
        o = m.generate(**inp, max_new_tokens=MAXOUT, num_beams=4)
    pred = tok.decode(o[0], skip_special_tokens=True).strip()
    gold = it["omni"].strip()
    if pred == gold: strict += 1
    if canon_addr(pred) == canon_addr(gold): func += 1

n = len(ev)
print(f"strict exact match:      {strict}/{n}  ({100*strict/n:.1f}%)")
print(f"functional (verb-canon): {func}/{n}  ({100*func/n:.1f}%)")
