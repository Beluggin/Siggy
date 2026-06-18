#!/usr/bin/env python3
"""
compare_rich_canon.py — apples-to-apples baseline for the canon-training experiment.

The new model trains on canon verbs and is judged on canon gold (15 verb classes).
The old rich model (62.2%) was judged on rich gold (248 classes) — an unfair, harder
target. To compare honestly we ask: what does the OLD rich model score if we
canon-collapse its predicted verb and judge it against the SAME canon gold?

Same seeded split (SEED=42, 20%) and same canon corpus the new model uses, so the
held-out examples are identical — only the model differs.

Usage: python3 compare_rich_canon.py [rich_model_path]
       default = omniaddress_model_rich_backup
"""
import sys, json, random
from pathlib import Path
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
from canon_verbs import CANON   # synonym -> canonical verb map

CORPUS = Path("omni_phase1_corpus.canon.jsonl")  # gold is already canon-collapsed
SEED, FRAC = 42, 0.20
rich_path = sys.argv[1] if len(sys.argv) > 1 else "omniaddress_model_rich_backup"

pairs = [json.loads(l) for l in open(CORPUS) if l.strip()]
rng = random.Random(SEED); rng.shuffle(pairs)
eval_pairs = pairs[:int(len(pairs) * FRAC)]

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[compare] rich model: {rich_path}  device: {dev}  held-out: {len(eval_pairs)}")
tok = AutoTokenizer.from_pretrained(rich_path)
m = T5ForConditionalGeneration.from_pretrained(rich_path).to(dev).eval()

correct = 0
field_ok = [0, 0, 0, 0, 0]
for it in eval_pairs:
    inp = tok("omniaddress: " + it["english"], return_tensors="pt",
              max_length=96, truncation=True).to(dev)
    with torch.no_grad():
        out = m.generate(**inp, max_new_tokens=24, num_beams=4)
    pred = tok.decode(out[0], skip_special_tokens=True).strip()
    gold = it["omni"].strip()              # already canon
    pf = pred.split(".")
    if len(pf) == 5:                       # collapse the predicted verb to its canonical
        pf[1] = CANON.get(pf[1], pf[1])
    pred_canon = ".".join(pf)
    if pred_canon == gold:
        correct += 1
    gf = gold.split(".")
    for i in range(min(5, len(gf))):
        if i < len(pf) and pf[i] == gf[i]:
            field_ok[i] += 1

n = len(eval_pairs)
labels = ["subject", "verb", "object", "tense", "negator"]
print(f"\n[compare] RICH model, verb canon-collapsed, judged on canon gold:")
print(f"          exact-match: {correct}/{n}  ({100.0*correct/n:.1f}%)")
for lab, fc in zip(labels, field_ok):
    print(f"          {lab:<8} {100.0*fc/n:5.1f}%")
